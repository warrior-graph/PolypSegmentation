import numpy as np
import pytest
from skimage.feature import graycomatrix, graycoprops

from polypseg.features import FeatureConfig, compute_features, feature_group, glcm_features, relief_features
from polypseg.metrics import dice, iou, number_of_clicks
from polypseg.preprocessing import inpaint_specular, specular_mask
from polypseg.segmenter import CLASSIFIERS, InteractiveSegmenter, SegmenterConfig, graph_propagate
from polypseg.simulation import next_click, simulate
from polypseg.superpixels import METHODS, SuperpixelConfig, adjacency, compute_superpixels


def test_metrics():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    assert dice(a, b) == 1.0 and iou(a, b) == 1.0
    a[:2] = True
    b[1:3] = True
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(1 / 3)
    assert number_of_clicks([0.5, 0.86, 0.95], 0.85, 10) == 2
    assert number_of_clicks([0.5, 0.6], 0.9, 10) == 10


@pytest.mark.parametrize("method", METHODS)
def test_superpixels_contiguous(synthetic, method):
    image, _ = synthetic
    labels = compute_superpixels(image, SuperpixelConfig(method=method, n_segments=80))
    assert labels.shape == image.shape[:2]
    assert labels.min() == 0
    assert np.array_equal(np.unique(labels), np.arange(labels.max() + 1))
    adj = adjacency(labels)
    assert adj.shape[0] == labels.max() + 1
    assert (adj != adj.T).nnz == 0


def test_features_shape_and_finite(synthetic):
    image, _ = synthetic
    labels = compute_superpixels(image, SuperpixelConfig(n_segments=60))
    feats, names = compute_features(image, labels)
    assert feats.shape == (labels.max() + 1, len(names))
    assert np.isfinite(feats).all()
    only_color, names_c = compute_features(image, labels, FeatureConfig(lbp=False, glcm=False, gabor=False))
    assert only_color.shape[1] == len(names_c) < feats.shape[1]
    with_relief, names_r = compute_features(image, labels, FeatureConfig(relief=True))
    assert with_relief.shape[1] == len(names_r) == feats.shape[1] + 20
    assert np.isfinite(with_relief).all()


def test_feature_groups_cover_all_names(synthetic):
    image, _ = synthetic
    labels = compute_superpixels(image, SuperpixelConfig(n_segments=60))
    _, names = compute_features(image, labels, FeatureConfig(relief=True))
    groups = [feature_group(n) for n in names]
    assert set(groups) == {"color", "lbp", "glcm", "gabor", "relief"}
    assert groups.count("color") == 18


def test_relief_marks_bright_dome():
    """Uma cúpula clara deve ter shape index perto de −1 e cúpula positiva só no centro."""
    yy, xx = np.mgrid[:64, :64]
    gray = 0.3 + 0.5 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 6.0**2))
    labels = np.where((yy - 32) ** 2 + (xx - 32) ** 2 < 16, 0, 1).astype(np.int32)
    counts = np.bincount(labels.ravel()).astype(float)
    feats, names = relief_features(gray, labels, counts, (4.0,), 4.0)
    col = {n: feats[:, i] for i, n in enumerate(names)}
    assert col["relief_s4_shape_mean"][0] < -0.9
    assert col["relief_s4_dome"][0] > 5 * col["relief_s4_dome"][1]
    assert col["relief_s4_center_surround"][0] > col["relief_s4_center_surround"][1]


def test_masked_glcm_matches_skimage_for_single_region():
    """Com um único superpixel, a GLCM restrita deve coincidir com a do scikit-image."""
    rng = np.random.default_rng(1)
    gray = rng.random((40, 50))
    labels = np.zeros(gray.shape, dtype=np.int32)
    levels = 16
    feats, names = glcm_features(gray, labels, levels)
    q = np.minimum((gray * levels).astype(np.uint8), levels - 1)
    ref = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels, symmetric=True, normed=False)
    p = ref.sum(axis=(2, 3)).astype(float)
    p /= p.sum()
    pooled = p[:, :, None, None]
    for prop in ("contrast", "homogeneity", "dissimilarity"):
        expected = graycoprops(pooled, prop)[0, 0]
        assert feats[0, names.index(f"glcm_{prop}")] == pytest.approx(expected, rel=1e-6)


def test_specular_inpaint():
    image = np.full((20, 20, 3), 0.5, dtype=np.float32)
    image[8:12, 8:12] = 1.0
    mask = specular_mask(image)
    assert mask[10, 10]
    assert np.allclose(inpaint_specular(image, mask)[10, 10], 0.5)


def test_graph_propagate_respects_seeds():
    from scipy import sparse

    # cadeia 0-1-2-3-4 com aresta fraca entre 2 e 3
    w = sparse.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0.01, 0],
        [0, 0, 0.01, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=float))
    x = graph_propagate(w, np.array([0, 4]), np.array([1.0, 0.0]))
    assert x[0] == 1.0 and x[4] == 0.0
    assert x[1] > 0.5 and x[2] > 0.5 and x[3] < 0.5


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_segmenter_segments_synthetic_polyp(synthetic, classifier):
    image, gt = synthetic
    # Sem a estrela: aqui o pólipo é mais escuro que a mucosa, o contrário do que ela supõe
    # (ver test_modules_segment_synthetic_polyp, que usa uma cúpula clara).
    seg = InteractiveSegmenter(
        SegmenterConfig(classifier=classifier, superpixels=SuperpixelConfig(n_segments=120), radial_prior=False)
    )
    seg.set_image(image)
    assert seg.predict().mask.sum() == 0  # sem cliques, sem máscara
    seg.add_click(80, 110, True)
    seg.add_click(20, 20, False)
    result = seg.predict()
    assert result.probability.shape == (seg.n_superpixels,)
    assert dice(result.mask, gt) > 0.85


def test_balanced_groups_segment_synthetic_polyp(synthetic):
    image, gt = synthetic
    config = SegmenterConfig(
        superpixels=SuperpixelConfig(n_segments=120), features=FeatureConfig(relief=True), balance_groups=True,
        radial_prior=False,  # pólipo escuro; ver test_segmenter_segments_synthetic_polyp
    )
    seg = InteractiveSegmenter(config)
    seg.set_image(image)
    weights = seg._column_weights()
    color_cols = [i for i, n in enumerate(seg.feature_names) if feature_group(n) == "color"]
    assert np.isclose(weights[color_cols].sum(), 1.0) and np.isclose(weights.sum(), 5.0)
    seg.add_click(80, 110, True)
    seg.add_click(20, 20, False)
    assert dice(seg.predict().mask, gt) > 0.85


def test_undo_and_clear(synthetic):
    image, _ = synthetic
    seg = InteractiveSegmenter()
    seg.set_image(image)
    seg.add_click(80, 110, True)
    seg.add_click(10, 10, False)
    seg.undo()
    assert len(seg.clicks) == 1
    seg.clear()
    assert seg.clicks == []


def test_last_click_on_superpixel_wins(synthetic):
    image, _ = synthetic
    seg = InteractiveSegmenter()
    seg.set_image(image)
    seg.add_click(80, 110, True)
    seg.add_click(80, 110, False)
    idx, val = seg.seeds()
    assert len(idx) == 1 and val[0] == 0


def test_next_click_targets_largest_error():
    gt = np.zeros((50, 50), bool)
    gt[10:40, 10:40] = True
    pred = np.zeros_like(gt)
    row, col, positive = next_click(pred, gt)
    assert positive and gt[row, col] and abs(row - 24.5) <= 1 and abs(col - 24.5) <= 1
    assert next_click(gt, gt) is None
    pred = gt.copy()
    pred[0:5, :] = True
    pred[10:40, 10:40] = True
    _, _, positive = next_click(pred, gt)
    assert not positive


def test_simulation_improves_with_clicks(synthetic):
    image, gt = synthetic
    seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=120)))
    seg.set_image(image)
    curve = simulate(seg, gt, max_clicks=5)
    assert len(curve.dice) == 5
    assert max(curve.dice) > 0.85


def test_border_includes_tissue_next_to_black_frame(synthetic):
    image, _ = synthetic
    framed = image.copy()
    framed[:, :20] = 0.0  # moldura preta à esquerda
    seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=120)))
    seg.set_image(framed)
    assert seg.labels[80, 22] in seg.border_superpixels  # tecido logo depois da moldura
    assert seg.labels[80, 110] not in seg.border_superpixels  # centro do pólipo


def _dark_rim_disc(size: int = 96, radius: int = 22):
    """Disco claro com anel escuro em volta, sobre mucosa média: um pólipo idealizado."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    gray = np.where(r < radius, 0.8, 0.5) - 0.3 * np.exp(-((r - radius) ** 2) / 4.0)
    image = np.stack([gray, gray * 0.6, gray * 0.5], axis=-1).astype(np.float32)
    return image, r < radius


def test_pair_strength_is_high_across_the_rim():
    from polypseg.boundary import boundary_map, pair_strength

    image, gt = _dark_rim_disc()
    labels = compute_superpixels(image, SuperpixelConfig(n_segments=80))
    strength = pair_strength(labels, boundary_map(image)).tocoo()
    inside = np.bincount(labels.ravel(), weights=gt.ravel()) / np.bincount(labels.ravel()) > 0.5
    crossing = inside[strength.row] != inside[strength.col]
    assert crossing.any() and (~crossing).any()
    assert strength.data[crossing].mean() > 2 * strength.data[~crossing].mean()


def test_star_contour_recovers_disc_radius():
    from polypseg.boundary import boundary_map
    from polypseg.radial import star_contour, star_mask

    image, gt = _dark_rim_disc()
    center = (48, 48)
    angles, radii = star_contour(boundary_map(image), center, luminance=image.mean(axis=-1), outward_weight=0.8)
    assert np.abs(np.median(radii) - 22) <= 3
    assert dice(star_mask(gt.shape, center, angles, radii), gt) > 0.9


def test_refine_band_fixes_blocky_mask():
    from polypseg.refine import refine_band

    image, gt = _dark_rim_disc()
    blocky = np.zeros_like(gt)
    blocky[30:66, 30:66] = True  # quadrado: erra os cantos e as laterais do disco
    refined = refine_band(image, blocky, [(48, 48, True)], band=8)
    assert dice(refined, gt) > dice(blocky, gt) + 0.03


def test_local_relief_is_positive_on_bright_dome():
    from polypseg.relief import local_relief, pseudo_elevation

    yy, xx = np.mgrid[:96, :96]
    dome = 0.3 + 0.4 * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 8.0**2))
    image = np.stack([dome, dome, dome], axis=-1)
    lrm = local_relief(pseudo_elevation(image))
    assert lrm[48, 48] > 0 > lrm[48, 75]


@pytest.mark.parametrize("flags", [
    {"boundary_edges": True},
    {"radial_prior": True},
    {"pixel_refine": True},
    {"boundary_edges": True, "radial_prior": True, "pixel_refine": True, "relief_map": True},
])
def test_modules_segment_synthetic_polyp(flags):
    # Cúpula clara com borda escura: o prior radial supõe essa polaridade (luz junto da câmera).
    image, gt = _dark_rim_disc(size=128, radius=30)
    seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=120), **flags))
    seg.set_image(image)
    seg.add_click(64, 64, True)
    seg.add_click(8, 8, False)
    assert dice(seg.predict().mask, gt) > 0.85


def test_smooth_mask_rounds_a_jagged_disc():
    from polypseg.refine import smooth_mask

    _, gt = _dark_rim_disc(size=128, radius=30)
    jagged = gt.copy()
    jagged[::6] = False  # cortes finos atravessando o disco
    jagged[60:64, 90:100] = True  # espinho para fora
    smooth = smooth_mask(jagged, 3.0, [(64, 64, True)])
    assert dice(smooth, gt) > dice(jagged, gt) + 0.03
    assert smooth[64, 64]
    assert smooth_mask(jagged, 0.0) is jagged
