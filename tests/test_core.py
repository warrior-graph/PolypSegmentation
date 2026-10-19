import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import color
from skimage.feature import graycomatrix, graycoprops

from polypseg.features import FeatureConfig, compute_features, feature_group, glcm_features, relief_features
from polypseg.metrics import dice, iou, number_of_clicks
from polypseg.preprocessing import inpaint_specular, specular_mask
from polypseg.segmenter import CLASSIFIERS, InteractiveSegmenter, SegmenterConfig, graph_cut, graph_propagate
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
    cut = graph_cut(w, np.array([0, 4]), np.array([1.0, 0.0]))
    assert cut.tolist() == [1, 1, 1, 0, 0]
    # Um prior forte o bastante vence a aresta fraca: o nó 3 passa para o pólipo.
    prior = np.array([0.5, 0.5, 0.5, 1.0, 0.5])
    cut = graph_cut(w, np.array([0, 4]), np.array([1.0, 0.0]), prior=prior, prior_weight=np.array([0, 0, 0, 2.0, 0]))
    assert cut.tolist() == [1, 1, 1, 1, 0]

    from polypseg.segmenter import geodesic_star

    # A aresta fraca 2-3 fica no caminho de 4 até a fonte: o nó 3 não pode passar do nó 2.
    star = geodesic_star(w, np.array([1.0, 0.9, 0.2, 0.8, 0.0]), np.array([0]))
    assert np.allclose(star, [1.0, 0.9, 0.2, 0.2, 0.0])


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
    {"boundary_edges": True, "radial_prior": True, "graph_solver": "cut"},
    {"boundary_edges": True, "radial_prior": True, "geodesic_star": True, "geodesic_prior": 0.1},
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


def test_boundary_connectivity_marks_border_regions_as_background():
    from polypseg.background import boundary_connectivity

    image, gt = _dark_rim_disc(size=128, radius=30)
    labels = compute_superpixels(image, SuperpixelConfig(n_segments=100))
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg = boundary_connectivity(image, labels, border)
    inside = np.bincount(labels.ravel(), weights=gt.ravel()) / np.bincount(labels.ravel()) > 0.5
    assert bg[inside].mean() < 0.2 < 0.5 < bg[~inside].mean()


def test_region_separation_prefers_the_true_disc():
    from polypseg.radial import region_separation

    image, gt = _dark_rim_disc(size=128, radius=30)
    lab = color.rgb2lab(image)
    yy, xx = np.mgrid[:128, :128]
    small = np.hypot(yy - 64, xx - 64) < 12
    big = np.hypot(yy - 64, xx - 64) < 50
    assert region_separation(lab, gt) > max(region_separation(lab, small), region_separation(lab, big))


def test_hemoglobin_index_and_vessel_options_keep_map_in_range():
    from polypseg.boundary import boundary_map, hemoglobin_index

    image, _ = _dark_rim_disc()
    ihb = hemoglobin_index(image)
    assert ihb[48, 48] > 0  # R > G no disco avermelhado
    bmap = boundary_map(image, hemoglobin=True, vessel_weight=1.0)
    assert bmap.min() >= 0 and bmap.max() <= 1


def test_snake_band_only_changes_the_band():
    from polypseg.refine import snake_band

    image, gt = _dark_rim_disc(size=128, radius=30)
    blocky = np.zeros_like(gt)
    blocky[38:90, 38:90] = True
    out = snake_band(image, blocky, [(64, 64, True)], iterations=20, band=8)
    far = ~ndi.binary_dilation(blocky, iterations=8) | ndi.binary_erosion(blocky, iterations=8)
    assert np.array_equal(out[far], blocky[far])


def test_region_term_pulls_star_to_the_color_model():
    from polypseg.radial import color_model, star_contour, star_mask

    image, gt = _dark_rim_disc(size=128, radius=30)
    lab = color.rgb2lab(image)
    yy, xx = np.mgrid[:128, :128]
    fg = np.hypot(yy - 64, xx - 64) < 8
    bg = np.zeros_like(gt)
    bg[:10] = True
    prob = color_model(lab, fg, bg)
    assert prob[gt].mean() > 0.8 and prob[~gt & bg].mean() < 0.2
    # Sem evidência de borda, só o termo de região decide o raio.
    angles, radii = star_contour(np.zeros(gt.shape), (64, 64), region=prob, region_weight=50.0)
    assert dice(star_mask(gt.shape, (64, 64), angles, radii), gt) > 0.9


def test_geodesic_prior_turns_off_after_max_clicks(synthetic):
    image, _ = synthetic
    seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=80), geodesic_max_clicks=1))
    seg.set_image(image)
    calls = []
    original = seg._geodesic_prior
    seg._geodesic_prior = lambda idx, val: calls.append(len(seg.clicks)) or original(idx, val)
    seg.add_click(80, 110, True)
    seg.predict()
    seg.add_click(10, 10, False)
    seg.predict()
    assert calls == [1]


def test_color_model_star_on_first_click():
    image, gt = _dark_rim_disc(size=128, radius=30)
    seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=120), radial_color_model=10.0))
    seg.set_image(image)
    seg.add_click(64, 64, True)
    assert dice(seg.predict().mask, gt) > 0.85


def _two_polyps():
    left, gl = _dark_rim_disc(size=96, radius=22)
    right, gr = _dark_rim_disc(size=96, radius=22)
    return np.concatenate([left, right], axis=1), np.concatenate([gl, gr], axis=1)


def test_box_segmenter_segments_each_box_and_ignores_outside():
    from polypseg.boxes import BoxSegmenter, boxes_from_mask

    image, gt = _two_polyps()
    boxes = boxes_from_mask(gt)
    assert len(boxes) == 2
    seg = BoxSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=120)))
    seg.set_image(image, boxes)
    assert seg.add_click(48, 48, True) == [0]
    assert seg.add_click(48, 144, True) == [1]
    assert seg.add_click(2, 2, False) == []  # fora das caixas: já é fundo
    mask = seg.predict().mask
    assert dice(mask, gt) > 0.85
    outside = np.ones_like(gt)
    for r0, c0, r1, c1 in boxes:
        outside[r0:r1, c0:c1] = False
    assert not mask[outside].any()
    seg.undo()
    assert len(seg.regions[1].segmenter.clicks) == 0


def test_box_click_routing_on_overlap():
    from polypseg.boxes import BoxSegmenter

    image, _ = _two_polyps()
    seg = BoxSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=60)))
    seg.set_image(image, [(20, 20, 76, 110), (20, 86, 76, 170)])  # cruzam nas colunas 86–110
    # Centros nas colunas 65 (largura 90) e 128 (largura 84).
    assert seg.add_click(48, 92, True) == [0]  # 27/90 = 0,30 contra 36/84 = 0,43
    assert seg.add_click(48, 104, True) == [1]  # 39/90 = 0,43 contra 24/84 = 0,29
    assert seg.add_click(48, 98, False) == [0, 1]  # fundo vale para as duas
