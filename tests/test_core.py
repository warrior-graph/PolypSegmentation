import numpy as np
import pytest
from skimage.feature import graycomatrix, graycoprops

from polypseg.features import FeatureConfig, compute_features, glcm_features
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
    seg = InteractiveSegmenter(SegmenterConfig(classifier=classifier, superpixels=SuperpixelConfig(n_segments=120)))
    seg.set_image(image)
    assert seg.predict().mask.sum() == 0  # sem cliques, sem máscara
    seg.add_click(80, 110, True)
    seg.add_click(20, 20, False)
    result = seg.predict()
    assert result.probability.shape == (seg.n_superpixels,)
    assert dice(result.mask, gt) > 0.85


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
