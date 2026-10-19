"""Geração de superpixels e grafo de adjacência entre eles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from skimage import color, segmentation
from skimage.filters import sobel

METHODS = ("slic", "felzenszwalb", "watershed")


@dataclass
class SuperpixelConfig:
    method: str = "slic"
    n_segments: int = 200
    compactness: float = 10.0
    sigma: float = 1.0


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Rótulos contíguos 0..K-1."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def compute_superpixels(image: np.ndarray, config: SuperpixelConfig | None = None) -> np.ndarray:
    """Retorna um mapa de rótulos (H, W) com valores 0..K-1."""
    config = config or SuperpixelConfig()
    if config.method == "slic":
        labels = segmentation.slic(
            image,
            n_segments=config.n_segments,
            compactness=config.compactness,
            sigma=config.sigma,
            start_label=0,
            convert2lab=True,
        )
    elif config.method == "felzenszwalb":
        # A escala controla o tamanho: ajusta para aproximar n_segments.
        area = image.shape[0] * image.shape[1]
        scale = max(area / max(config.n_segments, 1) / 4.0, 10.0)
        labels = segmentation.felzenszwalb(
            image, scale=scale, sigma=config.sigma, min_size=int(scale / 4)
        )
    elif config.method == "watershed":
        gradient = sobel(color.rgb2gray(image))
        labels = segmentation.watershed(
            gradient, markers=config.n_segments, compactness=config.compactness / 1000.0
        )
    else:
        raise ValueError(f"método de superpixel desconhecido: {config.method!r}")
    return _relabel(labels)


def adjacency(labels: np.ndarray) -> sparse.csr_matrix:
    """Matriz de adjacência (K, K) dos superpixels com vizinhança-4."""
    n = int(labels.max()) + 1
    pairs = [
        (labels[:, :-1].ravel(), labels[:, 1:].ravel()),
        (labels[:-1, :].ravel(), labels[1:, :].ravel()),
    ]
    rows, cols = [], []
    for a, b in pairs:
        diff = a != b
        rows.append(a[diff])
        cols.append(b[diff])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.size, dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj = adj + adj.T
    adj.data[:] = 1.0
    return adj


def boundaries(labels: np.ndarray) -> np.ndarray:
    """Máscara booleana das fronteiras entre superpixels."""
    return segmentation.find_boundaries(labels, mode="inner")
