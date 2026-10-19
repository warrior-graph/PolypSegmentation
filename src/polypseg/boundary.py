"""Mapa de borda do pólipo e força de borda entre superpixels vizinhos.

A borda de um pólipo aparece como um vale escuro entre a cúpula iluminada e a
mucosa ao redor (Bernal et al., 2015, WM-DOVA), mais visível no canal azul, onde
os vasos quase somem. O mapa combina esse vale com o gradiente de cor, e as arestas
do grafo passam a ser cortadas quando a fronteira entre dois superpixels coincide
com ele (princípio dos *intervening contours*, Leung e Malik, 1998).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from skimage import color
from skimage.feature import hessian_matrix

from polypseg.preprocessing import border_mask


def valley_map(channel: np.ndarray, scales: tuple[float, ...] = (2.0, 4.0)) -> np.ndarray:
    """Maior autovalor da Hessiana normalizada por σ², positivo só em vales escuros."""
    g = channel.astype(np.float64)
    out = np.zeros_like(g)
    for sigma in scales:
        hrr, hrc, hcc = hessian_matrix(g, sigma=sigma, mode="reflect", order="rc", use_gaussian_derivatives=False)
        l1 = ((hrr + hcc) / 2 + np.sqrt(((hrr - hcc) / 2) ** 2 + hrc**2)) * sigma**2
        out = np.maximum(out, l1)
    return out


def _normalize(values: np.ndarray, valid: np.ndarray, q: float = 99.0) -> np.ndarray:
    top = np.percentile(values[valid], q) if valid.any() else 1.0
    return np.clip(values / max(top, 1e-8), 0.0, 1.0)


def boundary_map(image: np.ndarray, valley_weight: float = 0.5, extra: np.ndarray | None = None) -> np.ndarray:
    """Evidência de borda em [0, 1]: vale no canal B e gradiente em Lab, zerados na moldura.

    ``extra`` (já em [0, 1]) entra com o mesmo peso do gradiente, para somar outras pistas
    de borda, como a descontinuidade do relevo.
    """
    frame = ndi.binary_dilation(border_mask(image), iterations=4)
    valid = ~frame
    valley = _normalize(valley_map(image[..., 2]), valid)
    lab = color.rgb2lab(image)
    grad = np.zeros(image.shape[:2])
    for k, scale in ((0, 100.0), (1, 128.0), (2, 128.0)):
        smooth = ndi.gaussian_filter(lab[..., k] / scale, 1.5)
        grad += ndi.sobel(smooth, 0) ** 2 + ndi.sobel(smooth, 1) ** 2
    grad = _normalize(np.sqrt(grad), valid)
    if extra is not None:
        grad = np.maximum(grad, extra)
    out = valley_weight * valley + (1 - valley_weight) * grad
    out[frame] = 0.0
    return out


def pair_strength(labels: np.ndarray, bmap: np.ndarray) -> sparse.csr_matrix:
    """Média do mapa de borda ao longo da fronteira de cada par de superpixels vizinhos.

    Cada par de pixels vizinhos-4 com rótulos diferentes contribui com o maior valor dos
    dois pixels. A matriz é simétrica e só tem entradas para superpixels vizinhos.
    """
    n = int(labels.max()) + 1
    keys, vals = [], []
    for a, b, va, vb in (
        (labels[:, :-1], labels[:, 1:], bmap[:, :-1], bmap[:, 1:]),
        (labels[:-1, :], labels[1:, :], bmap[:-1, :], bmap[1:, :]),
    ):
        diff = a != b
        lo = np.minimum(a[diff], b[diff]).astype(np.int64)
        hi = np.maximum(a[diff], b[diff]).astype(np.int64)
        keys.append(lo * n + hi)
        vals.append(np.maximum(va[diff], vb[diff]))
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    uniq, inverse = np.unique(keys, return_inverse=True)
    mean = np.bincount(inverse, weights=vals) / np.bincount(inverse)
    rows, cols = uniq // n, uniq % n
    s = sparse.coo_matrix((mean, (rows, cols)), shape=(n, n)).tocsr()
    return (s + s.T).tocsr()

