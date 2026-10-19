"""Refino da máscara no nível do pixel: faixa estreita em volta da borda e suavização.

A máscara de superpixels fica presa às fronteiras do SLIC (teto de IoU ≈ 0,89 com 400
superpixels). Um random walker por pixel (Grady, 2006) decide só a faixa de ±band px em
torno da borda: os pixels logo dentro e logo fora da faixa viram sementes, junto com os
cliques, e a borda volta a seguir a imagem. Só os pixels da faixa entram no sistema
linear, então o custo cresce com o perímetro e não com a área.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from skimage import color
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour


def refine_band(
    image: np.ndarray,
    mask: np.ndarray,
    clicks: list[tuple[int, int, bool]],
    band: int = 6,
    beta: float = 4.0,
) -> np.ndarray:
    """Máscara refinada por random walker nos pixels a até ``band`` px da borda."""
    from polypseg.segmenter import graph_propagate

    if not mask.any() or mask.all():
        return mask
    grown = ndi.binary_dilation(mask, iterations=band + 1)
    shrunk = ndi.binary_erosion(mask, iterations=band + 1)
    unknown = ndi.binary_dilation(mask, iterations=band) & ~ndi.binary_erosion(mask, iterations=band)
    nodes = grown & ~shrunk  # faixa mais um anel de sementes de cada lado
    seed_val = np.full(mask.shape, -1, dtype=np.int8)
    seed_val[nodes & ~unknown & mask] = 1
    seed_val[nodes & ~unknown & ~mask] = 0
    for row, col, positive in clicks:
        if nodes[row, col]:
            seed_val[row, col] = int(positive)

    index = np.full(mask.shape, -1, dtype=np.int64)
    index[nodes] = np.arange(int(nodes.sum()))
    lab = color.rgb2lab(image) / np.array([100.0, 128.0, 128.0])
    rows, cols, dist2 = [], [], []
    for a, b, la, lb in (
        (index[:, :-1], index[:, 1:], lab[:, :-1], lab[:, 1:]),
        (index[:-1, :], index[1:, :], lab[:-1, :], lab[1:, :]),
    ):
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok])
        cols.append(b[ok])
        dist2.append(((la - lb) ** 2).sum(axis=-1)[ok])
    rows, cols, dist2 = np.concatenate(rows), np.concatenate(cols), np.concatenate(dist2)
    w = np.exp(-beta * dist2 / max(float(dist2.mean()), 1e-12))
    n = int(nodes.sum())
    weights = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()

    flat_seed = seed_val[nodes]
    seed_idx = np.flatnonzero(flat_seed >= 0)
    if not ((flat_seed == 1).any() and (flat_seed == 0).any()):
        return mask
    prob = graph_propagate(weights, seed_idx, flat_seed[seed_idx].astype(float))
    out = mask.copy()
    out[nodes] = prob > 0.5
    return out


def smooth_mask(mask: np.ndarray, sigma: float, clicks: list[tuple[int, int, bool]] = ()) -> np.ndarray:
    """Arredonda a borda suavizando a distância assinada e cortando no nível zero.

    É um passo de fluxo por curvatura média (como no esquema de Merriman, Bence e Osher):
    degraus e reentrâncias com raio menor que ~σ somem, e formas grandes quase não encolhem.
    Os pixels clicados mantêm o rótulo do clique.
    """
    if sigma <= 0 or not mask.any() or mask.all():
        return mask
    signed = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    out = ndi.gaussian_filter(signed, sigma) > 0
    for row, col, positive in clicks:
        out[row, col] = positive
    return out


def snake_band(
    image: np.ndarray,
    mask: np.ndarray,
    clicks: list[tuple[int, int, bool]] = (),
    iterations: int = 20,
    band: int = 8,
    smoothing: int = 1,
) -> np.ndarray:
    """Contorno ativo geodésico morfológico (Márquez-Neila et al., 2014) a partir da máscara.

    O termo de curvatura arredonda o contorno e o de borda o prende ao gradiente. A evolução
    roda num recorte em volta da máscara e só pode mudar pixels a até ``band`` px da borda
    original; os pixels clicados mantêm o rótulo do clique.
    """
    if iterations <= 0 or not mask.any() or mask.all():
        return mask
    rows, cols = np.nonzero(ndi.binary_dilation(mask, iterations=band + 2))
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    gray = color.rgb2gray(image[r0:r1, c0:c1])
    edges = inverse_gaussian_gradient(gray, alpha=100.0, sigma=1.5)
    crop = mask[r0:r1, c0:c1]
    evolved = morphological_geodesic_active_contour(
        edges, iterations, init_level_set=crop.astype(np.int8), smoothing=smoothing, balloon=0, threshold="auto"
    ).astype(bool)
    near = ndi.binary_dilation(crop, iterations=band) & ~ndi.binary_erosion(crop, iterations=band)
    out = mask.copy()
    out[r0:r1, c0:c1] = np.where(near, evolved, crop)
    for row, col, positive in clicks:
        out[row, col] = positive
    return out
