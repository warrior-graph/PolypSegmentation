"""Probabilidade de fundo por conectividade com a borda (Zhu et al., CVPR 2014).

Uma região é fundo quando a área a que ela pertence toca muito a borda da imagem. Sobre o
grafo de superpixels, com distâncias geodésicas de cor d(p, q):

    Area(p)    = Σ_q exp(−d²(p, q) / 2σ²)
    LenBnd(p)  = Σ_{q na borda} exp(−d²(p, q) / 2σ²)
    BndCon(p)  = LenBnd(p) / √Area(p)
    P_fundo(p) = 1 − exp(−BndCon²(p) / 2σ_b²)

A "borda" aqui é a mesma da regra atual de fundo: superpixels na borda da imagem ou
encostados na moldura preta. Isso generaliza a regra de rótulo fixo para um prior suave
em todos os superpixels.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from skimage import color

from polypseg.superpixels import adjacency


def boundary_connectivity(
    image: np.ndarray,
    labels: np.ndarray,
    border: np.ndarray,
    sigma_color: float = 10.0,
    sigma_bnd: float = 1.0,
) -> np.ndarray:
    """P(fundo) por superpixel, em [0, 1]; ``border`` são os índices dos superpixels de borda."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    lab = color.rgb2lab(image).reshape(-1, 3)
    mean = np.stack([np.bincount(flat, weights=lab[:, k], minlength=n) / counts for k in range(3)], axis=1)

    adj = adjacency(labels).tocoo()
    dist = np.linalg.norm(mean[adj.row] - mean[adj.col], axis=1)
    graph = sparse.coo_matrix((np.maximum(dist, 1e-6), (adj.row, adj.col)), shape=(n, n)).tocsr()
    geo = dijkstra(graph, directed=False)
    geo[~np.isfinite(geo)] = 1e6
    sim = np.exp(-(geo**2) / (2 * sigma_color**2))

    area = sim.sum(axis=1)
    on_border = np.zeros(n, dtype=bool)
    on_border[np.asarray(border, dtype=int)] = True
    length = sim[:, on_border].sum(axis=1)
    bndcon = length / np.sqrt(area)
    return 1.0 - np.exp(-(bndcon**2) / (2 * sigma_bnd**2))
