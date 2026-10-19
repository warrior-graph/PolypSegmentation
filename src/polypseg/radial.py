"""Prior de forma em estrela centrado no clique, por transformada polar e programação dinâmica.

O WM-DOVA (Bernal et al., 2015) procura pólipos acumulando, em setores radiais, a evidência
de vale ao redor de cada centro candidato. Aqui o clique do usuário já dá o centro: a
evidência de borda é amostrada em coordenadas polares em torno dele e uma programação
dinâmica circular escolhe o contorno fechado r(θ) que passa pela borda mais forte, com
|r(θ+1) − r(θ)| limitado. O resultado é estrela-convexo em relação ao clique, a mesma
restrição de forma de Veksler (2008) e Gulshan et al. (2010), e serve de prior suave
para o random walker, que cliques posteriores podem contrariar.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon


def star_contour(
    bmap: np.ndarray,
    center: tuple[int, int],
    n_angles: int = 72,
    r_min: float = 4.0,
    r_max: float | None = None,
    max_step: int = 2,
    luminance: np.ndarray | None = None,
    outward_weight: float = 0.0,
    balloon: float = 0.0,
    constraints: list[tuple[int, int, bool]] = (),
    margin: float = 3.0,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ângulos e raios (em pixels) do contorno fechado de maior evidência de borda.

    ``constraints`` são os outros cliques (linha, coluna, positivo): o contorno precisa
    passar a pelo menos ``margin`` px por fora de cada clique de pólipo e por dentro de cada
    clique de fundo, nos ângulos vizinhos ao do clique. É assim que os cliques seguintes
    corrigem a estrela em vez de brigar com ela.

    ``valid`` marca o tecido (fora da moldura preta): o escurecimento na entrada da moldura não
    conta como borda, e cada raio fica mais caro depois de sair do tecido.

    Com ``luminance``, parte da evidência vem do escurecimento ao sair do clique
    (−∂L/∂r > 0), já que a cúpula iluminada do pólipo termina num vale. ``balloon``
    desconta o custo de raios maiores, para o contorno não parar em bordas internas.
    """
    h, w = bmap.shape
    if r_max is None:
        r_max = 0.6 * max(h, w)
    radii = np.arange(r_min, r_max, 1.0)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    rows = center[0] + np.sin(angles)[:, None] * radii[None, :]
    cols = center[1] + np.cos(angles)[:, None] * radii[None, :]
    evidence = ndi.map_coordinates(bmap, [rows, cols], order=1, mode="constant", cval=0.0)
    cost = 1.0 - evidence - balloon * (radii / radii[-1])[None, :]
    if luminance is not None and outward_weight > 0:
        # Com a luz junto da câmera, a cúpula, mais próxima, é mais clara que a mucosa em volta.
        # Escolher a polaridade por imagem piorou o contorno na base de desenvolvimento.
        lum = ndi.map_coordinates(ndi.gaussian_filter(luminance, 2.0), [rows, cols], order=1, mode="nearest")
        drop = np.clip(-np.gradient(lum, axis=1), 0.0, None)
        if valid is not None:
            drop[ndi.map_coordinates(valid.astype(float), [rows, cols], order=0, mode="constant", cval=0.0) < 0.5] = 0.0
        drop = np.clip(drop / max(float(np.percentile(drop, 99)), 1e-8), 0.0, 1.0)
        cost -= outward_weight * (drop - evidence)
    if valid is not None:
        inside = ndi.map_coordinates(valid.astype(float), [rows, cols], order=0, mode="constant", cval=0.0) >= 0.5
        cost += np.cumsum(~inside, axis=1) > 0
    step = 2 * np.pi / n_angles
    for row, col, positive in constraints:
        dist = np.hypot(row - center[0], col - center[1])
        if dist < r_min:
            continue
        theta = np.arctan2(row - center[0], col - center[1]) % (2 * np.pi)
        # Ângulos que o clique cobre: ao menos o vizinho de cada lado, mais se estiver perto.
        width = max(step, np.arctan2(margin, dist))
        near = np.abs((angles - theta + np.pi) % (2 * np.pi) - np.pi) <= width
        if positive:
            cost[np.ix_(near, radii < dist + margin)] += 1e3
        else:
            cost[np.ix_(near, radii > dist - margin)] += 1e3
    path, _ = _circular_dp(cost, max_step)
    return angles, radii[path]


def _circular_dp(cost: np.ndarray, max_step: int) -> tuple[np.ndarray, float]:
    """Caminho fechado de custo mínimo em ``cost`` (A, R), com |Δr| ≤ max_step entre ângulos.

    A DP corre sobre três voltas e devolve a do meio, que praticamente não depende do ponto
    de partida; o custo devolvido é o dessa volta.
    """
    n_angles, n_r = cost.shape
    turns = 3 * n_angles
    acc = cost[0].copy()
    back = np.zeros((turns, n_r), dtype=np.int32)
    offsets = np.arange(-max_step, max_step + 1)
    idx = np.arange(n_r)
    for t in range(1, turns):
        cand = np.full((offsets.size, n_r), np.inf)
        for k, off in enumerate(offsets):
            src = idx + off
            ok = (src >= 0) & (src < n_r)
            cand[k, ok] = acc[src[ok]]
        best = np.argmin(cand, axis=0)
        back[t] = idx + offsets[best]
        acc = cost[t % n_angles] + cand[best, idx]
    path = np.empty(turns, dtype=np.int32)
    path[-1] = int(np.argmin(acc))
    for t in range(turns - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    middle = path[n_angles : 2 * n_angles]
    return middle, float(cost[np.arange(n_angles), middle].sum())


def star_mask(shape: tuple[int, int], center: tuple[int, int], angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    rows = center[0] + np.sin(angles) * radii
    cols = center[1] + np.cos(angles) * radii
    rr, cc = polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask[center] = True
    return mask


def soft_prior(mask: np.ndarray, tau: float = 5.0) -> np.ndarray:
    """Probabilidade por pixel: sigmoide da distância assinada à borda da máscara."""
    signed = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    return 1.0 / (1.0 + np.exp(-signed / tau))
