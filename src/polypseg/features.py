"""Descritores de cor e textura por superpixel.

Todas as agregações são vetorizadas com ``np.bincount`` sobre o mapa de rótulos,
então o custo é praticamente linear no número de pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import color
from skimage.feature import hessian_matrix, local_binary_pattern
from skimage.filters import gabor_kernel


@dataclass
class FeatureConfig:
    color: bool = True
    lbp: bool = True
    glcm: bool = True
    gabor: bool = True
    relief: bool = False
    lbp_radii: tuple[int, ...] = (1, 2)
    glcm_levels: int = 16
    gabor_frequencies: tuple[float, ...] = (0.1, 0.25)
    gabor_orientations: int = 4
    relief_scales: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    relief_surround: float = 4.0


TEXTURE_GROUPS = ("lbp", "glcm", "gabor", "relief")


def feature_group(name: str) -> str:
    """Grupo de um descritor pelo prefixo do nome; o que não é textura nem relevo é cor."""
    prefix = name.split("_", 1)[0]
    return prefix if prefix in TEXTURE_GROUPS else "color"


def _mean_std(values: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = labels.ravel()
    v = values.ravel().astype(np.float64)
    n = counts.size
    s1 = np.bincount(flat, weights=v, minlength=n)
    s2 = np.bincount(flat, weights=v * v, minlength=n)
    mean = s1 / counts
    var = np.maximum(s2 / counts - mean**2, 0.0)
    return mean, np.sqrt(var)


def color_features(image: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Média e desvio em RGB, Lab e HSV (matiz como cos/sen, por ser circular)."""
    lab = color.rgb2lab(image)
    hsv = color.rgb2hsv(image)
    hue = hsv[..., 0] * 2 * np.pi
    channels = {
        "R": image[..., 0], "G": image[..., 1], "B": image[..., 2],
        "L": lab[..., 0] / 100.0, "a": lab[..., 1] / 128.0, "b": lab[..., 2] / 128.0,
        "S": hsv[..., 1], "V": hsv[..., 2],
    }
    cols, names = [], []
    for name, values in channels.items():
        mean, std = _mean_std(values, labels, counts)
        cols += [mean, std]
        names += [f"{name}_mean", f"{name}_std"]
    for name, values in (("hue_cos", np.cos(hue)), ("hue_sin", np.sin(hue))):
        mean, _ = _mean_std(values, labels, counts)
        cols.append(mean)
        names.append(name)
    return np.stack(cols, axis=1), names


def lbp_features(gray: np.ndarray, labels: np.ndarray, counts: np.ndarray, radii: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    """Histogramas normalizados de LBP uniforme invariante à rotação."""
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    n = counts.size
    cols, names = [], []
    for radius in radii:
        points = 8 * radius
        codes = local_binary_pattern(gray_u8, points, radius, method="uniform").astype(np.int64)
        bins = points + 2
        hist = np.bincount(labels.ravel() * bins + codes.ravel(), minlength=n * bins)
        hist = hist.reshape(n, bins) / counts[:, None]
        cols.append(hist)
        names += [f"lbp_r{radius}_{i}" for i in range(bins)]
    return np.concatenate(cols, axis=1), names


def glcm_features(gray: np.ndarray, labels: np.ndarray, levels: int) -> tuple[np.ndarray, list[str]]:
    """Propriedades de Haralick com GLCM restrita aos pares dentro de cada superpixel.

    Cada GLCM conta só pares de pixels vizinhos em que os dois pertencem ao mesmo
    superpixel, em 4 direções (0°, 45°, 90°, 135°), simétrica e normalizada.
    """
    n = int(labels.max()) + 1
    q = np.minimum((np.clip(gray, 0, 1) * levels).astype(np.int64), levels - 1)
    h, w = q.shape
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    glcm = np.zeros(n * levels * levels, dtype=np.float64)
    for dr, dc in offsets:
        r0, r1 = 0, h - dr
        c0, c1 = max(0, -dc), w - max(0, dc)
        a_q = q[r0:r1, c0:c1]
        b_q = q[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        a_l = labels[r0:r1, c0:c1]
        b_l = labels[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        same = a_l == b_l
        lab = a_l[same]
        i, j = a_q[same], b_q[same]
        base = lab * levels * levels
        glcm += np.bincount(base + i * levels + j, minlength=glcm.size)
        glcm += np.bincount(base + j * levels + i, minlength=glcm.size)
    glcm = glcm.reshape(n, levels, levels)
    total = glcm.sum(axis=(1, 2), keepdims=True)
    p = glcm / np.maximum(total, 1.0)

    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    diff = (i - j).astype(np.float64)
    contrast = (p * diff**2).sum(axis=(1, 2))
    dissimilarity = (p * np.abs(diff)).sum(axis=(1, 2))
    homogeneity = (p / (1.0 + diff**2)).sum(axis=(1, 2))
    energy = np.sqrt((p**2).sum(axis=(1, 2)))
    mu_i = (p * i).sum(axis=(1, 2))
    mu_j = (p * j).sum(axis=(1, 2))
    sd_i = np.sqrt((p * (i - mu_i[:, None, None]) ** 2).sum(axis=(1, 2)))
    sd_j = np.sqrt((p * (j - mu_j[:, None, None]) ** 2).sum(axis=(1, 2)))
    cov = (p * (i - mu_i[:, None, None]) * (j - mu_j[:, None, None])).sum(axis=(1, 2))
    denom = sd_i * sd_j
    correlation = np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 1.0)
    entropy = -(p * np.log2(p + 1e-12)).sum(axis=(1, 2))
    feats = np.stack([contrast, dissimilarity, homogeneity, energy, correlation, entropy], axis=1)
    names = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_entropy"]
    return feats, names


def gabor_features(
    gray: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    frequencies: tuple[float, ...],
    orientations: int,
) -> tuple[np.ndarray, list[str]]:
    """Média e desvio da magnitude da resposta a um banco de filtros de Gabor."""
    cols, names = [], []
    centered = gray - gray.mean()
    for freq in frequencies:
        for k in range(orientations):
            theta = np.pi * k / orientations
            kernel = gabor_kernel(freq, theta=theta)
            response = np.abs(fftconvolve(centered, kernel, mode="same"))
            mean, std = _mean_std(response, labels, counts)
            cols += [mean, std]
            deg = int(round(np.degrees(theta)))
            names += [f"gabor_f{freq}_t{deg}_mean", f"gabor_f{freq}_t{deg}_std"]
    return np.stack(cols, axis=1), names


def relief_features(
    gray: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    scales: tuple[float, ...],
    surround: float,
) -> tuple[np.ndarray, list[str]]:
    """Relevo aproximado pela luminância, já que a luz do endoscópio sai junto da câmera.

    Por escala σ, com a Hessiana normalizada por σ² e autovalores λ1 ≥ λ2:
    - shape index (2/π)·atan2(λ1 + λ2, λ1 − λ2): −1 é cúpula clara, +1 é bacia escura;
    - curvedness √((λ1² + λ2²)/2): quanto de curvatura há;
    - cúpula max(−λ1, 0): positiva só quando os dois autovalores são negativos (blob claro);
    - centro-entorno log(G_σ ∗ I / G_kσ ∗ I): brilho local relativo à vizinhança.
    """
    g = gray.astype(np.float64)
    cols, names = [], []
    for sigma in scales:
        hrr, hrc, hcc = hessian_matrix(g, sigma=sigma, mode="reflect", order="rc", use_gaussian_derivatives=False)
        half_tr = (hrr + hcc) / 2
        root = np.sqrt(((hrr - hcc) / 2) ** 2 + hrc**2)
        l1 = (half_tr + root) * sigma**2
        l2 = (half_tr - root) * sigma**2
        shape = (2 / np.pi) * np.arctan2(l1 + l2, l1 - l2)
        curvedness = np.sqrt((l1**2 + l2**2) / 2)
        dome = np.maximum(-l1, 0.0)
        s_mean, s_std = _mean_std(shape, labels, counts)
        c_mean, _ = _mean_std(curvedness, labels, counts)
        d_mean, _ = _mean_std(dome, labels, counts)
        cols += [s_mean, s_std, c_mean, d_mean]
        names += [f"relief_s{sigma:g}_{k}" for k in ("shape_mean", "shape_std", "curvedness", "dome")]
        center = ndi.gaussian_filter(g, sigma, mode="reflect")
        around = ndi.gaussian_filter(g, sigma * surround, mode="reflect")
        cs_mean, _ = _mean_std(np.log((center + 1e-3) / (around + 1e-3)), labels, counts)
        cols.append(cs_mean)
        names.append(f"relief_s{sigma:g}_center_surround")
    return np.stack(cols, axis=1), names


def compute_features(image: np.ndarray, labels: np.ndarray, config: FeatureConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Matriz (K, D) de descritores e a lista de nomes das D colunas."""
    config = config or FeatureConfig()
    n = int(labels.max()) + 1
    counts = np.maximum(np.bincount(labels.ravel(), minlength=n).astype(np.float64), 1.0)
    gray = color.rgb2gray(image)
    blocks, names = [], []
    if config.color:
        f, nm = color_features(image, labels, counts)
        blocks.append(f)
        names += nm
    if config.lbp:
        f, nm = lbp_features(gray, labels, counts, config.lbp_radii)
        blocks.append(f)
        names += nm
    if config.glcm:
        f, nm = glcm_features(gray, labels, config.glcm_levels)
        blocks.append(f)
        names += nm
    if config.gabor:
        f, nm = gabor_features(gray, labels, counts, config.gabor_frequencies, config.gabor_orientations)
        blocks.append(f)
        names += nm
    if config.relief:
        f, nm = relief_features(gray, labels, counts, config.relief_scales, config.relief_surround)
        blocks.append(f)
        names += nm
    if not blocks:
        raise ValueError("ative ao menos um grupo de descritores")
    return np.concatenate(blocks, axis=1).astype(np.float32), names
