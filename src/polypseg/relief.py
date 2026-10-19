"""Pseudo-elevação por sombreamento coaxial e modelo de relevo local.

No endoscópio a luz sai praticamente do centro de projeção, então o brilho cai com o
quadrado da distância, I ≈ ρ·cosθ / r² (Okatani e Deguchi, 1997), e r ∝ 1/√I dá uma
profundidade relativa. O canal R é o menos absorvido pela hemoglobina, logo o que menos
mistura vasos ao sombreamento. O *Local Relief Model* da arqueologia com LiDAR (Hesse,
2010) subtrai a tendência regional (o tubo do cólon, o lúmen, as pregas largas) e deixa
só o microrrelevo, em que o pólipo aparece como uma elevação.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def pseudo_elevation(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Elevação relativa −1/√R (maior = mais perto da câmera)."""
    red = ndi.gaussian_filter(image[..., 0].astype(np.float64), sigma)
    return -1.0 / np.sqrt(np.maximum(red, 1e-3))


def local_relief(elevation: np.ndarray, small: float = 2.0, large: float = 24.0) -> np.ndarray:
    """Elevação suavizada menos a tendência regional; positivo em protrusões."""
    return ndi.gaussian_filter(elevation, small) - ndi.gaussian_filter(elevation, large)


def relief_edges(image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """|∇ relevo local| normalizado em [0, 1]: a descontinuidade do relevo na borda da cúpula."""
    lrm = local_relief(pseudo_elevation(image))
    mag = np.hypot(ndi.sobel(lrm, 0), ndi.sobel(lrm, 1))
    ref = mag[valid] if valid is not None and valid.any() else mag
    return np.clip(mag / max(float(np.percentile(ref, 99)), 1e-8), 0.0, 1.0)
