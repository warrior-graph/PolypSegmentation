"""Pré-processamento de quadros de colonoscopia."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, io, util


def load_image(path: str) -> np.ndarray:
    """Carrega uma imagem como RGB float em [0, 1]."""
    image = io.imread(path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    return util.img_as_float(image).astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    """Carrega uma máscara binária (pixels claros = pólipo)."""
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=-1)
    return mask > (mask.max() / 2 if mask.max() > 0 else 0)


def specular_mask(image: np.ndarray, v_thresh: float = 0.9, s_thresh: float = 0.25) -> np.ndarray:
    """Reflexos especulares: pixels muito claros e pouco saturados."""
    hsv = color.rgb2hsv(image)
    mask = (hsv[..., 2] > v_thresh) & (hsv[..., 1] < s_thresh)
    return ndi.binary_dilation(mask, iterations=2)


def border_mask(image: np.ndarray, thresh: float = 0.08) -> np.ndarray:
    """Moldura preta do endoscópio: regiões escuras conectadas à borda da imagem."""
    dark = image.max(axis=-1) < thresh
    labels, _ = ndi.label(dark)
    edge_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    edge_labels = edge_labels[edge_labels > 0]
    return np.isin(labels, edge_labels)


def inpaint_specular(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Substitui reflexos pela cor do pixel válido mais próximo."""
    if mask is None:
        mask = specular_mask(image)
    if not mask.any() or mask.all():
        return image
    _, (rows, cols) = ndi.distance_transform_edt(mask, return_indices=True)
    return image[rows, cols]


def preprocess(image: np.ndarray, remove_specular: bool = True) -> np.ndarray:
    """Pré-processamento padrão aplicado antes dos superpixels."""
    if remove_specular:
        image = inpaint_specular(image)
    return image
