import numpy as np
import pytest


@pytest.fixture
def synthetic():
    """Imagem 160x200 com um "pólipo" circular avermelhado e texturizado sobre fundo rosado."""
    rng = np.random.default_rng(0)
    h, w = 160, 200
    rr, cc = np.mgrid[:h, :w]
    gt = (rr - 80) ** 2 + (cc - 110) ** 2 < 35**2
    image = np.empty((h, w, 3), dtype=np.float32)
    image[:] = (0.85, 0.55, 0.50)
    image[gt] = (0.65, 0.25, 0.22)
    stripes = 0.05 * np.sin(cc / 2.0)[..., None]
    image += np.where(gt[..., None], stripes, 0.0)
    image += rng.normal(0, 0.02, image.shape)
    return np.clip(image, 0, 1).astype(np.float32), gt
