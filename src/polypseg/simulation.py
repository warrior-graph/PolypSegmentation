"""Simulação de cliques para avaliar a segmentação interativa sem usuário.

Segue o protocolo usual da literatura de segmentação interativa: cada novo clique
cai no ponto mais interno da maior região de erro (falso negativo vira clique de
pólipo, falso positivo vira clique de fundo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from polypseg.metrics import dice, iou
from polypseg.segmenter import InteractiveSegmenter


def _interior_point(region: np.ndarray) -> tuple[int, float, tuple[int, int]]:
    """Ponto mais distante da borda da região (bordas da imagem contam como borda)."""
    padded = np.pad(region, 1, constant_values=False)
    dist = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    flat = int(np.argmax(dist))
    return flat, float(dist.ravel()[flat]), np.unravel_index(flat, region.shape)


def next_click(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, bool] | None:
    """Próximo clique simulado: (linha, coluna, positivo) ou None se não houver erro."""
    fn = gt & ~pred
    fp = pred & ~gt
    if not fn.any() and not fp.any():
        return None
    _, d_fn, p_fn = _interior_point(fn) if fn.any() else (0, -1.0, (0, 0))
    _, d_fp, p_fp = _interior_point(fp) if fp.any() else (0, -1.0, (0, 0))
    if d_fn >= d_fp:
        return int(p_fn[0]), int(p_fn[1]), True
    return int(p_fp[0]), int(p_fp[1]), False


@dataclass
class ClickCurve:
    dice: list[float]
    iou: list[float]


def simulate(segmenter: InteractiveSegmenter, gt: np.ndarray, max_clicks: int = 10) -> ClickCurve:
    """Roda até max_clicks cliques simulados e devolve Dice e IoU após cada um."""
    gt = gt.astype(bool)
    segmenter.clear()
    pred = np.zeros_like(gt)
    dices, ious = [], []
    for _ in range(max_clicks):
        click = next_click(pred, gt)
        if click is None:
            dices.append(dices[-1] if dices else 1.0)
            ious.append(ious[-1] if ious else 1.0)
            continue
        segmenter.add_click(*click)
        pred = segmenter.predict().mask
        dices.append(dice(pred, gt))
        ious.append(iou(pred, gt))
    return ClickCurve(dices, ious)
