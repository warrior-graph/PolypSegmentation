"""Métricas de segmentação."""

from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def number_of_clicks(ious: list[float], threshold: float, max_clicks: int) -> int:
    """NoC@thr: primeiro número de cliques em que o IoU atinge o limiar (ou max_clicks)."""
    for k, value in enumerate(ious, start=1):
        if value >= threshold:
            return k
    return max_clicks
