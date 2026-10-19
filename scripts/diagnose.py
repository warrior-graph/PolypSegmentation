"""Tetos e decomposição do erro do segmentador, para decidir onde vale investir.

- teto de superpixels: Dice ao rotular cada superpixel pela maioria do gabarito;
- teto de arestas: cliques simulados com o grafo cortado exatamente na borda do gabarito;
- decomposição do erro: fração dos pixels errados a até --band px da borda do gabarito,
  após 1, 3 e 10 cliques, com a configuração padrão.

Uso: python scripts/diagnose.py --data data/Kvasir-SEG/Kvasir-SEG --list data/dev.txt --jobs 12
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse

from polypseg.evaluation import find_pairs, resize_max
from polypseg.metrics import dice, iou, number_of_clicks
from polypseg.preprocessing import load_image, load_mask
from polypseg.segmenter import InteractiveSegmenter, SegmenterConfig
from polypseg.simulation import next_click
from polypseg.superpixels import SuperpixelConfig, adjacency

SEGMENTS = (200, 400, 800)
CHECKPOINTS = (1, 3, 10)


def superpixel_majority(labels: np.ndarray, gt: np.ndarray) -> np.ndarray:
    n = int(labels.max()) + 1
    frac = np.bincount(labels.ravel(), weights=gt.ravel(), minlength=n) / np.maximum(np.bincount(labels.ravel(), minlength=n), 1)
    return frac > 0.5


def oracle_weights(labels: np.ndarray, gt: np.ndarray, cut: float = 1e-4) -> sparse.csr_matrix:
    major = superpixel_majority(labels, gt)
    adj = adjacency(labels).tocoo()
    w = np.where(major[adj.row] == major[adj.col], 1.0, cut)
    n = int(labels.max()) + 1
    return sparse.coo_matrix((w, (adj.row, adj.col)), shape=(n, n)).tocsr()


def run_clicks(seg: InteractiveSegmenter, gt: np.ndarray, max_clicks: int, band: int):
    """Curva de IoU e, nos checkpoints, a fração do erro perto da borda do gabarito."""
    near = ndi.distance_transform_edt(gt) + ndi.distance_transform_edt(~gt) <= band + 1
    seg.clear()
    pred = np.zeros_like(gt)
    ious, dices, near_frac = [], [], {}
    for k in range(1, max_clicks + 1):
        click = next_click(pred, gt)
        if click is not None:
            seg.add_click(*click)
            pred = seg.predict().mask
        ious.append(iou(pred, gt))
        dices.append(dice(pred, gt))
        if k in CHECKPOINTS:
            err = pred ^ gt
            near_frac[k] = float((err & near).sum() / max(err.sum(), 1))
    return dices, ious, near_frac


def diagnose(pair, max_side: int, band: int):
    image, gt = load_image(str(pair[0])), load_mask(str(pair[1]))
    image, gt = resize_max(image, gt, max_side)
    out = {}
    for n in SEGMENTS:
        seg = InteractiveSegmenter(SegmenterConfig(superpixels=SuperpixelConfig(n_segments=n)))
        seg.set_image(image)
        major = superpixel_majority(seg.labels, gt)
        out[f"ceiling_{n}"] = (dice(major[seg.labels], gt), iou(major[seg.labels], gt))
        if n == 400:
            out["default"] = run_clicks(seg, gt, 10, band)
            seg.weights = oracle_weights(seg.labels, gt)
            out["edge_oracle"] = run_clicks(seg, gt, 10, band)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--list", type=Path)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--band", type=int, default=5)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    pairs = find_pairs(args.data)
    if args.list:
        wanted = {line.strip() for line in args.list.open() if line.strip()}
        pairs = [pair for pair in pairs if pair[0].stem in wanted]
    with ProcessPoolExecutor(max(args.jobs, 1)) as pool:
        results = list(pool.map(partial(diagnose, max_side=args.max_side, band=args.band), pairs))

    print(f"{len(pairs)} imagens")
    print("\nTeto de superpixels (maioria do gabarito)")
    for n in SEGMENTS:
        d = np.array([r[f"ceiling_{n}"] for r in results])
        print(f"  n={n:<4} Dice={d[:, 0].mean():.3f}  IoU={d[:, 1].mean():.3f}  IoU≥0,85 em {np.mean(d[:, 1] >= 0.85):.0%}  IoU≥0,90 em {np.mean(d[:, 1] >= 0.90):.0%}")
    for name in ("default", "edge_oracle"):
        dices = np.array([r[name][0] for r in results])
        ious = np.array([r[name][1] for r in results])
        noc85 = np.mean([number_of_clicks(list(row), 0.85, 10) for row in ious])
        noc90 = np.mean([number_of_clicks(list(row), 0.90, 10) for row in ious])
        print(f"\n{name}: " + " ".join(f"Dice@{k}={dices[:, k - 1].mean():.3f}" for k in (1, 3, 5, 10)) + f"  NoC@85={noc85:.2f} NoC@90={noc90:.2f}")
        for k in CHECKPOINTS:
            frac = np.mean([r[name][2][k] for r in results])
            print(f"  após {k:>2} cliques: {frac:.0%} dos pixels errados a ≤ {args.band} px da borda do gabarito")


if __name__ == "__main__":
    main()
