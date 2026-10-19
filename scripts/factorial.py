"""Fatorial completo dos quatro módulos na base de desenvolvimento.

Módulos: M1 relief_map (relevo local no mapa de borda), M2 boundary_edges (arestas
cortadas pela borda), M3 radial_prior (estrela restrita pelos cliques), M4 pixel_refine
(random walker por pixel na faixa da borda). Grava um CSV por combinação e imprime a
tabela, os efeitos principais e as interações de 2ª ordem sobre NoC@85 e Dice@1.

Uso: python scripts/factorial.py --data data/Kvasir-SEG/Kvasir-SEG --list data/dev.txt --jobs 12
"""

from __future__ import annotations

import argparse
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from polypseg.evaluation import evaluate_pair, find_pairs
from polypseg.metrics import number_of_clicks
from polypseg.segmenter import SegmenterConfig

MODULES = {"M1": "relief_map", "M2": "boundary_edges", "M3": "radial_prior", "M4": "pixel_refine"}


def summarize(curves) -> dict[str, float]:
    dice = np.array([c.dice for c in curves])
    iou = np.array([c.iou for c in curves])
    return {
        "dice1": dice[:, 0].mean(), "dice3": dice[:, 2].mean(), "dice5": dice[:, 4].mean(), "dice10": dice[:, 9].mean(),
        "noc85": np.mean([number_of_clicks(list(r), 0.85, 10) for r in iou]),
        "noc90": np.mean([number_of_clicks(list(r), 0.90, 10) for r in iou]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--list", type=Path)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/factorial"))
    args = p.parse_args()
    pairs = find_pairs(args.data)
    if args.list:
        wanted = {line.strip() for line in args.list.open() if line.strip()}
        pairs = [pair for pair in pairs if pair[0].stem in wanted]
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    with ProcessPoolExecutor(max(args.jobs, 1)) as pool:
        for levels in itertools.product((False, True), repeat=len(MODULES)):
            on = dict(zip(MODULES, levels))
            # Prior geodésico desligado, para "nenhum" continuar sendo o método anterior.
            config = SegmenterConfig(**{field: on[m] for m, field in MODULES.items()}, geodesic_prior=0.0)
            curves = list(pool.map(partial(evaluate_pair, config=config, max_clicks=10, max_side=512), pairs))
            tag = "+".join(m for m in MODULES if on[m]) or "nenhum"
            with (args.out / f"{tag}.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["image", "clicks", "dice", "iou"])
                for (img, _), c in zip(pairs, curves):
                    for k, (d, j) in enumerate(zip(c.dice, c.iou), start=1):
                        writer.writerow([img.name, k, d, j])
            results[levels] = summarize(curves)
            r = results[levels]
            print(f"{tag:14s} D@1={r['dice1']:.3f} D@3={r['dice3']:.3f} D@5={r['dice5']:.3f} D@10={r['dice10']:.3f} "
                  f"NoC85={r['noc85']:.2f} NoC90={r['noc90']:.2f}", flush=True)

    names = list(MODULES)
    signs = {lv: np.array([1 if x else -1 for x in lv]) for lv in results}
    for metric in ("noc85", "dice1"):
        y = {lv: r[metric] for lv, r in results.items()}
        print(f"\nEfeitos sobre {metric} (diferença média entre ligado e desligado)")
        for i, m in enumerate(names):
            print(f"  {m}: {np.mean([y[lv] * signs[lv][i] for lv in y]) * 2:+.3f}")
        for i, j in itertools.combinations(range(len(names)), 2):
            effect = np.mean([y[lv] * signs[lv][i] * signs[lv][j] for lv in y]) * 2
            print(f"  {names[i]}×{names[j]}: {effect:+.3f}")


if __name__ == "__main__":
    main()
