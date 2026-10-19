"""Avaliação com cliques simulados em uma base no formato do Kvasir-SEG.

Exemplo:
    polypseg-eval --data data/Kvasir-SEG --max-clicks 10 --limit 100 --out results/kvasir.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from polypseg.features import FeatureConfig
from polypseg.metrics import number_of_clicks
from polypseg.preprocessing import load_image, load_mask
from polypseg.segmenter import CLASSIFIERS, InteractiveSegmenter, SegmenterConfig
from polypseg.simulation import simulate
from polypseg.superpixels import METHODS, SuperpixelConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(root: Path, images_dir: str = "images", masks_dir: str = "masks") -> list[tuple[Path, Path]]:
    """Pares (imagem, máscara) com o mesmo nome-base em images/ e masks/."""
    masks = {p.stem: p for p in (root / masks_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = []
    for img in sorted((root / images_dir).iterdir()):
        if img.suffix.lower() in IMAGE_EXTS and img.stem in masks:
            pairs.append((img, masks[img.stem]))
    return pairs


def resize_max(image: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    from skimage.transform import resize

    h, w = mask.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return image, mask
    shape = (int(round(h * scale)), int(round(w * scale)))
    image = resize(image, shape, anti_aliasing=True).astype(np.float32)
    mask = resize(mask.astype(float), shape, order=0, anti_aliasing=False) > 0.5
    return image, mask


def evaluate_pair(pair: tuple[Path, Path], config: SegmenterConfig, max_clicks: int, max_side: int):
    image, gt = load_image(str(pair[0])), load_mask(str(pair[1]))
    if max_side:
        image, gt = resize_max(image, gt, max_side)
    segmenter = InteractiveSegmenter(config)
    segmenter.set_image(image)
    return simulate(segmenter, gt, max_clicks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="pasta com images/ e masks/")
    p.add_argument("--max-clicks", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="avalia só as N primeiras imagens (0 = todas)")
    p.add_argument("--max-side", type=int, default=512, help="redimensiona o maior lado (0 = original)")
    p.add_argument("--method", choices=METHODS, default="slic")
    p.add_argument("--n-segments", type=int, default=400)
    p.add_argument("--compactness", type=float, default=10.0)
    p.add_argument("--classifier", choices=CLASSIFIERS, default="graph")
    p.add_argument("--border-background", choices=("always", "auto", "never"), default="always")
    p.add_argument("--features", default="color,lbp,glcm,gabor", help="grupos separados por vírgula")
    p.add_argument("--jobs", type=int, default=1, help="processos em paralelo")
    p.add_argument("--out", type=Path, help="CSV com Dice e IoU por imagem e clique")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    groups = {g.strip() for g in args.features.split(",") if g.strip()}
    config = SegmenterConfig(
        superpixels=SuperpixelConfig(args.method, args.n_segments, args.compactness),
        features=FeatureConfig(
            color="color" in groups, lbp="lbp" in groups, glcm="glcm" in groups, gabor="gabor" in groups
        ),
        classifier=args.classifier,
        border_background=args.border_background,
    )
    pairs = find_pairs(args.data)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"nenhum par imagem/máscara encontrado em {args.data}")

    run = partial(evaluate_pair, config=config, max_clicks=args.max_clicks, max_side=args.max_side)
    all_dice, all_iou, rows = [], [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for i, ((img_path, _), curve) in enumerate(zip(pairs, pool.map(run, pairs)), start=1):
            all_dice.append(curve.dice)
            all_iou.append(curve.iou)
            for k, (d, j) in enumerate(zip(curve.dice, curve.iou), start=1):
                rows.append({"image": img_path.name, "clicks": k, "dice": d, "iou": j})
            print(f"[{i}/{len(pairs)}] {img_path.name}: Dice@1={curve.dice[0]:.3f} Dice@{args.max_clicks}={curve.dice[-1]:.3f}", flush=True)

    dice_arr, iou_arr = np.array(all_dice), np.array(all_iou)
    print(f"\n{len(pairs)} imagens, {time.perf_counter() - start:.1f}s, configuração: {args.method} "
          f"n={args.n_segments}, {args.classifier}, descritores={sorted(groups)}")
    print("cliques  mDice   mIoU")
    for k in range(args.max_clicks):
        print(f"{k + 1:>7}  {dice_arr[:, k].mean():.4f}  {iou_arr[:, k].mean():.4f}")
    for thr in (0.85, 0.90):
        noc = [number_of_clicks(list(row), thr, args.max_clicks) for row in iou_arr]
        print(f"NoC@{int(thr * 100)}: {np.mean(noc):.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["image", "clicks", "dice", "iou"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"resultados por imagem em {args.out}")


if __name__ == "__main__":
    main()
