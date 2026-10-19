"""Segmentação automática com YOLO-seg: sem cliques, cada pólipo vira uma caixa e uma máscara.

O treino usa a própria CLI do ultralytics (ver README); aqui ficam só a conversão das
máscaras para o formato do YOLO e a avaliação com as mesmas métricas da fase assistida.

Exemplo:
    polypseg-auto dataset --src data/TrainDataset --out data/yolo
    polypseg-auto eval --weights results/yolo/s/weights/best.pt --data data/TestDataset/* --out results/yolo.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from polypseg.boxes import Box
from polypseg.evaluation import find_pairs
from polypseg.metrics import dice, iou
from polypseg.preprocessing import load_mask


def mask_to_polygons(mask: np.ndarray, tolerance: float = 1.0) -> list[np.ndarray]:
    """Contorno externo de cada componente conexa, como (x, y) normalizados em [0, 1].

    ponytail: buracos são ignorados (o YOLO-seg não os representa e pólipos quase não têm).
    """
    h, w = mask.shape
    labels, n = ndi.label(mask)
    polygons = []
    for k in range(1, n + 1):
        padded = np.pad(labels == k, 1)
        contour = max(measure.find_contours(padded.astype(float), 0.5), key=len) - 1
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) >= 4:  # fechado: o 1º ponto se repete no fim
            xy = np.column_stack([contour[:, 1] / w, contour[:, 0] / h])
            polygons.append(np.clip(xy, 0, 1))
    return polygons


def build_dataset(src: Path, out: Path, val_frac: float = 0.1, seed: int = 0) -> None:
    """Cria out/{images,labels}/{train,val} (imagens por symlink) e out/data.yaml com a classe polyp.

    A validação serve só para o early stopping do ultralytics.
    """
    pairs = find_pairs(src)
    random.Random(seed).shuffle(pairs)
    n_val = round(val_frac * len(pairs))
    for split, chunk in (("val", pairs[:n_val]), ("train", pairs[n_val:])):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, msk in chunk:
            link = out / "images" / split / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
            lines = ["0 " + " ".join(f"{v:.6f}" for v in poly.ravel()) for poly in mask_to_polygons(load_mask(str(msk)))]
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{split}: {len(chunk)} imagens")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    )


def predict_mask(model, image_path: Path, conf: float = 0.25) -> tuple[np.ndarray, list[Box]]:
    """União das máscaras das instâncias na resolução original e as caixas (linha0, coluna0, linha1, coluna1)."""
    r = model.predict(str(image_path), conf=conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return np.zeros(r.orig_shape, dtype=bool), []
    mask = r.masks.data.cpu().numpy().any(axis=0)
    boxes = [(int(y0), int(x0), int(np.ceil(y1)), int(np.ceil(x1))) for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy()]
    return mask, boxes


def evaluate(weights: Path, roots: list[Path], conf: float, out: Path | None) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows = []
    print(f"{'base':<20} {'imagens':>7} {'mDice':>7} {'mIoU':>7} {'sem detecção':>13}")
    for root in roots:
        scores = []
        for img, msk in find_pairs(root):
            gt = load_mask(str(msk))
            pred, boxes = predict_mask(model, img, conf)
            scores.append((dice(pred, gt), iou(pred, gt), len(boxes)))
            rows.append({"base": root.name, "image": img.name, "dice": scores[-1][0], "iou": scores[-1][1], "boxes": len(boxes)})
        d, j, b = np.array(scores).T
        print(f"{root.name:<20} {len(scores):>7} {d.mean():>7.3f} {j.mean():>7.3f} {int((b == 0).sum()):>13}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    ds = sub.add_parser("dataset", help="converte images/ e masks/ para o formato do YOLO-seg")
    ds.add_argument("--src", type=Path, required=True, help="pasta com images/ e masks/ (ex.: data/TrainDataset)")
    ds.add_argument("--out", type=Path, default=Path("data/yolo"))
    ds.add_argument("--val-frac", type=float, default=0.1)
    ev = sub.add_parser("eval", help="Dice e IoU médios por base, sem cliques")
    ev.add_argument("--weights", type=Path, required=True)
    ev.add_argument("--data", type=Path, nargs="+", required=True, help="uma ou mais pastas com images/ e masks/")
    ev.add_argument("--conf", type=float, default=0.25)
    ev.add_argument("--out", type=Path, help="CSV com Dice e IoU por imagem")
    args = p.parse_args(argv)
    if args.command == "dataset":
        build_dataset(args.src, args.out, args.val_frac)
    else:
        evaluate(args.weights, args.data, args.conf, args.out)


if __name__ == "__main__":
    main()
