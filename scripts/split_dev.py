"""Sorteia a base de desenvolvimento: 200 imagens de data/train.txt do Kvasir-SEG.

Exclui as imagens que também estão no teste Kvasir do protocolo PraNet, para que
nenhuma decisão de método seja tomada olhando imagens de teste.

Uso: python scripts/split_dev.py [--n 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train = [line.strip() for line in (ROOT / "train.txt").open() if line.strip()]
    pranet = {f.stem for f in (ROOT / "TestDataset" / "Kvasir" / "images").iterdir()}
    pool = sorted(set(train) - pranet)
    dev = sorted(random.Random(args.seed).sample(pool, args.n))
    (ROOT / "dev.txt").write_text("\n".join(dev) + "\n")
    print(f"{len(dev)} imagens de {len(pool)} candidatas em {ROOT / 'dev.txt'}")


if __name__ == "__main__":
    main()
