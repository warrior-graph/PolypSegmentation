"""Baixa as bases públicas de pólipos usadas na avaliação para data/.

- data/Kvasir-SEG/{images,masks}: 1000 imagens (espelho berkaytrhn/kvasir-seg no Hugging Face).
- data/TestDataset/<base>/{images,masks}: conjuntos de teste do protocolo PraNet
  (CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir), extraídos do zip
  doantrongthai/Polyp_Segmentation sem baixar o arquivo inteiro (~1,1 GB).
- data/CVC-ClinicDB-dev/{images,masks}: base de desenvolvimento com 120 das 550 imagens
  do CVC-ClinicDB no treino do PraNet (nomes numéricos), sorteadas com semente 0. Não
  se sobrepõe a nenhum conjunto de teste.
- data/TrainDataset/{images,masks}: treino completo do PraNet (900 Kvasir + 550
  CVC-ClinicDB), para a segmentação automática. Só com --only pranet-train.

Uso: pip install -e ".[data]" && python scripts/download_data.py [--only kvasir|pranet|cvc-dev|pranet-train]
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRANET_ZIP = "https://huggingface.co/datasets/doantrongthai/Polyp_Segmentation/resolve/main/dataset.zip"
PRANET_SETS = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")
CVC_DEV_SIZE = 120


def download_kvasir(root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "berkaytrhn/kvasir-seg", repo_type="dataset", local_dir=root, allow_patterns=["Kvasir-SEG/*"], max_workers=8
    )


def _fetch_all(targets: dict[str, Path], workers: int) -> None:
    """Extrai cada membro do zip remoto para o caminho indicado, em paralelo."""
    from remotezip import RemoteZip

    names = sorted(targets)

    def fetch(chunk: list[str]) -> None:
        for name in chunk:
            target = targets[name]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):  # conexões caem de vez em quando em downloads parciais
                try:
                    with RemoteZip(PRANET_ZIP) as z:
                        target.write_bytes(z.read(name))
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, [names[i::workers] for i in range(workers)]))


def download_pranet(root: Path, workers: int = 8) -> None:
    from remotezip import RemoteZip

    prefix = "dataset/TestDataset/"
    with RemoteZip(PRANET_ZIP) as z:
        names = [
            n for n in z.namelist()
            if n.startswith(prefix) and not n.endswith("/") and n[len(prefix):].split("/")[0] in PRANET_SETS
        ]
    _fetch_all({n: root / "TestDataset" / n[len(prefix):] for n in names}, workers)


def download_cvc_dev(root: Path, workers: int = 8) -> None:
    from remotezip import RemoteZip

    prefix = "dataset/TrainDataset/"
    with RemoteZip(PRANET_ZIP) as z:
        stems = sorted(
            Path(n).stem for n in z.namelist()
            if n.startswith(prefix + "images/") and Path(n).stem.isdigit()
        )
    chosen = random.Random(0).sample(stems, CVC_DEV_SIZE)
    targets = {}
    for stem in chosen:
        for sub in ("images", "masks"):
            targets[f"{prefix}{sub}/{stem}.png"] = root / "CVC-ClinicDB-dev" / sub / f"{stem}.png"
    _fetch_all(targets, workers)


def download_pranet_train(root: Path, workers: int = 8) -> None:
    from remotezip import RemoteZip

    prefix = "dataset/TrainDataset/"
    with RemoteZip(PRANET_ZIP) as z:
        names = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
    _fetch_all({n: root / "TrainDataset" / n[len(prefix):] for n in names}, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--only", choices=("kvasir", "pranet", "cvc-dev", "pranet-train"))
    args = parser.parse_args()
    if args.only in (None, "kvasir"):
        download_kvasir(args.root)
    if args.only in (None, "pranet"):
        download_pranet(args.root)
    if args.only in (None, "cvc-dev"):
        download_cvc_dev(args.root)
    if args.only == "pranet-train":  # ponytail: fora do padrão, só a fase automática precisa
        download_pranet_train(args.root)
    for d in sorted(args.root.glob("**/images")):
        print(f"{d.parent}: {sum(1 for _ in d.iterdir())} imagens")


if __name__ == "__main__":
    main()
