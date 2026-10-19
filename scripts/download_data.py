"""Baixa as bases públicas de pólipos usadas na avaliação para data/.

- data/Kvasir-SEG/{images,masks}: 1000 imagens (espelho berkaytrhn/kvasir-seg no Hugging Face).
- data/TestDataset/<base>/{images,masks}: conjuntos de teste do protocolo PraNet
  (CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir), extraídos do zip
  doantrongthai/Polyp_Segmentation sem baixar o arquivo inteiro (~1,1 GB).

Uso: pip install -e ".[data]" && python scripts/download_data.py [--only kvasir|pranet]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRANET_ZIP = "https://huggingface.co/datasets/doantrongthai/Polyp_Segmentation/resolve/main/dataset.zip"
PRANET_SETS = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")


def download_kvasir(root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "berkaytrhn/kvasir-seg", repo_type="dataset", local_dir=root, allow_patterns=["Kvasir-SEG/*"], max_workers=8
    )


def download_pranet(root: Path, workers: int = 8) -> None:
    from remotezip import RemoteZip

    prefix = "dataset/TestDataset/"
    with RemoteZip(PRANET_ZIP) as z:
        names = [
            n for n in z.namelist()
            if n.startswith(prefix) and not n.endswith("/") and n[len(prefix):].split("/")[0] in PRANET_SETS
        ]

    def fetch(chunk: list[str]) -> None:
        for name in chunk:
            target = root / "TestDataset" / name[len(prefix):]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--only", choices=("kvasir", "pranet"))
    args = parser.parse_args()
    if args.only in (None, "kvasir"):
        download_kvasir(args.root)
    if args.only in (None, "pranet"):
        download_pranet(args.root)
    for d in sorted(args.root.glob("**/images")):
        print(f"{d.parent}: {sum(1 for _ in d.iterdir())} imagens")


if __name__ == "__main__":
    main()
