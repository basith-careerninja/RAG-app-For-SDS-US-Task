"""Pulls the AAPL 10-Q filing and its reference Q&A out of the KG-RAG
dataset zip (https://github.com/docugami/KG-RAG-datasets) into data/raw/.

Usage:
    python scripts/prepare_data.py --zip "C:\\path\\to\\KG-RAG-datasets-main.zip"
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "raw" / "aapl"

FILES = [
    "sec-10-q/data/v1/docs/2022 Q3 AAPL.pdf",
    "sec-10-q/data/v1/qna_data.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to KG-RAG-datasets-main.zip")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        top = zf.namelist()[0].split("/")[0]
        DEST_DIR.mkdir(parents=True, exist_ok=True)

        for rel_path in FILES:
            member_name = f"{top}/{rel_path}"
            dest = DEST_DIR / Path(rel_path).name
            dest.write_bytes(zf.read(member_name))
            print(f"wrote {dest}")

    print(f"\nDone. Files in {DEST_DIR}")


if __name__ == "__main__":
    main()
