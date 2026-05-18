#!/usr/bin/env python
"""Dataset download guide and file validator for thgnn-shilling.

Usage
-----
Print instructions for all datasets::

    python scripts/download_data.py

Print instructions for a specific dataset::

    python scripts/download_data.py --dataset ml-1m

Validate downloaded files (checks presence and registered checksums)::

    python scripts/download_data.py --validate
    python scripts/download_data.py --validate --raw-dir /path/to/raw

Register a checksum after downloading (prints SHA-256 for a file)::

    python scripts/download_data.py --checksum data/raw/ml-1m/ratings.dat
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry:
#   description  : human-readable name
#   instructions : list of strings printed verbatim (use <RAW_DIR> placeholder)
#   required_files : relative paths under raw_dir that must exist
#   checksums    : {rel_path: (algo, hex_digest)}  — empty means "skip"
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict] = {
    "ml-1m": {
        "description": "MovieLens-1M  (GroupLens Research)",
        "instructions": [
            "  wget https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "  unzip ml-1m.zip -d <RAW_DIR>",
            "",
            "  Result:",
            "    <RAW_DIR>/ml-1m/ratings.dat",
            "    <RAW_DIR>/ml-1m/movies.dat",
            "    <RAW_DIR>/ml-1m/users.dat",
        ],
        "required_files": [
            "ml-1m/ratings.dat",
            "ml-1m/movies.dat",
            "ml-1m/users.dat",
        ],
        # SHA-256 of the files extracted from ml-1m.zip (register after download).
        "checksums": {},
    },
    "amazon-books": {
        "description": "Amazon Books 2023  (McAuley Lab / UCSD)",
        "instructions": [
            "  mkdir -p <RAW_DIR>/amazon_2023",
            "",
            "  # Reviews (~35 GB uncompressed — download the .gz form):",
            "  wget -P <RAW_DIR>/amazon_2023 \\",
            "    https://mcauleylab.ucsd.edu/data/amazon_2023/data/review_categories/Books.jsonl.gz",
            "",
            "  # Metadata:",
            "  wget -P <RAW_DIR>/amazon_2023 \\",
            "    https://mcauleylab.ucsd.edu/data/amazon_2023/data/meta_categories/meta_Books.jsonl.gz",
            "",
            "  Alternatively browse: https://amazon-reviews-2023.github.io/",
        ],
        "required_files": [
            "amazon_2023/Books.jsonl.gz",
            "amazon_2023/meta_Books.jsonl.gz",
        ],
        "checksums": {},
    },
    "amazon-clothing": {
        "description": "Amazon Clothing, Shoes and Jewelry 2023  (McAuley Lab / UCSD)",
        "instructions": [
            "  mkdir -p <RAW_DIR>/amazon_2023",
            "",
            "  # Reviews:",
            "  wget -P <RAW_DIR>/amazon_2023 \\",
            "    https://mcauleylab.ucsd.edu/data/amazon_2023/data/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz",
            "",
            "  # Metadata:",
            "  wget -P <RAW_DIR>/amazon_2023 \\",
            "    https://mcauleylab.ucsd.edu/data/amazon_2023/data/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
            "",
            "  Alternatively browse: https://amazon-reviews-2023.github.io/",
        ],
        "required_files": [
            "amazon_2023/Clothing_Shoes_and_Jewelry.jsonl.gz",
            "amazon_2023/meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        ],
        "checksums": {},
    },
    "yelp2018": {
        "description": "Yelp2018  (Yelp Academic Dataset, reviews ≤ 2018-12-31)",
        "instructions": [
            "  1. Request access at https://www.yelp.com/dataset/download",
            "     (free for non-commercial use; requires accepting their licence).",
            "",
            "  2. Download and extract the archive.  Locate:",
            "       yelp_academic_dataset_review.json",
            "       yelp_academic_dataset_business.json",
            "",
            "  3. Copy them to:",
            "       <RAW_DIR>/yelp2018/yelp_academic_dataset_review.json",
            "       <RAW_DIR>/yelp2018/yelp_academic_dataset_business.json",
            "",
            "  Note: the loader automatically filters to reviews ≤ 2018-12-31",
            "        to match the standard Yelp2018 benchmark split.",
        ],
        "required_files": [
            "yelp2018/yelp_academic_dataset_review.json",
            "yelp2018/yelp_academic_dataset_business.json",
        ],
        "checksums": {},
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_checksum(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _hr() -> str:
    return "─" * 68


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def print_instructions(raw_dir: str, dataset_filter: Optional[str]) -> None:
    target = DATASETS
    if dataset_filter:
        if dataset_filter not in DATASETS:
            print(f"Unknown dataset '{dataset_filter}'. Choose from: {list(DATASETS)}")
            sys.exit(1)
        target = {dataset_filter: DATASETS[dataset_filter]}

    print()
    print("=" * 68)
    print("  thgnn-shilling — Dataset Download Instructions")
    print(f"  Target raw directory: {raw_dir}")
    print("=" * 68)

    for key, ds in target.items():
        print(f"\n{_hr()}")
        print(f"  [{key}]  {ds['description']}")
        print()
        for line in ds["instructions"]:
            print(line.replace("<RAW_DIR>", raw_dir))

    print(f"\n{_hr()}")
    print("\nTo validate after downloading:")
    print(f"  python scripts/download_data.py --validate --raw-dir {raw_dir}")
    print()


def validate(raw_dir: str, dataset_filter: Optional[str]) -> bool:
    root = Path(raw_dir)
    target = DATASETS
    if dataset_filter:
        if dataset_filter not in DATASETS:
            print(f"Unknown dataset '{dataset_filter}'. Choose from: {list(DATASETS)}")
            sys.exit(1)
        target = {dataset_filter: DATASETS[dataset_filter]}

    print(f"\nValidating files under: {root.resolve()}\n")
    all_ok = True

    for key, ds in target.items():
        print(f"[{key}]  {ds['description']}")
        for rel in ds["required_files"]:
            p = root / rel
            if not p.exists():
                print(f"  ✗  MISSING    {rel}")
                all_ok = False
                continue

            size_mb = p.stat().st_size / (1 << 20)
            cs_info: Optional[Tuple[str, str]] = ds["checksums"].get(rel)

            if cs_info:
                algo, expected = cs_info
                computed = compute_checksum(p, algo)
                if computed == expected:
                    print(f"  ✓  OK         {rel}  ({size_mb:.1f} MB)")
                else:
                    print(f"  ✗  MISMATCH   {rel}  ({size_mb:.1f} MB)")
                    print(f"               expected  {algo}:{expected}")
                    print(f"               computed  {algo}:{computed}")
                    all_ok = False
            else:
                print(f"  ~  PRESENT    {rel}  ({size_mb:.1f} MB)  [no checksum registered]")
        print()

    if all_ok:
        print("All required files are present.")
    else:
        print("Some files are missing or have checksum mismatches — see above.")
    return all_ok


def print_checksum(file_path: str) -> None:
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    digest = compute_checksum(p, "sha256")
    print(f"sha256:{digest}  {file_path}")
    print()
    print("Add this to DATASETS[<key>]['checksums'] in scripts/download_data.py:")
    print(f'  "{p.name}": ("sha256", "{digest}"),')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download instructions and file validator for thgnn-shilling datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", default="data/raw", metavar="DIR",
        help="Raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--dataset", metavar="NAME",
        choices=list(DATASETS),
        help="Show instructions / validate a single dataset",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Check that required files exist and pass registered checksums",
    )
    parser.add_argument(
        "--checksum", metavar="FILE",
        help="Compute and print the SHA-256 of a file for registering in checksums",
    )
    args = parser.parse_args()

    if args.checksum:
        print_checksum(args.checksum)
    elif args.validate:
        ok = validate(args.raw_dir, args.dataset)
        sys.exit(0 if ok else 1)
    else:
        print_instructions(args.raw_dir, args.dataset)


if __name__ == "__main__":
    main()
