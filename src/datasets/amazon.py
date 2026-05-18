from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .base import BaseLoader

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield parsed JSON objects from a plain or gzip-compressed JSONL file."""
    opener = gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_file(directory: Path, *candidates: str) -> Path:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of {candidates} found in {directory}.\n"
        "Run: python scripts/download_data.py --raw-dir data/raw"
    )


class AmazonBaseLoader(BaseLoader):
    """Base loader for Amazon 2023 review datasets (McAuley Lab / UCSD format).

    Review JSONL schema (2023 release)
    -----------------------------------
    rating            float   — star rating
    title             str     — review headline
    text              str     — review body
    asin              str     — product ASIN
    user_id           str     — anonymised reviewer ID
    timestamp         int     — milliseconds since Unix epoch
    helpful_vote      int
    verified_purchase bool

    Metadata JSONL schema
    ---------------------
    parent_asin / asin   str     — product ASIN (use parent_asin when present)
    average_rating        float
    rating_number         int
    price                 str | null   — e.g. "$12.99"
    categories            List[str]
    title                 str

    Extra item features appended to the base 5-dim stats
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dim 5 : log(price + 1)           — price signal
    Dim 6 : average_rating from meta — catalogue-level rating
    Dim 7 : log(rating_number + 1)   — catalogue-level popularity
    """

    category: str = ""  # e.g. "Books", "Clothing_Shoes_and_Jewelry"
    dataset_name: str = "amazon"

    def _review_path(self) -> Path:
        base = self.raw_dir / "amazon_2023"
        return _find_file(
            base,
            f"{self.category}.jsonl.gz",
            f"{self.category}.jsonl",
        )

    def _meta_path(self) -> Path:
        base = self.raw_dir / "amazon_2023"
        return _find_file(
            base,
            f"meta_{self.category}.jsonl.gz",
            f"meta_{self.category}.jsonl",
        )

    def _load_raw(self) -> Tuple:
        # ── Metadata (load first so reviews can be validated) ─────────────
        logger.info("[%s] Reading metadata", self.category)
        meta: Dict[str, Dict] = {}
        for row in tqdm(_iter_jsonl(self._meta_path()), desc=f"meta/{self.category}", leave=False):
            asin = row.get("parent_asin") or row.get("asin")
            if not asin:
                continue
            raw_price = row.get("price") or "0"
            try:
                price = float(str(raw_price).replace("$", "").replace(",", "").strip() or "0")
            except ValueError:
                price = 0.0
            meta[asin] = {
                "price": price,
                "avg_rating": float(row.get("average_rating") or 0.0),
                "rating_number": int(row.get("rating_number") or 0),
            }

        # ── Reviews ───────────────────────────────────────────────────────
        logger.info("[%s] Reading reviews", self.category)
        user_ids: List[str] = []
        item_ids: List[str] = []
        ratings: List[float] = []
        timestamps: List[float] = []
        review_texts: List[str] = []

        for row in tqdm(_iter_jsonl(self._review_path()), desc=f"reviews/{self.category}", leave=False):
            uid = row.get("user_id")
            asin = row.get("asin")
            rat = row.get("rating")
            ts_ms = row.get("timestamp")
            if not (uid and asin and rat is not None and ts_ms is not None):
                continue
            user_ids.append(uid)
            item_ids.append(asin)
            ratings.append(float(rat))
            timestamps.append(float(ts_ms) / 1000.0)  # ms → Unix seconds
            headline = (row.get("title") or "").strip()
            body = (row.get("text") or "").strip()
            review_texts.append(f"{headline} {body}".strip())

        return (
            np.array(user_ids),
            np.array(item_ids),
            np.array(ratings, dtype=np.float32),
            np.array(timestamps, dtype=np.float64),
            review_texts,
            {},      # Amazon 2023 has no structured per-user metadata
            meta,
        )

    def _extra_item_features(
        self, item_meta: Any, item_id_map: Dict, num_items: int,
    ) -> Optional[np.ndarray]:
        feats = np.zeros((num_items, 3), dtype=np.float32)
        for orig_id, idx in item_id_map.items():
            m = item_meta.get(orig_id, {})
            feats[idx, 0] = float(np.log1p(m.get("price", 0.0) or 0.0))
            feats[idx, 1] = float(m.get("avg_rating", 0.0))
            feats[idx, 2] = float(np.log1p(m.get("rating_number", 0)))
        return feats


class AmazonBooksLoader(AmazonBaseLoader):
    """Amazon Books (2023 reviews, McAuley Lab).

    Required files in ``<raw_dir>/amazon_2023/``::

        Books.jsonl.gz
        meta_Books.jsonl.gz
    """

    category = "Books"
    dataset_name = "amazon-books"


class AmazonClothingLoader(AmazonBaseLoader):
    """Amazon Clothing, Shoes and Jewelry (2023 reviews, McAuley Lab).

    Required files in ``<raw_dir>/amazon_2023/``::

        Clothing_Shoes_and_Jewelry.jsonl.gz
        meta_Clothing_Shoes_and_Jewelry.jsonl.gz
    """

    category = "Clothing_Shoes_and_Jewelry"
    dataset_name = "amazon-clothing"
