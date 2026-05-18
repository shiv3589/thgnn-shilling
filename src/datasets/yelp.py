from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .base import BaseLoader

logger = logging.getLogger(__name__)

# Yelp2018 research benchmark: reviews up to end of 2018.
_CUTOFF_TS: float = datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()

_DATE_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def _parse_yelp_date(date_str: str) -> Optional[float]:
    """Return Unix timestamp or None if unparseable."""
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class Yelp2018Loader(BaseLoader):
    """Loader for the Yelp2018 recommendation benchmark.

    The benchmark is the standard research subset of the Yelp Academic Dataset
    that retains only reviews submitted on or before 2018-12-31.

    Expected raw files (after extracting the Yelp dataset archive)::

        <raw_dir>/yelp2018/yelp_academic_dataset_review.json
        <raw_dir>/yelp2018/yelp_academic_dataset_business.json

    Each file is newline-delimited JSON (one JSON object per line).

    Extra item features appended to the base 5-dim stats
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dim 5 : business star rating from Yelp (float, 1-5)
    Dim 6 : log(review_count + 1) from Yelp metadata
    Dim 7 : is_open flag (0 or 1)
    """

    dataset_name = "yelp2018"

    def _load_raw(self) -> Tuple:
        base = self.raw_dir / "yelp2018"
        self._check_files(base)

        # ── Business metadata ──────────────────────────────────────────────
        logger.info("Reading Yelp business metadata")
        biz_path = base / "yelp_academic_dataset_business.json"
        businesses: Dict[str, Dict] = {}
        with open(biz_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="businesses", leave=False):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                bid = row.get("business_id")
                if not bid:
                    continue
                businesses[bid] = {
                    "stars": float(row.get("stars") or 0.0),
                    "review_count": int(row.get("review_count") or 0),
                    "is_open": int(row.get("is_open") or 0),
                }

        # ── Reviews (filtered to ≤ 2018-12-31) ───────────────────────────
        logger.info("Reading Yelp reviews (cutoff: 2018-12-31)")
        review_path = base / "yelp_academic_dataset_review.json"
        user_ids: List[str] = []
        item_ids: List[str] = []
        ratings: List[float] = []
        timestamps: List[float] = []
        review_texts: List[str] = []

        n_skipped_date = 0
        with open(review_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="reviews", leave=False):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = row.get("user_id")
                bid = row.get("business_id")
                stars = row.get("stars")
                date_str = row.get("date")
                text = (row.get("text") or "").strip()

                if not (uid and bid and stars is not None and date_str):
                    continue

                ts = _parse_yelp_date(date_str)
                if ts is None:
                    n_skipped_date += 1
                    continue
                if ts > _CUTOFF_TS:
                    continue

                user_ids.append(uid)
                item_ids.append(bid)
                ratings.append(float(stars))
                timestamps.append(ts)
                review_texts.append(text)

        if n_skipped_date:
            logger.warning("Skipped %d reviews with unparseable dates", n_skipped_date)
        logger.info("Yelp2018 reviews retained: %d", len(ratings))

        return (
            np.array(user_ids),
            np.array(item_ids),
            np.array(ratings, dtype=np.float32),
            np.array(timestamps, dtype=np.float64),
            review_texts,
            {},
            businesses,
        )

    def _extra_item_features(
        self, item_meta: Any, item_id_map: Dict, num_items: int,
    ) -> Optional[np.ndarray]:
        feats = np.zeros((num_items, 3), dtype=np.float32)
        for orig_id, idx in item_id_map.items():
            m = item_meta.get(orig_id, {})
            feats[idx, 0] = float(m.get("stars", 0.0))
            feats[idx, 1] = float(np.log1p(m.get("review_count", 0)))
            feats[idx, 2] = float(m.get("is_open", 0))
        return feats

    def get_item_texts(self, item_id_map: Dict) -> List[str]:
        """Return ``"<business name> <categories>"`` for each business/item."""
        base = self.raw_dir / "yelp2018"
        texts: List[str] = [""] * len(item_id_map)
        with open(base / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                bid = row.get("business_id")
                if bid not in item_id_map:
                    continue
                idx = item_id_map[bid]
                name = (row.get("name") or "").strip()
                cats = (row.get("categories") or "").strip()
                texts[idx] = f"{name} {cats}".strip()
        return texts

    @staticmethod
    def _check_files(base: Path) -> None:
        required = [
            "yelp_academic_dataset_review.json",
            "yelp_academic_dataset_business.json",
        ]
        missing = [name for name in required if not (base / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing Yelp files in {base}: {missing}\n"
                "Run: python scripts/download_data.py --raw-dir data/raw"
            )
