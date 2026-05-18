from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseLoader

logger = logging.getLogger(__name__)

# 18 genres in ML-1M, in the order they appear in movies.dat
_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_GENRE_IDX = {g: i for i, g in enumerate(_GENRES)}
_NUM_GENRES = len(_GENRES)

# ML-1M age-group codes → 0-based index (normalised to [0,1] later)
_AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
_NUM_AGE_GROUPS = 7
_NUM_OCCUPATIONS = 21  # 0-20


class MovieLens1MLoader(BaseLoader):
    """Loader for the MovieLens-1M dataset.

    Expected raw files (after unzipping ``ml-1m.zip``)::

        <raw_dir>/ml-1m/ratings.dat   — UserID::MovieID::Rating::Timestamp
        <raw_dir>/ml-1m/movies.dat    — MovieID::Title::Genres
        <raw_dir>/ml-1m/users.dat     — UserID::Gender::Age::Occupation::Zip

    Timestamps in ratings.dat are already Unix seconds (int).

    Extra user features appended to the base 5-dim stats
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dim 5  : gender (0=M, 1=F)
    Dim 6  : age-group index normalised to [0, 1]
    Dim 7  : occupation index normalised to [0, 1]

    Extra item features appended to the base 5-dim stats
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dim 5  : release year normalised to [0, 1] (range 1919-2000)
    Dim 6-23 : genre multi-hot (18 dims)
    """

    dataset_name = "ml-1m"

    def _load_raw(self) -> Tuple:
        base = self.raw_dir / "ml-1m"
        self._check_files(base)

        # ── ratings ───────────────────────────────────────────────────────
        logger.info("Reading ratings.dat")
        user_ids, item_ids, ratings, timestamps = [], [], [], []

        with open(base / "ratings.dat", encoding="latin-1") as f:
            for line in f:
                uid, mid, rat, ts = line.strip().split("::")
                user_ids.append(uid)
                item_ids.append(mid)
                ratings.append(float(rat))
                timestamps.append(float(ts))

        review_texts: List[str] = [""] * len(ratings)  # no review text in ML-1M

        # ── movies metadata ───────────────────────────────────────────────
        logger.info("Reading movies.dat")
        movies: Dict[str, Dict] = {}
        with open(base / "movies.dat", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                mid, title, genre_str = parts[0], parts[1], parts[2]
                m = re.search(r"\((\d{4})\)$", title.strip())
                movies[mid] = {
                    "genres": genre_str.split("|"),
                    "year": int(m.group(1)) if m else 0,
                }

        # ── users metadata ────────────────────────────────────────────────
        logger.info("Reading users.dat")
        users: Dict[str, Dict] = {}
        with open(base / "users.dat", encoding="latin-1") as f:
            for line in f:
                uid, gender, age, occ, _ = line.strip().split("::")
                users[uid] = {
                    "gender": 0 if gender == "M" else 1,
                    "age": int(age),
                    "occupation": int(occ),
                }

        return (
            np.array(user_ids),
            np.array(item_ids),
            np.array(ratings, dtype=np.float32),
            np.array(timestamps, dtype=np.float64),
            review_texts,
            users,
            movies,
        )

    def _extra_user_features(
        self, user_meta: Any, user_id_map: Dict, num_users: int,
    ) -> Optional[np.ndarray]:
        feats = np.zeros((num_users, 3), dtype=np.float32)
        for orig_id, idx in user_id_map.items():
            m = user_meta.get(str(orig_id), {})
            feats[idx, 0] = float(m.get("gender", 0))
            feats[idx, 1] = _AGE_MAP.get(m.get("age", 1), 0) / (_NUM_AGE_GROUPS - 1)
            feats[idx, 2] = m.get("occupation", 0) / (_NUM_OCCUPATIONS - 1)
        return feats

    def _extra_item_features(
        self, item_meta: Any, item_id_map: Dict, num_items: int,
    ) -> Optional[np.ndarray]:
        # 1 (year) + 18 (genre multi-hot)
        feats = np.zeros((num_items, 1 + _NUM_GENRES), dtype=np.float32)
        for orig_id, idx in item_id_map.items():
            m = item_meta.get(str(orig_id), {})
            year = m.get("year", 0)
            if year > 0:
                # ML-1M movies span roughly 1919-2000
                feats[idx, 0] = np.clip((year - 1919) / (2000 - 1919), 0.0, 1.0)
            for g in m.get("genres", []):
                gi = _GENRE_IDX.get(g)
                if gi is not None:
                    feats[idx, 1 + gi] = 1.0
        return feats

    @staticmethod
    def _check_files(base: Path) -> None:
        missing = [
            name for name in ("ratings.dat", "movies.dat", "users.dat")
            if not (base / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing ML-1M files in {base}: {missing}\n"
                "Run: python scripts/download_data.py --raw-dir data/raw"
            )
