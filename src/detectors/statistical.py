"""Statistical baseline detector.

Extracts per-user statistical features (rating distribution, temporal burstiness,
review-text length) and trains a logistic regression classifier.

Features
--------
  0  review_count        — total reviews written
  1  mean_rating         — average star rating
  2  std_rating          — std dev of ratings (shillers rate uniformly)
  3  frac_max_rating     — fraction of reviews at max stars
  4  frac_min_rating     — fraction of reviews at min stars (1.0)
  5  mean_log_iat        — mean log(1 + inter-arrival time)
  6  std_log_iat         — std  log(1 + inter-arrival time)
  7  mean_text_len       — average review text character length
  8  rating_entropy      — Shannon entropy of the rating histogram
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.stats import entropy as _scipy_entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.datasets.base import HeteroDataset
from .base import BaseDetector


class StatisticalBaseline(BaseDetector):
    """Unsupervised outlier scoring (IsolationForest) on 9 per-user statistical
    features.  Never sees ground-truth labels — contamination is set from the
    known injection rate, mirroring what a deployed detector would assume."""

    def __init__(self, seed: int = 42, injection_rate: float = 0.01) -> None:
        self.seed       = seed
        self.scaler     = StandardScaler()
        contamination   = float(np.clip(injection_rate, 1e-3, 0.5))
        self.clf        = IsolationForest(contamination=contamination,
                                          random_state=seed)
        self._fitted    = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(self, dataset: HeteroDataset) -> np.ndarray:
        num_users = dataset.num_users
        ratings   = dataset.ratings.numpy().astype(np.float32)
        rev_users = dataset.review_user_idx.numpy()
        timestamps = dataset.timestamps.double().numpy()
        texts      = dataset.review_texts
        max_r      = float(np.max(ratings)) if len(ratings) > 0 else 5.0

        user_r:    defaultdict = defaultdict(list)
        user_ts:   defaultdict = defaultdict(list)
        user_tlen: defaultdict = defaultdict(list)

        for idx, u in enumerate(rev_users):
            user_r[u].append(ratings[idx])
            user_ts[u].append(timestamps[idx])
            user_tlen[u].append(len(texts[idx]) if texts else 0)

        feats = np.zeros((num_users, 9), dtype=np.float32)

        for u in range(num_users):
            rs  = np.array(user_r[u],  dtype=np.float32)
            ts  = np.sort(np.array(user_ts[u],   dtype=np.float64))
            tls = user_tlen[u]
            n   = len(rs)

            feats[u, 0] = n
            if n > 0:
                feats[u, 1] = rs.mean()
                feats[u, 2] = rs.std()
                feats[u, 3] = (rs == max_r).mean()
                feats[u, 4] = (rs == 1.0).mean()
                feats[u, 7] = np.mean(tls) if tls else 0.0

                # rating entropy (smoothed histogram over stars 1–5)
                bins = np.bincount(np.round(rs).astype(int), minlength=7)[1:7]
                feats[u, 8] = float(_scipy_entropy(bins.astype(float) + 1e-8))

            if n > 1:
                iat        = np.log1p(np.clip(np.diff(ts), 0, None))
                feats[u, 5] = iat.mean()
                feats[u, 6] = iat.std() if len(iat) > 1 else 0.0

        return feats

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: HeteroDataset,
        *,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        X  = self._extract(dataset)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs)
        self._fitted = True

    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        X  = self._extract(dataset)
        Xs = self.scaler.transform(X)
        # IsolationForest: lower score_samples = more anomalous; negate so
        # higher output = more shilling-like, consistent with other detectors.
        return (-self.clf.score_samples(Xs)).astype(np.float32)
