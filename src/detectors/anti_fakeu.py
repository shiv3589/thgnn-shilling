"""Anti-FakeU baseline detector (simplified).

Reference
---------
Zhang et al. "FakeReview Detector: A Yelp Case Study." WWW 2020.
The full Anti-FakeU model uses a GCN + auxiliary tasks.  This simplified
implementation captures the key insight — that shillers tend to rate a narrow
cluster of high-profile items uniformly — through graph neighbourhood features
combined with statistical profile features, fed into a two-layer MLP.

Graph neighbourhood features (per user)
----------------------------------------
  item_deg_mean   — mean interaction degree of rated items (popularity bias)
  item_deg_max    — max item degree among rated items
  item_deg_std    — std of item degrees
  frac_top10_items— fraction of rated items in top-10 % by degree
  corated_users   — average number of users who share ≥1 item with this user
                    (high for shillers who rate the same popular targets)

Combined with the 9 statistical features from StatisticalBaseline, this gives
a 14-dimensional feature vector used to train a 2-layer MLP classifier.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.datasets.base import HeteroDataset
from .base import BaseDetector


class AntiFakeUBaseline(BaseDetector):
    """Graph-structural + statistical features → unsupervised outlier scoring
    (IsolationForest).  Never sees ground-truth labels — contamination is set
    from the known injection rate, mirroring what a deployed detector would
    assume."""

    def __init__(self, hidden: int = 64, seed: int = 42,
                 injection_rate: float = 0.01) -> None:
        self.hidden   = hidden
        self.seed     = seed
        self.scaler   = StandardScaler()
        contamination = float(np.clip(injection_rate, 1e-3, 0.5))
        self.clf      = IsolationForest(contamination=contamination, random_state=seed)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract(self, dataset: HeteroDataset) -> np.ndarray:
        num_users = dataset.num_users
        num_items = dataset.num_items
        ratings   = dataset.ratings.numpy().astype(np.float32)
        rev_users = dataset.review_user_idx.numpy()
        rev_items = dataset.review_item_idx.numpy()
        timestamps = dataset.timestamps.double().numpy()
        texts      = dataset.review_texts
        max_r      = float(np.max(ratings)) if len(ratings) > 0 else 5.0

        # ── Statistical features (9-dim) ──────────────────────────────────
        user_r:    defaultdict = defaultdict(list)
        user_ts:   defaultdict = defaultdict(list)
        user_tlen: defaultdict = defaultdict(list)
        user_items: defaultdict = defaultdict(list)

        for idx, u in enumerate(rev_users):
            user_r[u].append(ratings[idx])
            user_ts[u].append(timestamps[idx])
            user_tlen[u].append(len(texts[idx]) if texts else 0)
            user_items[u].append(int(rev_items[idx]))

        # Item degree = total number of reviews for each item
        item_deg = np.bincount(rev_items, minlength=num_items).astype(np.float32)
        top10_thr = float(np.percentile(item_deg[item_deg > 0], 90)) if item_deg.max() > 0 else 1.0

        # Item → set of users (for co-rating)
        item_users: defaultdict = defaultdict(set)
        for u, i in zip(rev_users, rev_items):
            item_users[i].add(int(u))

        stat  = np.zeros((num_users,  9), dtype=np.float32)
        graph = np.zeros((num_users,  5), dtype=np.float32)

        for u in range(num_users):
            rs   = np.array(user_r[u],  dtype=np.float32)
            ts   = np.sort(np.array(user_ts[u], dtype=np.float64))
            tls  = user_tlen[u]
            itms = user_items[u]
            n    = len(rs)

            # Statistical
            stat[u, 0] = n
            if n > 0:
                stat[u, 1] = rs.mean()
                stat[u, 2] = rs.std()
                stat[u, 3] = (rs == max_r).mean()
                stat[u, 4] = (rs == 1.0).mean()
                stat[u, 7] = np.mean(tls) if tls else 0.0
                bins = np.bincount(np.round(rs).astype(int), minlength=7)[1:7]
                from scipy.stats import entropy as _e
                stat[u, 8] = float(_e(bins.astype(float) + 1e-8))
            if n > 1:
                iat        = np.log1p(np.clip(np.diff(ts), 0, None))
                stat[u, 5] = iat.mean()
                stat[u, 6] = iat.std() if len(iat) > 1 else 0.0

            # Graph: item popularity statistics
            if itms:
                degs = item_deg[itms]
                graph[u, 0] = degs.mean()
                graph[u, 1] = degs.max()
                graph[u, 2] = degs.std() if len(degs) > 1 else 0.0
                graph[u, 3] = float((degs >= top10_thr).mean())

                # Average co-raters: for each item, how many other users rated it
                co_sizes = [len(item_users[i]) - 1 for i in itms]  # -1 = exclude self
                graph[u, 4] = float(np.mean(co_sizes)) if co_sizes else 0.0

        return np.concatenate([stat, graph], axis=1)   # (num_users, 14)

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

    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        X  = self._extract(dataset)
        Xs = self.scaler.transform(X)
        return (-self.clf.score_samples(Xs)).astype(np.float32)
