"""SemanticShield baseline detector.

Represents each user as the TF-IDF vector of their concatenated review texts,
then trains a logistic regression classifier on those text features.  Users
who write generic, repetitive, or empty reviews (typical of rating-only attacks
such as AUSH or GraphAttack) cluster at the origin and are easily flagged.

For users with no review text the document is the empty string, yielding an
all-zero TF-IDF vector — a signal in itself because legitimate users almost
always write some text.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.datasets.base import HeteroDataset
from .base import BaseDetector


class SemanticShieldBaseline(BaseDetector):
    """TF-IDF user-document features + logistic regression."""

    def __init__(self, max_features: int = 500, seed: int = 42) -> None:
        self.max_features = max_features
        self.seed         = seed
        self.vectorizer   = TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
        )
        self.clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                      random_state=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _user_docs(self, dataset: HeteroDataset) -> List[str]:
        """Concatenate all review texts per user → one document per user."""
        user_texts: defaultdict = defaultdict(list)
        for u, text in zip(dataset.review_user_idx.tolist(), dataset.review_texts):
            user_texts[u].append(str(text))
        return [
            " ".join(user_texts.get(u, [""])) for u in range(dataset.num_users)
        ]

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
        docs = self._user_docs(dataset)
        y    = dataset.labels.numpy()
        X    = self.vectorizer.fit_transform(docs)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            self._degenerate   = True
            self._fallback_val = float(y.max())
            return
        self._degenerate = False
        self.clf.fit(X, y)

    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        if getattr(self, "_degenerate", False):
            return np.full(dataset.num_users, self._fallback_val, dtype=np.float32)
        docs = self._user_docs(dataset)
        X    = self.vectorizer.transform(docs)
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)
