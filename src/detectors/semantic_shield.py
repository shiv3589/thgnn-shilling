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
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.datasets.base import HeteroDataset
from .base import BaseDetector


class SemanticShieldBaseline(BaseDetector):
    """TF-IDF user-document features + unsupervised outlier scoring
    (IsolationForest).  Never sees ground-truth labels — contamination is set
    from the known injection rate, mirroring what a deployed detector would
    assume."""

    def __init__(self, max_features: int = 500, seed: int = 42,
                 injection_rate: float = 0.01) -> None:
        self.max_features = max_features
        self.seed         = seed
        self.vectorizer   = TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
        )
        contamination = float(np.clip(injection_rate, 1e-3, 0.5))
        self.clf = IsolationForest(contamination=contamination, random_state=seed)

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
        X    = self.vectorizer.fit_transform(docs)
        self.clf.fit(X)

    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        docs = self._user_docs(dataset)
        X    = self.vectorizer.transform(docs)
        return (-self.clf.score_samples(X)).astype(np.float32)
