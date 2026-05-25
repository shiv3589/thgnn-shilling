"""Abstract base class for all shilling-attack detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.datasets.base import HeteroDataset


class BaseDetector(ABC):
    """Common interface for all detectors used in the experiment matrix.

    A detector receives a (possibly poisoned) ``HeteroDataset`` whose
    ``labels`` tensor contains ground-truth shilling flags (0=real, 1=fake),
    fits itself on that data, and scores every user with an anomaly probability.

    Parameters to ``fit`` and ``score_users`` are intentionally minimal so the
    experiment runner can call them uniformly across all detector types.
    """

    @abstractmethod
    def fit(
        self,
        dataset: HeteroDataset,
        *,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        """Train or fit the detector on *dataset*.

        *dataset* is the poisoned graph; ``dataset.labels`` provides
        ground-truth class labels for supervised detectors.
        """

    @abstractmethod
    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        """Return a per-user anomaly score in ``[0, 1]``.

        Higher values indicate greater suspicion of shilling.
        Shape: ``(dataset.num_users,)``, dtype float32.
        """

    def predict(
        self,
        dataset: HeteroDataset,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Binary predictions (1 = fake) using *threshold*. Shape: (num_users,)."""
        return (self.score_users(dataset) >= threshold).astype(np.int64)
