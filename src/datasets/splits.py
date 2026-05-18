from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def temporal_split(
    timestamps: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition reviews into train/val/test by chronological order.

    Reviews are sorted by timestamp; the first ``train_ratio`` fraction go to
    train, the next ``val_ratio`` to val, and the remainder to test.  Ties in
    timestamp are broken by their original position (stable sort).

    Returns three boolean masks of shape ``(n,)`` that are mutually exclusive
    and collectively exhaustive.
    """
    n = len(timestamps)
    order = np.argsort(timestamps, kind="stable")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[order[:train_end]] = True
    val_mask[order[train_end:val_end]] = True
    test_mask[order[val_end:]] = True

    return train_mask, val_mask, test_mask
