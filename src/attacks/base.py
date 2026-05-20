"""Abstract base for shilling attack generators.

A ShillingAttack subclass must implement ``generate_profiles`` to produce a
list of ``FakeProfile`` objects, each describing one synthetic attacker.
The concrete ``inject`` method on this base class handles all graph surgery
(node / edge / feature augmentation) in a dataset-agnostic way.

Injection contract
------------------
* Fake users receive contiguous global indices [num_real_users, …).
* Fake reviews receive contiguous global indices [num_real_reviews, …).
* All fake timestamps are strictly greater than the last real timestamp.
* Fake user labels are set to 1; all existing labels are preserved.
* Fake reviews are assigned to the *test* split (train_mask = val_mask = False,
  test_mask = True) so they do not contaminate the learning signal.
* Item features (x_item) are not recomputed; item nodes do not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.datasets.base import HeteroDataset, _node_stats


# ---------------------------------------------------------------------------
# FakeProfile
# ---------------------------------------------------------------------------

@dataclass
class FakeProfile:
    """One synthetic attacker profile.

    Attributes
    ----------
    user_id      : str  — unique identifier (e.g. ``"fake_42"``)
    rated_items  : Dict[int, float]  — item global index → star rating
    review_texts : List[str]         — one placeholder text per rated item;
                   order must match ``list(rated_items.items())``.
    timestamps   : List[float]       — Unix-second timestamp per rated item;
                   must all be > max real timestamp and ordered within the profile.
    """
    user_id: str
    rated_items: Dict[int, float] = field(default_factory=dict)
    review_texts: List[str]       = field(default_factory=list)
    timestamps: List[float]       = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.rated_items)
        if len(self.review_texts) != n:
            raise ValueError(
                f"review_texts length {len(self.review_texts)} != rated_items length {n}"
            )
        if len(self.timestamps) != n:
            raise ValueError(
                f"timestamps length {len(self.timestamps)} != rated_items length {n}"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ShillingAttack(ABC):
    """Abstract base for shilling attack generators.

    Subclasses implement ``generate_profiles``; the base provides ``inject``.

    Parameters
    ----------
    dataset : HeteroDataset  — clean reference dataset (used for statistics)
    seed    : int            — RNG seed for reproducibility
    """

    # Subclasses should override this with the injection rates they support.
    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(self, dataset: HeteroDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Create *n_fake_users* attacker profiles.

        Parameters
        ----------
        n_fake_users : int   — how many fake users to generate
        budget_pct   : float — fraction of the item catalog each fake user rates
                               (includes target items; filler = budget − targets)
        target_items : List[int]  — item global indices to push to max rating

        Returns
        -------
        List[FakeProfile] of length *n_fake_users*
        """
        ...

    # ------------------------------------------------------------------
    # Concrete injection
    # ------------------------------------------------------------------

    def inject(
        self,
        dataset: HeteroDataset,
        profiles: List[FakeProfile],
    ) -> HeteroDataset:
        """Augment *dataset* with the synthetic attacker profiles.

        Returns a new HeteroDataset; the original is not modified.
        """
        n_real_u = dataset.num_users
        n_real_r = dataset.num_reviews
        n_fake_u = len(profiles)

        # ── Flatten all fake reviews ──────────────────────────────────────
        fake_ratings:   List[float] = []
        fake_texts:     List[str]   = []
        fake_ts_list:   List[float] = []
        fake_rev_user:  List[int]   = []
        fake_rev_item:  List[int]   = []

        for uid_offset, profile in enumerate(profiles):
            u_global = n_real_u + uid_offset
            for (item_idx, rating), text, ts in zip(
                profile.rated_items.items(),
                profile.review_texts,
                profile.timestamps,
            ):
                fake_ratings.append(rating)
                fake_texts.append(text)
                fake_ts_list.append(ts)
                fake_rev_user.append(u_global)
                fake_rev_item.append(item_idx)

        n_fake_r = len(fake_ratings)

        if n_fake_r == 0:
            # No-op: nothing to inject
            return dataset

        fake_r_idx = torch.arange(n_real_r, n_real_r + n_fake_r, dtype=torch.long)
        fake_u_t   = torch.tensor(fake_rev_user, dtype=torch.long)
        fake_i_t   = torch.tensor(fake_rev_item, dtype=torch.long)
        fake_ts_t  = torch.tensor(fake_ts_list,
                                  dtype=dataset.timestamps.dtype)
        fake_rat_t = torch.tensor(fake_ratings, dtype=torch.float32)

        # ── Augment edges ─────────────────────────────────────────────────
        _new_edges = {
            ("user", "writes", "review"): torch.stack([fake_u_t, fake_r_idx]),
            ("review", "about", "item"):  torch.stack([fake_r_idx, fake_i_t]),
            ("user", "rates", "item"):    torch.stack([fake_u_t, fake_i_t]),
        }
        new_eid = {
            et: torch.cat([dataset.edge_index_dict[et], _new_edges[et]], dim=1)
            for et in dataset.edge_index_dict
        }
        new_ea = {
            et: torch.cat([dataset.edge_attr_dict[et], fake_ts_t])
            for et in dataset.edge_attr_dict
        }

        # ── Compute x_user for fake users ────────────────────────────────
        # Reuse the same _node_stats helper as the dataset loaders.
        local_uid = np.array(fake_rev_user, dtype=np.int64) - n_real_u  # 0-based local
        fake_x_base = _node_stats(
            local_uid,
            np.array(fake_ratings, dtype=np.float32),
            np.array(fake_ts_list, dtype=np.float64),
            n_fake_u,
        )
        # Pad any dataset-specific extra dims with zeros
        extra_dims = dataset.x_user.shape[1] - 5
        if extra_dims > 0:
            fake_x_user_np = np.concatenate(
                [fake_x_base, np.zeros((n_fake_u, extra_dims), dtype=np.float32)],
                axis=1,
            )
        else:
            fake_x_user_np = fake_x_base
        new_x_user = torch.cat(
            [dataset.x_user, torch.from_numpy(fake_x_user_np)], dim=0
        )

        # ── Labels (fake users → 1) ───────────────────────────────────────
        new_labels = torch.cat([
            dataset.labels,
            torch.ones(n_fake_u, dtype=torch.long),
        ])

        # ── Review-level arrays ───────────────────────────────────────────
        new_review_texts  = dataset.review_texts + fake_texts
        new_ratings       = torch.cat([dataset.ratings, fake_rat_t])
        new_timestamps    = torch.cat([dataset.timestamps, fake_ts_t])
        new_rev_user_idx  = torch.cat([dataset.review_user_idx, fake_u_t])
        new_rev_item_idx  = torch.cat([dataset.review_item_idx, fake_i_t])

        # ── Masks: all fake reviews → test split ─────────────────────────
        _f = torch.zeros(n_fake_r, dtype=torch.bool)
        new_train_mask = torch.cat([dataset.train_mask, _f])
        new_val_mask   = torch.cat([dataset.val_mask,   _f])
        new_test_mask  = torch.cat([dataset.test_mask,  torch.ones(n_fake_r, dtype=torch.bool)])

        # ── user_id_map ───────────────────────────────────────────────────
        new_user_id_map = dict(dataset.user_id_map)
        for offset, profile in enumerate(profiles):
            new_user_id_map[profile.user_id] = n_real_u + offset

        return HeteroDataset(
            dataset_name=dataset.dataset_name,
            num_users=n_real_u + n_fake_u,
            num_items=dataset.num_items,
            num_reviews=n_real_r + n_fake_r,
            edge_index_dict=new_eid,
            edge_attr_dict=new_ea,
            x_user=new_x_user,
            x_item=dataset.x_item,
            review_texts=new_review_texts,
            ratings=new_ratings,
            timestamps=new_timestamps,
            review_user_idx=new_rev_user_idx,
            review_item_idx=new_rev_item_idx,
            labels=new_labels,
            train_mask=new_train_mask,
            val_mask=new_val_mask,
            test_mask=new_test_mask,
            user_id_map=new_user_id_map,
            item_id_map=dataset.item_id_map,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _placeholder_text(self, item_idx: int, rating: float) -> str:
        """Return a short generic review text for a given rating."""
        _positive = [
            "good product",
            "great item",
            "highly recommend",
            "really enjoyed this",
            "excellent quality",
            "would buy again",
            "very satisfied",
            "works perfectly",
        ]
        _neutral = [
            "decent product",
            "okay item",
            "meets expectations",
            "fair value",
        ]
        _negative = [
            "not great",
            "below average",
            "disappointed",
            "would not recommend",
        ]
        if rating >= 4.0:
            pool = _positive
        elif rating >= 3.0:
            pool = _neutral
        else:
            pool = _negative
        idx = int(self.rng.integers(0, len(pool)))
        return pool[idx]

    def _spread_timestamps(
        self,
        n_reviews: int,
        base_ts: float,
        *,
        min_gap_secs: float = 3_600.0,   # 1 hour between reviews
        start_offset: float = 86_400.0,  # first review ≥ 1 day after base
    ) -> List[float]:
        """Return *n_reviews* timestamps strictly greater than *base_ts*.

        Reviews are spaced *min_gap_secs* apart, with a random jitter of up to
        *min_gap_secs* added to each successive timestamp so the sequence is
        realistic but not perfectly uniform.
        """
        ts = float(base_ts) + start_offset
        result: List[float] = []
        for _ in range(n_reviews):
            result.append(ts)
            ts += min_gap_secs + float(self.rng.uniform(0, min_gap_secs))
        return result
