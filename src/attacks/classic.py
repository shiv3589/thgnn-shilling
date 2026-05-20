"""Classic shilling attack strategies.

RandomAttack
    Each fake user rates a random sample of filler items at approximately the
    dataset mean rating (± Gaussian noise), then rates all target items at the
    maximum rating.  Establishes the baseline injection difficulty.

BandwagonAttack
    Identical to RandomAttack but filler items are drawn from the *top-N most
    popular* items in the dataset.  Popularity bias makes profiles appear more
    legitimate to collaborative-filter models, increasing attack potency.

Both classes expose:
    injection_rates : List[float]   — canonical fractions of real users to inject
    generate_profiles(n, budget_pct, target_items) → List[FakeProfile]
    inject(dataset, profiles) → HeteroDataset        (inherited from base)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.datasets.base import HeteroDataset
from .base import FakeProfile, ShillingAttack


# ---------------------------------------------------------------------------
# RandomAttack
# ---------------------------------------------------------------------------

class RandomAttack(ShillingAttack):
    """Shilling attack with uniformly random filler items.

    Parameters
    ----------
    dataset     : HeteroDataset  — clean reference dataset
    max_rating  : float          — rating assigned to target items (default 5.0)
    rating_noise: float          — std of Gaussian noise added to mean rating for
                                   filler items; clipped to [1, max_rating]
    seed        : int            — RNG seed
    """

    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(
        self,
        dataset: HeteroDataset,
        max_rating: float = 5.0,
        rating_noise: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, seed)
        self.max_rating   = max_rating
        self.rating_noise = rating_noise
        self._mean_rating = float(dataset.ratings.mean().item())
        self._num_items   = dataset.num_items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Generate *n_fake_users* random-attack profiles.

        Parameters
        ----------
        n_fake_users : int   — number of synthetic attackers
        budget_pct   : float — fraction of the item catalog rated per fake user;
                               filler_size = max(0, round(budget_pct * num_items)
                                             − len(target_items))
        target_items : List[int]  — item global indices to push to max rating

        Returns
        -------
        List[FakeProfile]
        """
        target_set  = set(target_items)
        filler_pool = [i for i in range(self._num_items) if i not in target_set]
        filler_size = max(0, round(budget_pct * self._num_items) - len(target_items))

        base_ts = float(self.dataset.timestamps.max().item())
        profiles: List[FakeProfile] = []

        for uid in range(n_fake_users):
            filler_idx = _sample_fillers(
                filler_pool, filler_size, self.rng, replace=False
            )
            rated   = _build_rated_items(
                filler_idx, target_items,
                self._mean_rating, self.rating_noise, self.max_rating, self.rng,
            )
            texts  = [self._placeholder_text(it, r) for it, r in rated.items()]
            # Stagger each fake user's start time by uid * 30 min so profiles
            # are temporally distinct even when many users share the same base.
            user_base = base_ts + uid * 1_800.0
            tss = self._spread_timestamps(len(rated), user_base)
            profiles.append(FakeProfile(
                user_id=f"fake_{uid}",
                rated_items=rated,
                review_texts=texts,
                timestamps=tss,
            ))

        return profiles


# ---------------------------------------------------------------------------
# BandwagonAttack
# ---------------------------------------------------------------------------

class BandwagonAttack(ShillingAttack):
    """Shilling attack that uses popular items as filler.

    Filler items are sampled (without replacement) from the *top_n* most
    frequently rated items in the clean dataset, making fake profiles resemble
    genuine power-users and harder to detect by simple popularity filters.

    Parameters
    ----------
    dataset     : HeteroDataset
    max_rating  : float  — rating for target items (default 5.0)
    rating_noise: float  — Gaussian noise std for filler ratings
    top_n       : int    — size of the popular-item pool (default 50)
    seed        : int
    """

    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(
        self,
        dataset: HeteroDataset,
        max_rating: float = 5.0,
        rating_noise: float = 0.5,
        top_n: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, seed)
        self.max_rating   = max_rating
        self.rating_noise = rating_noise
        self._mean_rating = float(dataset.ratings.mean().item())
        self._num_items   = dataset.num_items

        # Pre-compute item popularity (degree) and sort descending
        item_counts = np.bincount(
            dataset.review_item_idx.numpy(), minlength=dataset.num_items
        )
        popular_order = np.argsort(item_counts)[::-1]
        self._popular_pool: List[int] = popular_order[:top_n].tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Generate *n_fake_users* bandwagon-attack profiles.

        Filler items are drawn exclusively from the top-N popular items,
        excluding any that are already target items.
        """
        target_set    = set(target_items)
        filler_pool   = [i for i in self._popular_pool if i not in target_set]
        filler_size   = max(0, round(budget_pct * self._num_items) - len(target_items))

        base_ts = float(self.dataset.timestamps.max().item())
        profiles: List[FakeProfile] = []

        for uid in range(n_fake_users):
            filler_idx = _sample_fillers(
                filler_pool, filler_size, self.rng, replace=False
            )
            rated  = _build_rated_items(
                filler_idx, target_items,
                self._mean_rating, self.rating_noise, self.max_rating, self.rng,
            )
            texts  = [self._placeholder_text(it, r) for it, r in rated.items()]
            user_base = base_ts + uid * 1_800.0
            tss = self._spread_timestamps(len(rated), user_base)
            profiles.append(FakeProfile(
                user_id=f"fake_{uid}",
                rated_items=rated,
                review_texts=texts,
                timestamps=tss,
            ))

        return profiles


# ---------------------------------------------------------------------------
# Private helpers shared by both attacks
# ---------------------------------------------------------------------------

def _sample_fillers(
    pool: List[int],
    k: int,
    rng: np.random.Generator,
    replace: bool = False,
) -> List[int]:
    """Sample up to *k* items from *pool* (shrink if pool is smaller)."""
    if k == 0 or not pool:
        return []
    k = min(k, len(pool))
    return rng.choice(pool, size=k, replace=replace).tolist()


def _build_rated_items(
    filler_idx: List[int],
    target_items: List[int],
    mean_rating: float,
    noise_std: float,
    max_rating: float,
    rng: np.random.Generator,
) -> dict:
    """Combine filler + target ratings into an ordered dict.

    Filler ratings = clip(N(mean, noise_std), 1, max_rating), rounded to 0.5.
    Target ratings = max_rating.
    """
    rated: dict = {}
    for item in filler_idx:
        r = float(np.clip(rng.normal(mean_rating, noise_std), 1.0, max_rating))
        # Round to nearest 0.5 (common in star-rating systems)
        rated[item] = round(r * 2) / 2
    for item in target_items:
        rated[item] = max_rating
    return rated
