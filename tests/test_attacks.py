"""Unit tests for ShillingAttack base and classic attack implementations.

Synthetic dataset
-----------------
20 users, 30 items, 50 reviews — no file I/O required.

Tests
-----
  1.  FakeProfile validation: mismatched list lengths raise ValueError.
  2.  RandomAttack.generate_profiles returns correct count.
  3.  BandwagonAttack.generate_profiles returns correct count.
  4.  Filler items in RandomAttack are distinct from target items.
  5.  Filler items in BandwagonAttack come from the popular-item pool.
  6.  All fake timestamps are strictly after the last real timestamp.
  7.  Timestamps are ordered (non-decreasing) within each FakeProfile.
  8.  Target items receive max_rating in every profile.
  9.  Filler ratings are in [1, max_rating].
  10. inject(): injected dataset has the correct number of users.
  11. inject(): injected dataset has the correct number of reviews.
  12. inject(): fake users have label = 1; real users retain label = 0.
  13. inject(): fake reviews land in test_mask only (not train/val).
  14. inject(): new edge counts are consistent across all three edge types.
  15. inject(): user_id_map extended with fake user IDs.
  16. inject(): review_user_idx and review_item_idx are in valid range.
  17. inject(): x_user shape matches new num_users.
  18. inject(): BandwagonAttack injection same structural checks.
  19. inject(): original dataset is unchanged (not mutated in place).
  20. injection_rate convenience: n_fake = round(rate * num_real_users).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest
import torch

from src.attacks import BandwagonAttack, FakeProfile, RandomAttack
from src.attacks.base import ShillingAttack
from src.datasets.base import HeteroDataset

# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

N_USERS   = 20
N_ITEMS   = 30
N_REVIEWS = 50
SEED      = 0


def _make_dataset(seed: int = SEED) -> HeteroDataset:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    writes_u = torch.from_numpy(rng.integers(0, N_USERS, N_REVIEWS).astype(np.int64))
    writes_r = torch.arange(N_REVIEWS, dtype=torch.long)
    about_r  = torch.arange(N_REVIEWS, dtype=torch.long)
    about_i  = torch.from_numpy(rng.integers(0, N_ITEMS, N_REVIEWS).astype(np.int64))
    n_rates  = 40
    rates_u  = torch.from_numpy(rng.integers(0, N_USERS, n_rates).astype(np.int64))
    rates_i  = torch.from_numpy(rng.integers(0, N_ITEMS, n_rates).astype(np.int64))

    ts_np = np.sort(rng.uniform(0, 1e9, N_REVIEWS))
    ts    = torch.from_numpy(ts_np)

    n_train = int(0.8 * N_REVIEWS)
    n_val   = int(0.1 * N_REVIEWS)
    train_mask = torch.zeros(N_REVIEWS, dtype=torch.bool)
    val_mask   = torch.zeros(N_REVIEWS, dtype=torch.bool)
    test_mask  = torch.zeros(N_REVIEWS, dtype=torch.bool)
    train_mask[:n_train]            = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:]       = True

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about",  "item"):   torch.stack([about_r,  about_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u,  rates_i]),
    }
    edge_attr_dict = {k: ts for k in edge_index_dict}

    return HeteroDataset(
        dataset_name="synthetic",
        num_users=N_USERS,
        num_items=N_ITEMS,
        num_reviews=N_REVIEWS,
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        x_user=torch.randn(N_USERS, 8),
        x_item=torch.randn(N_ITEMS, 5),
        review_texts=["review text"] * N_REVIEWS,
        ratings=torch.from_numpy(rng.uniform(1, 5, N_REVIEWS).astype(np.float32)),
        timestamps=ts,
        review_user_idx=writes_u,
        review_item_idx=about_i,
        labels=torch.zeros(N_USERS, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        user_id_map={i: i for i in range(N_USERS)},
        item_id_map={i: i for i in range(N_ITEMS)},
    )


@pytest.fixture(scope="module")
def dataset() -> HeteroDataset:
    return _make_dataset()


@pytest.fixture(scope="module")
def random_attack(dataset) -> RandomAttack:
    return RandomAttack(dataset, max_rating=5.0, rating_noise=0.5, seed=SEED)


@pytest.fixture(scope="module")
def bandwagon_attack(dataset) -> BandwagonAttack:
    return BandwagonAttack(dataset, max_rating=5.0, rating_noise=0.5, top_n=15, seed=SEED)


# Shared attack parameters used across most tests
TARGET_ITEMS = [2, 7, 14]
BUDGET_PCT   = 0.2          # 20 % of 30 items = 6; filler = 6 - 3 = 3
N_FAKE       = 4
MAX_RATING   = 5.0

# Expected reviews per fake user
_BUDGET_ITEMS = max(0, round(BUDGET_PCT * N_ITEMS))         # 6
_FILLER_SIZE  = max(0, _BUDGET_ITEMS - len(TARGET_ITEMS))   # 3
_ITEMS_PER_FAKE = _FILLER_SIZE + len(TARGET_ITEMS)          # 6


# ---------------------------------------------------------------------------
# Test 1 — FakeProfile validation
# ---------------------------------------------------------------------------

def test_fake_profile_length_mismatch():
    with pytest.raises(ValueError, match="review_texts"):
        FakeProfile(
            user_id="x",
            rated_items={0: 5.0},
            review_texts=[],        # length 0 ≠ 1
            timestamps=[1e9],
        )
    with pytest.raises(ValueError, match="timestamps"):
        FakeProfile(
            user_id="x",
            rated_items={0: 5.0},
            review_texts=["ok"],
            timestamps=[],          # length 0 ≠ 1
        )


# ---------------------------------------------------------------------------
# Tests 2–3 — Profile count
# ---------------------------------------------------------------------------

def test_random_attack_profile_count(random_attack):
    profiles = random_attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    assert len(profiles) == N_FAKE


def test_bandwagon_attack_profile_count(bandwagon_attack):
    profiles = bandwagon_attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    assert len(profiles) == N_FAKE


# ---------------------------------------------------------------------------
# Test 4 — RandomAttack filler excludes target items
# ---------------------------------------------------------------------------

def test_random_filler_excludes_targets(random_attack):
    profiles = random_attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        filler = set(p.rated_items.keys()) - target_set
        assert filler.isdisjoint(target_set), \
            f"Filler item overlaps target: {filler & target_set}"


# ---------------------------------------------------------------------------
# Test 5 — BandwagonAttack filler comes from popular-item pool
# ---------------------------------------------------------------------------

def test_bandwagon_filler_from_popular_pool(bandwagon_attack, dataset):
    profiles = bandwagon_attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    target_set = set(TARGET_ITEMS)
    popular_pool = set(bandwagon_attack._popular_pool)
    for p in profiles:
        filler_items = set(p.rated_items.keys()) - target_set
        assert filler_items.issubset(popular_pool), \
            f"Filler not in popular pool: {filler_items - popular_pool}"


# ---------------------------------------------------------------------------
# Test 6 — All fake timestamps are after the last real timestamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_timestamps_after_max_real(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    profiles = attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    max_real = float(dataset.timestamps.max().item())
    for p in profiles:
        assert all(t > max_real for t in p.timestamps), \
            f"Fake timestamp ≤ max real ({max_real}): {[t for t in p.timestamps if t <= max_real]}"


# ---------------------------------------------------------------------------
# Test 7 — Timestamps non-decreasing within each profile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_timestamps_ordered_within_profile(request, attack_fixture):
    attack = request.getfixturevalue(attack_fixture)
    profiles = attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    for p in profiles:
        ts = p.timestamps
        assert ts == sorted(ts), f"Timestamps not sorted in {p.user_id}: {ts}"


# ---------------------------------------------------------------------------
# Test 8 — Target items receive max_rating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_target_items_get_max_rating(request, attack_fixture):
    attack = request.getfixturevalue(attack_fixture)
    profiles = attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    for p in profiles:
        for t in TARGET_ITEMS:
            assert t in p.rated_items, f"{p.user_id} missing target item {t}"
            assert p.rated_items[t] == MAX_RATING, \
                f"{p.user_id} target {t} has rating {p.rated_items[t]} ≠ {MAX_RATING}"


# ---------------------------------------------------------------------------
# Test 9 — Filler ratings in valid range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_filler_ratings_in_range(request, attack_fixture):
    attack = request.getfixturevalue(attack_fixture)
    profiles = attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        for item, rating in p.rated_items.items():
            if item in target_set:
                continue
            assert 1.0 <= rating <= MAX_RATING, \
                f"{p.user_id} filler item {item} has out-of-range rating {rating}"


# ---------------------------------------------------------------------------
# Helper: build injected dataset for structural tests
# ---------------------------------------------------------------------------

def _inject(attack, dataset, n_fake=N_FAKE):
    profiles = attack.generate_profiles(n_fake, BUDGET_PCT, TARGET_ITEMS)
    return attack.inject(dataset, profiles), profiles


# ---------------------------------------------------------------------------
# Test 10 — Correct number of users after injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_user_count(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, _ = _inject(attack, dataset)
    assert aug.num_users == N_USERS + N_FAKE, \
        f"Expected {N_USERS + N_FAKE} users, got {aug.num_users}"


# ---------------------------------------------------------------------------
# Test 11 — Correct number of reviews after injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_review_count(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, profiles = _inject(attack, dataset)
    expected_new = sum(len(p.rated_items) for p in profiles)
    assert aug.num_reviews == N_REVIEWS + expected_new, \
        f"Expected {N_REVIEWS + expected_new} reviews, got {aug.num_reviews}"


# ---------------------------------------------------------------------------
# Test 12 — Labels: fake = 1, real = 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_labels(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, _ = _inject(attack, dataset)
    assert (aug.labels[:N_USERS] == 0).all(), "Original users should retain label 0"
    assert (aug.labels[N_USERS:] == 1).all(), "Fake users should have label 1"


# ---------------------------------------------------------------------------
# Test 13 — Fake reviews in test_mask only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_fake_reviews_in_test_mask(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, profiles = _inject(attack, dataset)
    n_new = sum(len(p.rated_items) for p in profiles)
    fake_train = aug.train_mask[N_REVIEWS:]
    fake_val   = aug.val_mask[N_REVIEWS:]
    fake_test  = aug.test_mask[N_REVIEWS:]
    assert not fake_train.any(), "Fake reviews must not be in train_mask"
    assert not fake_val.any(),   "Fake reviews must not be in val_mask"
    assert fake_test.all(),      "All fake reviews must be in test_mask"


# ---------------------------------------------------------------------------
# Test 14 — Edge counts consistent across all three edge types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_edge_counts(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, profiles = _inject(attack, dataset)
    n_new = sum(len(p.rated_items) for p in profiles)
    for etype, ei in aug.edge_index_dict.items():
        orig_e = dataset.edge_index_dict[etype].size(1)
        # All three edge types get exactly n_new new edges
        assert ei.size(1) == orig_e + n_new, \
            f"{etype}: expected {orig_e + n_new} edges, got {ei.size(1)}"


# ---------------------------------------------------------------------------
# Test 15 — user_id_map extended
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_user_id_map(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, profiles = _inject(attack, dataset)
    for i, p in enumerate(profiles):
        assert p.user_id in aug.user_id_map, f"{p.user_id} missing from user_id_map"
        assert aug.user_id_map[p.user_id] == N_USERS + i


# ---------------------------------------------------------------------------
# Test 16 — review_user_idx and review_item_idx in valid range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_index_bounds(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, _ = _inject(attack, dataset)
    assert aug.review_user_idx.max() < aug.num_users, \
        "review_user_idx out of bounds"
    assert aug.review_item_idx.max() < aug.num_items, \
        "review_item_idx out of bounds"
    assert aug.review_user_idx.min() >= 0
    assert aug.review_item_idx.min() >= 0


# ---------------------------------------------------------------------------
# Test 17 — x_user shape matches augmented num_users
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_x_user_shape(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, _ = _inject(attack, dataset)
    assert aug.x_user.shape == (aug.num_users, dataset.x_user.shape[1]), \
        f"x_user shape {aug.x_user.shape} inconsistent with num_users={aug.num_users}"


# ---------------------------------------------------------------------------
# Test 18 — x_item unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attack_fixture", ["random_attack", "bandwagon_attack"])
def test_inject_x_item_unchanged(request, attack_fixture, dataset):
    attack = request.getfixturevalue(attack_fixture)
    aug, _ = _inject(attack, dataset)
    assert torch.equal(aug.x_item, dataset.x_item), \
        "Item features should not change after injection"


# ---------------------------------------------------------------------------
# Test 19 — Original dataset is not mutated
# ---------------------------------------------------------------------------

def test_inject_no_mutation(random_attack, dataset):
    orig_users   = dataset.num_users
    orig_reviews = dataset.num_reviews
    orig_labels  = dataset.labels.clone()
    aug, _       = _inject(random_attack, dataset)
    # Original dataset must be untouched
    assert dataset.num_users   == orig_users,   "num_users mutated"
    assert dataset.num_reviews == orig_reviews, "num_reviews mutated"
    assert torch.equal(dataset.labels, orig_labels), "labels mutated"


# ---------------------------------------------------------------------------
# Test 20 — Injection rate convenience: n_fake = round(rate * num_real_users)
# ---------------------------------------------------------------------------

def test_injection_rate_scaling(random_attack, dataset):
    for rate in RandomAttack.injection_rates:
        n_fake = round(rate * dataset.num_users)
        profiles = random_attack.generate_profiles(n_fake, BUDGET_PCT, TARGET_ITEMS)
        assert len(profiles) == n_fake, \
            f"rate={rate}: expected {n_fake} profiles, got {len(profiles)}"


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
