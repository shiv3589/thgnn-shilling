"""Unit tests for GraphAttack.

Tests
-----
  1.  Profile count matches n_fake_users.
  2.  All fake timestamps are strictly after the last real timestamp.
  3.  Timestamps are non-decreasing within each profile.
  4.  Target items are present and rated at max_rating in every profile.
  5.  Filler ratings are in [1, max_rating] and rounded to nearest 0.5.
  6.  Filler items are distinct from target items.
  7.  Budget constraint: len(rated_items) == filler_size + len(target_items).
  8.  inject(): correct user / review counts after injection.
  9.  inject(): fake users labelled 1, real users 0.
  10. inject(): fake reviews in test_mask only.
  11. inject(): review_user_idx and review_item_idx in valid range.
  12. loss_trace() returns a list of length n_steps with finite values.
  13. loss_trace() shows a decreasing trend (first loss > last loss on average).
  14. Different seeds produce different profiles.
  15. Zero filler: when budget covers only target items, no filler is added.
  16. generate_profiles works with n_fake_users=1 (edge case).
  17. review_texts are non-empty strings (GraphAttack uses placeholder texts).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attacks import FakeProfile, GraphAttack
from src.attacks.graph_attack import _discretise, _soft_ndcg_loss
from src.datasets.base import HeteroDataset
from src.models.lightgcn import LightGCN

# ---------------------------------------------------------------------------
# Synthetic dataset & surrogate
# ---------------------------------------------------------------------------

N_USERS   = 50
N_ITEMS   = 80
N_REVIEWS = 120
SEED      = 0


def _make_surrogate(dataset: HeteroDataset, embed_dim: int = 16) -> LightGCN:
    """Tiny LightGCN surrogate (randomly initialised, not trained)."""
    torch.manual_seed(SEED)
    return LightGCN(dataset.num_users, dataset.num_items,
                    embed_dim=embed_dim, n_layers=1)


# Shared test parameters
TARGET_ITEMS  = [1, 5, 12]
BUDGET_PCT    = 0.2              # 20% of 80 = 16; filler = 16 - 3 = 13
MAX_RATING    = 5.0
N_FAKE        = 6
_BUDGET_ITEMS = max(0, round(BUDGET_PCT * N_ITEMS))        # 16
_FILLER_SIZE  = max(0, _BUDGET_ITEMS - len(TARGET_ITEMS))  # 13
_ITEMS_PER    = _FILLER_SIZE + len(TARGET_ITEMS)           # 16


@pytest.fixture(scope="module")
def dataset(synthetic_graph) -> HeteroDataset:
    return synthetic_graph


@pytest.fixture(scope="module")
def surrogate(dataset) -> LightGCN:
    return _make_surrogate(dataset)


@pytest.fixture(scope="module")
def attack(surrogate, dataset) -> GraphAttack:
    return GraphAttack(
        surrogate, dataset,
        n_steps=5,          # fast for tests
        lr=0.1,
        temp=0.5,
        max_rating=MAX_RATING,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def profiles(attack) -> list:
    return attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)


# ---------------------------------------------------------------------------
# Test 1 — Profile count
# ---------------------------------------------------------------------------

def test_profile_count(profiles):
    assert len(profiles) == N_FAKE


# ---------------------------------------------------------------------------
# Test 2 — Timestamps after max real timestamp
# ---------------------------------------------------------------------------

def test_timestamps_after_max_real(profiles, dataset):
    max_real = float(dataset.timestamps.max().item())
    for p in profiles:
        assert all(t > max_real for t in p.timestamps), \
            f"{p.user_id}: fake timestamp ≤ max real ({max_real})"


# ---------------------------------------------------------------------------
# Test 3 — Timestamps non-decreasing within each profile
# ---------------------------------------------------------------------------

def test_timestamps_ordered(profiles):
    for p in profiles:
        assert p.timestamps == sorted(p.timestamps), \
            f"{p.user_id}: timestamps not sorted"


# ---------------------------------------------------------------------------
# Test 4 — Target items present and at max_rating
# ---------------------------------------------------------------------------

def test_target_items_max_rating(profiles):
    for p in profiles:
        for t in TARGET_ITEMS:
            assert t in p.rated_items, f"{p.user_id} missing target {t}"
            assert p.rated_items[t] == MAX_RATING, \
                f"{p.user_id} target {t}: {p.rated_items[t]} ≠ {MAX_RATING}"


# ---------------------------------------------------------------------------
# Test 5 — Filler ratings in [1, max_rating], rounded to 0.5
# ---------------------------------------------------------------------------

def test_filler_rating_range(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        for item, r in p.rated_items.items():
            if item in target_set:
                continue
            assert 1.0 <= r <= MAX_RATING, \
                f"{p.user_id} item {item} out-of-range rating {r}"
            assert abs(r * 2 - round(r * 2)) < 1e-6, \
                f"{p.user_id} item {item} not rounded to 0.5: {r}"


# ---------------------------------------------------------------------------
# Test 6 — Filler items distinct from targets
# ---------------------------------------------------------------------------

def test_filler_excludes_targets(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        filler = set(p.rated_items.keys()) - target_set
        assert filler.isdisjoint(target_set), \
            f"{p.user_id} filler overlaps targets: {filler & target_set}"


# ---------------------------------------------------------------------------
# Test 7 — Budget constraint
# ---------------------------------------------------------------------------

def test_budget_constraint(profiles):
    for p in profiles:
        assert len(p.rated_items) == _ITEMS_PER, \
            f"{p.user_id}: expected {_ITEMS_PER} items, got {len(p.rated_items)}"


# ---------------------------------------------------------------------------
# Tests 8–11 — inject() structural invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def augmented(attack, dataset, profiles):
    return attack.inject(dataset, profiles)


def test_inject_user_count(augmented):
    assert augmented.num_users == N_USERS + N_FAKE


def test_inject_review_count(augmented, profiles):
    n_new = sum(len(p.rated_items) for p in profiles)
    assert augmented.num_reviews == N_REVIEWS + n_new


def test_inject_labels(augmented):
    assert (augmented.labels[:N_USERS] == 0).all()
    assert (augmented.labels[N_USERS:] == 1).all()


def test_inject_masks(augmented):
    assert not augmented.train_mask[N_REVIEWS:].any()
    assert not augmented.val_mask[N_REVIEWS:].any()
    assert augmented.test_mask[N_REVIEWS:].all()


def test_inject_index_bounds(augmented):
    assert augmented.review_user_idx.max() < augmented.num_users
    assert augmented.review_item_idx.max() < augmented.num_items
    assert augmented.review_user_idx.min() >= 0
    assert augmented.review_item_idx.min() >= 0


# ---------------------------------------------------------------------------
# Test 12 — loss_trace() length and finiteness
# ---------------------------------------------------------------------------

def test_loss_trace_length_and_finite(surrogate, dataset):
    n_steps = 8
    trace = GraphAttack.loss_trace(
        surrogate, dataset, TARGET_ITEMS,
        n_steps=n_steps, lr=0.1, temp=0.5, max_rating=MAX_RATING, seed=1,
    )
    assert len(trace) == n_steps, f"Expected {n_steps} losses, got {len(trace)}"
    assert all(np.isfinite(v) for v in trace), f"Non-finite losses: {trace}"


# ---------------------------------------------------------------------------
# Test 13 — loss_trace() shows optimisation progress
# ---------------------------------------------------------------------------

def test_loss_trace_decreasing(surrogate, dataset):
    """Mean of the first half should be ≥ mean of the second half."""
    trace = GraphAttack.loss_trace(
        surrogate, dataset, TARGET_ITEMS,
        n_steps=20, lr=0.2, temp=0.5, max_rating=MAX_RATING, seed=0,
    )
    first_half  = float(np.mean(trace[:10]))
    second_half = float(np.mean(trace[10:]))
    assert first_half >= second_half, \
        f"Loss did not decrease: first_half={first_half:.4f} > second_half={second_half:.4f}"


# ---------------------------------------------------------------------------
# Test 14 — Different seeds → different profiles
# ---------------------------------------------------------------------------

def test_different_seeds_differ(dataset, surrogate):
    a1 = GraphAttack(surrogate, dataset, n_steps=5, seed=10, max_rating=MAX_RATING)
    a2 = GraphAttack(surrogate, dataset, n_steps=5, seed=20, max_rating=MAX_RATING)
    p1 = a1.generate_profiles(3, BUDGET_PCT, TARGET_ITEMS)
    p2 = a2.generate_profiles(3, BUDGET_PCT, TARGET_ITEMS)
    r1 = list(p1[0].rated_items.values())
    r2 = list(p2[0].rated_items.values())
    assert r1 != r2, "Different seeds produced identical profiles"


# ---------------------------------------------------------------------------
# Test 15 — Zero filler (budget exactly covers target items)
# ---------------------------------------------------------------------------

def test_zero_filler(dataset, surrogate):
    exact_budget = len(TARGET_ITEMS) / N_ITEMS   # filler_size = 0
    attack = GraphAttack(surrogate, dataset, n_steps=3, seed=7, max_rating=MAX_RATING)
    profiles = attack.generate_profiles(3, exact_budget, TARGET_ITEMS)
    for p in profiles:
        assert set(p.rated_items.keys()) == set(TARGET_ITEMS), \
            f"Expected only target items, got {set(p.rated_items.keys())}"
        assert all(r == MAX_RATING for r in p.rated_items.values())


# ---------------------------------------------------------------------------
# Test 16 — Single fake user (n_fake_users=1)
# ---------------------------------------------------------------------------

def test_single_fake_user(dataset, surrogate):
    attack = GraphAttack(surrogate, dataset, n_steps=3, seed=3, max_rating=MAX_RATING)
    profiles = attack.generate_profiles(1, BUDGET_PCT, TARGET_ITEMS)
    assert len(profiles) == 1
    p = profiles[0]
    assert len(p.rated_items) == _ITEMS_PER
    for t in TARGET_ITEMS:
        assert p.rated_items[t] == MAX_RATING


# ---------------------------------------------------------------------------
# Test 17 — review_texts are non-empty strings (placeholder texts)
# ---------------------------------------------------------------------------

def test_review_texts_nonempty(profiles):
    for p in profiles:
        assert len(p.review_texts) == len(p.rated_items)
        assert all(isinstance(t, str) and len(t) > 0 for t in p.review_texts), \
            f"{p.user_id}: found empty review text(s)"


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers
# ---------------------------------------------------------------------------

def test_discretise_rounding():
    assert _discretise(4.3, 5.0) == 4.5
    assert _discretise(3.7, 5.0) == 3.5
    assert _discretise(1.0, 5.0) == 1.0
    assert _discretise(5.0, 5.0) == 5.0
    assert _discretise(0.1, 5.0) == 1.0   # clipped to min
    assert _discretise(9.9, 5.0) == 5.0   # clipped to max


def test_soft_ndcg_loss_scalar():
    """_soft_ndcg_loss should return a finite non-positive scalar."""
    n_fake, n_items = 4, 20
    target_idx     = torch.tensor([0, 5, 10], dtype=torch.long)
    non_target_mask = torch.ones(n_items, dtype=torch.bool)
    non_target_mask[target_idx] = False
    non_target_idx = torch.where(non_target_mask)[0]

    scores    = torch.randn(n_fake, n_items)
    K_t       = len(target_idx)
    ideal_dcg = float(sum(1.0 / np.log2(r + 2) for r in range(K_t)))

    loss = _soft_ndcg_loss(scores, target_idx, non_target_idx, ideal_dcg, temp=0.5)
    assert loss.shape == (), "Expected scalar loss"
    assert float(loss.item()) <= 0.0, "−NDCG should be ≤ 0"
    assert np.isfinite(loss.item())


def test_soft_ndcg_loss_gradient():
    """Gradient of _soft_ndcg_loss should flow through scores."""
    n_fake, n_items = 2, 15
    target_idx     = torch.tensor([3, 7], dtype=torch.long)
    non_target_mask = torch.ones(n_items, dtype=torch.bool)
    non_target_mask[target_idx] = False
    non_target_idx = torch.where(non_target_mask)[0]
    ideal_dcg      = float(sum(1.0 / np.log2(r + 2) for r in range(2)))

    scores = torch.randn(n_fake, n_items, requires_grad=True)
    loss   = _soft_ndcg_loss(scores, target_idx, non_target_idx, ideal_dcg, temp=0.5)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.shape == (n_fake, n_items)
    assert scores.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
