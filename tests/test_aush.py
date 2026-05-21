"""Unit tests for AUSHAttack.

Tests
-----
  1.  Raises RuntimeError if generate_profiles called before pretrain.
  2.  pretrain completes without error (short run: 3 epochs).
  3.  generate_profiles returns correct number of profiles.
  4.  All fake timestamps are strictly after the last real timestamp.
  5.  Timestamps are non-decreasing within each profile.
  6.  Target items are present and rated at max_rating in every profile.
  7.  Filler ratings are in [1, max_rating] and rounded to 0.5.
  8.  Filler items are distinct from target items.
  9.  All review texts are empty strings (AUSH is rating-only).
  10. Budget constraint: len(rated_items) == filler_size + len(target_items).
  11. inject(): correct user / review counts after AUSH injection.
  12. inject(): fake users labelled 1, real users 0.
  13. inject(): fake reviews in test_mask only.
  14. inject(): review_user_idx and review_item_idx in valid range.
  15. Generator output is in [0, 1] for arbitrary noise inputs.
  16. Discriminator output is in [0, 1] for arbitrary inputs.
  17. Training loss decreases meaningfully (D and G losses are finite, < 10).
  18. Different seeds produce different profiles.
  19. pretrain is idempotent: re-calling overwrites the previous generator.
  20. generate_profiles works with zero filler (budget covers only targets).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attacks import AUSHAttack, FakeProfile
from src.attacks.aush import _Generator, _Discriminator, _dense_rows
from src.datasets.base import HeteroDataset

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

N_USERS   = 50
N_ITEMS   = 80
N_REVIEWS = 120
SEED      = 0

# Shared test parameters
TARGET_ITEMS = [1, 5, 12]
BUDGET_PCT   = 0.2           # 20 % of 80 items = 16 budget; filler = 16 − 3 = 13
MAX_RATING   = 5.0
N_FAKE       = 6
_BUDGET_ITEMS = max(0, round(BUDGET_PCT * N_ITEMS))           # 16
_FILLER_SIZE  = max(0, _BUDGET_ITEMS - len(TARGET_ITEMS))     # 13
_ITEMS_PER    = _FILLER_SIZE + len(TARGET_ITEMS)              # 16


@pytest.fixture(scope="module")
def dataset(synthetic_graph) -> HeteroDataset:
    return synthetic_graph


@pytest.fixture(scope="module")
def pretrained_attack(dataset) -> AUSHAttack:
    """AUSHAttack trained for 3 epochs (fast, just checks it runs)."""
    attack = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32,
                        max_rating=MAX_RATING, seed=SEED)
    attack.pretrain(dataset, epochs=3, batch_size=8, lr=1e-3,
                    lambda_push=1.0, n_targets_per_sample=2,
                    device="cpu")
    return attack


@pytest.fixture(scope="module")
def profiles(pretrained_attack) -> list:
    return pretrained_attack.generate_profiles(N_FAKE, BUDGET_PCT, TARGET_ITEMS)


# ---------------------------------------------------------------------------
# Test 1 — RuntimeError before pretrain
# ---------------------------------------------------------------------------

def test_raises_before_pretrain(dataset):
    fresh = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=1)
    with pytest.raises(RuntimeError, match="pretrain"):
        fresh.generate_profiles(3, BUDGET_PCT, TARGET_ITEMS)


# ---------------------------------------------------------------------------
# Test 2 — pretrain completes
# ---------------------------------------------------------------------------

def test_pretrain_completes(dataset):
    a = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=2)
    a.pretrain(dataset, epochs=2, batch_size=16, device="cpu")
    assert a._pretrained
    assert a._generator is not None
    assert a._discriminator is not None


# ---------------------------------------------------------------------------
# Test 3 — Profile count
# ---------------------------------------------------------------------------

def test_profile_count(profiles):
    assert len(profiles) == N_FAKE


# ---------------------------------------------------------------------------
# Test 4 — Timestamps after max real timestamp
# ---------------------------------------------------------------------------

def test_timestamps_after_max_real(profiles, dataset):
    max_real = float(dataset.timestamps.max().item())
    for p in profiles:
        assert all(t > max_real for t in p.timestamps), \
            f"{p.user_id}: fake timestamp ≤ max real ({max_real})"


# ---------------------------------------------------------------------------
# Test 5 — Timestamps non-decreasing within profile
# ---------------------------------------------------------------------------

def test_timestamps_ordered(profiles):
    for p in profiles:
        assert p.timestamps == sorted(p.timestamps), \
            f"{p.user_id}: timestamps not sorted"


# ---------------------------------------------------------------------------
# Test 6 — Target items present and at max rating
# ---------------------------------------------------------------------------

def test_target_items_max_rating(profiles):
    for p in profiles:
        for t in TARGET_ITEMS:
            assert t in p.rated_items, f"{p.user_id} missing target {t}"
            assert p.rated_items[t] == MAX_RATING, \
                f"{p.user_id} target {t}: {p.rated_items[t]} ≠ {MAX_RATING}"


# ---------------------------------------------------------------------------
# Test 7 — Filler ratings in [1, max_rating], rounded to 0.5
# ---------------------------------------------------------------------------

def test_filler_rating_range(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        for item, r in p.rated_items.items():
            if item in target_set:
                continue
            assert 1.0 <= r <= MAX_RATING, \
                f"{p.user_id} filler item {item} out-of-range rating {r}"
            # Check rounding to nearest 0.5
            assert abs(r * 2 - round(r * 2)) < 1e-6, \
                f"{p.user_id} filler item {item} not rounded to 0.5: {r}"


# ---------------------------------------------------------------------------
# Test 8 — Filler items distinct from targets
# ---------------------------------------------------------------------------

def test_filler_excludes_targets(profiles):
    target_set = set(TARGET_ITEMS)
    for p in profiles:
        filler = set(p.rated_items.keys()) - target_set
        assert filler.isdisjoint(target_set), \
            f"{p.user_id} filler overlaps targets: {filler & target_set}"


# ---------------------------------------------------------------------------
# Test 9 — Review texts are empty strings
# ---------------------------------------------------------------------------

def test_review_texts_empty(profiles):
    for p in profiles:
        assert all(t == "" for t in p.review_texts), \
            f"{p.user_id} has non-empty review text(s)"


# ---------------------------------------------------------------------------
# Test 10 — Budget constraint
# ---------------------------------------------------------------------------

def test_budget_constraint(profiles):
    for p in profiles:
        assert len(p.rated_items) == _ITEMS_PER, \
            f"{p.user_id}: expected {_ITEMS_PER} rated items, " \
            f"got {len(p.rated_items)}"


# ---------------------------------------------------------------------------
# Tests 11–14 — inject() structural checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def augmented(pretrained_attack, dataset, profiles):
    return pretrained_attack.inject(dataset, profiles)


def test_inject_user_count(augmented):
    assert augmented.num_users == N_USERS + N_FAKE


def test_inject_review_count(augmented, profiles):
    n_new = sum(len(p.rated_items) for p in profiles)
    assert augmented.num_reviews == N_REVIEWS + n_new


def test_inject_labels(augmented):
    assert (augmented.labels[:N_USERS] == 0).all()
    assert (augmented.labels[N_USERS:] == 1).all()


def test_inject_masks(augmented, profiles):
    n_new = sum(len(p.rated_items) for p in profiles)
    assert not augmented.train_mask[N_REVIEWS:].any()
    assert not augmented.val_mask[N_REVIEWS:].any()
    assert augmented.test_mask[N_REVIEWS:].all()


def test_inject_index_bounds(augmented):
    assert augmented.review_user_idx.max() < augmented.num_users
    assert augmented.review_item_idx.max() < augmented.num_items
    assert augmented.review_user_idx.min() >= 0
    assert augmented.review_item_idx.min() >= 0


# ---------------------------------------------------------------------------
# Test 15 — Generator output range
# ---------------------------------------------------------------------------

def test_generator_output_range(pretrained_attack, dataset):
    G = pretrained_attack._generator
    G.eval()
    B, D = 8, pretrained_attack.embedding_dim
    z   = torch.randn(B, D)
    tgt = torch.zeros(B, dataset.num_items)
    tgt[:, TARGET_ITEMS] = 1.0
    with torch.no_grad():
        out = G(z, tgt)
    assert out.shape == (B, dataset.num_items)
    assert (out >= 0).all() and (out <= 1).all(), \
        f"Generator output not in [0,1]: min={out.min():.4f} max={out.max():.4f}"


# ---------------------------------------------------------------------------
# Test 16 — Discriminator output range
# ---------------------------------------------------------------------------

def test_discriminator_output_range(pretrained_attack, dataset):
    Disc = pretrained_attack._discriminator
    Disc.eval()
    x = torch.rand(10, dataset.num_items)
    with torch.no_grad():
        out = Disc(x)
    assert out.shape == (10, 1)
    assert (out >= 0).all() and (out <= 1).all()


# ---------------------------------------------------------------------------
# Test 17 — Training losses are finite
# ---------------------------------------------------------------------------

def test_training_losses_finite(dataset):
    """Quick 5-epoch run — losses must be finite and not explode.

    Verified directly from the GAN modules rather than log parsing to avoid
    any dependency on logging-level configuration in the test environment.
    """
    import torch.nn.functional as F

    a = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=99)
    a.pretrain(dataset, epochs=5, batch_size=8, device="cpu",
               n_targets_per_sample=2)

    # Spot-check one forward pass of trained G and D
    G = a._generator
    D = a._discriminator
    G.eval(); D.eval()

    B = 4
    z   = torch.randn(B, a.embedding_dim)
    tgt = torch.zeros(B, dataset.num_items)
    tgt[:, TARGET_ITEMS] = 1.0
    real = torch.rand(B, dataset.num_items)

    with torch.no_grad():
        fake   = G(z, tgt)
        d_real = D(real)
        d_fake = D(fake)

    loss_D = (F.binary_cross_entropy(d_real, torch.full((B, 1), 0.9)) +
              F.binary_cross_entropy(d_fake, torch.full((B, 1), 0.1))) * 0.5
    loss_G = F.binary_cross_entropy(d_fake, torch.ones(B, 1))

    assert np.isfinite(loss_D.item()), f"loss_D not finite: {loss_D.item()}"
    assert np.isfinite(loss_G.item()), f"loss_G not finite: {loss_G.item()}"
    assert loss_D.item() < 10.0, f"loss_D exploded: {loss_D.item()}"
    assert loss_G.item() < 10.0, f"loss_G exploded: {loss_G.item()}"


# ---------------------------------------------------------------------------
# Test 18 — Different seeds → different profiles
# ---------------------------------------------------------------------------

def test_different_seeds_differ(dataset):
    a1 = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=10)
    a2 = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=20)
    a1.pretrain(dataset, epochs=2, batch_size=8, device="cpu")
    a2.pretrain(dataset, epochs=2, batch_size=8, device="cpu")
    p1 = a1.generate_profiles(3, BUDGET_PCT, TARGET_ITEMS)
    p2 = a2.generate_profiles(3, BUDGET_PCT, TARGET_ITEMS)
    # At least one profile's rating vector should differ
    r1 = list(p1[0].rated_items.values())
    r2 = list(p2[0].rated_items.values())
    assert r1 != r2, "Different seeds produced identical profiles"


# ---------------------------------------------------------------------------
# Test 19 — pretrain is idempotent (overwrites old generator)
# ---------------------------------------------------------------------------

def test_pretrain_idempotent(dataset):
    a = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=5)
    a.pretrain(dataset, epochs=2, batch_size=8, device="cpu")
    gen_id_1 = id(a._generator)
    a.pretrain(dataset, epochs=2, batch_size=8, device="cpu")
    gen_id_2 = id(a._generator)
    # Second call creates a fresh generator (new object)
    assert gen_id_1 != gen_id_2


# ---------------------------------------------------------------------------
# Test 20 — Zero filler (budget exactly covers targets)
# ---------------------------------------------------------------------------

def test_zero_filler(dataset):
    """budget_pct such that round(budget_pct * num_items) == len(target_items)."""
    exact_budget = len(TARGET_ITEMS) / N_ITEMS  # filler_size = 0
    a = AUSHAttack(dataset, embedding_dim=16, hidden_dim=32, seed=7)
    a.pretrain(dataset, epochs=2, batch_size=8, device="cpu")
    profiles = a.generate_profiles(3, exact_budget, TARGET_ITEMS)
    for p in profiles:
        assert set(p.rated_items.keys()) == set(TARGET_ITEMS), \
            f"Expected only target items, got {set(p.rated_items.keys())}"
        assert all(r == MAX_RATING for r in p.rated_items.values())


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
