"""Unit tests for TemporalContrastiveLoss.

Tests
-----
  1. L_TC is scalar and non-negative.
  2. L_TC decreases when embeddings of same-campaign (temporal-positive)
     users are moved closer together.
  3. The < 2 fake-user guard fires correctly (returns a zero scalar).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.losses import TemporalContrastiveLoss

EMB_DIM = 16
WINDOW_HOURS = 24.0
WINDOW_SECONDS = WINDOW_HOURS * 3600.0


# ---------------------------------------------------------------------------
# Test 1 — scalar and non-negative
# ---------------------------------------------------------------------------

def test_tc_loss_scalar_and_nonnegative():
    torch.manual_seed(0)
    n_users = 20
    embeddings = torch.randn(n_users, EMB_DIM)

    # First 8 users are fake and clustered in time (within the window);
    # remaining users are real, spread far outside the window.
    labels = torch.zeros(n_users, dtype=torch.long)
    labels[:8] = 1

    timestamps = torch.zeros(n_users)
    timestamps[:8] = torch.arange(8) * 60.0  # fake users a minute apart
    timestamps[8:] = 10 * WINDOW_SECONDS + torch.arange(n_users - 8) * WINDOW_SECONDS

    loss_fn = TemporalContrastiveLoss(tau=0.07, window_hours=WINDOW_HOURS, n_negatives=4)
    loss = loss_fn(embeddings, timestamps, labels)

    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"Loss must be non-negative, got {loss.item()}"
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


# ---------------------------------------------------------------------------
# Test 2 — loss decreases as same-campaign embeddings are pulled together
# ---------------------------------------------------------------------------

def test_tc_loss_decreases_when_positives_move_closer():
    n_real = 6
    n_users = n_real + 2   # indices 0, 1 are the fake temporal-positive pair
    labels = torch.zeros(n_users, dtype=torch.long)
    labels[0] = 1
    labels[1] = 1

    timestamps = torch.zeros(n_users)
    timestamps[0] = 0.0
    timestamps[1] = 100.0                          # within window of user 0
    timestamps[2:] = 10 * WINDOW_SECONDS + torch.arange(n_real) * WINDOW_SECONDS

    torch.manual_seed(42)
    real_embs = torch.randn(n_real, EMB_DIM)

    loss_fn = TemporalContrastiveLoss(tau=0.07, window_hours=WINDOW_HOURS, n_negatives=3)

    def build_embeddings(sim: float) -> torch.Tensor:
        """Two fake-user vectors with controlled cosine similarity ~ sim."""
        u0 = F.normalize(torch.tensor([1.0] + [0.0] * (EMB_DIM - 1)), dim=0)
        orth = F.normalize(torch.tensor([0.0, 1.0] + [0.0] * (EMB_DIM - 2)), dim=0)
        u1 = sim * u0 + (1 - abs(sim)) * orth
        return torch.cat([u0.unsqueeze(0), u1.unsqueeze(0), real_embs], dim=0)

    far_embeddings   = build_embeddings(sim=-0.9)   # nearly opposite
    close_embeddings = build_embeddings(sim=0.9)     # nearly identical

    torch.manual_seed(123)
    loss_far = loss_fn(far_embeddings, timestamps, labels)
    torch.manual_seed(123)
    loss_close = loss_fn(close_embeddings, timestamps, labels)

    assert loss_close.item() < loss_far.item(), (
        f"Expected loss to decrease as temporal positives move closer: "
        f"far={loss_far.item():.4f}  close={loss_close.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — guard: < 2 fake users in batch
# ---------------------------------------------------------------------------

def test_tc_loss_guard_fewer_than_two_fake_users():
    torch.manual_seed(0)
    n_users = 10
    embeddings = torch.randn(n_users, EMB_DIM)
    timestamps = torch.arange(n_users).float() * WINDOW_SECONDS

    loss_fn = TemporalContrastiveLoss(tau=0.07, window_hours=WINDOW_HOURS, n_negatives=4)

    # Zero fake users
    labels_zero = torch.zeros(n_users, dtype=torch.long)
    loss_zero = loss_fn(embeddings, timestamps, labels_zero)
    assert loss_zero.dim() == 0
    assert loss_zero.item() == 0.0

    # Exactly one fake user
    labels_one = torch.zeros(n_users, dtype=torch.long)
    labels_one[0] = 1
    loss_one = loss_fn(embeddings, timestamps, labels_one)
    assert loss_one.dim() == 0
    assert loss_one.item() == 0.0


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
