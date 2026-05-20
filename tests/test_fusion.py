"""Unit tests for MultimodalFusion, TemporalBurstiness, and THGNNDetector.

Synthetic data mirrors the test_hgt.py fixture (20 users, 30 items, 50 reviews)
so the two test modules share conceptual parity.

Tests
-----
  1. MultimodalFusion output shape and no NaN/Inf.
  2. MultimodalFusion attention weights sum to 1 per sample.
  3. TemporalBurstiness output shape and no NaN/Inf.
  4. TemporalBurstiness handles users with 0 or 1 review.
  5. THGNNDetector forward: score shape, embedding dict keys, no NaN/Inf.
  6. THGNNDetector: item_embs=None fallback produces valid outputs.
  7. THGNNDetector gradients flow to all parameters.
  8. Determinism in eval mode (dropout=0).
"""
from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch import Tensor

from src.models.hgt import EDGE_TYPES, NODE_TYPES, HGTEncoder
from src.models.fusion import (
    MultimodalFusion,
    TemporalBurstiness,
    THGNNDetector,
    _mean_pool,
    _group_timestamps,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_USERS   = 20
N_ITEMS   = 30
N_REVIEWS = 50
HIDDEN    = 128
TEXT_DIM  = 768
PROJ_DIM  = 128
BURST_DIM = 32
SEED      = 0

D_USER   = 8
D_ITEM   = 12
D_REVIEW = 16

N_RATES  = 40


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_graph():
    torch.manual_seed(SEED)
    node_feats = {
        "user":   torch.randn(N_USERS,   D_USER),
        "item":   torch.randn(N_ITEMS,   D_ITEM),
        "review": torch.randn(N_REVIEWS, D_REVIEW),
    }
    writes_u = torch.randint(0, N_USERS,   (N_REVIEWS,))
    writes_r = torch.arange(N_REVIEWS)
    about_r  = torch.arange(N_REVIEWS)
    about_i  = torch.randint(0, N_ITEMS,   (N_REVIEWS,))
    rates_u  = torch.randint(0, N_USERS,   (N_RATES,))
    rates_i  = torch.randint(0, N_ITEMS,   (N_RATES,))

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about",  "item"):   torch.stack([about_r,  about_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u,  rates_i]),
    }
    edge_attr_dict = {
        ("user",   "writes", "review"): torch.rand(N_REVIEWS) * 1e9,
        ("review", "about",  "item"):   torch.rand(N_REVIEWS) * 1e9,
        ("user",   "rates",  "item"):   torch.rand(N_RATES)   * 1e9,
    }
    node_timestamps = {
        "user":   torch.rand(N_USERS)   * 1e9,
        "item":   torch.rand(N_ITEMS)   * 1e9,
        "review": torch.rand(N_REVIEWS) * 1e9,
    }
    review_user_idx = writes_u                          # (N_REVIEWS,)
    review_item_idx = about_i                           # (N_REVIEWS,)
    timestamps      = torch.rand(N_REVIEWS) * 1e9

    return (
        node_feats, edge_index_dict, edge_attr_dict,
        node_timestamps, review_user_idx, review_item_idx, timestamps,
    )


@pytest.fixture(scope="module")
def hgt_encoder():
    torch.manual_seed(SEED)
    return HGTEncoder(
        hidden_dim=HIDDEN,
        num_heads=4,
        num_layers=2,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        input_dims={"user": D_USER, "item": D_ITEM, "review": D_REVIEW},
        dropout=0.0,
        time_emb_dim=64,
    )


@pytest.fixture(scope="module")
def fusion():
    torch.manual_seed(SEED)
    return MultimodalFusion(
        subgraph_dim=HIDDEN,
        text_dim=TEXT_DIM,
        proj_dim=PROJ_DIM,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def burstiness():
    torch.manual_seed(SEED)
    return TemporalBurstiness(hidden_dim=BURST_DIM, dropout=0.0)


@pytest.fixture(scope="module")
def detector(hgt_encoder, fusion, burstiness):
    torch.manual_seed(SEED)
    return THGNNDetector(
        hgt_encoder=hgt_encoder,
        fusion=fusion,
        burstiness_encoder=burstiness,
        mlp_hidden=64,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1 — MultimodalFusion output shape and no NaN/Inf
# ---------------------------------------------------------------------------

def test_fusion_output_shape_and_clean(fusion):
    torch.manual_seed(SEED)
    sub  = torch.randn(N_USERS, HIDDEN)
    rev  = torch.randn(N_USERS, TEXT_DIM)
    item = torch.randn(N_USERS, TEXT_DIM)
    fused, attn_w = fusion(sub, rev, item)

    assert fused.shape  == (N_USERS, PROJ_DIM), fused.shape
    assert attn_w.shape == (N_USERS, 1, 2),     attn_w.shape
    assert not torch.isnan(fused).any()
    assert not torch.isinf(fused).any()


# ---------------------------------------------------------------------------
# Test 2 — Attention weights sum to 1
# ---------------------------------------------------------------------------

def test_fusion_attn_weights_sum_to_one(fusion):
    torch.manual_seed(SEED)
    sub  = torch.randn(N_USERS, HIDDEN)
    rev  = torch.randn(N_USERS, TEXT_DIM)
    item = torch.randn(N_USERS, TEXT_DIM)
    _, attn_w = fusion(sub, rev, item)

    sums = attn_w.sum(dim=-1)   # (N_USERS, 1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Attention weights do not sum to 1: {sums}"


# ---------------------------------------------------------------------------
# Test 3 — TemporalBurstiness output shape and no NaN/Inf
# ---------------------------------------------------------------------------

def test_burstiness_output_shape_and_clean(burstiness, synthetic_graph):
    _, _, _, _, review_user_idx, _, timestamps = synthetic_graph
    ts_list = _group_timestamps(timestamps, review_user_idx, N_USERS)
    out = burstiness(ts_list)

    assert out.shape == (N_USERS, BURST_DIM), out.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Test 4 — Burstiness handles users with 0 or 1 review
# ---------------------------------------------------------------------------

def test_burstiness_sparse_users(burstiness):
    """Users with 0 or 1 review must not crash or produce NaN."""
    ts_list = [
        torch.tensor([], dtype=torch.float32),          # 0 reviews
        torch.tensor([1e6], dtype=torch.float32),        # 1 review
        torch.tensor([1e6, 2e6, 5e6], dtype=torch.float32),  # 3 reviews
    ]
    out = burstiness(ts_list)
    assert out.shape == (3, BURST_DIM)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Test 5 — THGNNDetector forward: shapes, keys, clean values
# ---------------------------------------------------------------------------

def test_detector_forward(detector, synthetic_graph):
    (
        node_feats, edge_index_dict, edge_attr_dict,
        node_timestamps, review_user_idx, review_item_idx, timestamps,
    ) = synthetic_graph

    review_embs = torch.randn(N_REVIEWS, TEXT_DIM)
    item_embs   = torch.randn(N_ITEMS,   TEXT_DIM)

    scores, embs = detector(
        node_feats, edge_index_dict, edge_attr_dict,
        review_embs, item_embs,
        review_user_idx, review_item_idx, timestamps,
        node_timestamps=node_timestamps,
    )

    assert scores.shape == (N_USERS,), scores.shape
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()

    expected_keys = {"user", "item", "review", "user_fused", "user_burst"}
    assert set(embs.keys()) == expected_keys, set(embs.keys())

    assert embs["user"].shape       == (N_USERS,   HIDDEN),    embs["user"].shape
    assert embs["item"].shape       == (N_ITEMS,   HIDDEN),    embs["item"].shape
    assert embs["review"].shape     == (N_REVIEWS, HIDDEN),    embs["review"].shape
    assert embs["user_fused"].shape == (N_USERS,   PROJ_DIM),  embs["user_fused"].shape
    assert embs["user_burst"].shape == (N_USERS,   BURST_DIM), embs["user_burst"].shape


# ---------------------------------------------------------------------------
# Test 6 — item_embs=None fallback
# ---------------------------------------------------------------------------

def test_detector_no_item_embs(detector, synthetic_graph):
    (
        node_feats, edge_index_dict, edge_attr_dict,
        node_timestamps, review_user_idx, review_item_idx, timestamps,
    ) = synthetic_graph

    review_embs = torch.randn(N_REVIEWS, TEXT_DIM)
    scores, embs = detector(
        node_feats, edge_index_dict, edge_attr_dict,
        review_embs, None,
        review_user_idx, review_item_idx, timestamps,
        node_timestamps=node_timestamps,
    )

    assert scores.shape == (N_USERS,)
    assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# Test 7 — Gradient flow
# ---------------------------------------------------------------------------

def test_detector_gradients(synthetic_graph):
    """Verify gradient flow for all reachable parameters.

    THGNNDetector only uses hgt_out["user"] in its forward pass; the item and
    review HGT outputs are stored in the returned embeddings dict but do not
    contribute to the scalar user scores.  Therefore, all HGT parameters that
    *only* affect item or review node outputs are structurally dead from the
    detector's loss perspective.

    Live HGT sub-paths (user residual path):
        hgt.input_projs.user.*
        hgt.layers.*.W_res.user.*
        hgt.layers.*.norm.user.*

    Everything else in hgt.* is dead; fusion.*, burstiness.*, and mlp.* must
    all be live.
    """
    _HGT_LIVE_SUFFIXES = ("input_projs.user", "W_res.user", "norm.user")

    torch.manual_seed(SEED)
    hgt = HGTEncoder(
        hidden_dim=HIDDEN, num_heads=4, num_layers=2,
        node_types=NODE_TYPES, edge_types=EDGE_TYPES,
        input_dims={"user": D_USER, "item": D_ITEM, "review": D_REVIEW},
        dropout=0.0, time_emb_dim=64,
    )
    fus = MultimodalFusion(subgraph_dim=HIDDEN, text_dim=TEXT_DIM,
                           proj_dim=PROJ_DIM, num_heads=4, dropout=0.0)
    burst = TemporalBurstiness(hidden_dim=BURST_DIM, dropout=0.0)
    det   = THGNNDetector(hgt, fus, burst, mlp_hidden=64, dropout=0.0)

    (
        node_feats, edge_index_dict, edge_attr_dict,
        node_timestamps, review_user_idx, review_item_idx, timestamps,
    ) = synthetic_graph

    review_embs = torch.randn(N_REVIEWS, TEXT_DIM, requires_grad=True)
    item_embs   = torch.randn(N_ITEMS,   TEXT_DIM, requires_grad=True)

    scores, _ = det(
        node_feats, edge_index_dict, edge_attr_dict,
        review_embs, item_embs,
        review_user_idx, review_item_idx, timestamps,
        node_timestamps=node_timestamps,
    )
    scores.sum().backward()

    unexpected_dead = []
    for name, p in det.named_parameters():
        if p.grad is not None:
            continue
        # HGT params that only affect item/review outputs: expected dead
        if name.startswith("hgt.") and not any(s in name for s in _HGT_LIVE_SUFFIXES):
            continue
        unexpected_dead.append(name)
    assert not unexpected_dead, f"Unexpected dead parameters: {unexpected_dead}"

    # User-path HGT params must be live
    for name, p in det.named_parameters():
        if name.startswith("hgt.") and any(s in name for s in _HGT_LIVE_SUFFIXES):
            assert p.grad is not None, f"Expected gradient in user-path HGT param: {name}"

    nan_grads = [
        name for name, p in det.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not nan_grads, f"NaN gradients in: {nan_grads}"


# ---------------------------------------------------------------------------
# Test 8 — Determinism in eval mode
# ---------------------------------------------------------------------------

def test_detector_deterministic(detector, synthetic_graph):
    (
        node_feats, edge_index_dict, edge_attr_dict,
        node_timestamps, review_user_idx, review_item_idx, timestamps,
    ) = synthetic_graph
    review_embs = torch.randn(N_REVIEWS, TEXT_DIM)
    item_embs   = torch.randn(N_ITEMS,   TEXT_DIM)

    kwargs = dict(
        node_feats=node_feats, edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict, review_embs=review_embs,
        item_embs=item_embs, review_user_idx=review_user_idx,
        review_item_idx=review_item_idx, timestamps=timestamps,
        node_timestamps=node_timestamps,
    )

    detector.eval()
    with torch.no_grad():
        s1, _ = detector(**kwargs)
        s2, _ = detector(**kwargs)
    assert torch.allclose(s1, s2), "Non-deterministic scores in eval mode"
    detector.train()


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
