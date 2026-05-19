"""Unit tests for HGTEncoder.

Builds a tiny synthetic heterogeneous graph (20 users, 30 items, 50 reviews)
and verifies:

  1. Output shapes are (N_t, hidden_dim) for every node type.
  2. No NaN or Inf values in outputs.
  3. Gradients flow to every parameter (no dead parameters).
  4. Temporal encoding is active and produces a non-trivial bias.
  5. Fallback behaviour when node_timestamps=None (use edge timestamps).
  6. Robustness when an edge type is absent from the input graph.
  7. Determinism: two identical calls produce identical outputs.
  8. TemporalEncoding: shape and no-NaN check.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor

from src.models.hgt import (
    EDGE_TYPES,
    NODE_TYPES,
    HGTEncoder,
    TemporalEncoding,
    _scatter_softmax,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_USERS   = 20
N_ITEMS   = 30
N_REVIEWS = 50
HIDDEN    = 128
N_HEADS   = 4
N_LAYERS  = 2

D_USER   = 8
D_ITEM   = 12
D_REVIEW = 16

SEED = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_graph():
    """Return a consistent synthetic graph used across tests."""
    torch.manual_seed(SEED)

    # Node features (different input dims per type)
    node_feats = {
        "user":   torch.randn(N_USERS,   D_USER),
        "item":   torch.randn(N_ITEMS,   D_ITEM),
        "review": torch.randn(N_REVIEWS, D_REVIEW),
    }

    # Edge indices — kept structurally valid (indices in range)
    # user-writes-review: each review comes from exactly one user
    writes_u = torch.randint(0, N_USERS,   (N_REVIEWS,))
    writes_r = torch.arange(N_REVIEWS)

    # review-about-item: each review is about one item
    about_r = torch.arange(N_REVIEWS)
    about_i = torch.randint(0, N_ITEMS, (N_REVIEWS,))

    # user-rates-item: 40 interactions (may repeat user-item pairs)
    n_rates  = 40
    rates_u  = torch.randint(0, N_USERS, (n_rates,))
    rates_i  = torch.randint(0, N_ITEMS, (n_rates,))

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about",  "item"):   torch.stack([about_r,  about_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u,  rates_i]),
    }

    # Edge timestamps (random Unix-second values)
    edge_attr_dict = {
        ("user",   "writes", "review"): torch.rand(N_REVIEWS) * 1e9,
        ("review", "about",  "item"):   torch.rand(N_REVIEWS) * 1e9,
        ("user",   "rates",  "item"):   torch.rand(n_rates)   * 1e9,
    }

    # Per-node timestamps (simulate nodes having a "creation" time)
    node_timestamps = {
        "user":   torch.rand(N_USERS)   * 1e9,
        "item":   torch.rand(N_ITEMS)   * 1e9,
        "review": torch.rand(N_REVIEWS) * 1e9,
    }

    return node_feats, edge_index_dict, edge_attr_dict, node_timestamps


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(SEED)
    return HGTEncoder(
        hidden_dim=HIDDEN,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        input_dims={"user": D_USER, "item": D_ITEM, "review": D_REVIEW},
        dropout=0.0,        # disable for determinism in tests
        time_emb_dim=64,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(encoder, graph, timestamps=True):
    nf, ei, ea, nt = graph
    return encoder(nf, ei, ea, node_timestamps=nt if timestamps else None)


# ---------------------------------------------------------------------------
# Test 1 — Output shapes
# ---------------------------------------------------------------------------

def test_output_shapes(encoder, synthetic_graph):
    out = _run(encoder, synthetic_graph)
    assert set(out.keys()) == {"user", "item", "review"}, \
        f"Expected all three node types, got {set(out.keys())}"
    assert out["user"].shape   == (N_USERS,   HIDDEN), out["user"].shape
    assert out["item"].shape   == (N_ITEMS,   HIDDEN), out["item"].shape
    assert out["review"].shape == (N_REVIEWS, HIDDEN), out["review"].shape


# ---------------------------------------------------------------------------
# Test 2 — No NaN / Inf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_timestamps", [True, False])
def test_no_nan_inf(encoder, synthetic_graph, use_timestamps):
    out = _run(encoder, synthetic_graph, timestamps=use_timestamps)
    for nt, emb in out.items():
        assert not torch.isnan(emb).any(),  f"NaN in {nt} (timestamps={use_timestamps})"
        assert not torch.isinf(emb).any(),  f"Inf in {nt} (timestamps={use_timestamps})"


# ---------------------------------------------------------------------------
# Test 3 — Gradients reach every parameter
# ---------------------------------------------------------------------------

def test_gradients_flow(encoder, synthetic_graph):
    """Verify gradient flow for all parameters reachable by the canonical edges.

    With edge types {user→review, review→item, user→item}, three parameter
    groups are *structurally* unreachable and must have grad=None:
      - W_Q.user   : users are never a destination  (query = target type)
      - W_K.item   : items  are never a source       (key   = source type)
      - W_V.item   : items  are never a source       (value = source type)
      - W_out.user : users take the residual-only path; output proj unused

    All other parameters must receive a gradient.
    """
    # Patterns that identify the structurally unreachable parameters
    _STRUCTURAL_DEAD = ("W_Q.user", "W_K.item", "W_V.item", "W_out.user")

    enc = HGTEncoder(
        hidden_dim=HIDDEN,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        input_dims={"user": D_USER, "item": D_ITEM, "review": D_REVIEW},
        dropout=0.0,
        time_emb_dim=64,
    )
    out = _run(enc, synthetic_graph)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    unexpected_dead = [
        name for name, p in enc.named_parameters()
        if p.grad is None
        and not any(pat in name for pat in _STRUCTURAL_DEAD)
    ]
    assert not unexpected_dead, f"Unexpected dead parameters: {unexpected_dead}"

    # Confirm structurally dead params are indeed dead (sanity check)
    for name, p in enc.named_parameters():
        if any(pat in name for pat in _STRUCTURAL_DEAD):
            assert p.grad is None, f"{name!r} should be unreachable but has grad"

    nan_grads = [
        name for name, p in enc.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not nan_grads, f"NaN gradients in: {nan_grads}"


# ---------------------------------------------------------------------------
# Test 4 — Temporal encoding produces non-trivial bias
# ---------------------------------------------------------------------------

def test_temporal_encoding_nontrivial():
    """Different Δt values must produce different biases after training-init."""
    enc = TemporalEncoding(emb_dim=64)
    # proj is zero-initialised for stability; give it non-zero weights so the
    # test is not trivially satisfied by a constant-zero output.
    torch.nn.init.normal_(enc.proj.weight, std=0.1)
    torch.nn.init.normal_(enc.proj.bias,   std=0.01)

    delta = torch.tensor([0.0, 86_400.0, 365.0 * 86_400.0])
    bias  = enc(delta)
    assert bias.shape == (3,), bias.shape
    assert not torch.isnan(bias).any()
    # Biases must differ for different time deltas
    assert not torch.allclose(bias[0:1], bias[1:2]), \
        "Δt=0 and Δt=1day produced identical temporal bias"
    assert not torch.allclose(bias[1:2], bias[2:3]), \
        "Δt=1day and Δt=1year produced identical temporal bias"


# ---------------------------------------------------------------------------
# Test 5 — node_timestamps=None fallback
# ---------------------------------------------------------------------------

def test_no_node_timestamps_shape(encoder, synthetic_graph):
    out = _run(encoder, synthetic_graph, timestamps=False)
    assert out["user"].shape   == (N_USERS,   HIDDEN)
    assert out["item"].shape   == (N_ITEMS,   HIDDEN)
    assert out["review"].shape == (N_REVIEWS, HIDDEN)


# ---------------------------------------------------------------------------
# Test 6 — Missing edge type in input graph
# ---------------------------------------------------------------------------

def test_missing_edge_type(encoder, synthetic_graph):
    """Dropping user-rates-item edges must still produce valid item embeddings
    (items still receive messages from reviews via review-about-item)."""
    nf, ei, ea, nt = synthetic_graph
    ei_partial = {k: v for k, v in ei.items() if k != ("user", "rates", "item")}
    ea_partial = {k: v for k, v in ea.items() if k != ("user", "rates", "item")}

    out = encoder(nf, ei_partial, ea_partial, node_timestamps=nt)
    assert out["item"].shape == (N_ITEMS, HIDDEN)
    assert not torch.isnan(out["item"]).any()


# ---------------------------------------------------------------------------
# Test 7 — Determinism (dropout=0)
# ---------------------------------------------------------------------------

def test_deterministic(encoder, synthetic_graph):
    encoder.eval()
    with torch.no_grad():
        out1 = _run(encoder, synthetic_graph)
        out2 = _run(encoder, synthetic_graph)
    for nt in out1:
        assert torch.allclose(out1[nt], out2[nt]), \
            f"Non-deterministic output for node type '{nt}'"
    encoder.train()


# ---------------------------------------------------------------------------
# Test 8 — _scatter_softmax correctness
# ---------------------------------------------------------------------------

def test_scatter_softmax_sums_to_one():
    """Each group's softmax weights must sum to 1."""
    torch.manual_seed(42)
    E, H, N = 100, 4, 20
    logits = torch.randn(E, H)
    index  = torch.randint(0, N, (E,))

    weights = _scatter_softmax(logits, index, N)
    assert weights.shape == (E, H)
    assert not torch.isnan(weights).any()

    # Sum per destination node per head should be 1 (for nodes with ≥1 edge)
    sums = torch.zeros(N, H).scatter_add_(0, index.unsqueeze(1).expand(-1, H), weights)
    nodes_with_edges = index.unique()
    assert torch.allclose(sums[nodes_with_edges], torch.ones_like(sums[nodes_with_edges]), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9 — Hyperparameter variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hidden,heads,layers", [
    (64,  2, 1),
    (128, 4, 2),
    (256, 8, 3),
])
def test_hyperparameter_variants(hidden, heads, layers, synthetic_graph):
    torch.manual_seed(SEED)
    enc = HGTEncoder(
        hidden_dim=hidden,
        num_heads=heads,
        num_layers=layers,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        input_dims={"user": D_USER, "item": D_ITEM, "review": D_REVIEW},
        dropout=0.0,
    )
    nf, ei, ea, nt = synthetic_graph
    out = enc(nf, ei, ea, node_timestamps=nt)
    for node_type, emb in out.items():
        n = {"user": N_USERS, "item": N_ITEMS, "review": N_REVIEWS}[node_type]
        assert emb.shape == (n, hidden), \
            f"Shape mismatch: hidden={hidden} heads={heads} layers={layers}"
        assert not torch.isnan(emb).any()


# ---------------------------------------------------------------------------
# Allow running directly with python tests/test_hgt.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
