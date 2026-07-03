"""Unit tests for AttackAwareAnomalyAttention.

Builds a tiny synthetic heterogeneous graph (20 users, 30 items, 50 reviews)
and verifies:

  1. forward() produces the expected output shape for every node type.
  2. No NaN/Inf values in outputs.
  3. lambda_t, lambda_s, lambda_g are present in model.parameters() and
     initialised to 0.1.
  4. HGTEncoder(use_aaa=True) produces correctly-shaped outputs end-to-end.
"""
from __future__ import annotations

import pytest
import torch

from src.models.aaa import AttackAwareAnomalyAttention
from src.models.hgt import EDGE_TYPES, NODE_TYPES, HGTEncoder

N_USERS   = 20
N_ITEMS   = 30
N_REVIEWS = 50
HIDDEN    = 32
N_HEADS   = 4
TEXT_DIM  = 16

SEED = 0


@pytest.fixture(scope="module")
def synthetic_graph():
    torch.manual_seed(SEED)

    x_dict = {
        "user":   torch.randn(N_USERS,   HIDDEN),
        "item":   torch.randn(N_ITEMS,   HIDDEN),
        "review": torch.randn(N_REVIEWS, HIDDEN),
    }

    writes_u = torch.randint(0, N_USERS, (N_REVIEWS,))
    writes_r = torch.arange(N_REVIEWS)

    about_r = torch.arange(N_REVIEWS)
    about_i = torch.randint(0, N_ITEMS, (N_REVIEWS,))

    n_rates = 40
    rates_u = torch.randint(0, N_USERS, (n_rates,))
    rates_i = torch.randint(0, N_ITEMS, (n_rates,))

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about",  "item"):   torch.stack([about_r,  about_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u,  rates_i]),
    }
    edge_attr_dict = {
        ("user",   "writes", "review"): torch.rand(N_REVIEWS) * 1e6 + 1.0,
        ("review", "about",  "item"):   torch.rand(N_REVIEWS) * 1e6 + 1.0,
        ("user",   "rates",  "item"):   torch.rand(n_rates)   * 1e6 + 1.0,
    }

    review_embs = torch.randn(N_REVIEWS, TEXT_DIM)
    item_embs   = torch.randn(N_ITEMS,   TEXT_DIM)

    return x_dict, edge_index_dict, edge_attr_dict, review_embs, item_embs


@pytest.fixture
def layer():
    torch.manual_seed(SEED)
    return AttackAwareAnomalyAttention(
        hidden_dim=HIDDEN, num_heads=N_HEADS, num_edge_types=len(EDGE_TYPES), dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1 — Output shapes
# ---------------------------------------------------------------------------

def test_output_shapes(layer, synthetic_graph):
    x_dict, ei, ea, review_embs, item_embs = synthetic_graph
    out = layer(x_dict, ei, ea, review_embs, item_embs)

    assert set(out.keys()) == {"user", "item", "review"}
    assert out["user"].shape   == (N_USERS,   HIDDEN)
    assert out["item"].shape   == (N_ITEMS,   HIDDEN)
    assert out["review"].shape == (N_REVIEWS, HIDDEN)


# ---------------------------------------------------------------------------
# Test 2 — No NaN / Inf, with and without review/item embeddings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_semantic", [True, False])
def test_no_nan_inf(layer, synthetic_graph, with_semantic):
    x_dict, ei, ea, review_embs, item_embs = synthetic_graph
    if with_semantic:
        out = layer(x_dict, ei, ea, review_embs, item_embs)
    else:
        out = layer(x_dict, ei, ea, None, None)

    for nt, emb in out.items():
        assert not torch.isnan(emb).any(), f"NaN in {nt} (with_semantic={with_semantic})"
        assert not torch.isinf(emb).any(), f"Inf in {nt} (with_semantic={with_semantic})"


# ---------------------------------------------------------------------------
# Test 3 — Lambda parameters are learnable and correctly initialised
# ---------------------------------------------------------------------------

def test_lambda_parameters_present_and_initialised(layer):
    params = dict(layer.named_parameters())
    for name in ("lambda_t", "lambda_s", "lambda_g"):
        assert name in params, f"{name} missing from model.parameters()"
        assert params[name].requires_grad
        assert torch.isclose(params[name], torch.tensor(0.1)), \
            f"{name} initial value {params[name].item()} != 0.1"


def test_lambda_gradients_flow(layer, synthetic_graph):
    x_dict, ei, ea, review_embs, item_embs = synthetic_graph
    out = layer(x_dict, ei, ea, review_embs, item_embs)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    for name in ("lambda_t", "lambda_s", "lambda_g"):
        p = dict(layer.named_parameters())[name]
        assert p.grad is not None, f"{name} received no gradient"
        assert not torch.isnan(p.grad).any()


# ---------------------------------------------------------------------------
# Test 4 — HGTEncoder(use_aaa=True) end-to-end
# ---------------------------------------------------------------------------

def test_hgt_encoder_with_aaa(synthetic_graph):
    x_dict, ei, ea, review_embs, item_embs = synthetic_graph
    torch.manual_seed(SEED)
    enc = HGTEncoder(
        hidden_dim=HIDDEN,
        num_heads=N_HEADS,
        num_layers=2,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        dropout=0.0,
        use_aaa=True,
    )
    out = enc(x_dict, ei, ea, review_embs=review_embs, item_embs=item_embs)

    assert out["user"].shape   == (N_USERS,   HIDDEN)
    assert out["item"].shape   == (N_ITEMS,   HIDDEN)
    assert out["review"].shape == (N_REVIEWS, HIDDEN)
    for nt, emb in out.items():
        assert not torch.isnan(emb).any()

    lambda_names = {n for n, _ in enc.named_parameters() if "lambda_" in n}
    assert len(lambda_names) == 3 * 2, "Expected 3 lambdas per layer across 2 layers"


# ---------------------------------------------------------------------------
# Test 5 — Too many distinct edge types raises
# ---------------------------------------------------------------------------

def test_too_many_edge_types_raises(synthetic_graph):
    x_dict, ei, ea, review_embs, item_embs = synthetic_graph
    small_layer = AttackAwareAnomalyAttention(hidden_dim=HIDDEN, num_heads=N_HEADS, num_edge_types=1)
    with pytest.raises(ValueError):
        small_layer(x_dict, ei, ea, review_embs, item_embs)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
