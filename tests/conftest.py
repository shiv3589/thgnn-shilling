"""Shared pytest fixtures available to all test modules."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.datasets.base import HeteroDataset


@pytest.fixture(scope="session")
def synthetic_graph() -> HeteroDataset:
    """Deterministic heterogeneous graph shared across all test modules.

    Dimensions: 50 users, 80 items, 120 reviews, seed=0.
    """
    n_users, n_items, n_reviews = 50, 80, 120
    seed = 0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    writes_u = torch.from_numpy(rng.integers(0, n_users, n_reviews).astype(np.int64))
    writes_r = torch.arange(n_reviews, dtype=torch.long)
    about_r  = torch.arange(n_reviews, dtype=torch.long)
    about_i  = torch.from_numpy(rng.integers(0, n_items, n_reviews).astype(np.int64))
    n_rates  = int(n_reviews * 0.7)
    rates_u  = torch.from_numpy(rng.integers(0, n_users, n_rates).astype(np.int64))
    rates_i  = torch.from_numpy(rng.integers(0, n_items, n_rates).astype(np.int64))

    ts_np = np.sort(rng.uniform(0, 1e9, n_reviews))
    ts    = torch.from_numpy(ts_np)

    n_train = int(0.7 * n_reviews)
    n_val   = int(0.1 * n_reviews)
    train_mask = torch.zeros(n_reviews, dtype=torch.bool)
    val_mask   = torch.zeros(n_reviews, dtype=torch.bool)
    test_mask  = torch.zeros(n_reviews, dtype=torch.bool)
    train_mask[:n_train]                = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :]        = True

    edge_index_dict = {
        ("user",   "writes", "review"): torch.stack([writes_u, writes_r]),
        ("review", "about",  "item"):   torch.stack([about_r,  about_i]),
        ("user",   "rates",  "item"):   torch.stack([rates_u,  rates_i]),
    }
    edge_attr_dict = {k: ts for k in edge_index_dict}

    return HeteroDataset(
        dataset_name    = "synthetic",
        num_users       = n_users,
        num_items       = n_items,
        num_reviews     = n_reviews,
        edge_index_dict = edge_index_dict,
        edge_attr_dict  = edge_attr_dict,
        x_user          = torch.randn(n_users, 8),
        x_item          = torch.randn(n_items, 5),
        review_texts    = ["review text"] * n_reviews,
        ratings         = torch.from_numpy(rng.uniform(1, 5, n_reviews).astype(np.float32)),
        timestamps      = ts,
        review_user_idx = writes_u,
        review_item_idx = about_i,
        labels          = torch.zeros(n_users, dtype=torch.long),
        train_mask      = train_mask,
        val_mask        = val_mask,
        test_mask       = test_mask,
        user_id_map     = {i: i for i in range(n_users)},
        item_id_map     = {i: i for i in range(n_items)},
    )
