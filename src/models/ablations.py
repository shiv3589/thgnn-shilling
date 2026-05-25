"""Ablation variants of THGNNDetector.

Each variant isolates one architectural component to measure its contribution:

  THGNNNoText      — zeros both review and item BERT text streams; tests the
                     contribution of the text modality to detection.

  THGNNNoTemporal  — passes an empty edge_attr_dict to HGT (disabling the
                     sinusoidal temporal bias in every attention layer) and
                     removes the TemporalBurstiness GRU stream entirely; tests
                     the contribution of all temporal signals.

  THGNNStaticGCN   — replaces the HGT backbone with a 2-layer homogeneous GCN
                     that treats the graph as a single node/edge type (no
                     relation-specific Q/K/V or type-specific W_att/W_msg);
                     tests the contribution of heterogeneity.

All three share the same forward() signature as THGNNDetector and can be used
as drop-in replacements in experiment wrappers.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fusion import (
    MultimodalFusion,
    TemporalBurstiness,
    THGNNDetector,
    _group_timestamps,
    _mean_pool,
)
from .hgt import HGTEncoder


# ---------------------------------------------------------------------------
# 1. THGNNNoText — zero the text modality
# ---------------------------------------------------------------------------

class THGNNNoText(THGNNDetector):
    """THGNNDetector with both BERT text streams replaced by zero vectors.

    Inherits the full HGT + fusion + burstiness architecture; the only change
    is that review_embs and item_embs are zeroed before the fusion cross-
    attention so no text signal reaches the MLP.
    """

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
        review_embs: Tensor,
        item_embs: Optional[Tensor],
        review_user_idx: Tensor,
        review_item_idx: Tensor,
        timestamps: Tensor,
        node_timestamps: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # Zero text; item_embs=None is zeroed inside parent's forward
        return super().forward(
            node_feats,
            edge_index_dict,
            edge_attr_dict,
            torch.zeros_like(review_embs),
            None,
            review_user_idx,
            review_item_idx,
            timestamps,
            node_timestamps,
        )


# ---------------------------------------------------------------------------
# 2. THGNNNoTemporal — disable all temporal signals
# ---------------------------------------------------------------------------

class THGNNNoTemporal(nn.Module):
    """THGNNDetector without temporal encoding in HGT or TemporalBurstiness GRU.

    Temporal signals are eliminated in two places:
      1. HGT receives an empty edge_attr_dict so every HGTLayer skips the
         sinusoidal temporal bias branch (ts_edge is None for all edges).
      2. No TemporalBurstiness module is created; the MLP input is
         fusion.proj_dim only (no burst concatenation).

    Parameters
    ----------
    hgt_encoder : HGTEncoder — same architecture as the full model
    fusion      : MultimodalFusion
    mlp_hidden  : int
    dropout     : float
    """

    def __init__(
        self,
        hgt_encoder: HGTEncoder,
        fusion: MultimodalFusion,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hgt    = hgt_encoder
        self.fusion = fusion

        combined_dim = fusion.proj_dim  # no burstiness term
        self.mlp = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
        review_embs: Tensor,
        item_embs: Optional[Tensor],
        review_user_idx: Tensor,
        review_item_idx: Tensor,
        timestamps: Tensor,
        node_timestamps: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        num_users = node_feats["user"].size(0)
        num_items = node_feats["item"].size(0)
        device    = node_feats["user"].device

        # HGT with empty edge_attr_dict → temporal bias never applied
        hgt_out     = self.hgt(node_feats, edge_index_dict, {}, None)
        user_struct = hgt_out["user"]

        # Text streams
        user_rev_emb = _mean_pool(review_embs, review_user_idx, num_users)
        if item_embs is None:
            item_embs = torch.zeros(num_items, review_embs.size(1), device=device)
        user_item_emb = _mean_pool(
            item_embs[review_item_idx], review_user_idx, num_users
        )

        fused, _ = self.fusion(user_struct, user_rev_emb, user_item_emb)

        scores = self.mlp(fused).squeeze(1)
        return scores, {**hgt_out, "user_fused": fused}


# ---------------------------------------------------------------------------
# 3. THGNNStaticGCN — homogeneous 2-layer GCN backbone
# ---------------------------------------------------------------------------

class _StaticGCNEncoder(nn.Module):
    """2-layer homogeneous GCN over the flattened heterogeneous graph.

    All node types are projected to ``hidden_dim`` and concatenated into a
    single feature matrix.  A symmetrically-normalised adjacency (with self-
    loops, using all edge types treated as undirected) is built on the first
    call and cached for the life of the training run.

    Node ordering in the homogeneous matrix: [users | items | reviews].

    Parameters
    ----------
    hidden_dim  : int
    input_dims  : {node_type: input_feature_dim}
    dropout     : float
    """

    _NODE_ORDER = ("user", "item", "review")

    def __init__(
        self,
        hidden_dim: int,
        input_dims: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        _in = input_dims or {}
        self.input_projs = nn.ModuleDict({
            nt: nn.Linear(_in.get(nt, hidden_dim), hidden_dim)
            for nt in self._NODE_ORDER
        })
        # Two GCN weight matrices (no bias — bias absorbed into LayerNorm)
        self.gcn1  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gcn2  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

        self._cached_adj: Optional[Tensor] = None
        self._cached_key: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------------

    @staticmethod
    def _spmm(adj: Tensor, x: Tensor) -> Tensor:
        if x.device.type == "mps":
            return torch.sparse.mm(adj.cpu(), x.cpu()).to(x.device)
        return torch.sparse.mm(adj, x)

    def _build_adj(
        self,
        n_u: int,
        n_i: int,
        n_r: int,
        edge_index_dict: Dict,
        device: torch.device,
    ) -> Tensor:
        """Build D^{-½} A D^{-½} for the homogeneous flattened graph.

        Self-loops are added; all heterogeneous edges are treated as
        undirected (both src→dst and dst→src are added).
        """
        n_total = n_u + n_i + n_r
        offsets: Dict[str, int] = {"user": 0, "item": n_u, "review": n_u + n_i}

        rows: List[Tensor] = []
        cols: List[Tensor] = []

        for (src_type, _, dst_type), ei in edge_index_dict.items():
            src = (ei[0] + offsets[src_type]).long().to(device)
            dst = (ei[1] + offsets[dst_type]).long().to(device)
            rows += [src, dst]   # both directions
            cols += [dst, src]

        # Self-loops
        self_idx = torch.arange(n_total, dtype=torch.long, device=device)
        rows.append(self_idx)
        cols.append(self_idx)

        row = torch.cat(rows)
        col = torch.cat(cols)

        # Degree normalisation: d^{-½}
        deg = torch.zeros(n_total, device=device)
        deg.scatter_add_(0, row, torch.ones(len(row), device=device))
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[~torch.isfinite(d_inv_sqrt)] = 0.0

        vals = d_inv_sqrt[row] * d_inv_sqrt[col]
        adj  = torch.sparse_coo_tensor(
            torch.stack([row, col]), vals, (n_total, n_total), device=device
        ).coalesce()
        return adj

    # ------------------------------------------------------------------

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        device = next(self.parameters()).device

        n_u = node_feats["user"].shape[0]
        n_i = node_feats["item"].shape[0]
        n_r = node_feats["review"].shape[0]
        n_total = n_u + n_i + n_r

        # Rebuild adjacency only when graph size changes (fixed during training)
        key = (n_u, n_i, n_r)
        if self._cached_adj is None or self._cached_key != key:
            self._cached_adj = self._build_adj(n_u, n_i, n_r, edge_index_dict, device)
            self._cached_key = key
        adj = self._cached_adj

        # Project each node type and concatenate in canonical order
        h = torch.empty(n_total, self.hidden_dim, device=device)
        h[0       : n_u]        = self.input_projs["user"](node_feats["user"].to(device))
        h[n_u     : n_u + n_i]  = self.input_projs["item"](node_feats["item"].to(device))
        h[n_u + n_i:]           = self.input_projs["review"](node_feats["review"].to(device))

        # GCN layer 1: H' = GELU(Â H W) + H  (residual) → LayerNorm → dropout
        h1 = self.drop(self.norm1(F.gelu(self.gcn1(self._spmm(adj, h))) + h))

        # GCN layer 2: H'' = GELU(Â H' W) + H'  → LayerNorm
        h2 = self.norm2(F.gelu(self.gcn2(self._spmm(adj, h1))) + h1)

        return {
            "user":   h2[0       : n_u],
            "item":   h2[n_u     : n_u + n_i],
            "review": h2[n_u + n_i:],
        }


class THGNNStaticGCN(nn.Module):
    """THGNNDetector with HGT replaced by a homogeneous 2-layer GCN.

    Text fusion (MultimodalFusion) and temporal burstiness (TemporalBurstiness)
    are kept unchanged; only the structural backbone changes.  This isolates the
    contribution of heterogeneous message-passing (relation-specific Q/K/V
    and type-specific attention matrices in HGT).

    Parameters
    ----------
    gcn_encoder        : _StaticGCNEncoder
    fusion             : MultimodalFusion
    burstiness_encoder : TemporalBurstiness
    mlp_hidden         : int
    dropout            : float
    """

    def __init__(
        self,
        gcn_encoder: _StaticGCNEncoder,
        fusion: MultimodalFusion,
        burstiness_encoder: TemporalBurstiness,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gcn        = gcn_encoder
        self.fusion     = fusion
        self.burstiness = burstiness_encoder

        combined_dim = fusion.proj_dim + burstiness_encoder.hidden_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,           # not used — GCN is non-temporal
        review_embs: Tensor,
        item_embs: Optional[Tensor],
        review_user_idx: Tensor,
        review_item_idx: Tensor,
        timestamps: Tensor,
        node_timestamps: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        num_users = node_feats["user"].size(0)
        num_items = node_feats["item"].size(0)
        device    = node_feats["user"].device

        # Static GCN backbone (homogeneous, no temporal)
        gcn_out     = self.gcn(node_feats, edge_index_dict)
        user_struct = gcn_out["user"]

        # Text fusion (unchanged)
        user_rev_emb = _mean_pool(review_embs, review_user_idx, num_users)
        if item_embs is None:
            item_embs = torch.zeros(num_items, review_embs.size(1), device=device)
        user_item_emb = _mean_pool(
            item_embs[review_item_idx], review_user_idx, num_users
        )
        fused, _ = self.fusion(user_struct, user_rev_emb, user_item_emb)

        # Temporal burstiness (kept — only backbone is ablated)
        ts_per_user = _group_timestamps(timestamps, review_user_idx, num_users)
        burst = self.burstiness(ts_per_user)

        combined = torch.cat([fused, burst], dim=1)
        scores   = self.mlp(combined).squeeze(1)

        return scores, {**gcn_out, "user_fused": fused, "user_burst": burst}
