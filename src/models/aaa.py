"""Attack-Aware Anomaly Attention — HGT attention augmented with anomaly priors.

Replaces the standard per-relation bilinear attention logit with the same
logit plus three learnably-weighted anomaly scores, all computed *inside*
the softmax (i.e. added to the logit before normalisation, not fused as a
separate stream afterwards):

    alpha(u, v, phi) = softmax_u [
        W_base(q_v, k_u, phi)
      + lambda_t * f_temporal(dt_uv)
      + lambda_s * f_semantic(r_uv)
      + lambda_g * f_structural(d_u)
    ]

    f_temporal(dt)   = -log(dt / mu_dt + eps)          — bursty edges score high
    f_semantic(r)    = 1 - cosine_sim(r_emb, item_emb) — incoherent review/item pairs score high
    f_structural(du) = exp(-du / d_mean)               — thin-neighbourhood users score high

W_base is the same per-relation bilinear form used by HGTLayer
(q^T W_att^phi k), so this module is a drop-in replacement for HGTLayer's
attention when substituted via HGTEncoder(use_aaa=True).

Design note on node types
--------------------------
Unlike HGTLayer, this module's constructor does not take a ``node_types``
list (per the required 3-argument signature). Q/K/V/output/residual
projections are therefore shared across all node types rather than
per-type — valid here because HGTEncoder already projects every node type
into the same ``hidden_dim`` space before any layer (this one included)
runs.

Design note on edge types
--------------------------
The constructor takes ``num_edge_types`` (a count) rather than the list of
edge-type tuples, so per-relation W_att / W_msg tensors are allocated
eagerly as a single stacked parameter indexed by an integer slot. The
mapping from an edge-type tuple to its slot is learned lazily the first
time that edge type is seen in ``forward`` (capped at ``num_edge_types``
distinct types) — but the underlying parameters themselves already exist
at construction time, so they are visible to an optimiser built right
after ``__init__``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hgt import EdgeKey, _scatter_softmax

__all__ = ["AttackAwareAnomalyAttention"]


class AttackAwareAnomalyAttention(nn.Module):
    """HGT-style attention layer with joint temporal/semantic/structural anomaly scoring.

    Parameters
    ----------
    hidden_dim     : int  — shared embedding dimension for every node type
    num_heads      : int  — attention heads; hidden_dim must be divisible by num_heads
    num_edge_types : int  — upper bound on the number of distinct edge types
                             this layer will see across all forward() calls
    dropout        : float
    eps            : float — numerical-stability epsilon used in f_temporal

    forward(x_dict, edge_index_dict, edge_attr_dict, review_embs=None, item_embs=None)
    ------------------------------------------------------------------------------------
    x_dict           : Dict[str, FloatTensor(N_t, hidden_dim)]
    edge_index_dict  : Dict[EdgeKey, LongTensor(2, E)]
    edge_attr_dict   : Dict[EdgeKey, FloatTensor(E,)]  — per-edge Δt (seconds)
    review_embs      : Optional[FloatTensor(N_review, D)] — review text embeddings
    item_embs        : Optional[FloatTensor(N_item, D)]   — item description embeddings

    Returns
    -------
    Dict[str, FloatTensor(N_t, hidden_dim)]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_edge_types: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim     = hidden_dim
        self.num_heads      = num_heads
        self.d_head         = hidden_dim // num_heads
        self.num_edge_types = num_edge_types
        self.eps            = eps

        # ── Shared (type-agnostic) Q / K / V / output / residual projections ──
        self.W_Q   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.W_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm  = nn.LayerNorm(hidden_dim)

        # ── Per-relation bilinear attention / message matrices ────────────────
        # Stacked and indexed by integer slot (see _edge_slot below).
        scale = 1.0 / math.sqrt(self.d_head)
        self.W_att = nn.Parameter(
            torch.empty(num_edge_types, num_heads, self.d_head, self.d_head).normal_(std=scale)
        )
        self.W_msg = nn.Parameter(
            torch.empty(num_edge_types, num_heads, self.d_head, self.d_head).normal_(std=scale)
        )

        # ── Learnable anomaly-term weights (initialised to 0.1) ───────────────
        self.lambda_t = nn.Parameter(torch.tensor(0.1))
        self.lambda_s = nn.Parameter(torch.tensor(0.1))
        self.lambda_g = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)

        # Lazily populated: edge-type tuple -> integer slot in W_att / W_msg
        self._edge_slot: Dict[EdgeKey, int] = {}

    # ------------------------------------------------------------------
    # Edge-type -> parameter-slot mapping
    # ------------------------------------------------------------------

    def _edge_type_slot(self, et: EdgeKey) -> int:
        if et not in self._edge_slot:
            if len(self._edge_slot) >= self.num_edge_types:
                raise ValueError(
                    f"Encountered more distinct edge types ({len(self._edge_slot) + 1}) "
                    f"than num_edge_types={self.num_edge_types}"
                )
            self._edge_slot[et] = len(self._edge_slot)
        return self._edge_slot[et]

    # ------------------------------------------------------------------
    # Anomaly-score helpers
    # ------------------------------------------------------------------

    def _f_temporal(self, delta_t: Optional[Tensor], num_edges: int, device, dtype) -> Tensor:
        """f_temporal(Δt) = -log(Δt / μ_Δt + eps); μ_Δt = mean over this edge type."""
        if delta_t is None:
            return torch.zeros(num_edges, device=device, dtype=dtype)
        dt = delta_t.float()
        mu = dt.mean().clamp_min(self.eps)
        return -torch.log(dt / mu + self.eps)

    def _f_semantic(
        self,
        et: EdgeKey,
        src_n: Tensor,
        dst_n: Tensor,
        review_embs: Optional[Tensor],
        item_embs: Optional[Tensor],
        num_edges: int,
        device,
        dtype,
    ) -> Tensor:
        """f_semantic(r) = 1 - cosine_sim(review_emb, item_emb) for review<->item edges."""
        if review_embs is None or item_embs is None:
            return torch.zeros(num_edges, device=device, dtype=dtype)

        src_type, _, dst_type = et
        if src_type == "review":
            r_idx = src_n
        elif dst_type == "review":
            r_idx = dst_n
        else:
            return torch.zeros(num_edges, device=device, dtype=dtype)

        if dst_type == "item":
            i_idx = dst_n
        elif src_type == "item":
            i_idx = src_n
        else:
            return torch.zeros(num_edges, device=device, dtype=dtype)

        r_emb = review_embs[r_idx]
        i_emb = item_embs[i_idx]
        cos_sim = F.cosine_similarity(r_emb, i_emb, dim=-1)
        return 1.0 - cos_sim

    @staticmethod
    def _compute_degrees(x_dict: Dict[str, Tensor], edge_index_dict: Dict[EdgeKey, Tensor]) -> Dict[str, Tensor]:
        """Total incident-edge count per node, per node type (both directions)."""
        degree = {nt: torch.zeros(feat.shape[0], device=feat.device) for nt, feat in x_dict.items()}
        for (src_type, _, dst_type), ei in edge_index_dict.items():
            if ei.shape[1] == 0:
                continue
            src_n, dst_n = ei[0], ei[1]
            degree[src_type].scatter_add_(0, src_n, torch.ones_like(src_n, dtype=degree[src_type].dtype))
            degree[dst_type].scatter_add_(0, dst_n, torch.ones_like(dst_n, dtype=degree[dst_type].dtype))
        return degree

    def _f_structural(self, src_type: str, src_n: Tensor, degree: Dict[str, Tensor]) -> Tensor:
        """f_structural(d_u) = exp(-d_u / d_mean); d_mean over the source node type."""
        deg = degree[src_type]
        d_mean = deg.mean().clamp_min(self.eps)
        d_u = deg[src_n]
        return torch.exp(-d_u / d_mean)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeKey, Tensor],
        edge_attr_dict: Dict[EdgeKey, Tensor],
        review_embs: Optional[Tensor] = None,
        item_embs: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        d_sqrt = math.sqrt(self.d_head)
        degree = self._compute_degrees(x_dict, edge_index_dict)
        new_h: Dict[str, Tensor] = {}

        node_types = list(x_dict.keys())

        for dst_type in node_types:
            N_dst = x_dict[dst_type].shape[0]

            agg_logits: list = []
            agg_msgs:   list = []
            agg_dst:    list = []

            for et, ei in edge_index_dict.items():
                src_type, _, dst_type_ = et
                if dst_type_ != dst_type:
                    continue
                E = ei.shape[1]
                if E == 0:
                    continue

                src_n = ei[0]
                dst_n = ei[1]
                device, dtype = x_dict[dst_type].device, x_dict[dst_type].dtype
                slot = self._edge_type_slot(et)

                # ── Q / K / V ──────────────────────────────────────────────
                k = self.W_K(x_dict[src_type])[src_n].view(E, self.num_heads, self.d_head)
                q = self.W_Q(x_dict[dst_type])[dst_n].view(E, self.num_heads, self.d_head)
                v = self.W_V(x_dict[src_type])[src_n].view(E, self.num_heads, self.d_head)

                # ── W_base: standard bilinear heterogeneous attention ───────
                W_a = self.W_att[slot]                                     # (H, d, d)
                base_logit = torch.einsum("ehd,hdf,ehf->eh", q, W_a, k) / d_sqrt  # (E, H)

                # ── Anomaly scores, computed inside the softmax logit ───────
                delta_t = edge_attr_dict.get(et)
                f_temp = self._f_temporal(delta_t, E, device, dtype)                       # (E,)
                f_sem  = self._f_semantic(et, src_n, dst_n, review_embs, item_embs, E, device, dtype)  # (E,)
                f_struct = self._f_structural(src_type, src_n, degree)                     # (E,)

                anomaly_bias = (
                    self.lambda_t * f_temp
                    + self.lambda_s * f_sem
                    + self.lambda_g * f_struct
                )                                                          # (E,)
                logit = base_logit + anomaly_bias.unsqueeze(1)             # (E, H)

                # ── Message ──────────────────────────────────────────────
                W_m = self.W_msg[slot]                                     # (H, d, d)
                msg = torch.einsum("hdf,ehf->ehd", W_m, v)                 # (E, H, d)

                agg_logits.append(logit)
                agg_msgs.append(msg)
                agg_dst.append(dst_n)

            if not agg_logits:
                new_h[dst_type] = self.norm(self.W_res(x_dict[dst_type]))
                continue

            all_logits = torch.cat(agg_logits, dim=0)   # (E_tot, H)
            all_msgs   = torch.cat(agg_msgs,   dim=0)   # (E_tot, H, d)
            all_dst    = torch.cat(agg_dst,    dim=0)   # (E_tot,)

            weights = _scatter_softmax(all_logits, all_dst, N_dst)   # (E_tot, H)
            weights = self.dropout(weights)

            weighted = weights.unsqueeze(-1) * all_msgs
            idx_exp  = all_dst.unsqueeze(1).unsqueeze(2).expand_as(weighted)
            agg      = weighted.new_zeros(N_dst, self.num_heads, self.d_head)
            agg.scatter_add_(0, idx_exp, weighted)
            agg      = agg.reshape(N_dst, self.hidden_dim)

            out = self.W_out(F.gelu(agg))
            res = self.W_res(x_dict[dst_type])
            new_h[dst_type] = self.norm(out + res)

        return new_h
