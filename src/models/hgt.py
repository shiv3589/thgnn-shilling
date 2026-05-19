"""Heterogeneous Graph Transformer (HGT) backbone — Hu et al., WWW 2020.

Reference
---------
Hu et al. "Heterogeneous Graph Transformer." WWW 2020.
https://arxiv.org/abs/2003.01332

Architecture summary
--------------------
Each HGT layer computes, for every target node t of type τ(t):

    ATT(s, e, t)  =  q(t)ᵀ W_att^{ϕ(e)} k(s)  /  √d_head  +  φ(Δtₛₜ)
    MSG(s, e, t)  =  W_msg^{ϕ(e)} v(s)
    Agg(t)        =  Σ_s  softmax_s[ATT(s,e,t)] · MSG(s,e,t)
    H^{l+1}(t)   =  LayerNorm( W_out(GELU(Agg(t))) + W_res · H^l(t) )

where:
  q(t) = W_Q^{τ(t)} H^l(t)      — query from the *target* node type
  k(s) = W_K^{τ(s)} H^l(s)      — key   from the *source* node type
  v(s) = W_V^{τ(s)} H^l(s)      — value from the *source* node type
  φ(Δt) = scalar temporal bias   — learnable sinusoidal encoding of |t_s − t_t|

Softmax is computed jointly over *all* incoming edge types for a given target.

Temporal encoding
-----------------
The learnable sinusoidal encoding maps a scalar time delta (in seconds) to a
per-edge attention bias that is *added* to the raw logit — not concatenated
to node features.  Frequencies ω and phases φ are nn.Parameters; the
combined sinusoidal vector is then projected to a scalar by a learned linear.

When node_timestamps is None the raw edge timestamp is used as the temporal
signal (treating t=0 as reference), so temporal encoding is always active.

Graph topology note
-------------------
The three canonical edge types give items two incoming relation types
(review→item, user→item) and reviews one (user→review), while users receive
no incoming messages under the default edge set.  Adding reverse edges
(review→user, item→user) before calling forward() is recommended for richer
user representations in practice.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Matches the type alias in src/datasets/base.py
EdgeKey = Tuple[str, str, str]

# Default node / edge types for this project
NODE_TYPES: List[str] = ["user", "item", "review"]
EDGE_TYPES: List[EdgeKey] = [
    ("user",   "writes", "review"),
    ("review", "about",  "item"),
    ("user",   "rates",  "item"),
]


# ---------------------------------------------------------------------------
# Helper: edge-type key
# ---------------------------------------------------------------------------

def _ek(et: EdgeKey) -> str:
    """Convert an EdgeKey tuple to a ParameterDict-safe string key."""
    return f"{et[0]}__{et[1]}__{et[2]}"


# ---------------------------------------------------------------------------
# Temporal encoding
# ---------------------------------------------------------------------------

class TemporalEncoding(nn.Module):
    """Learnable-frequency sinusoidal encoding of a scalar time delta.

    φ(Δt) = Linear( [sin(ω·(Δt/86400) + α) ; cos(ω·(Δt/86400) + α)] )

    where ω (frequencies) and α (phases) are learnable parameters.  The
    linear projection outputs a single scalar that is *added* to the
    attention logit before softmax.

    Inputs are normalised to days (÷ 86 400) so initial frequency values
    near zero are meaningful.

    Parameters
    ----------
    emb_dim : int — internal sinusoidal dimension (must be even, default 64)
    """

    def __init__(self, emb_dim: int = 64) -> None:
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError("emb_dim must be even")
        half = emb_dim // 2
        self.omega = nn.Parameter(torch.randn(half) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(half))
        self.proj  = nn.Linear(emb_dim, 1, bias=True)
        # Start near zero so temporal bias doesn't dominate at init
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, delta_t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        delta_t : FloatTensor (E,) — |t_src − t_dst| in seconds (or abs. ts)

        Returns
        -------
        FloatTensor (E,) — scalar per-edge attention bias
        """
        dt    = delta_t.float().unsqueeze(-1) / 86_400.0   # (E, 1), in days
        arg   = dt * self.omega + self.alpha               # (E, half)
        emb   = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (E, emb_dim)
        return self.proj(emb).squeeze(-1)                  # (E,)


# ---------------------------------------------------------------------------
# Scatter-softmax helper
# ---------------------------------------------------------------------------

def _scatter_softmax(logits: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Numerically-stable softmax over groups defined by ``index``.

    For each unique value g in ``index``, computes softmax over all positions
    where ``index == g``.  Uses the max-subtraction trick per group.

    Parameters
    ----------
    logits    : FloatTensor (E, H)
    index     : LongTensor  (E,)   — destination node index for each edge
    num_nodes : int

    Returns
    -------
    FloatTensor (E, H) — attention weights; each group sums to 1
    """
    H   = logits.shape[1]
    idx = index.unsqueeze(1).expand(-1, H)          # (E, H)

    # Per-group maximum for numerical stability
    max_val = logits.new_full((num_nodes, H), float("-inf"))
    max_val.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)
    # Nodes with no incoming edges keep -inf; they are never in `index` so safe.

    exp_l   = (logits - max_val[index]).exp()       # (E, H)
    sum_exp = logits.new_zeros(num_nodes, H)
    sum_exp.scatter_add_(0, idx, exp_l)

    return exp_l / (sum_exp[index] + 1e-8)          # (E, H)


# ---------------------------------------------------------------------------
# Single HGT layer
# ---------------------------------------------------------------------------

class HGTLayer(nn.Module):
    """One HGT message-passing layer.

    Parameters
    ----------
    hidden_dim   : int
    num_heads    : int              — hidden_dim must be divisible by num_heads
    node_types   : List[str]
    edge_types   : List[EdgeKey]
    dropout      : float
    time_emb_dim : int
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_types: List[str],
        edge_types: List[EdgeKey],
        dropout: float = 0.1,
        time_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_head     = hidden_dim // num_heads
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        # ── Per-node-type Q / K / V projections ───────────────────────────
        self.W_Q = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim, bias=False) for nt in node_types})
        self.W_K = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim, bias=False) for nt in node_types})
        self.W_V = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim, bias=False) for nt in node_types})

        # ── Per-relation W_att and W_msg matrices ─────────────────────────
        #   shape (num_heads, d_head, d_head) — one square matrix per head
        scale = 1.0 / math.sqrt(self.d_head)
        self.W_att = nn.ParameterDict({
            _ek(et): nn.Parameter(torch.empty(num_heads, self.d_head, self.d_head).normal_(std=scale))
            for et in edge_types
        })
        self.W_msg = nn.ParameterDict({
            _ek(et): nn.Parameter(torch.empty(num_heads, self.d_head, self.d_head).normal_(std=scale))
            for et in edge_types
        })

        # ── Per-node-type output projection + residual + norm ─────────────
        self.W_out  = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types})
        self.W_res  = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim, bias=False) for nt in node_types})
        self.norm   = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})

        self.temporal_enc = TemporalEncoding(time_emb_dim)
        self.dropout      = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        h: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeKey, Tensor],
        edge_attr_dict:  Dict[EdgeKey, Tensor],
        node_timestamps: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        d_sqrt  = math.sqrt(self.d_head)
        new_h: Dict[str, Tensor] = {}

        for dst_type in self.node_types:
            if dst_type not in h:
                continue
            N_dst = h[dst_type].shape[0]

            agg_logits: List[Tensor] = []
            agg_msgs:   List[Tensor] = []
            agg_dst:    List[Tensor] = []

            for et in self.edge_types:
                src_type, _, dst_type_ = et
                if dst_type_ != dst_type:
                    continue
                if et not in edge_index_dict:
                    continue

                ei   = edge_index_dict[et]          # (2, E)
                E    = ei.shape[1]
                if E == 0:
                    continue

                src_n = ei[0]                        # (E,)
                dst_n = ei[1]                        # (E,)
                ek    = _ek(et)

                # ── Q / K / V per edge ────────────────────────────────────
                # k, v: from source node type;  q: from target node type
                k = self.W_K[src_type](h[src_type])[src_n].view(E, self.num_heads, self.d_head)
                q = self.W_Q[dst_type](h[dst_type])[dst_n].view(E, self.num_heads, self.d_head)
                v = self.W_V[src_type](h[src_type])[src_n].view(E, self.num_heads, self.d_head)

                # ── Attention logit: q^T W_att k / √d ────────────────────
                # einsum 'ehd, hdf, ehf -> eh'
                #   e=edge, h=head, d=query-dim, f=key-dim (both = d_head)
                W_a   = self.W_att[ek]                           # (H, d, d)
                logit = torch.einsum("ehd,hdf,ehf->eh", q, W_a, k) / d_sqrt  # (E, H)

                # ── Temporal bias (scalar, same across heads) ─────────────
                ts_edge = edge_attr_dict.get(et)
                if ts_edge is not None:
                    if node_timestamps is not None:
                        zero_u = h[src_type].new_zeros(h[src_type].shape[0])
                        zero_t = h[dst_type].new_zeros(h[dst_type].shape[0])
                        ts_s   = node_timestamps.get(src_type, zero_u)[src_n]
                        ts_d   = node_timestamps.get(dst_type, zero_t)[dst_n]
                        delta  = (ts_s - ts_d).abs()
                    else:
                        # Use absolute edge timestamp when node timestamps absent
                        delta = ts_edge.float()
                    bias  = self.temporal_enc(delta)             # (E,)
                    logit = logit + bias.unsqueeze(1)            # (E, H)

                # ── Message: W_msg v ──────────────────────────────────────
                # einsum 'hdf, ehf -> ehd'
                #   h=head, d=output-dim, f=input-dim (both = d_head), e=edge
                W_m = self.W_msg[ek]                             # (H, d, d)
                msg = torch.einsum("hdf,ehf->ehd", W_m, v)      # (E, H, d)

                agg_logits.append(logit)
                agg_msgs.append(msg)
                agg_dst.append(dst_n)

            # ── Residual-only update when no incoming edges ───────────────
            if not agg_logits:
                new_h[dst_type] = self.norm[dst_type](
                    self.W_res[dst_type](h[dst_type])
                )
                continue

            all_logits = torch.cat(agg_logits, dim=0)   # (E_tot, H)
            all_msgs   = torch.cat(agg_msgs,   dim=0)   # (E_tot, H, d)
            all_dst    = torch.cat(agg_dst,    dim=0)   # (E_tot,)

            # ── Scatter-softmax over destination nodes ────────────────────
            weights = _scatter_softmax(all_logits, all_dst, N_dst)   # (E_tot, H)
            weights = self.dropout(weights)

            # ── Weighted message aggregation ──────────────────────────────
            weighted = weights.unsqueeze(-1) * all_msgs               # (E_tot, H, d)
            idx_exp  = all_dst.unsqueeze(1).unsqueeze(2).expand_as(weighted)
            agg      = weighted.new_zeros(N_dst, self.num_heads, self.d_head)
            agg.scatter_add_(0, idx_exp, weighted)
            agg      = agg.reshape(N_dst, self.hidden_dim)            # (N_dst, D)

            # ── Output projection + skip + layer norm ─────────────────────
            out = self.W_out[dst_type](F.gelu(agg))
            res = self.W_res[dst_type](h[dst_type])
            new_h[dst_type] = self.norm[dst_type](out + res)

        return new_h


# ---------------------------------------------------------------------------
# HGT encoder — public API
# ---------------------------------------------------------------------------

class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder.

    Parameters
    ----------
    hidden_dim   : int          — shared embedding dimension (all node types)
    num_heads    : int          — attention heads; hidden_dim % num_heads == 0
    num_layers   : int          — stacked HGT layers
    node_types   : List[str]    — e.g. ["user", "item", "review"]
    edge_types   : List[EdgeKey]— e.g. [("user","writes","review"), ...]
    input_dims   : Dict[str, int] or None
                    Per-node-type input feature dimension.  Defaults to
                    hidden_dim for every unspecified type (assumes features
                    are already projected to hidden_dim before calling).
    dropout      : float
    time_emb_dim : int          — sinusoidal encoding internal dimension

    forward(node_feats, edge_index_dict, edge_attr_dict, node_timestamps=None)
    --------------------------------------------------------------------------
    node_feats       : Dict[str, FloatTensor(N_t, D_t)]
    edge_index_dict  : Dict[EdgeKey, LongTensor(2, E)]
    edge_attr_dict   : Dict[EdgeKey, FloatTensor(E,)]  — Unix-second timestamps
    node_timestamps  : Dict[str, FloatTensor(N_t,)] or None
                         When given, Δt per edge = |ts_src − ts_dst|.
                         When None, the raw edge timestamp is used directly.

    Returns
    -------
    Dict[str, FloatTensor(N_t, hidden_dim)]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        node_types: List[str],
        edge_types: List[EdgeKey],
        input_dims: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
        time_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        _in = input_dims or {}
        self.input_projs = nn.ModuleDict({
            nt: nn.Linear(_in.get(nt, hidden_dim), hidden_dim)
            for nt in node_types
        })

        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, num_heads, node_types, edge_types, dropout, time_emb_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeKey, Tensor],
        edge_attr_dict: Dict[EdgeKey, Tensor],
        node_timestamps: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        # Project each node type from its input dim to hidden_dim
        h: Dict[str, Tensor] = {
            nt: self.input_projs[nt](feat)
            for nt, feat in node_feats.items()
            if nt in self.input_projs
        }

        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_attr_dict, node_timestamps)

        return h
