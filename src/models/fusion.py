"""Multimodal fusion and shilling-attack detection head.

Architecture
------------
MultimodalFusion
    Three 128-dim projections of:
      1. subgraph_emb  : HGT user node output           (hidden_dim)
      2. review_emb    : mean-pooled review BERT CLS     (text_dim=768)
      3. item_emb      : mean-pooled item BERT CLS       (text_dim=768)
    Cross-modal attention: query=subgraph, keys/values=stacked text streams
    Add-Norm → position-wise FF → Add-Norm → 128-dim fused vector.

TemporalBurstiness
    Per-user inter-arrival times (log1p seconds) encoded by a 1-layer GRU
    → 32-dim burstiness vector.

THGNNDetector
    HGT encodes the heterogeneous subgraph → user embeddings (128-dim).
    MultimodalFusion fuses with BERT text streams (128-dim).
    TemporalBurstiness encodes review timestamps (32-dim).
    Concatenation (160-dim) → LayerNorm → 2-layer MLP → scalar shilling score.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .hgt import HGTEncoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_pool(embs: Tensor, idx: Tensor, num_nodes: int) -> Tensor:
    """Scatter-mean: aggregate rows of *embs* by integer *idx* → (num_nodes, D).

    Nodes with no entries receive a zero vector.
    """
    D = embs.size(1)
    out = torch.zeros(num_nodes, D, dtype=embs.dtype, device=embs.device)
    cnt = torch.zeros(num_nodes, 1, dtype=embs.dtype, device=embs.device)
    expanded = idx.unsqueeze(1).expand(-1, D)
    out.scatter_add_(0, expanded, embs)
    cnt.scatter_add_(0, idx.unsqueeze(1), torch.ones(len(idx), 1, dtype=embs.dtype, device=embs.device))
    cnt.clamp_(min=1.0)
    return out / cnt


def _group_timestamps(
    timestamps: Tensor,
    review_user_idx: Tensor,
    num_users: int,
) -> List[Tensor]:
    """Return a list of length *num_users* where element i holds the sorted
    review timestamps (float32, seconds) for user i."""
    groups: List[List[float]] = [[] for _ in range(num_users)]
    ts_list = timestamps.tolist()
    uid_list = review_user_idx.tolist()
    for t, u in zip(ts_list, uid_list):
        groups[u].append(t)
    device = timestamps.device
    return [
        torch.tensor(sorted(g), dtype=torch.float32, device=device)
        for g in groups
    ]


# ---------------------------------------------------------------------------
# MultimodalFusion
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """Cross-modal attention fusion of subgraph + two text embedding streams.

    Parameters
    ----------
    subgraph_dim : dimension of incoming HGT user embedding
    text_dim     : dimension of BERT CLS vectors (review and item streams)
    proj_dim     : shared projection dimension (default 128)
    num_heads    : attention heads (must divide proj_dim)
    dropout      : dropout probability on attention and FF layers
    """

    def __init__(
        self,
        subgraph_dim: int = 128,
        text_dim: int = 768,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim

        # Input projections → proj_dim
        self.proj_sub  = nn.Linear(subgraph_dim, proj_dim)
        self.proj_rev  = nn.Linear(text_dim,     proj_dim)
        self.proj_item = nn.Linear(text_dim,     proj_dim)

        # Cross-modal attention: query=subgraph, keys/values=text streams
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)

        # Position-wise feed-forward
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        subgraph_emb: Tensor,
        review_emb: Tensor,
        item_emb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        subgraph_emb : (N, subgraph_dim)
        review_emb   : (N, text_dim)
        item_emb     : (N, text_dim)

        Returns
        -------
        fused        : (N, proj_dim)
        attn_weights : (N, 1, 2)  — attention over [review, item] streams
        """
        q = self.proj_sub(subgraph_emb).unsqueeze(1)   # (N, 1, proj_dim)
        k = torch.stack([
            self.proj_rev(review_emb),
            self.proj_item(item_emb),
        ], dim=1)                                        # (N, 2, proj_dim)
        v = k

        attn_out, attn_w = self.attn(q, k, v)           # (N,1,D), (N,1,2)
        q2 = self.norm1(q + attn_out)                    # Add-Norm
        out = self.norm2(q2 + self.ff(q2))               # FF + Add-Norm
        return out.squeeze(1), attn_w                    # (N, proj_dim)


# ---------------------------------------------------------------------------
# TemporalBurstiness
# ---------------------------------------------------------------------------

class TemporalBurstiness(nn.Module):
    """Encodes per-user review inter-arrival times with a 1-layer GRU.

    Input  : list of N per-user timestamp tensors (sorted ascending, seconds)
    Output : (N, hidden_dim) burstiness vectors
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, ts_list: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        ts_list : list of N float32 tensors, each of shape (L_i,)

        Returns
        -------
        out : (N, hidden_dim)
        """
        device = next(self.parameters()).device
        N = len(ts_list)
        seqs: List[Tensor] = []
        lengths: List[int] = []

        for ts in ts_list:
            if ts.numel() < 2:
                # No inter-arrival time available; use a single zero step
                seqs.append(torch.zeros(1, 1, device=device))
                lengths.append(1)
            else:
                iat = torch.diff(ts.to(device))
                iat = torch.log1p(iat.clamp(min=0.0)).unsqueeze(1)  # (L-1, 1)
                seqs.append(iat)
                lengths.append(iat.size(0))

        max_len = max(lengths)
        padded = torch.zeros(N, max_len, 1, device=device)
        for i, (seq, ln) in enumerate(zip(seqs, lengths)):
            padded[i, :ln] = seq

        packed = pack_padded_sequence(
            padded,
            lengths=torch.tensor(lengths, dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)            # h_n: (1, N, hidden_dim)
        return self.drop(h_n.squeeze(0))     # (N, hidden_dim)


# ---------------------------------------------------------------------------
# THGNNDetector
# ---------------------------------------------------------------------------

class THGNNDetector(nn.Module):
    """Temporal Heterogeneous GNN shilling-attack detector.

    Combines:
      - HGT structural embeddings  (proj_dim = 128)
      - Multimodal BERT text fusion (proj_dim = 128)
      - Temporal burstiness GRU    (burst_dim = 32)

    The concatenation (128 + 32 = 160-dim) is scored by a 2-layer MLP.

    Parameters
    ----------
    hgt_encoder        : pre-built HGTEncoder
    fusion             : MultimodalFusion
    burstiness_encoder : TemporalBurstiness
    mlp_hidden         : hidden size of the classification MLP (default 64)
    dropout            : dropout for MLP (default 0.1)
    """

    def __init__(
        self,
        hgt_encoder: HGTEncoder,
        fusion: MultimodalFusion,
        burstiness_encoder: TemporalBurstiness,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hgt       = hgt_encoder
        self.fusion    = fusion
        self.burstiness = burstiness_encoder

        combined_dim = fusion.proj_dim + burstiness_encoder.hidden_dim  # 128+32

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
        """
        Parameters
        ----------
        node_feats       : {node_type: (N_t, D_t)} numeric features
        edge_index_dict  : {(src, rel, dst): LongTensor(2, E)}
        edge_attr_dict   : {(src, rel, dst): FloatTensor(E,)} edge timestamps
        review_embs      : (N_reviews, 768) BERT CLS vectors for review texts
        item_embs        : (N_items, 768) BERT CLS vectors for item texts;
                           pass None to use zeros (text stream disabled)
        review_user_idx  : (N_reviews,) user index for each review
        review_item_idx  : (N_reviews,) item index for each review
        timestamps       : (N_reviews,) review timestamps (seconds, float)
        node_timestamps  : optional per-node timestamps for HGT temporal attn

        Returns
        -------
        scores           : (N_users,) raw logit shilling scores (unbounded)
        embeddings       : dict with keys "user", "item", "review" —
                           the final HGT node embeddings plus "user_fused"
                           (128-dim post-fusion) and "user_burst" (32-dim)
        """
        num_users = node_feats["user"].size(0)
        num_items = node_feats["item"].size(0)
        device    = node_feats["user"].device

        # ── 1. HGT structural embeddings ──────────────────────────────────
        hgt_out = self.hgt(
            node_feats, edge_index_dict, edge_attr_dict,
            node_timestamps=node_timestamps,
        )
        user_struct = hgt_out["user"]   # (N_users, hidden_dim)

        # ── 2. Text stream: per-user mean-pooled BERT embeddings ──────────
        # Review stream: mean-pool review_embs per user
        user_rev_emb = _mean_pool(review_embs, review_user_idx, num_users)

        # Item stream: mean-pool item_embs per user via review→item link
        if item_embs is None:
            item_embs = torch.zeros(num_items, review_embs.size(1), device=device)
        # For each review, get item emb, then scatter-mean to users
        review_item_embs = item_embs[review_item_idx]            # (N_reviews, 768)
        user_item_emb    = _mean_pool(review_item_embs, review_user_idx, num_users)

        # ── 3. Multimodal fusion ───────────────────────────────────────────
        fused, attn_w = self.fusion(user_struct, user_rev_emb, user_item_emb)

        # ── 4. Temporal burstiness ─────────────────────────────────────────
        ts_per_user = _group_timestamps(timestamps, review_user_idx, num_users)
        burst = self.burstiness(ts_per_user)                     # (N_users, 32)

        # ── 5. Classification MLP ─────────────────────────────────────────
        combined = torch.cat([fused, burst], dim=1)              # (N_users, 160)
        scores   = self.mlp(combined).squeeze(1)                 # (N_users,)

        embeddings = {
            **hgt_out,
            "user_fused": fused,
            "user_burst": burst,
        }
        return scores, embeddings
