"""AblationWrapper — BaseDetector adapter for THGNNDetector ablation variants.

Subclasses THGNNWrapper and overrides only ``_build()`` so that the full-graph
training loop, focal-loss optimiser, and ``score_users()`` logic are reused
without modification.

Supported variants
------------------
  "no_text"      → THGNNNoText      (BERT streams zeroed)
  "no_temporal"  → THGNNNoTemporal  (HGT temporal bias + GRU burstiness removed)
  "static_gcn"   → THGNNStaticGCN   (HGT backbone → homogeneous 2-layer GCN)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.datasets.base import HeteroDataset
from src.models import EDGE_TYPES, NODE_TYPES, HGTEncoder, MultimodalFusion, TemporalBurstiness
from src.models.ablations import (
    THGNNNoText,
    THGNNNoTemporal,
    THGNNStaticGCN,
    _StaticGCNEncoder,
)
from .thgnn_wrapper import THGNNWrapper


class AblationWrapper(THGNNWrapper):
    """Wraps a THGNNDetector ablation variant inside the standard training loop.

    Parameters
    ----------
    variant   : str   — one of "no_text", "no_temporal", "static_gcn"
    epochs    : int   — training epochs (default 30, inherited)
    lr        : float — AdamW learning rate (default 1e-3, inherited)
    hidden_dim: int   — HGT/GCN hidden dim (default 64, inherited)
    text_dim  : int   — BERT embedding dim (default 768, inherited)
    seed      : int
    """

    VARIANTS = ("no_text", "no_temporal", "static_gcn")

    def __init__(self, variant: str, **kwargs) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"variant must be one of {self.VARIANTS!r}, got {variant!r}"
            )
        super().__init__(**kwargs)
        self.variant = variant

    # ------------------------------------------------------------------
    # Override _build() — everything else (fit, score_users) is inherited
    # ------------------------------------------------------------------

    def _build(self, dataset: HeteroDataset, device: torch.device) -> nn.Module:  # type: ignore[override]
        input_dims = {
            "user":   dataset.x_user.shape[1],
            "item":   dataset.x_item.shape[1],
            "review": 1,
        }
        hd = self.hidden_dim
        nh = max(1, hd // 32)

        if self.variant == "no_text":
            hgt = HGTEncoder(
                hidden_dim=hd, num_heads=nh, num_layers=2,
                node_types=NODE_TYPES, edge_types=EDGE_TYPES,
                input_dims=input_dims, dropout=0.1, time_emb_dim=32,
            )
            fusion = MultimodalFusion(
                subgraph_dim=hd, text_dim=self.text_dim,
                proj_dim=hd, num_heads=nh, dropout=0.1,
            )
            burst = TemporalBurstiness(hidden_dim=32, dropout=0.1)
            return THGNNNoText(
                hgt, fusion, burst, mlp_hidden=hd // 2, dropout=0.1
            ).to(device)

        if self.variant == "no_temporal":
            hgt = HGTEncoder(
                hidden_dim=hd, num_heads=nh, num_layers=2,
                node_types=NODE_TYPES, edge_types=EDGE_TYPES,
                input_dims=input_dims, dropout=0.1, time_emb_dim=32,
            )
            fusion = MultimodalFusion(
                subgraph_dim=hd, text_dim=self.text_dim,
                proj_dim=hd, num_heads=nh, dropout=0.1,
            )
            return THGNNNoTemporal(
                hgt, fusion, mlp_hidden=hd // 2, dropout=0.1
            ).to(device)

        # "static_gcn"
        gcn = _StaticGCNEncoder(
            hidden_dim=hd, input_dims=input_dims, dropout=0.1,
        )
        fusion = MultimodalFusion(
            subgraph_dim=hd, text_dim=self.text_dim,
            proj_dim=hd, num_heads=nh, dropout=0.1,
        )
        burst = TemporalBurstiness(hidden_dim=32, dropout=0.1)
        return THGNNStaticGCN(
            gcn, fusion, burst, mlp_hidden=hd // 2, dropout=0.1
        ).to(device)
