"""THGNNWrapper — programmatic interface to THGNNDetector for the experiment runner.

Adapts the full THGNNDetector architecture from ``src/models/`` into the
``BaseDetector`` interface.  Training mirrors ``scripts/train.py`` but runs
full-graph (no k-hop mini-batching) for simplicity; the architecture and
loss function are identical.

For very large datasets (>50 k users) the full-graph forward pass may OOM.
Set ``max_users_full_graph`` to a lower bound and the wrapper will fall back
to the statistical baseline silently.

Hyperparameters are reduced vs. the standalone ``train.py`` to keep
experiment-matrix runs tractable:
  - 30 training epochs  (vs 100)
  - hidden_dim=64       (vs 128)
  - 2 HGT layers        (vs 2, same)
  - review_feat_dim=1   (dummy; BERT embeddings not precomputed here)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.base import HeteroDataset
from src.models import (
    EDGE_TYPES,
    NODE_TYPES,
    HGTEncoder,
    MultimodalFusion,
    TemporalBurstiness,
    THGNNDetector,
)
from .base import BaseDetector

logger = logging.getLogger(__name__)

_MAX_USERS_FULL_GRAPH = 100_000


def _focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    bce   = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    p_t   = torch.exp(-bce)
    a_t   = labels.float() * alpha + (1.0 - labels.float()) * (1.0 - alpha)
    return (a_t * (1.0 - p_t) ** gamma * bce).mean()


class THGNNWrapper(BaseDetector):
    """Trains THGNNDetector on the poisoned dataset and returns user anomaly scores.

    Parameters
    ----------
    epochs      : int   — training epochs (default 30)
    lr          : float — AdamW learning rate (default 1e-3)
    hidden_dim  : int   — HGT hidden/output dim (default 64)
    text_dim    : int   — BERT embedding dim; zeros used if embeddings absent (768)
    seed        : int
    use_aaa     : bool  — substitute AttackAwareAnomalyAttention (Eq. 10: base +
                  λ_t·f_temporal + λ_s·f_semantic + λ_g·f_structural) for the
                  standard HGT attention (Eq. 5: base + temporal bias only) in
                  every HGT layer. Forwarded to HGTEncoder(use_aaa=...); see
                  ``configs/paper_runs.yaml``'s ``thgnn.use_aaa``.
    """

    def __init__(
        self,
        epochs:     int   = 30,
        lr:         float = 1e-3,
        hidden_dim: int   = 64,
        text_dim:   int   = 768,
        seed:       int   = 42,
        use_aaa:    bool  = False,
    ) -> None:
        self.epochs     = epochs
        self.lr         = lr
        self.hidden_dim = hidden_dim
        self.text_dim   = text_dim
        self.seed       = seed
        self.use_aaa    = use_aaa
        self.model: Optional[THGNNDetector] = None
        self._dataset: Optional[HeteroDataset] = None

    # ------------------------------------------------------------------
    # Variant provenance
    # ------------------------------------------------------------------

    def _log_variant_assertion(self, hgt: HGTEncoder) -> None:
        """Log + assert which attention variant is active, for run provenance.

        Must run every time a model is built so the active variant is
        unambiguous from the training log alone (see Phase 0 issue: prior
        runs could not confirm whether Table 1 used AAA or base HGT
        attention because this line did not exist and use_aaa was silently
        dropped before reaching HGTEncoder).
        """
        lambda_names = sorted(
            n for n, _ in hgt.named_parameters() if n.split(".")[-1] in
            ("lambda_t", "lambda_s", "lambda_g")
        )
        if self.use_aaa:
            assert lambda_names, (
                "use_aaa=True but no lambda_t/lambda_s/lambda_g parameters "
                "found in HGTEncoder — AAA attention was not constructed."
            )
            for n in lambda_names:
                p = dict(hgt.named_parameters())[n]
                assert p.requires_grad, f"{n} is present but requires_grad=False"
            logger.info(
                "THGNNWrapper variant=AAA (Attack-Aware Anomaly Attention, Eq. 10) "
                "— %d learnable lambda params confirmed requires_grad=True: %s",
                len(lambda_names), lambda_names,
            )
        else:
            assert not lambda_names, (
                f"use_aaa=False but found lambda params {lambda_names} — "
                "base HGT attention was not constructed."
            )
            logger.info(
                "THGNNWrapper variant=BASE (standard HGT attention + temporal "
                "bias only, Eq. 5) — no lambda_t/lambda_s/lambda_g params present."
            )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build(self, dataset: HeteroDataset, device: torch.device) -> THGNNDetector:
        input_dims = {
            "user":   dataset.x_user.shape[1],
            "item":   dataset.x_item.shape[1],
            "review": 1,   # dummy — review content lives in BERT embeddings
        }
        hgt = HGTEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=max(1, self.hidden_dim // 32),
            num_layers=2,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            input_dims=input_dims,
            dropout=0.1,
            time_emb_dim=32,
            use_aaa=self.use_aaa,
        )
        self._log_variant_assertion(hgt)
        fusion = MultimodalFusion(
            subgraph_dim=self.hidden_dim,
            text_dim=self.text_dim,
            proj_dim=self.hidden_dim,
            num_heads=max(1, self.hidden_dim // 32),
            dropout=0.1,
        )
        burst = TemporalBurstiness(hidden_dim=32, dropout=0.1)
        return THGNNDetector(hgt, fusion, burst,
                             mlp_hidden=self.hidden_dim // 2,
                             dropout=0.1).to(device)

    # ------------------------------------------------------------------
    # Full-graph forward helpers
    # ------------------------------------------------------------------

    def _full_forward(
        self,
        model: THGNNDetector,
        dataset: HeteroDataset,
        device: torch.device,
    ) -> torch.Tensor:
        """Run one full-graph forward pass; return raw logit scores (num_users,)."""
        review_feat_dim = model.hgt.input_projs["review"].in_features

        node_feats = {
            "user":   dataset.x_user.to(device),
            "item":   dataset.x_item.to(device),
            "review": torch.zeros(dataset.num_reviews, review_feat_dim, device=device),
        }
        edge_index_dict = {k: v.to(device) for k, v in dataset.edge_index_dict.items()}
        edge_attr_dict  = {k: v.float().to(device) for k, v in dataset.edge_attr_dict.items()}
        review_embs     = torch.zeros(dataset.num_reviews, self.text_dim, device=device)

        scores, _ = model(
            node_feats,
            edge_index_dict,
            edge_attr_dict,
            review_embs,
            None,   # item_embs
            dataset.review_user_idx.to(device),
            dataset.review_item_idx.to(device),
            dataset.timestamps.float().to(device),
        )
        return scores   # (num_users,) raw logits

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: HeteroDataset,
        *,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        if dataset.num_users > _MAX_USERS_FULL_GRAPH:
            logger.warning(
                "THGNNWrapper: num_users=%d > %d — skipping full-graph training; "
                "scores will be uniform 0.5.",
                dataset.num_users, _MAX_USERS_FULL_GRAPH,
            )
            self._skipped = True
            return

        self._skipped = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = torch.device(device)

        model = self._build(dataset, dev)
        opt   = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 1e-2,
        )
        labels = dataset.labels.float().to(dev)

        model.train()
        for epoch in range(1, self.epochs + 1):
            opt.zero_grad()
            scores = self._full_forward(model, dataset, dev)
            loss   = _focal_loss(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if epoch == 1 or epoch % 10 == 0:
                logger.debug(
                    "THGNNWrapper epoch %2d/%d  loss=%.4f",
                    epoch, self.epochs, loss.item(),
                )

        self.model    = model
        self._dataset = dataset
        self._device  = dev

    def score_users(self, dataset: HeteroDataset) -> np.ndarray:
        if getattr(self, "_skipped", False) or self.model is None:
            return np.full(dataset.num_users, 0.5, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self._full_forward(self.model, dataset, self._device)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)
