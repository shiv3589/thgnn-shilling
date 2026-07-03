"""Temporal contrastive loss for shilling-attack detection.

TemporalContrastiveLoss
    Pulls together fused user embeddings from THGNNDetector for fake users
    whose first-interaction timestamps fall within a short window of one
    another (temporal positives — likely coordinated members of the same
    injection campaign), and pushes them apart from users whose timestamps
    fall outside that window (temporal negatives, real users preferred as
    hard negatives):

        L_TC = -log[ exp(sim(h_u, h_u+) / tau) / sum_k exp(sim(h_u, h_u-_k) / tau) ]
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss over fused user embeddings.

    Parameters
    ----------
    tau          : initial value of the learnable softmax temperature
    window_hours : positive-pair window W, in hours
    n_negatives  : number of temporal negatives sampled per anchor
    """

    def __init__(
        self,
        tau: float = 0.07,
        window_hours: float = 24.0,
        n_negatives: int = 8,
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600.0
        self.n_negatives = n_negatives

    def forward(self, embeddings: Tensor, timestamps: Tensor, labels: Tensor) -> Tensor:
        """
        Parameters
        ----------
        embeddings : (N, D) fused user embeddings (e.g. THGNNDetector's
                     "user_fused" output)
        timestamps : (N,) first-interaction timestamp per user (seconds)
        labels     : (N,) binary label per user (1 = fake, 0 = real)

        Returns
        -------
        loss : scalar tensor. Falls back to 0.0 if fewer than 2 fake users
               are present in the batch, or if no anchor ends up with both a
               temporal positive and at least one temporal negative.
        """
        device = embeddings.device
        labels = labels.to(device)
        timestamps = timestamps.to(device)

        fake_idx = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)
        if fake_idx.numel() < 2:
            return embeddings.new_zeros(())

        emb_norm = F.normalize(embeddings, dim=-1)
        tau = self.tau.clamp(min=1e-4)

        per_anchor_losses: List[Tensor] = []

        for u in fake_idx.tolist():
            dt = (timestamps - timestamps[u]).abs()   # (N,)

            # ── Positive: temporally closest other fake user within W ──────
            pos_mask = (labels == 1) & (dt <= self.window_seconds)
            pos_mask[u] = False
            if not pos_mask.any():
                continue
            pos_candidates = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
            pos_idx = pos_candidates[dt[pos_candidates].argmin()]

            # ── Negatives: outside the window, real users preferred ────────
            neg_mask = dt > self.window_seconds
            neg_mask[u] = False
            real_neg = torch.nonzero(neg_mask & (labels == 0), as_tuple=False).squeeze(-1)
            fake_neg = torch.nonzero(neg_mask & (labels == 1), as_tuple=False).squeeze(-1)

            if real_neg.numel() >= self.n_negatives:
                perm = torch.randperm(real_neg.numel(), device=device)[: self.n_negatives]
                sel = real_neg[perm]
            else:
                remaining = self.n_negatives - real_neg.numel()
                if fake_neg.numel() == 0:
                    sel = real_neg
                elif fake_neg.numel() >= remaining:
                    perm = torch.randperm(fake_neg.numel(), device=device)[:remaining]
                    sel = torch.cat([real_neg, fake_neg[perm]])
                else:
                    # Not enough distinct out-of-window users — sample with
                    # replacement to fill the negative set.
                    idx = torch.randint(0, fake_neg.numel(), (remaining,), device=device)
                    sel = torch.cat([real_neg, fake_neg[idx]])

            if sel.numel() == 0:
                continue

            sim_pos = emb_norm[u] @ emb_norm[pos_idx]
            sim_neg = emb_norm[sel] @ emb_norm[u]            # (n_sel,)

            numerator = torch.exp(sim_pos / tau)
            denominator = torch.exp(sim_neg / tau).sum()
            per_anchor_losses.append(-torch.log(numerator / denominator + 1e-12))

        if not per_anchor_losses:
            return embeddings.new_zeros(())

        return torch.stack(per_anchor_losses).mean()
