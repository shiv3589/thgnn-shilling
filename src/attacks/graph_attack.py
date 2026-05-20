"""GraphAttack — gradient-based attack crafted against the LightGCN recommender.

The attack treats the fake user's rating vector r ∈ [1, max_rating]^{num_items}
as a continuous parameter and optimises it via projected gradient ascent to
maximise the predicted rank of target items in the surrogate LightGCN.

Differentiable surrogate
------------------------
LightGCN's adjacency is binary (presence/absence of an interaction), so we
cannot directly differentiate through it.  Instead we:

  1. Run one forward pass on the *clean* graph to obtain frozen item embeddings
     E ∈ R^{num_items × d} from the surrogate model.

  2. Map the continuous rating vector r to soft interaction weights:
         w_i = (r_i − 1) / (max_rating − 1)    ∈ [0, 1]

  3. Approximate the fake user's embedding as the normalised weighted mean of
     interacted item embeddings (LightGCN 1-hop message-passing approximation):
         e_fake = (E^T w) / (‖w‖₁ + ε)

  4. Compute item scores:  s = E · e_fake    ∈ R^{num_items}

  5. Use a differentiable NDCG approximation:
         softRank_t ≈ 1 + Σ_{j ∉ targets} σ((s_j − s_t) / τ)
         approxNDCG = Σ_t  1 / log₂(softRank_t + 1)  /  IDCG

     Minimising −approxNDCG pushes target items toward rank 1.

Optimisation loop
-----------------
  * Initialise R ~ Uniform(1, max_rating) per fake user
  * n_steps of Adam + PGD (r.clamp_(1, max_rating) after every step)
  * Batch over all n_fake_users simultaneously for efficiency

Profile selection
-----------------
After optimisation the ``filler_size`` non-target items with the highest r
values are selected; their r values are discretised to the nearest 0.5 star.
Target items are always included at max_rating.

Note
----
This attack is crafted against a *surrogate* LightGCN and has no knowledge of
the THGNNDetector — it is a strong black-box threat from the detector's perspective.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import HeteroDataset
from src.models.lightgcn import LightGCN, build_norm_adj
from .base import FakeProfile, ShillingAttack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable NDCG surrogate
# ---------------------------------------------------------------------------

def _soft_ndcg_loss(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    non_target_idx: torch.Tensor,
    ideal_dcg: float,
    temp: float,
) -> torch.Tensor:
    """Differentiable −NDCG approximation for a batch of fake users.

    Uses a sigmoid-based soft-rank estimator (Qin et al., 2010):
        softRank_t ≈ 1 + Σ_{j ∉ targets} σ((s_j − s_t) / τ)

    Parameters
    ----------
    scores       : (n_fake, num_items) — predicted scores for all items
    target_idx   : (K_t,) LongTensor  — indices of target items
    non_target_idx : (N_nt,) LongTensor — all non-target item indices
    ideal_dcg    : float — IDCG (max achievable DCG for K_t targets)
    temp         : float — softmax temperature τ; smaller → sharper rank estimate

    Returns
    -------
    Scalar: mean negative approxNDCG over the batch
    """
    target_s     = scores[:, target_idx]      # (n_fake, K_t)
    non_target_s = scores[:, non_target_idx]  # (n_fake, N_nt)

    # gap[n, k, j] = s_j − s_t[k]  (positive → j ranked above target k)
    # (n_fake, 1, N_nt) − (n_fake, K_t, 1) → (n_fake, K_t, N_nt)
    gap        = non_target_s.unsqueeze(1) - target_s.unsqueeze(2)
    soft_above = torch.sigmoid(gap / temp).sum(dim=2)   # (n_fake, K_t)
    soft_rank  = soft_above + 1.0                        # (n_fake, K_t)

    discount = 1.0 / torch.log2(soft_rank + 1.0)        # (n_fake, K_t)
    approx_ndcg = discount.sum(dim=1) / max(ideal_dcg, 1e-8)  # (n_fake,)
    return -approx_ndcg.mean()


# ---------------------------------------------------------------------------
# GraphAttack
# ---------------------------------------------------------------------------

class GraphAttack(ShillingAttack):
    """Gradient-based white-box attack against a surrogate LightGCN.

    Parameters
    ----------
    surrogate_lightgcn : LightGCN  — trained model (weights are frozen during attack)
    dataset            : HeteroDataset  — clean reference dataset
    n_steps            : int   — projected gradient ascent steps (default 30)
    lr                 : float — inner optimiser learning rate (default 0.1)
    temp               : float — soft-rank temperature τ (default 0.5);
                         smaller = sharper rank estimate but weaker gradients
    max_rating         : float — upper bound of the rating scale (default 5.0)
    seed               : int   — RNG seed for rating initialisation
    """

    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(
        self,
        surrogate_lightgcn: LightGCN,
        dataset: HeteroDataset,
        n_steps: int = 30,
        lr: float = 0.1,
        temp: float = 0.5,
        max_rating: float = 5.0,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, seed)
        self.surrogate  = surrogate_lightgcn
        self.n_steps    = n_steps
        self.attack_lr  = lr
        self.temp       = temp
        self.max_rating = max_rating

    # ------------------------------------------------------------------
    # ShillingAttack interface
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Craft *n_fake_users* adversarial profiles via projected gradient ascent.

        Parameters
        ----------
        n_fake_users : int   — number of fake profiles to generate
        budget_pct   : float — fraction of the item catalog rated per fake user
        target_items : List[int] — global item indices to promote

        Returns
        -------
        List[FakeProfile]
        """
        dataset   = self.dataset
        num_items = dataset.num_items
        target_set  = set(target_items)
        filler_size = max(0, round(budget_pct * num_items) - len(target_items))

        dev = next(self.surrogate.parameters()).device

        # ── Step 1: frozen item embeddings from surrogate ─────────────────
        E = self._frozen_item_embs(dev)                 # (num_items, embed_dim)

        # ── Step 2: precompute target / non-target index tensors ──────────
        target_t     = torch.tensor(target_items, dtype=torch.long, device=dev)
        non_target_mask = torch.ones(num_items, dtype=torch.bool, device=dev)
        non_target_mask[target_t] = False
        non_target_idx = torch.where(non_target_mask)[0]    # (N_nt,)

        K_t       = len(target_items)
        ideal_dcg = float(sum(1.0 / np.log2(r + 2) for r in range(K_t)))

        # ── Step 3: initialise R ∈ [1, max_rating]^{n_fake × num_items} ──
        torch_seed = int(self.rng.integers(2**31))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(torch_seed)
        R = (torch.rand(n_fake_users, num_items, generator=gen)
             * (self.max_rating - 1.0) + 1.0).to(dev)
        R[:, target_t] = self.max_rating         # bias targets high from the start
        R.requires_grad_(True)

        # ── Step 4: projected gradient ascent ─────────────────────────────
        optimizer = torch.optim.Adam([R], lr=self.attack_lr)

        for step in range(self.n_steps):
            # Projection (also done after the step; this handles the first iter)
            R_proj = R.clamp(1.0, self.max_rating)

            # Soft interaction weights: linear mapping [1, max_rating] → [0, 1]
            w = (R_proj - 1.0) / (self.max_rating - 1.0)   # (n_fake, num_items)

            # Fake user embedding: normalised weighted mean of item embeddings
            w_sum  = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            E_fake = (w / w_sum) @ E                         # (n_fake, embed_dim)

            # Scores for every item: (n_fake, num_items)
            scores = E_fake @ E.t()

            loss = _soft_ndcg_loss(
                scores, target_t, non_target_idx, ideal_dcg, self.temp
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # PGD: project back to valid rating range
            with torch.no_grad():
                R.data.clamp_(1.0, self.max_rating)

            if step == 0 or (step + 1) % 10 == 0:
                logger.debug(
                    "GraphAttack step %2d/%d  loss=%.4f",
                    step + 1, self.n_steps, loss.item(),
                )

        # ── Step 5: build profiles ────────────────────────────────────────
        R_final = R.detach()
        R_final[:, target_t] = self.max_rating     # guarantee max on targets

        base_ts  = float(dataset.timestamps.max().item())
        profiles: List[FakeProfile] = []

        for uid in range(n_fake_users):
            r_uid = R_final[uid]                    # (num_items,)

            if filler_size > 0 and non_target_idx.numel() > 0:
                non_target_r = r_uid[non_target_idx]
                k = min(filler_size, non_target_idx.numel())
                top_local = torch.topk(non_target_r, k=k).indices  # local indices
                filler_items = non_target_idx[top_local].cpu().tolist()
            else:
                filler_items = []

            rated: Dict[int, float] = {}
            for item in filler_items:
                r_val = float(r_uid[item].item())
                rated[item] = _discretise(r_val, self.max_rating)
            for item in target_items:
                rated[item] = self.max_rating

            texts = [self._placeholder_text(it, r) for it, r in rated.items()]
            user_base = base_ts + uid * 1_800.0
            tss       = self._spread_timestamps(len(rated), user_base)

            profiles.append(FakeProfile(
                user_id=f"graph_{uid}",
                rated_items=rated,
                review_texts=texts,
                timestamps=tss,
            ))

        return profiles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frozen_item_embs(self, device: torch.device) -> torch.Tensor:
        """Return detached item embeddings from one clean-graph forward pass."""
        ds = self.dataset
        train_mask = ds.train_mask
        u_np = ds.review_user_idx[train_mask].numpy()
        i_np = ds.review_item_idx[train_mask].numpy()
        adj  = build_norm_adj(u_np, i_np, ds.num_users, ds.num_items, device)

        self.surrogate.eval()
        with torch.no_grad():
            _, E = self.surrogate(adj)          # (num_items, embed_dim)
        return E.detach()

    @staticmethod
    def loss_trace(
        surrogate_lightgcn: LightGCN,
        dataset: HeteroDataset,
        target_items: List[int],
        n_steps: int = 30,
        lr: float = 0.1,
        temp: float = 0.5,
        max_rating: float = 5.0,
        seed: int = 0,
    ) -> List[float]:
        """Return the per-step loss trace for a single fake user.

        Useful for debugging/testing gradient flow without building a full profile.

        Returns
        -------
        List[float] of length n_steps
        """
        attack = GraphAttack(
            surrogate_lightgcn, dataset,
            n_steps=n_steps, lr=lr, temp=temp,
            max_rating=max_rating, seed=seed,
        )
        dev = next(surrogate_lightgcn.parameters()).device
        E   = attack._frozen_item_embs(dev)

        target_t   = torch.tensor(target_items, dtype=torch.long, device=dev)
        nt_mask    = torch.ones(dataset.num_items, dtype=torch.bool, device=dev)
        nt_mask[target_t] = False
        non_target_idx = torch.where(nt_mask)[0]
        K_t        = len(target_items)
        ideal_dcg  = float(sum(1.0 / np.log2(r + 2) for r in range(K_t)))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        R = (torch.rand(1, dataset.num_items, generator=gen)
             * (max_rating - 1.0) + 1.0).to(dev)
        R[:, target_t] = max_rating
        R.requires_grad_(True)
        opt = torch.optim.Adam([R], lr=lr)

        trace: List[float] = []
        for _ in range(n_steps):
            R_proj = R.clamp(1.0, max_rating)
            w      = (R_proj - 1.0) / (max_rating - 1.0)
            w_sum  = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            E_fake = (w / w_sum) @ E
            scores = E_fake @ E.t()
            loss   = _soft_ndcg_loss(scores, target_t, non_target_idx,
                                     ideal_dcg, temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                R.data.clamp_(1.0, max_rating)
            trace.append(loss.item())

        return trace


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _discretise(r: float, max_rating: float) -> float:
    """Round *r* to the nearest 0.5 star, clipped to [1, max_rating]."""
    clipped = float(np.clip(r, 1.0, max_rating))
    return round(clipped * 2) / 2
