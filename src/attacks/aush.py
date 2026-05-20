"""AUSH — Adversarial User Profile Shilling Attack (Chen et al. 2021).

Reference
---------
Chen et al. "Revisiting the Performance of iALS on Item Recommendation
Benchmarks." RecSys 2021.  AUSH is introduced as a stronger baseline attack
that trains a GAN on the real rating matrix so generated profiles are
statistically indistinguishable from genuine users.

Architecture
------------
Generator  (2-layer MLP)
    Input  : concat(noise z ∈ R^{emb_dim}, target indicator t ∈ {0,1}^{num_items})
    Hidden : Linear(emb_dim + num_items → hidden_dim) → ReLU
    Output : Linear(hidden_dim → num_items) → Sigmoid   ∈ [0,1]^{num_items}

Discriminator  (2-layer MLP)
    Input  : normalised rating vector ∈ [0,1]^{num_items}
    Hidden : Linear(num_items → hidden_dim) → LeakyReLU(0.2)
    Output : Linear(hidden_dim → 1) → Sigmoid   ∈ (0,1)

Training objective
------------------
D  : standard binary cross-entropy, label-smoothed (real=0.9, fake=0.1).
G  : L_G = L_adv + λ · L_push
     L_adv  = BCE(D(G(z,t)), 1)          — fool the discriminator
     L_push = −mean(G(z,t)[:, t_idx])    — maximise target-item ratings

Pretraining uses *randomly sampled* target items so the generator learns a
conditional distribution over arbitrary target sets.  The true attack targets
are injected only at generate_profiles() time.

Usage
-----
    attack = AUSHAttack(clean_dataset)
    attack.pretrain(clean_dataset, epochs=50)
    profiles = attack.generate_profiles(n_fake_users=100,
                                        budget_pct=0.1,
                                        target_items=[42, 7, 15])
    augmented = attack.inject(clean_dataset, profiles)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.base import HeteroDataset
from .base import FakeProfile, ShillingAttack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GAN modules
# ---------------------------------------------------------------------------

class _Generator(nn.Module):
    """2-layer MLP: (noise_dim + num_items) → hidden_dim → num_items."""

    def __init__(self, noise_dim: int, num_items: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, target_vec: torch.Tensor) -> torch.Tensor:
        """Returns normalised rating vector in [0, 1]^{num_items}."""
        return self.net(torch.cat([z, target_vec], dim=1))


class _Discriminator(nn.Module):
    """2-layer MLP: num_items → hidden_dim → 1."""

    def __init__(self, num_items: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# AUSHAttack
# ---------------------------------------------------------------------------

class AUSHAttack(ShillingAttack):
    """GAN-based shilling attack following the AUSH paper.

    Parameters
    ----------
    dataset       : clean HeteroDataset used for statistics and GAN training
    embedding_dim : noise vector dimension fed to the generator (default 64)
    hidden_dim    : hidden layer size for both G and D (default 128)
    max_rating    : maximum star rating in the dataset (default 5.0)
    seed          : RNG seed for reproducibility
    """

    injection_rates: List[float] = [0.005, 0.01, 0.05]

    def __init__(
        self,
        dataset: HeteroDataset,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        max_rating: float = 5.0,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, seed)
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.max_rating    = max_rating

        self._generator:     Optional[_Generator]     = None
        self._discriminator: Optional[_Discriminator] = None
        self._pretrained     = False
        self._device         = torch.device("cpu")

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def pretrain(
        self,
        dataset: HeteroDataset,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        lambda_push: float = 1.0,
        n_targets_per_sample: int = 3,
        device: Optional[str] = None,
        max_train_users: Optional[int] = None,
    ) -> None:
        """Train the GAN on real rating profiles from *dataset*.

        The generator is conditioned on a randomly sampled target item indicator
        at every training step, teaching it to produce convincing profiles for
        *any* target set.  Call :meth:`generate_profiles` afterwards.

        Parameters
        ----------
        dataset               : HeteroDataset to train on (typically the clean dataset)
        epochs                : GAN training epochs (default 50)
        batch_size            : users per mini-batch (default 64)
        lr                    : Adam learning rate (default 1e-3)
        lambda_push           : weight of the push loss in L_G (default 1.0)
        n_targets_per_sample  : number of random targets to condition on per step
                                (default 3; approximate injection budget)
        device                : torch device string; auto-detects if None
        max_train_users       : cap number of training users for large datasets
                                (default None = use all)
        """
        # ── Device selection ─────────────────────────────────────────────
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        dev = self._device

        num_items = dataset.num_items
        num_users = dataset.num_users

        if num_items > 20_000:
            logger.warning(
                "num_items=%d is large; the GAN's first linear layer will have "
                "%d parameters.  Consider reducing hidden_dim or setting "
                "max_train_users to limit memory usage.",
                num_items, (self.embedding_dim + num_items) * self.hidden_dim,
            )

        # ── Build sparse user→ratings lookup (no full dense matrix) ──────
        user_rating_dict: List[Dict[int, float]] = [
            {} for _ in range(num_users)
        ]
        for u, item, r in zip(
            dataset.review_user_idx.tolist(),
            dataset.review_item_idx.tolist(),
            dataset.ratings.tolist(),
        ):
            user_rating_dict[u][item] = r / self.max_rating  # normalise to [0,1]

        # ── Optionally subsample training users ───────────────────────────
        all_uids = np.arange(num_users)
        if max_train_users is not None and max_train_users < num_users:
            all_uids = self.rng.choice(all_uids, size=max_train_users, replace=False)
            logger.info(
                "AUSH: subsampling %d / %d users for GAN training.",
                max_train_users, num_users,
            )
        train_uids = all_uids.tolist()
        n_train = len(train_uids)

        # ── Build models ──────────────────────────────────────────────────
        torch_seed = int(self.rng.integers(0, 2**31))
        torch.manual_seed(torch_seed)
        G = _Generator(self.embedding_dim, num_items, self.hidden_dim).to(dev)
        D = _Discriminator(num_items, self.hidden_dim).to(dev)

        # Adam with GAN-recommended betas (Radford et al. 2015)
        opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

        n_targets_clamped = min(n_targets_per_sample, num_items)

        # ── GAN training loop ─────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            G.train()
            D.train()
            d_loss_sum = 0.0
            g_loss_sum = 0.0
            n_batches  = 0

            perm = self.rng.permutation(n_train)

            for start in range(0, n_train, batch_size):
                batch_uids = [train_uids[i] for i in perm[start : start + batch_size]]
                B = len(batch_uids)

                # Dense rating rows for this batch  (B, num_items)
                real = _dense_rows(batch_uids, user_rating_dict, num_items, dev)

                # Random target items — same set for all users in the batch
                t_idx = torch.from_numpy(
                    self.rng.choice(num_items, size=n_targets_clamped, replace=False)
                    .astype(np.int64)
                ).to(dev)
                target_vec = torch.zeros(B, num_items, device=dev)
                target_vec[:, t_idx] = 1.0

                z = torch.randn(B, self.embedding_dim, device=dev)

                # ── Discriminator step ─────────────────────────────────────
                fake_d = G(z, target_vec).detach()   # no gradient through G

                real_lbl = torch.full((B, 1), 0.9, device=dev)  # label smoothing
                fake_lbl = torch.full((B, 1), 0.1, device=dev)

                opt_D.zero_grad()
                loss_D = (
                    F.binary_cross_entropy(D(real),   real_lbl) +
                    F.binary_cross_entropy(D(fake_d), fake_lbl)
                ) * 0.5
                loss_D.backward()
                opt_D.step()

                # ── Generator step ─────────────────────────────────────────
                fake_g = G(z, target_vec)

                loss_adv  = F.binary_cross_entropy(
                    D(fake_g), torch.ones(B, 1, device=dev)
                )
                # Maximise predicted ratings on target items
                loss_push = -fake_g[:, t_idx].mean()
                loss_G    = loss_adv + lambda_push * loss_push

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                d_loss_sum += loss_D.item()
                g_loss_sum += loss_G.item()
                n_batches  += 1

            if epoch == 1 or epoch % 10 == 0:
                logger.info(
                    "AUSH pretrain  epoch %3d/%d  "
                    "loss_D=%.4f  loss_G=%.4f",
                    epoch, epochs,
                    d_loss_sum / max(n_batches, 1),
                    g_loss_sum / max(n_batches, 1),
                )

        self._generator     = G
        self._discriminator = D
        self._pretrained    = True
        logger.info("AUSH pretraining complete (%d epochs).", epochs)

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        n_fake_users: int,
        budget_pct: float,
        target_items: List[int],
    ) -> List[FakeProfile]:
        """Sample *n_fake_users* fake profiles from the trained generator.

        The generator is conditioned on *target_items*.  Filler items are the
        ``filler_size = round(budget_pct * num_items) − len(target_items)``
        non-target items with the highest predicted (normalised) rating.

        Review texts are set to empty strings because AUSH operates on rating
        vectors only.  The inherited ``inject()`` method will store these
        empty strings in the dataset's ``review_texts`` list; downstream BERT
        encoders treat empty strings as ``"[PAD]"``.

        Raises
        ------
        RuntimeError
            If called before :meth:`pretrain`.
        """
        if not self._pretrained or self._generator is None:
            raise RuntimeError(
                "Generator not trained.  Call pretrain(dataset) before "
                "generate_profiles()."
            )

        num_items   = self.dataset.num_items
        target_set  = set(target_items)
        filler_size = max(0, round(budget_pct * num_items) - len(target_items))

        # Condition the generator on the actual attack targets
        target_vec = torch.zeros(n_fake_users, num_items, device=self._device)
        for t in target_items:
            target_vec[:, t] = 1.0

        z = torch.randn(n_fake_users, self.embedding_dim, device=self._device)

        self._generator.eval()
        with torch.no_grad():
            # Shape: (n_fake_users, num_items), values in [0, 1]
            pred_norm = self._generator(z, target_vec).cpu().numpy()

        # Non-target item indices available for filler selection (constant)
        non_targets = [i for i in range(num_items) if i not in target_set]

        base_ts  = float(self.dataset.timestamps.max().item())
        profiles: List[FakeProfile] = []

        for uid in range(n_fake_users):
            scores = pred_norm[uid]   # (num_items,) normalised ratings

            # Filler: top-filler_size non-target items by predicted score
            if filler_size > 0 and non_targets:
                # argpartition is O(n) to get top-k indices
                if filler_size < len(non_targets):
                    nt_arr = np.array(non_targets, dtype=np.int64)
                    nt_scores = scores[nt_arr]
                    # top-k via partition (avoids full sort for large item spaces)
                    top_k_local = np.argpartition(nt_scores, -filler_size)[-filler_size:]
                    filler_idx = nt_arr[top_k_local].tolist()
                else:
                    filler_idx = list(non_targets)
            else:
                filler_idx = []

            rated: Dict[int, float] = {}
            for item in filler_idx:
                # Denormalise and round to nearest 0.5 star
                r = float(np.clip(scores[item] * self.max_rating,
                                  1.0, self.max_rating))
                rated[item] = round(r * 2) / 2
            for item in target_items:
                rated[item] = self.max_rating

            # AUSH is rating-only: texts are empty strings
            texts = [""] * len(rated)

            user_base = base_ts + uid * 1_800.0   # stagger profiles by 30 min
            tss = self._spread_timestamps(len(rated), user_base)

            profiles.append(FakeProfile(
                user_id=f"aush_{uid}",
                rated_items=rated,
                review_texts=texts,
                timestamps=tss,
            ))

        return profiles


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _dense_rows(
    user_ids: List[int],
    user_rating_dict: List[Dict[int, float]],
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense (B, num_items) normalised rating tensor for a user batch.

    Unrated items stay at 0.  Constructed on CPU then moved to *device* to
    avoid repeated host-device transfers inside the Python loop.
    """
    B = len(user_ids)
    rows = torch.zeros(B, num_items)
    for i, uid in enumerate(user_ids):
        for item, r in user_rating_dict[uid].items():
            rows[i, item] = r
    return rows.to(device)
