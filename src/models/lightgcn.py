"""LightGCN recommender model (He et al., SIGIR 2020).

Reference
---------
He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation." SIGIR 2020.  https://arxiv.org/abs/2002.02126

Design notes
------------
* The normalised adjacency is built from *training edges only* and injected
  into forward() at call time, so the same model object can be re-evaluated
  on a modified graph (e.g. with injected attack profiles) without reloading.
* Evaluation uses 999 randomly sampled negatives per positive (sampled-ranking
  protocol, standard in recommendation papers).  Metrics are therefore
  higher than full-ranking equivalents; compare only within the same protocol.
* MPS (Apple Silicon) falls back to CPU for sparse matmul (limited MPS sparse
  support in PyTorch ≤ 2.3); dense ops remain on MPS.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.base import HeteroDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_norm_adj(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the D^{-½} A D^{-½} sparse tensor for the bipartite graph.

    The matrix has shape (num_users + num_items) × (num_users + num_items)
    and represents the symmetric bipartite adjacency::

        A = [ 0,   R  ]
            [ R^T, 0  ]

    Duplicate interactions are deduplicated (one unweighted edge per pair).

    Parameters
    ----------
    user_idx, item_idx : 1-D int64 arrays of aligned interaction pairs
    num_users, num_items : int
    device : torch.device
    """
    N = num_users + num_items
    shifted = item_idx + num_users  # place items in the lower block

    # Both directions of the bipartite edge
    row = np.concatenate([user_idx, shifted], dtype=np.int64)
    col = np.concatenate([shifted, user_idx], dtype=np.int64)

    # Deduplicate (a user may rate the same item multiple times)
    edges = np.unique(np.stack([row, col], axis=1), axis=0)
    row, col = edges[:, 0], edges[:, 1]

    # Symmetric normalisation: d_i^{-½} · d_j^{-½}
    deg = np.bincount(row, minlength=N).astype(np.float32)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    vals = (d_inv_sqrt[row] * d_inv_sqrt[col]).astype(np.float32)

    indices = torch.from_numpy(np.stack([row, col]))
    values  = torch.from_numpy(vals)
    adj = torch.sparse_coo_tensor(indices, values, (N, N), dtype=torch.float32)
    return adj.coalesce().to(device)


def _spmm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse × dense matmul with CPU fallback for MPS."""
    if x.device.type == "mps":
        return torch.sparse.mm(adj.cpu(), x.cpu()).to(x.device)
    return torch.sparse.mm(adj, x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LightGCN(nn.Module):
    """LightGCN with mean-pooled layer embeddings.

    Parameters
    ----------
    num_users, num_items : int
    embed_dim : int    — embedding dimension (default: 64)
    n_layers  : int    — propagation depth   (default: 3)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers  = n_layers

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    # ------------------------------------------------------------------

    def forward(
        self, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate and return mean-pooled user / item embeddings.

        Parameters
        ----------
        adj : sparse FloatTensor, shape (num_users+num_items, num_users+num_items)
            Symmetrically normalised bipartite adjacency (see ``build_norm_adj``).

        Returns
        -------
        user_emb : FloatTensor (num_users, embed_dim)
        item_emb : FloatTensor (num_items, embed_dim)
        """
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layers = [ego]
        for _ in range(self.n_layers):
            ego = _spmm(adj, ego)
            layers.append(ego)

        # E = (1 / K+1) · Σ_k E^(k)   (He et al., Eq. 11)
        out = torch.stack(layers, dim=1).mean(dim=1)
        return out[: self.num_users], out[self.num_users :]

    # ------------------------------------------------------------------

    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """Bayesian Personalised Ranking loss."""
        u   = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]
        return -torch.log(
            torch.sigmoid((u * pos).sum(-1) - (u * neg).sum(-1)) + 1e-8
        ).mean()

    def reg_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """L2 regularisation on raw (layer-0) embeddings of sampled nodes."""
        nrm = (
            self.user_emb.weight[users].norm(2).pow(2)
            + self.item_emb.weight[pos_items].norm(2).pow(2)
            + self.item_emb.weight[neg_items].norm(2).pow(2)
        )
        return weight * nrm / (2 * len(users))


# ---------------------------------------------------------------------------
# BPR sampler
# ---------------------------------------------------------------------------

class _BPRSampler:
    """In-memory BPR triplet sampler with rejection-based negative sampling.

    For users with very dense positive sets the rejection loop is capped at
    ``max_tries`` iterations; the last drawn item is accepted unconditionally
    so sampling never blocks.
    """

    def __init__(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        num_items: int,
        max_tries: int = 30,
    ) -> None:
        self.num_items = num_items
        self.max_tries = max_tries

        pos: Dict[int, List[int]] = defaultdict(list)
        for u, i in zip(user_idx.tolist(), item_idx.tolist()):
            pos[u].append(i)

        self.user_pos: Dict[int, np.ndarray] = {u: np.array(v) for u, v in pos.items()}
        self.user_pos_set: Dict[int, set]    = {u: set(v) for u, v in pos.items()}
        self.users_arr = np.array(list(self.user_pos.keys()), dtype=np.int64)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users     = np.random.choice(self.users_arr, size=n, replace=True)
        pos_items = np.array([np.random.choice(self.user_pos[u]) for u in users], dtype=np.int64)
        neg_items = np.empty(n, dtype=np.int64)
        for k, u in enumerate(users):
            pos_set = self.user_pos_set[u]
            j = int(np.random.randint(self.num_items))
            for _ in range(self.max_tries - 1):
                if j not in pos_set:
                    break
                j = int(np.random.randint(self.num_items))
            neg_items[k] = j
        return users, pos_items, neg_items


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: LightGCN,
    adj: torch.Tensor,
    eval_user: np.ndarray,
    eval_item: np.ndarray,
    exclude_per_user: Dict[int, np.ndarray],
    num_items: int,
    k: int = 10,
    n_neg: int = 999,
    desc: str = "eval",
) -> Dict[str, float]:
    """Sampled-ranking evaluation: rank positives against ``n_neg`` negatives.

    Parameters
    ----------
    eval_user, eval_item : aligned positive (user, item) arrays
    exclude_per_user     : items to exclude from negative sampling, keyed by user
    n_neg                : number of sampled negatives per user (default 999)

    Returns
    -------
    {"hr@k": float, "ndcg@k": float}
    """
    model.eval()
    user_emb, item_emb = model(adj)
    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()

    # Group positives by user
    user_to_pos: Dict[int, List[int]] = defaultdict(list)
    for u, i in zip(eval_user.tolist(), eval_item.tolist()):
        user_to_pos[u].append(i)

    all_items = np.arange(num_items, dtype=np.int64)
    hr_list: List[float]   = []
    ndcg_list: List[float] = []

    for user, pos_items in tqdm(user_to_pos.items(), desc=desc, leave=False, dynamic_ncols=True):
        n_pos   = len(pos_items)
        excl    = set(pos_items) | set(exclude_per_user.get(user, np.array([], dtype=np.int64)).tolist())
        neg_pool = np.setdiff1d(all_items, np.fromiter(excl, dtype=np.int64), assume_unique=False)

        n_sample = min(n_neg, len(neg_pool))
        neg_items = (
            np.random.choice(neg_pool, size=n_sample, replace=False)
            if n_sample > 0 else neg_pool
        )

        # Candidates: positives first, then negatives
        candidates = np.concatenate([np.array(pos_items, dtype=np.int64), neg_items])
        scores = item_emb[candidates] @ user_emb[user]      # (C,)
        top_k  = np.argsort(-scores.numpy())[:k]             # indices into candidates

        # Hit = candidate index < n_pos (those are the positives)
        hit_ranks = [r for r, ci in enumerate(top_k) if ci < n_pos]

        hr_list.append(float(len(hit_ranks) > 0))

        if hit_ranks:
            dcg  = sum(1.0 / np.log2(r + 2) for r in hit_ranks)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(n_pos, k)))
            ndcg_list.append(dcg / idcg)
        else:
            ndcg_list.append(0.0)

    return {
        f"hr@{k}":   float(np.mean(hr_list))   if hr_list else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_lightgcn(
    dataset: HeteroDataset,
    epochs: int = 100,
    lr: float = 1e-3,
    embed_dim: int = 64,
    n_layers: int = 3,
    batch_size: int = 2048,
    reg_weight: float = 1e-4,
    eval_every: int = 5,
    k: int = 10,
    n_neg_eval: int = 999,
    patience: int = 10,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """Train LightGCN on a HeteroDataset.

    Uses training edges for graph propagation, validates on val_mask, and
    reports test metrics from the best-validation checkpoint.

    Parameters
    ----------
    dataset      : HeteroDataset
    epochs       : int     — maximum training epochs
    lr           : float   — Adam learning rate
    embed_dim    : int     — embedding dimension (default 64)
    n_layers     : int     — LightGCN propagation layers (default 3)
    batch_size   : int     — BPR triplets per gradient step
    reg_weight   : float   — L2 regularisation weight λ (default 1e-4)
    eval_every   : int     — validate every N epochs
    k            : int     — cutoff for HR@k / NDCG@k
    n_neg_eval   : int     — sampled negatives per positive during evaluation
    patience     : int     — early-stopping patience in evaluation rounds
    device       : str     — torch device (default: auto-detect)
    seed         : int

    Returns
    -------
    dict with keys
        ``ndcg@{k}``, ``hr@{k}``           — test-set metrics (best val ckpt)
        ``val_ndcg@{k}``, ``val_hr@{k}``   — validation metrics at best ckpt
        ``best_epoch``                       — epoch of best val NDCG
        ``train_loss``                       — list of per-epoch mean BPR losses
        ``train_time_secs``                  — wall-clock seconds for training
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ────────────────────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    logger.info(
        "[%s] LightGCN  device=%s  users=%d  items=%d",
        dataset.dataset_name, device, dataset.num_users, dataset.num_items,
    )

    # ── Interaction splits ────────────────────────────────────────────────
    train_u = dataset.review_user_idx[dataset.train_mask].numpy()
    train_i = dataset.review_item_idx[dataset.train_mask].numpy()
    val_u   = dataset.review_user_idx[dataset.val_mask].numpy()
    val_i   = dataset.review_item_idx[dataset.val_mask].numpy()
    test_u  = dataset.review_user_idx[dataset.test_mask].numpy()
    test_i  = dataset.review_item_idx[dataset.test_mask].numpy()
    logger.info(
        "[%s] train=%d  val=%d  test=%d",
        dataset.dataset_name, len(train_u), len(val_u), len(test_u),
    )

    # ── Normalised adjacency (training edges only) ────────────────────────
    adj = build_norm_adj(train_u, train_i, dataset.num_users, dataset.num_items, dev)

    # ── Positive sets for negative sampling and evaluation ────────────────
    # train_pos: used for BPR negatives + val evaluation exclusion
    train_pos: Dict[int, np.ndarray] = {}
    _tp: Dict[int, List[int]] = defaultdict(list)
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        _tp[u].append(i)
    train_pos = {u: np.array(v, dtype=np.int64) for u, v in _tp.items()}

    # train_val_pos: excludes both train and val items from test negatives
    _tvp: Dict[int, List[int]] = {u: v.tolist() for u, v in train_pos.items()}
    for u, i in zip(val_u.tolist(), val_i.tolist()):
        _tvp.setdefault(u, []).append(i)
    train_val_pos = {u: np.array(v, dtype=np.int64) for u, v in _tvp.items()}

    # ── Model + optimiser ─────────────────────────────────────────────────
    model   = LightGCN(dataset.num_users, dataset.num_items, embed_dim, n_layers).to(dev)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = _BPRSampler(train_u, train_i, dataset.num_items)

    steps_per_epoch = max(1, len(train_u) // batch_size)

    best_val_ndcg: float = -1.0
    best_state: Optional[Dict] = None
    best_epoch: int = 0
    no_improve: int = 0
    epoch_losses: List[float] = []

    t0 = time.perf_counter()

    pbar = tqdm(range(1, epochs + 1), desc=f"LightGCN/{dataset.dataset_name}",
                unit="ep", dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        running = 0.0

        for _ in range(steps_per_epoch):
            u_np, pos_np, neg_np = sampler.sample(batch_size)
            u_t   = torch.from_numpy(u_np).long().to(dev)
            pos_t = torch.from_numpy(pos_np).long().to(dev)
            neg_t = torch.from_numpy(neg_np).long().to(dev)

            user_emb, item_emb = model(adj)
            loss = (
                model.bpr_loss(user_emb, item_emb, u_t, pos_t, neg_t)
                + model.reg_loss(u_t, pos_t, neg_t, reg_weight)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        mean_loss = running / steps_per_epoch
        epoch_losses.append(mean_loss)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == epochs:
            val_m = _evaluate(
                model, adj, val_u, val_i, train_pos,
                dataset.num_items, k=k, n_neg=n_neg_eval, desc="val",
            )
            val_ndcg = val_m[f"ndcg@{k}"]

            pbar.set_postfix({
                "loss":         f"{mean_loss:.4f}",
                f"val_ndcg@{k}": f"{val_ndcg:.4f}",
            })

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_state = {nm: v.cpu().clone() for nm, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "[%s] early stop @ epoch %d (patience=%d)",
                        dataset.dataset_name, epoch, patience,
                    )
                    break
        else:
            pbar.set_postfix({"loss": f"{mean_loss:.4f}"})

    train_time = time.perf_counter() - t0

    # ── Test evaluation from best checkpoint ──────────────────────────────
    if best_state is not None:
        model.load_state_dict({nm: v.to(dev) for nm, v in best_state.items()})

    test_m = _evaluate(
        model, adj, test_u, test_i, train_val_pos,
        dataset.num_items, k=k, n_neg=n_neg_eval, desc="test",
    )
    val_m_best = _evaluate(
        model, adj, val_u, val_i, train_pos,
        dataset.num_items, k=k, n_neg=n_neg_eval, desc="val-final",
    )

    results: Dict = {
        **test_m,                                           # ndcg@k, hr@k
        **{f"val_{key}": val for key, val in val_m_best.items()},
        "best_epoch":      best_epoch,
        "train_loss":      epoch_losses,
        "train_time_secs": round(train_time, 2),
    }
    logger.info(
        "[%s] done  best_epoch=%d  ndcg@%d=%.4f  hr@%d=%.4f  (%.0f s)",
        dataset.dataset_name, best_epoch,
        k, test_m[f"ndcg@{k}"], k, test_m[f"hr@{k}"], train_time,
    )
    return results
