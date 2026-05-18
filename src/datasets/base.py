from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .splits import temporal_split

logger = logging.getLogger(__name__)

EdgeKey = Tuple[str, str, str]

# The three canonical edge types in the heterogeneous graph.
EDGE_TYPES: Tuple[EdgeKey, ...] = (
    ("user", "writes", "review"),
    ("review", "about", "item"),
    ("user", "rates", "item"),
)


@dataclass
class HeteroDataset:
    """Unified container for a heterogeneous rating graph.

    Node types
    ----------
    user, item, review

    Edge types
    ----------
    (user, writes, review)  — one edge per review, user → review node
    (review, about, item)   — one edge per review, review node → item
    (user, rates, item)     — one edge per review, user → item (direct)

    All edge timestamps are float Unix seconds.  The train/val/test masks
    partition *reviews* (and their associated edges) by timestamp: 80/10/10.
    Labels are per-user shilling flags; all zeros for genuine data.

    Attributes
    ----------
    dataset_name : str
    num_users, num_items, num_reviews : int
    edge_index_dict : {EdgeKey: LongTensor(2, E)}
        Source row 0, destination row 1, in the convention of PyG.
    edge_attr_dict : {EdgeKey: FloatTensor(E,)}
        Unix-second timestamps aligned with the corresponding edge_index.
    x_user : FloatTensor(num_users, D_u)
        Base stats: [degree, avg_rating, rating_std, span_days, avg_gap_days]
        plus optional dataset-specific dims appended by the subclass.
    x_item : FloatTensor(num_items, D_i)
        Same base stats plus optional dataset-specific dims.
    review_texts : List[str]  length == num_reviews
    ratings : FloatTensor(num_reviews,)
    timestamps : DoubleTensor(num_reviews,)  Unix seconds
    review_user_idx : LongTensor(num_reviews,)  which user wrote each review
    review_item_idx : LongTensor(num_reviews,)  which item each review covers
    labels : LongTensor(num_users,)  0 = genuine, 1 = shilling
    train_mask, val_mask, test_mask : BoolTensor(num_reviews,)
    user_id_map : Dict  original ID → contiguous index
    item_id_map : Dict  original ID → contiguous index
    """

    dataset_name: str

    num_users: int
    num_items: int
    num_reviews: int

    edge_index_dict: Dict[EdgeKey, torch.Tensor]
    edge_attr_dict: Dict[EdgeKey, torch.Tensor]

    x_user: torch.Tensor
    x_item: torch.Tensor

    review_texts: List[str]
    ratings: torch.Tensor
    timestamps: torch.Tensor

    review_user_idx: torch.Tensor
    review_item_idx: torch.Tensor

    labels: torch.Tensor

    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    user_id_map: Dict
    item_id_map: Dict

    def summary(self) -> str:
        lines = [
            f"HeteroDataset({self.dataset_name})",
            f"  users={self.num_users:,}  items={self.num_items:,}  reviews={self.num_reviews:,}",
            f"  x_user: {tuple(self.x_user.shape)}  x_item: {tuple(self.x_item.shape)}",
            f"  train={self.train_mask.sum():,}  val={self.val_mask.sum():,}  "
            f"test={self.test_mask.sum():,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kcore_filter(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    timestamps: np.ndarray,
    review_texts: List[str],
    min_user: int,
    min_item: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Iteratively remove users/items below the interaction threshold."""
    if min_user <= 1 and min_item <= 1:
        return user_ids, item_ids, ratings, timestamps, review_texts

    prev_n = -1
    while len(user_ids) != prev_n:
        prev_n = len(user_ids)
        u_vals, u_cnts = np.unique(user_ids, return_counts=True)
        i_vals, i_cnts = np.unique(item_ids, return_counts=True)
        valid_u = set(u_vals[u_cnts >= min_user])
        valid_i = set(i_vals[i_cnts >= min_item])
        mask = np.fromiter(
            (u in valid_u and i in valid_i for u, i in zip(user_ids, item_ids)),
            dtype=bool, count=len(user_ids),
        )
        user_ids = user_ids[mask]
        item_ids = item_ids[mask]
        ratings = ratings[mask]
        timestamps = timestamps[mask]
        review_texts = [t for t, m in zip(review_texts, mask) if m]
        logger.debug("k-core iteration: %d interactions remaining", len(user_ids))

    return user_ids, item_ids, ratings, timestamps, review_texts


def _node_stats(
    node_idx: np.ndarray,
    ratings: np.ndarray,
    ts: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Compute per-node stats: [degree, avg_rating, rating_std, span_days, avg_gap_days].

    Uses vectorised numpy for degree/mean/variance and a single sorted pass for
    temporal features.
    """
    feats = np.zeros((n_nodes, 5), dtype=np.float32)

    # degree
    np.add.at(feats[:, 0], node_idx, 1.0)

    # avg_rating via sum then divide
    np.add.at(feats[:, 1], node_idx, ratings)
    safe_deg = np.maximum(feats[:, 0], 1.0)
    feats[:, 1] /= safe_deg

    # rating_std via E[x²] - E[x]² (population std, unbiased when deg>1)
    sq_sum = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(sq_sum, node_idx, ratings ** 2)
    sq_sum /= safe_deg
    feats[:, 2] = np.sqrt(np.maximum(sq_sum - feats[:, 1] ** 2, 0.0))

    # temporal features — sort jointly by (node, time) then slice per node
    order = np.lexsort((ts, node_idx))
    sorted_node = node_idx[order]
    sorted_ts = ts[order]
    starts = np.searchsorted(sorted_node, np.arange(n_nodes), side="left")
    ends = np.searchsorted(sorted_node, np.arange(n_nodes), side="right")

    for n in range(n_nodes):
        s, e = starts[n], ends[n]
        if e - s < 1:
            continue
        tss = sorted_ts[s:e]
        feats[n, 3] = float(tss[-1] - tss[0]) / 86400.0
        if e - s > 1:
            feats[n, 4] = float(np.diff(tss).mean()) / 86400.0

    return feats


# ---------------------------------------------------------------------------
# Base loader
# ---------------------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for heterogeneous graph dataset loaders.

    Subclasses must implement ``_load_raw`` and may optionally override
    ``_extra_user_features`` / ``_extra_item_features`` to append
    dataset-specific feature columns to the base stats vector.
    """

    dataset_name: str = "base"

    def __init__(self, raw_dir: str, processed_dir: Optional[str] = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = (
            Path(processed_dir) if processed_dir else self.raw_dir / "processed"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
    ) -> HeteroDataset:
        """Parse raw files, build the heterogeneous graph, and return a dataset.

        Parameters
        ----------
        min_user_interactions : int
            k-core threshold for users.
        min_item_interactions : int
            k-core threshold for items.
        """
        logger.info("[%s] Reading raw files from %s", self.dataset_name, self.raw_dir)
        user_ids, item_ids, ratings, timestamps, review_texts, user_meta, item_meta = (
            self._load_raw()
        )
        logger.info("[%s] Raw interactions: %d", self.dataset_name, len(ratings))

        user_ids, item_ids, ratings, timestamps, review_texts = _kcore_filter(
            user_ids, item_ids, ratings, timestamps, review_texts,
            min_user_interactions, min_item_interactions,
        )
        logger.info(
            "[%s] After %d-core filter: %d interactions",
            self.dataset_name, min_user_interactions, len(ratings),
        )

        # Contiguous ID mappings
        unique_users = sorted(set(user_ids.tolist()))
        unique_items = sorted(set(item_ids.tolist()))
        user_id_map = {u: i for i, u in enumerate(unique_users)}
        item_id_map = {it: i for i, it in enumerate(unique_items)}
        num_users = len(unique_users)
        num_items = len(unique_items)
        num_reviews = len(ratings)

        u_idx = np.fromiter((user_id_map[u] for u in user_ids), dtype=np.int64, count=num_reviews)
        i_idx = np.fromiter((item_id_map[it] for it in item_ids), dtype=np.int64, count=num_reviews)
        r_idx = np.arange(num_reviews, dtype=np.int64)
        ts = timestamps.astype(np.float64)

        # Graph
        ts_f = torch.from_numpy(ts.astype(np.float32))
        edge_index_dict: Dict[EdgeKey, torch.Tensor] = {
            ("user", "writes", "review"): torch.from_numpy(np.stack([u_idx, r_idx])),
            ("review", "about", "item"):  torch.from_numpy(np.stack([r_idx, i_idx])),
            ("user", "rates", "item"):    torch.from_numpy(np.stack([u_idx, i_idx])),
        }
        edge_attr_dict: Dict[EdgeKey, torch.Tensor] = {k: ts_f for k in edge_index_dict}

        # Node features
        x_user_base = _node_stats(u_idx, ratings.astype(np.float32), ts, num_users)
        x_item_base = _node_stats(i_idx, ratings.astype(np.float32), ts, num_items)

        x_user_extra = self._extra_user_features(user_meta, user_id_map, num_users)
        x_item_extra = self._extra_item_features(item_meta, item_id_map, num_items)

        x_user = torch.from_numpy(
            np.concatenate([x_user_base, x_user_extra], axis=1)
            if x_user_extra is not None else x_user_base
        )
        x_item = torch.from_numpy(
            np.concatenate([x_item_base, x_item_extra], axis=1)
            if x_item_extra is not None else x_item_base
        )

        train_mask, val_mask, test_mask = temporal_split(ts)

        return HeteroDataset(
            dataset_name=self.dataset_name,
            num_users=num_users,
            num_items=num_items,
            num_reviews=num_reviews,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            x_user=x_user,
            x_item=x_item,
            review_texts=list(review_texts),
            ratings=torch.from_numpy(ratings.astype(np.float32)),
            timestamps=torch.from_numpy(ts),
            review_user_idx=torch.from_numpy(u_idx),
            review_item_idx=torch.from_numpy(i_idx),
            labels=torch.zeros(num_users, dtype=torch.long),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            user_id_map=user_id_map,
            item_id_map=item_id_map,
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_raw(self) -> Tuple[
        np.ndarray,   # user_ids  — object dtype (strings or ints)
        np.ndarray,   # item_ids  — object dtype
        np.ndarray,   # ratings   — float32
        np.ndarray,   # timestamps — float64, Unix seconds
        List[str],    # review_texts
        Any,          # user_meta — passed to _extra_user_features
        Any,          # item_meta — passed to _extra_item_features
    ]: ...

    def _extra_user_features(
        self,
        user_meta: Any,
        user_id_map: Dict,
        num_users: int,
    ) -> Optional[np.ndarray]:
        """Return float32 array of shape (num_users, D) or None."""
        return None

    def _extra_item_features(
        self,
        item_meta: Any,
        item_id_map: Dict,
        num_items: int,
    ) -> Optional[np.ndarray]:
        """Return float32 array of shape (num_items, D) or None."""
        return None

    def get_item_texts(self, item_id_map: Dict) -> List[str]:
        """Return a descriptive text string for each item in contiguous index order.

        The returned list has ``len(item_id_map)`` entries where index ``i``
        corresponds to the item whose contiguous index is ``i``.

        Base implementation returns empty strings.  Subclasses override this
        so that ``ItemEncoder`` has meaningful input even when item features are
        purely numeric (e.g. ML-1M) or stored in separate metadata files.
        """
        return [""] * len(item_id_map)
