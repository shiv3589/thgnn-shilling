"""Detection and ranking evaluation metrics.

Detection metrics (binary classification of shilling users)
------------------------------------------------------------
detection_f1        : F1 score at a given threshold
detection_rate      : Recall / true positive rate
false_alarm_rate    : False positive rate (FPR)

Ranking metrics (downstream recommender task)
---------------------------------------------
ndcg_at_k  : Normalised Discounted Cumulative Gain @ K
hr_at_k    : Hit Rate @ K (fraction of positives in top-K)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def detection_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Binary F1 for shilling-user detection.

    Parameters
    ----------
    y_true     : (N,) int array — ground-truth labels {0, 1}
    y_score    : (N,) float array — predicted probability or raw logit
    threshold  : decision boundary (applied to y_score directly)

    Returns
    -------
    F1 score in [0, 1]
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def detection_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Recall / detection rate = TP / (TP + FN).

    Returns 0.0 if there are no positive samples.
    """
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def false_alarm_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """False positive rate = FP / (FP + TN).

    Returns 0.0 if there are no negative samples.
    """
    y_pred = (y_score >= threshold).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def best_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Sweep thresholds and return (best_f1, best_threshold).

    Useful when positive-class prevalence is low.
    """
    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = detection_f1(y_true, y_score, threshold=float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """NDCG@K with binary relevance.

    Parameters
    ----------
    scores : (N,) predicted relevance scores (higher = more relevant)
    labels : (N,) binary ground-truth relevance
    k      : cutoff rank

    Returns
    -------
    NDCG@K in [0, 1]; 0 if no relevant items exist.
    """
    order = np.argsort(scores)[::-1][:k]
    rel   = labels[order].astype(float)

    ideal_rel = np.sort(labels.astype(float))[::-1][:k]

    log2_ranks = np.log2(np.arange(2, len(rel) + 2))
    dcg  = float((rel  / log2_ranks).sum())
    idcg = float((ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def hr_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """Hit Rate@K: fraction of all positive items that appear in the top-K list.

    Equivalently: recall@K when there is at most one positive (standard in
    sampled evaluation), or the proportion of positives retrieved for multiple.

    Returns 0.0 if there are no positive items.
    """
    order = np.argsort(scores)[::-1][:k]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    hits = int(labels[order].sum())
    return float(hits / n_pos)
