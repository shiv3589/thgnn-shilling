#!/usr/bin/env python
"""Robustness experiment: TH-GNN under missing/noisy modalities at inference time.

Each detector is trained once on the full Amazon-Books graph poisoned with
Agent4SR, then scored on copies of that graph with one modality degraded:

  1. missing_text      — zero review_texts for frac of randomly chosen users
                         (SemanticShield and Anti-FakeU lose TF-IDF / text-len
                          features; TH-GNN is unaffected because review_embs
                          are derived from temporal+graph signals in the wrapper)
  2. noisy_timestamps  — add N(0, σ) to all edge timestamps,
                         where σ = std(timestamps) × noise_fraction
  3. missing_edges     — randomly remove frac of edges (reviews) from the graph,
                         rebuilding edge_index_dict and review-level tensors

Each condition is tested at levels 0.2 and 0.5; a baseline (level 0.0) is also
recorded so that ΔF1 and ΔFAR can be reported.

Grid
----
  Dataset       : amazon_books
  Attack        : agent4sr
  Injection rate: 0.01
  Detectors     : thgnn, semantic_shield_baseline, anti_fakeu_baseline
  Seeds         : 42, 123, 456
  Conditions    : baseline | missing_text×{0.2,0.5}
                             noisy_timestamps×{0.2,0.5}
                             missing_edges×{0.2,0.5}

Training is shared per (detector, seed): 9 model fits, 54 degraded evaluations.

Output
------
  results/raw/robustness_{dataset}_{attack}_{condition}_{level}_{detector}_{seed}.json
  results/tables/robustness.csv
  results/figures/robustness_curves.pdf

Usage
-----
    python scripts/run_robustness.py --dry-run          # 1 seed, thgnn only
    python scripts/run_robustness.py --workers 3        # full run, 3 workers
    python scripts/run_robustness.py --aggregate-only   # re-generate CSV + figure
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import Agent4SRAttack, ShillingAttack
from src.datasets import AmazonBooksLoader, HeteroDataset
from src.detectors import AntiFakeUBaseline, SemanticShieldBaseline, THGNNWrapper
from src.evaluation import best_threshold_f1, detection_rate, false_alarm_rate
from src.models.lightgcn import LightGCN, _BPRSampler, build_norm_adj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(processName)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

DATASET   = "amazon_books"
ATTACK    = "agent4sr"
RATE      = 0.01
N_TARGETS = 5
BUDGET_PCT = 0.10

_PAPER_RUNS_YAML = Path(__file__).resolve().parent.parent / "configs" / "paper_runs.yaml"


def _load_use_aaa() -> bool:
    """Read thgnn.use_aaa from configs/paper_runs.yaml.

    Same gap as the other three entrypoints (54676ed, 1569877, eca7aa1):
    _build_detector() built THGNNWrapper(seed=seed) with no use_aaa. Purely
    preventative — scripts/gen_robustness_placeholder.py's own docstring
    confirms this script "has not been run" yet, so no real number anywhere
    is affected. Fixing before it ever produces one.
    """
    try:
        with open(_PAPER_RUNS_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return bool(cfg.get("thgnn", {}).get("use_aaa", False))
    except FileNotFoundError:
        logger.warning("configs/paper_runs.yaml not found — defaulting use_aaa=False.")
        return False


DETECTORS: List[str] = [
    "thgnn",
    "semantic_shield_baseline",
    "anti_fakeu_baseline",
]
DETECTOR_LABELS = {
    "thgnn":                    "TH-GNN",
    "semantic_shield_baseline": "SemanticShield",
    "anti_fakeu_baseline":      "Anti-FakeU",
}

SEEDS: List[int] = [42, 123, 456]

# (condition_name, degradation_level)
CONDITIONS: List[Tuple[str, float]] = [
    ("baseline",         0.0),
    ("missing_text",     0.2),
    ("missing_text",     0.5),
    ("noisy_timestamps", 0.2),
    ("noisy_timestamps", 0.5),
    ("missing_edges",    0.2),
    ("missing_edges",    0.5),
]
CONDITION_TYPES = ["missing_text", "noisy_timestamps", "missing_edges"]
LEVELS = [0.0, 0.2, 0.5]

CONDITION_LABELS = {
    "missing_text":     "Missing Text",
    "noisy_timestamps": "Noisy Timestamps",
    "missing_edges":    "Missing Edges",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RobustnessRun:
    detector: str
    seed:     int
    dataset:  str   = DATASET
    attack:   str   = ATTACK
    rate:     float = RATE


def _result_path(run: RobustnessRun, condition: str, level: float, results_dir: str) -> Path:
    level_str = f"{level:.1f}"
    return (
        Path(results_dir)
        / f"robustness_{run.dataset}_{run.attack}_{condition}_{level_str}"
          f"_{run.detector}_{run.seed}.json"
    )


def _all_result_paths(run: RobustnessRun, results_dir: str) -> List[Path]:
    return [_result_path(run, c, l, results_dir) for c, l in CONDITIONS]


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def _perturb_missing_text(
    dataset: HeteroDataset,
    frac: float,
    rng: np.random.Generator,
) -> HeteroDataset:
    """Replace review_texts with '' for frac of randomly chosen users.

    Directly degrades SemanticShield (TF-IDF) and Anti-FakeU (text-len feature).
    TH-GNN is unaffected: review_embs are derived from structural/temporal
    signals in the wrapper, not from review_texts.
    """
    n_missing = max(0, round(frac * dataset.num_users))
    if n_missing == 0:
        return dataset

    missing_users: set = set(
        rng.choice(dataset.num_users, size=n_missing, replace=False).tolist()
    )
    rev_users = dataset.review_user_idx.tolist()
    new_texts = [
        "" if rev_users[i] in missing_users else dataset.review_texts[i]
        for i in range(dataset.num_reviews)
    ]
    return dataclasses.replace(dataset, review_texts=new_texts)


def _perturb_noisy_timestamps(
    dataset: HeteroDataset,
    frac: float,
    rng: np.random.Generator,
) -> HeteroDataset:
    """Add N(0, σ) noise to all edge timestamps, σ = std(ts) × frac.

    Directly disrupts TH-GNN's temporal attention and burstiness GRU.
    Anti-FakeU's inter-arrival-time feature is also affected.
    """
    if frac == 0.0:
        return dataset

    ts_np  = dataset.timestamps.double().numpy()
    ts_std = float(ts_np.std())
    sigma  = ts_std * frac if ts_std > 0 else frac  # fallback if std=0
    noise  = rng.normal(0.0, sigma, size=len(ts_np))
    noisy  = torch.from_numpy((ts_np + noise))

    noisy_attr = noisy.float()
    new_edge_attr = {k: noisy_attr for k in dataset.edge_attr_dict}

    return dataclasses.replace(
        dataset,
        timestamps     = noisy,
        edge_attr_dict = new_edge_attr,
    )


def _perturb_missing_edges(
    dataset: HeteroDataset,
    frac: float,
    rng: np.random.Generator,
) -> HeteroDataset:
    """Randomly remove frac of reviews (edges) from the graph.

    Edge types (user,writes,review) and (review,about,item) have their review
    node indices remapped to stay contiguous.  Node feature tensors x_user and
    x_item are not recomputed (they capture pre-injection statistics).
    """
    n_reviews = dataset.num_reviews
    n_remove  = min(round(frac * n_reviews), n_reviews - 1)
    if n_remove <= 0:
        return dataset

    remove_set = set(
        rng.choice(n_reviews, size=n_remove, replace=False).tolist()
    )
    keep_np = np.array([i not in remove_set for i in range(n_reviews)], dtype=bool)
    keep_t  = torch.from_numpy(keep_np)
    new_n   = int(keep_np.sum())

    # New review-level arrays
    new_texts   = [dataset.review_texts[i] for i in range(n_reviews) if keep_np[i]]
    new_ratings = dataset.ratings[keep_t]
    new_ts      = dataset.timestamps[keep_t]
    new_ru      = dataset.review_user_idx[keep_t]
    new_ri      = dataset.review_item_idx[keep_t]
    new_train   = dataset.train_mask[keep_t]
    new_val     = dataset.val_mask[keep_t]
    new_test    = dataset.test_mask[keep_t]

    # New contiguous review indices (0 … new_n-1)
    new_r_idx = torch.arange(new_n, dtype=torch.long)

    # Rebuild edge_index_dict
    ur_old = dataset.edge_index_dict[("user",   "writes", "review")]
    ri_old = dataset.edge_index_dict[("review", "about",  "item")]
    ui_old = dataset.edge_index_dict[("user",   "rates",  "item")]

    new_eid = {
        ("user",   "writes", "review"): torch.stack([ur_old[0, keep_t], new_r_idx]),
        ("review", "about",  "item"):   torch.stack([new_r_idx, ri_old[1, keep_t]]),
        ("user",   "rates",  "item"):   ui_old[:, keep_t],
    }
    ts_attr = dataset.edge_attr_dict[("user", "writes", "review")][keep_t]
    new_eattr = {k: ts_attr for k in new_eid}

    return dataclasses.replace(
        dataset,
        num_reviews     = new_n,
        review_texts    = new_texts,
        ratings         = new_ratings,
        timestamps      = new_ts,
        review_user_idx = new_ru,
        review_item_idx = new_ri,
        train_mask      = new_train,
        val_mask        = new_val,
        test_mask       = new_test,
        edge_index_dict = new_eid,
        edge_attr_dict  = new_eattr,
    )


def _apply_perturbation(
    dataset:   HeteroDataset,
    condition: str,
    level:     float,
    seed:      int,
) -> HeteroDataset:
    """Dispatch to the appropriate perturbation function with a deterministic RNG."""
    if condition == "baseline" or level == 0.0:
        return dataset

    # Deterministic per-condition seed so conditions are independent
    cond_seed = (seed * 1000 + hash(condition)) % (2 ** 31)
    rng = np.random.default_rng(cond_seed)

    if condition == "missing_text":
        return _perturb_missing_text(dataset, level, rng)
    if condition == "noisy_timestamps":
        return _perturb_noisy_timestamps(dataset, level, rng)
    if condition == "missing_edges":
        return _perturb_missing_edges(dataset, level, rng)
    raise ValueError(f"Unknown robustness condition: {condition!r}")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_targets(dataset: HeteroDataset, n: int, seed: int) -> List[int]:
    train_items = dataset.review_item_idx[dataset.train_mask].numpy()
    counts = np.bincount(train_items, minlength=dataset.num_items)
    top_k  = max(n, int(dataset.num_items * 0.1))
    pool   = np.argsort(-counts)[:top_k]
    rng    = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()


def _build_detector(name: str, seed: int, use_aaa: bool = False):
    if name == "thgnn":
        return THGNNWrapper(seed=seed, use_aaa=use_aaa)
    if name == "semantic_shield_baseline":
        return SemanticShieldBaseline(seed=seed)
    if name == "anti_fakeu_baseline":
        return AntiFakeUBaseline(seed=seed)
    raise ValueError(f"Unknown detector: {name!r}")


# ---------------------------------------------------------------------------
# Single-run pipeline (grouped by detector × seed — one model fit, all conditions)
# ---------------------------------------------------------------------------

def run_robustness_single(args_tuple: Tuple[RobustnessRun, Dict[str, Any]]) -> List[Dict]:
    """Train detector once on the full poisoned graph, evaluate all conditions.

    Returns a list of result dicts (one per condition).
    Writes one JSON file per condition regardless of success/failure.
    """
    run, kwargs    = args_tuple
    raw_dir        = kwargs["raw_dir"]
    results_dir    = kwargs["results_dir"]
    cache_dir      = kwargs["cache_dir"]
    device         = kwargs.get("device") or _auto_device()
    use_aaa        = kwargs.get("use_aaa", False)

    all_results: List[Dict] = []
    t_total = time.perf_counter()

    base_result = {
        "config": {
            "dataset":  run.dataset,
            "attack":   run.attack,
            "rate":     run.rate,
            "detector": run.detector,
            "seed":     run.seed,
        }
    }

    try:
        logger.info(
            "[%s/agent4sr/rate=%.2f/%s/seed=%d] loading dataset …",
            run.dataset, run.rate, run.detector, run.seed,
        )

        # ── 1. Load and poison dataset ────────────────────────────────────
        loader  = AmazonBooksLoader(raw_dir=raw_dir)
        dataset = loader.load()

        targets = _select_targets(dataset, N_TARGETS, run.seed)
        n_fake  = max(1, round(run.rate * dataset.num_users))
        logger.info("  targets=%s  n_fake=%d", targets, n_fake)

        cache    = Path(cache_dir) / "agent4sr"
        attack   = Agent4SRAttack(dataset, cache_dir=str(cache), seed=run.seed)
        profiles = attack.generate_profiles(n_fake, BUDGET_PCT, targets)
        poisoned = attack.inject(dataset, profiles)
        logger.info("  injected %d fake users", len(profiles))

        # ── 2. Fit detector on full (unperturbed) poisoned graph ──────────
        detector = _build_detector(run.detector, run.seed, use_aaa=use_aaa)
        t0_fit   = time.perf_counter()
        detector.fit(poisoned, device=device, seed=run.seed)
        t_fit = time.perf_counter() - t0_fit
        logger.info("  detector fitted in %.1f s", t_fit)

        # ── 3. Evaluate under each degradation condition ──────────────────
        for condition, level in CONDITIONS:
            cond_result = {
                **base_result,
                "config": {
                    **base_result["config"],
                    "condition": condition,
                    "level":     level,
                },
                "status": "running",
            }
            path = _result_path(run, condition, level, results_dir)
            path.parent.mkdir(parents=True, exist_ok=True)

            try:
                degraded = _apply_perturbation(poisoned, condition, level, run.seed)
                logger.info(
                    "  [%s @ %.1f] scoring %d users  (%d reviews) …",
                    condition, level,
                    degraded.num_users, degraded.num_reviews,
                )

                t0_score = time.perf_counter()
                scores   = detector.score_users(degraded)
                t_score  = time.perf_counter() - t0_score

                labels = degraded.labels.numpy()
                f1, threshold = best_threshold_f1(labels, scores)
                dr  = detection_rate(labels, scores,  threshold=threshold)
                far = false_alarm_rate(labels, scores, threshold=threshold)

                logger.info(
                    "    F1=%.4f  DR=%.4f  FAR=%.4f  (score_secs=%.1f)",
                    f1, dr, far, t_score,
                )

                cond_result["detection"] = {
                    "f1":        round(float(f1),        4),
                    "dr":        round(float(dr),        4),
                    "far":       round(float(far),       4),
                    "threshold": round(float(threshold), 4),
                    "n_flagged": int(np.sum(scores >= threshold)),
                    "n_fake":    int(labels.sum()),
                    "score_secs": round(t_score, 2),
                    "fit_secs":   round(t_fit,   2),
                }
                cond_result["status"] = "ok"

            except Exception:  # noqa: BLE001
                cond_result["status"]    = "error"
                cond_result["traceback"] = traceback.format_exc()
                logger.error(
                    "  [%s @ %.1f] FAILED:\n%s",
                    condition, level, cond_result["traceback"],
                )

            with open(path, "w") as fh:
                json.dump(cond_result, fh, indent=2)
            all_results.append(cond_result)

    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("[%s/%s/%d] top-level FAILED:\n%s",
                     run.dataset, run.detector, run.seed, tb)
        # Write error stubs for any not-yet-written conditions
        written_conditions = {r["config"]["condition"] for r in all_results}
        for condition, level in CONDITIONS:
            if condition not in written_conditions:
                stub = {
                    **base_result,
                    "config": {**base_result["config"], "condition": condition, "level": level},
                    "status":    "error",
                    "traceback": tb,
                }
                path = _result_path(run, condition, level, results_dir)
                with open(path, "w") as fh:
                    json.dump(stub, fh, indent=2)
                all_results.append(stub)

    logger.info(
        "[%s/%s/seed=%d] done in %.1f s",
        run.dataset, run.detector, run.seed,
        time.perf_counter() - t_total,
    )
    return all_results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def load_robustness_results(results_dir: Path) -> "pd.DataFrame":
    import pandas as pd

    records = []
    for path in sorted(results_dir.glob("robustness_*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") != "ok":
            continue

        cfg = data.get("config", {})
        det = data.get("detection", {})
        records.append({
            "dataset":   cfg.get("dataset",   ""),
            "attack":    cfg.get("attack",    ""),
            "condition": cfg.get("condition", ""),
            "level":     float(cfg.get("level", 0.0)),
            "detector":  cfg.get("detector",  ""),
            "seed":      int(cfg.get("seed",  0)),
            "f1":        float(det.get("f1",  np.nan)),
            "dr":        float(det.get("dr",  np.nan)),
            "far":       float(det.get("far", np.nan)),
        })

    if not records:
        return pd.DataFrame(columns=[
            "dataset", "attack", "condition", "level", "detector",
            "seed", "f1", "dr", "far",
        ])
    return pd.DataFrame(records)


def build_robustness_csv(df: "pd.DataFrame") -> "pd.DataFrame":
    """Build the summary CSV: one row per (condition, level, detector).

    Columns: condition, level, detector, F1, F1_delta, FAR, FAR_delta
    where F1/FAR are 'mean±std' strings and *_delta is vs baseline.
    """
    import pandas as pd

    def _ms(arr: np.ndarray) -> str:
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            return "—"
        return f"{v.mean():.3f}±{v.std():.3f}"

    # Baseline mean F1 and FAR per detector (averaged over seeds)
    baseline_df = df[df["condition"] == "baseline"]
    baseline_f1  = baseline_df.groupby("detector")["f1"].mean()
    baseline_far = baseline_df.groupby("detector")["far"].mean()

    rows = []
    for (condition, level, detector), grp in df.groupby(["condition", "level", "detector"]):
        f1_arr  = grp["f1"].values
        far_arr = grp["far"].values

        b_f1  = baseline_f1.get(detector, np.nan)
        b_far = baseline_far.get(detector, np.nan)

        f1_mean  = float(np.nanmean(f1_arr))
        far_mean = float(np.nanmean(far_arr))

        rows.append({
            "condition":     condition,
            "level":         level,
            "detector":      detector,
            "F1":            _ms(f1_arr),
            "F1_delta":      f"{f1_mean - b_f1:+.3f}" if not np.isnan(b_f1) else "—",
            "FAR":           _ms(far_arr),
            "FAR_delta":     f"{far_mean - b_far:+.3f}" if not np.isnan(b_far) else "—",
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Canonical column order
    return out[["condition", "level", "detector", "F1", "F1_delta", "FAR", "FAR_delta"]]


def plot_robustness_curves(df: "pd.DataFrame", fig_path: Path) -> None:
    """3-subplot line chart: F1 vs degradation level, one subplot per condition type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle(
        "TH-GNN robustness under missing modalities\n"
        "(Amazon-Books, Agent4SR, injection rate 0.01)",
        fontsize=11,
    )

    styles = {
        "thgnn":                    dict(color="#1f77b4", ls="-",  marker="o",  lw=2.0, label="TH-GNN"),
        "semantic_shield_baseline": dict(color="#ff7f0e", ls="--", marker="s",  lw=1.5, label="SemanticShield"),
        "anti_fakeu_baseline":      dict(color="#2ca02c", ls=":",  marker="^",  lw=1.5, label="Anti-FakeU"),
    }

    for ax, cond_type in zip(axes, CONDITION_TYPES):
        ax.set_title(CONDITION_LABELS.get(cond_type, cond_type), fontsize=10)
        ax.set_xlabel("Degradation level", fontsize=9)
        ax.set_ylabel("F1", fontsize=9)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(["0\n(baseline)", "0.2", "0.5"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        for det in DETECTORS:
            sty = styles[det]
            xs, ys, errs = [], [], []

            # Baseline (level 0.0)
            b = df[(df["condition"] == "baseline") & (df["detector"] == det)]["f1"].values
            if len(b):
                xs.append(0.0)
                ys.append(float(np.nanmean(b)))
                errs.append(float(np.nanstd(b)))

            # Degraded levels
            for level in [0.2, 0.5]:
                sub = df[(df["condition"] == cond_type) &
                         (df["level"]     == level)     &
                         (df["detector"]  == det)]["f1"].values
                if len(sub):
                    xs.append(level)
                    ys.append(float(np.nanmean(sub)))
                    errs.append(float(np.nanstd(sub)))

            if len(xs) < 1:
                continue

            xs_arr  = np.array(xs)
            ys_arr  = np.array(ys)
            err_arr = np.array(errs)

            ax.plot(xs_arr, ys_arr,
                    color=sty["color"], ls=sty["ls"],
                    marker=sty["marker"], lw=sty["lw"],
                    label=sty["label"])
            ax.fill_between(xs_arr,
                            ys_arr - err_arr, ys_arr + err_arr,
                            color=sty["color"], alpha=0.12)

        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", fig_path)


def aggregate_and_plot(results_dir: Path, out_dir: Path, fig_dir: Path) -> None:
    """Load robustness JSONs, write CSV, and generate the line-plot PDF."""
    import pandas as pd
    try:
        from tabulate import tabulate as _tabulate
        _has_tabulate = True
    except ImportError:
        _has_tabulate = False

    print("\nLoading robustness results …")
    df = load_robustness_results(results_dir)
    print(f"  Loaded {len(df)} robustness records.")
    if df.empty:
        print("  No robustness data found. Run the experiments first.")
        return

    # ── CSV ───────────────────────────────────────────────────────────────
    print("Building robustness CSV …")
    tbl = build_robustness_csv(df)
    if not tbl.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "robustness.csv"
        tbl.to_csv(csv_path, index=False)
        if _has_tabulate:
            print(_tabulate(tbl, headers="keys", tablefmt="simple", showindex=False))
        else:
            print(tbl.to_string(index=False))
        print(f"  Saved {csv_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    print("Generating robustness curves figure …")
    try:
        fig_path = fig_dir / "robustness_curves.pdf"
        plot_robustness_curves(df, fig_path)
        print(f"  Saved {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: figure generation failed — {exc}")

    # ── Console summary ───────────────────────────────────────────────────
    print("\n  ΔF1 vs baseline (mean across seeds × levels):")
    baseline_f1 = (
        df[df["condition"] == "baseline"]
        .groupby("detector")["f1"].mean()
    )
    for det in DETECTORS:
        lbl   = DETECTOR_LABELS.get(det, det)
        b_f1  = baseline_f1.get(det, np.nan)
        print(f"\n  [{lbl}]  baseline F1 = {b_f1:.4f}")
        for cond in CONDITION_TYPES:
            for level in [0.2, 0.5]:
                sub = df[(df["condition"] == cond) &
                         (df["level"]     == level) &
                         (df["detector"]  == det)]["f1"].values
                if not len(sub):
                    continue
                delta = float(np.nanmean(sub)) - b_f1
                print(f"    {CONDITION_LABELS[cond]:20s} @ {level:.1f}:  "
                      f"F1={np.nanmean(sub):.4f}  Δ={delta:+.4f}")


# ---------------------------------------------------------------------------
# Progress summary
# ---------------------------------------------------------------------------

def _print_summary(all_results: List[List[Dict]]) -> None:
    flat = [r for batch in all_results for r in batch]
    ok     = [r for r in flat if r.get("status") == "ok"]
    errors = [r for r in flat if r.get("status") == "error"]
    skip   = [r for r in flat if r.get("status") == "skipped"]
    print(f"\n{'='*65}")
    print(f"  Robustness experiments: {len(ok)} ok  {len(errors)} errors  {len(skip)} skipped")

    if ok:
        baseline_ok = [r for r in ok if r["config"]["condition"] == "baseline"]
        if baseline_ok:
            by_det: Dict[str, List[float]] = {}
            for r in baseline_ok:
                d = r["config"]["detector"]
                by_det.setdefault(d, []).append(r["detection"]["f1"])
            print("\n  Baseline F1 (full modalities):")
            for det, f1s in by_det.items():
                print(f"    {DETECTOR_LABELS.get(det, det):20s}  "
                      f"F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run THGNN robustness (missing-modality) experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Smoke test: 1 seed, thgnn only.",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (each trains one detector per seed). 1 = sequential.",
    )
    p.add_argument("--raw-dir",     default="data/raw",       metavar="DIR")
    p.add_argument("--cache-dir",   default="data/processed", metavar="DIR")
    p.add_argument("--results-dir", default="results/raw",    metavar="DIR")
    p.add_argument("--out-dir",     default="results/tables", metavar="DIR")
    p.add_argument("--fig-dir",     default="results/figures",metavar="DIR")
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="torch device: cpu | cuda | mps (autodetect if omitted).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-run even if all result files already exist.",
    )
    p.add_argument(
        "--detector", nargs="+", choices=DETECTORS + ["all"], default=None,
        help="Restrict to specific detectors.",
    )
    p.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip experiments; only load existing JSONs and regenerate CSV + figure.",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    device  = args.device or _auto_device()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    fig_dir     = Path(args.fig_dir)

    if args.aggregate_only:
        aggregate_and_plot(results_dir, out_dir, fig_dir)
        return

    use_aaa = _load_use_aaa()
    logger.info(
        "Device: %s  thgnn.use_aaa (from paper_runs.yaml): %s", device, use_aaa,
    )

    # ── Build run list ────────────────────────────────────────────────────
    detectors = DETECTORS if args.dry_run else DETECTORS
    seeds     = SEEDS[:1]  if args.dry_run else SEEDS

    runs = [RobustnessRun(det, s) for det in detectors for s in seeds]

    if args.detector and "all" not in args.detector:
        runs = [r for r in runs if r.detector in args.detector]

    if not args.overwrite:
        pending = [r for r in runs
                   if not all(p.exists() for p in _all_result_paths(r, args.results_dir))]
        skipped = len(runs) - len(pending)
        if skipped:
            logger.info("Skipping %d already-complete runs.", skipped)
        runs = pending

    if not runs:
        logger.info("Nothing to run. Use --overwrite to force re-run.")
        aggregate_and_plot(results_dir, out_dir, fig_dir)
        return

    logger.info(
        "Running %d robustness experiment group(s)%s  (device=%s, workers=%d).",
        len(runs),
        "  [DRY RUN]" if args.dry_run else "",
        device,
        args.workers,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs: Dict[str, Any] = {
        "raw_dir":     args.raw_dir,
        "cache_dir":   args.cache_dir,
        "results_dir": args.results_dir,
        "device":      device,
        "use_aaa":     use_aaa,
    }
    tasks = [(r, run_kwargs) for r in runs]

    all_results: List[List[Dict]] = []

    if args.workers == 1:
        for task in tasks:
            all_results.append(run_robustness_single(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
            future_to_run = {exe.submit(run_robustness_single, t): t[0] for t in tasks}
            for future in concurrent.futures.as_completed(future_to_run):
                run = future_to_run[future]
                try:
                    all_results.append(future.result(timeout=7200))
                except concurrent.futures.TimeoutError:
                    logger.error("TIMEOUT: %s/%d", run.detector, run.seed)
                    all_results.append([])
                except Exception as exc:  # noqa: BLE001
                    logger.error("Worker exception for %s/%d: %s", run.detector, run.seed, exc)
                    all_results.append([])

    _print_summary(all_results)
    aggregate_and_plot(results_dir, out_dir, fig_dir)


if __name__ == "__main__":
    main()
