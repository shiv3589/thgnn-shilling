#!/usr/bin/env python
"""Train LightGCN on all four datasets and record clean baseline metrics.

The output JSON is the reference used to measure recommendation performance
*degradation* after shilling attacks are injected.

Output
------
    results/lightgcn_clean.json

    Schema::

        {
          "metadata": { model, embed_dim, n_layers, epochs, lr, timestamp },
          "results": {
            "<dataset>": {
              "ndcg@10": float,
              "hr@10":   float,
              "val_ndcg@10": float,
              "val_hr@10":   float,
              "best_epoch":  int,
              "train_time_secs": float
            }
          }
        }

Usage
-----
    python scripts/run_baseline.py
    python scripts/run_baseline.py --datasets ml-1m yelp2018 --device cuda
    python scripts/run_baseline.py --epochs 50 --batch-size 4096
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import (
    AmazonBooksLoader,
    AmazonClothingLoader,
    BaseLoader,
    MovieLens1MLoader,
    Yelp2018Loader,
)
from src.models.lightgcn import train_lightgcn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {
    "ml-1m":           MovieLens1MLoader,
    "amazon-books":    AmazonBooksLoader,
    "amazon-clothing": AmazonClothingLoader,
    "yelp2018":        Yelp2018Loader,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LightGCN clean-baseline training across all datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",       default="data/raw",       metavar="DIR")
    p.add_argument("--output",        default="results/lightgcn_clean.json", metavar="FILE")
    p.add_argument(
        "--datasets", nargs="+", default=list(LOADER_REGISTRY),
        choices=list(LOADER_REGISTRY), metavar="NAME",
    )
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--embed-dim",     type=int,   default=64)
    p.add_argument("--n-layers",      type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=2048)
    p.add_argument("--reg-weight",    type=float, default=1e-4)
    p.add_argument("--eval-every",    type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--min-interactions", type=int, default=5, metavar="K")
    p.add_argument("--device",        default=None, metavar="DEVICE")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _train_one(
    name: str,
    loader_cls: Type[BaseLoader],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict], str]:
    """Load dataset and run training.  Returns (metrics_dict, status_str)."""
    loader = loader_cls(raw_dir=args.raw_dir)
    logger.info("━━━  %s  ━━━", name)

    try:
        dataset = loader.load(
            min_user_interactions=args.min_interactions,
            min_item_interactions=args.min_interactions,
        )
    except FileNotFoundError as exc:
        logger.error("[%s] raw files not found — skipping.\n  %s", name, exc)
        return None, "MISSING"

    metrics = train_lightgcn(
        dataset,
        epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        reg_weight=args.reg_weight,
        eval_every=args.eval_every,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
    return metrics, "OK"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(
    rows: List[Tuple[str, Optional[Dict], str]],
    k: int,
    total_secs: float,
) -> None:
    W = 20
    cols = [
        ("Dataset", W),
        (f"NDCG@{k}", 10),
        (f"HR@{k}", 9),
        ("Best ep.", 9),
        ("Time", 9),
    ]

    def sep(char="─"):
        return "┼".join(char * (w + 2) for _, w in cols)

    def row_str(vals):
        parts = []
        for v, (_, w) in zip(vals, cols):
            parts.append(f" {v:<{w}} " if isinstance(v, str) and len(v) < w else f" {v:>{w}} ")
        return "│".join(parts)

    hdr   = "┌" + "┬".join("─" * (w + 2) for _, w in cols) + "┐"
    div   = "├" + sep() + "┤"
    foot  = "└" + "┴".join("─" * (w + 2) for _, w in cols) + "┘"

    print()
    print(hdr)
    print("│" + row_str([c for c, _ in cols]) + "│")
    print(div)

    for name, m, status in rows:
        if m is None:
            vals = [name, status, "–", "–", "–"]
        else:
            vals = [
                name,
                f"{m[f'ndcg@{k}']:.4f}",
                f"{m[f'hr@{k}']:.4f}",
                str(m["best_epoch"]),
                _fmt(m["train_time_secs"]),
            ]
        print("│" + row_str(vals) + "│")

    print(div)
    print("│" + row_str(["Total", "", "", "", _fmt(total_secs)]) + "│")
    print(foot)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    k = 10  # fixed evaluation cutoff

    wall_t0 = time.perf_counter()
    result_rows: List[Tuple[str, Optional[Dict], str]] = []

    for name in args.datasets:
        metrics, status = _train_one(name, LOADER_REGISTRY[name], args)
        result_rows.append((name, metrics, status))

    total_secs = time.perf_counter() - wall_t0

    # ── Print summary ──────────────────────────────────────────────────────
    _print_table(result_rows, k, total_secs)

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict = {
        "metadata": {
            "model":      "LightGCN",
            "embed_dim":  args.embed_dim,
            "n_layers":   args.n_layers,
            "epochs":     args.epochs,
            "lr":         args.lr,
            "reg_weight": args.reg_weight,
            "batch_size": args.batch_size,
            "eval_protocol": f"sampled-999-neg@{k}",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        },
        "results": {},
    }

    for name, metrics, status in result_rows:
        if metrics is None:
            payload["results"][name] = {"status": status}
            continue
        payload["results"][name] = {
            f"ndcg@{k}":     metrics[f"ndcg@{k}"],
            f"hr@{k}":       metrics[f"hr@{k}"],
            f"val_ndcg@{k}": metrics.get(f"val_ndcg@{k}"),
            f"val_hr@{k}":   metrics.get(f"val_hr@{k}"),
            "best_epoch":    metrics["best_epoch"],
            "train_time_secs": metrics["train_time_secs"],
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Results saved → %s", out_path.resolve())

    # Print per-dataset summary lines for quick log scanning
    for name, metrics, status in result_rows:
        if metrics:
            logger.info(
                "%-20s  ndcg@%d=%.4f  hr@%d=%.4f  (epoch %d)",
                name, k, metrics[f"ndcg@{k}"], k, metrics[f"hr@{k}"],
                metrics["best_epoch"],
            )


if __name__ == "__main__":
    main()
