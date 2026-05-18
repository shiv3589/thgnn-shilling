#!/usr/bin/env python
"""Pre-compute and cache RoBERTa embeddings for all four datasets.

For each dataset this script:
  1. Loads the HeteroDataset (applies k-core filtering, builds the graph).
  2. Encodes all review texts with ReviewEncoder → ``reviews.npy``.
  3. Builds per-item description strings, encodes with ItemEncoder → ``items.npy``.

Both encoders cache their outputs so re-running is a no-op unless ``--force``
is passed.

Output layout
-------------
    data/processed/{dataset}/embeddings/reviews.npy   shape (N_reviews, 768)
    data/processed/{dataset}/embeddings/items.npy     shape (N_items,   768)

Usage
-----
    # All datasets, auto-detect device
    python scripts/precompute_embeddings.py

    # Specific datasets on GPU, larger batch
    python scripts/precompute_embeddings.py \\
        --datasets ml-1m yelp2018 \\
        --device cuda \\
        --batch-size 128

    # Force re-encode even if cache exists
    python scripts/precompute_embeddings.py --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

# Allow running directly from repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import (
    AmazonBooksLoader,
    AmazonClothingLoader,
    BaseLoader,
    MovieLens1MLoader,
    Yelp2018Loader,
)
from src.utils.text_encoder import ItemEncoder, ReviewEncoder

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
        description="Pre-compute RoBERTa embeddings for review texts and item descriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",       default="data/raw",       metavar="DIR",
                   help="Root directory for raw dataset files")
    p.add_argument("--processed-dir", default="data/processed", metavar="DIR",
                   help="Root directory for processed outputs")
    p.add_argument(
        "--datasets", nargs="+", default=list(LOADER_REGISTRY),
        choices=list(LOADER_REGISTRY), metavar="NAME",
        help="Datasets to process (default: all four)",
    )
    p.add_argument("--batch-size", type=int, default=64, metavar="N",
                   help="Texts per encoder forward pass")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="Torch device string (default: auto-detect CUDA > MPS > CPU)")
    p.add_argument("--force", action="store_true",
                   help="Re-encode and overwrite existing cache files")
    p.add_argument("--min-interactions", type=int, default=5, metavar="K",
                   help="k-core threshold passed to each dataset loader")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-dataset work
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def process_dataset(
    name: str,
    loader_cls: Type[BaseLoader],
    *,
    raw_dir: str,
    processed_dir: str,
    review_enc: ReviewEncoder,
    item_enc: ItemEncoder,
    min_interactions: int,
    force: bool,
) -> Tuple[int, int, float]:
    """Run both encoders for one dataset.  Returns (n_reviews, n_items, secs)."""
    t0 = time.perf_counter()
    bar = "━" * 52
    logger.info("%s  %s  %s", bar, name, bar)

    loader = loader_cls(raw_dir=raw_dir)

    # ── Step 1 : load graph ────────────────────────────────────────────
    logger.info("[%s] Loading dataset (k=%d)…", name, min_interactions)
    ds = loader.load(
        min_user_interactions=min_interactions,
        min_item_interactions=min_interactions,
    )
    logger.info("[%s] %s", name, ds.summary())

    # ── Step 2 : review embeddings ─────────────────────────────────────
    n_empty = sum(1 for t in ds.review_texts if not t.strip())
    if n_empty:
        logger.info(
            "[%s] Note: %d / %d review texts are empty (ML-1M has no text).",
            name, n_empty, ds.num_reviews,
        )
    logger.info("[%s] Encoding %d review texts…", name, ds.num_reviews)
    review_enc.encode(
        ds.review_texts,
        dataset_name=name,
        show_progress=True,
        force=force,
    )

    # ── Step 3 : item embeddings ───────────────────────────────────────
    logger.info("[%s] Building %d item description strings…", name, ds.num_items)
    item_texts: List[str] = loader.get_item_texts(ds.item_id_map)
    n_empty_items = sum(1 for t in item_texts if not t.strip())
    if n_empty_items:
        logger.warning(
            "[%s] %d / %d item descriptions are empty.", name, n_empty_items, ds.num_items,
        )
    logger.info("[%s] Encoding %d item descriptions…", name, ds.num_items)
    item_enc.encode(
        item_texts,
        dataset_name=name,
        show_progress=True,
        force=force,
    )

    elapsed = time.perf_counter() - t0
    return ds.num_reviews, ds.num_items, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    enc_kwargs = dict(
        cache_dir=args.processed_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    # Both encoders share the same underlying roberta-base model instance
    # (lazy-loaded on first encode call, so startup is fast).
    review_enc = ReviewEncoder(**enc_kwargs)
    item_enc = ItemEncoder(**enc_kwargs)

    # We share the same model object between the two encoders to avoid
    # loading roberta-base twice.  They differ only in their _SUFFIX.
    item_enc._tokenizer = review_enc.tokenizer   # type: ignore[assignment]
    item_enc._model = review_enc.model           # type: ignore[assignment]

    wall_t0 = time.perf_counter()
    rows: List[Tuple[str, Optional[int], Optional[int], str]] = []

    for name in args.datasets:
        loader_cls = LOADER_REGISTRY[name]
        try:
            n_r, n_i, elapsed = process_dataset(
                name,
                loader_cls,
                raw_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                review_enc=review_enc,
                item_enc=item_enc,
                min_interactions=args.min_interactions,
                force=args.force,
            )
            rows.append((name, n_r, n_i, _fmt(elapsed)))
        except FileNotFoundError as exc:
            logger.error("[%s] Skipped — raw files missing:\n  %s", name, exc)
            rows.append((name, None, None, "SKIPPED"))

    total_secs = time.perf_counter() - wall_t0

    # ── Summary table ──────────────────────────────────────────────────
    print()
    W = 22
    print(f"┌{'─'*W}┬{'─'*12}┬{'─'*11}┬{'─'*10}┐")
    print(f"│ {'Dataset':<{W-2}} │ {'Reviews':>10} │ {'Items':>9} │ {'Time':>8} │")
    print(f"├{'─'*W}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
    for ds, n_r, n_i, t in rows:
        r_s = f"{n_r:,}" if n_r is not None else "–"
        i_s = f"{n_i:,}" if n_i is not None else "–"
        print(f"│ {ds:<{W-2}} │ {r_s:>10} │ {i_s:>9} │ {t:>8} │")
    print(f"├{'─'*W}┼{'─'*12}┼{'─'*11}┼{'─'*10}┤")
    print(f"│ {'Total':<{W-2}} │ {'':>10} │ {'':>9} │ {_fmt(total_secs):>8} │")
    print(f"└{'─'*W}┴{'─'*12}┴{'─'*11}┴{'─'*10}┘")

    out_root = Path(args.processed_dir)
    print()
    print("Embeddings saved under:", out_root.resolve())
    for name, n_r, *_ in rows:
        if n_r is not None:
            emb = out_root / name / "embeddings"
            print(f"  {emb}/")
            print(f"    reviews.npy")
            print(f"    items.npy")
    print()


if __name__ == "__main__":
    main()
