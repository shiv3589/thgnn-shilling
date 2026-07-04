#!/usr/bin/env python3
"""Inject shilling attack profiles into a clean dataset and save the result.

Usage
-----
    python scripts/run_attack.py --dataset ml1m --attack random --injection-rate 0.01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.classic import RandomAttack, BandwagonAttack
from src.datasets import (
    AmazonArtsCraftsLoader,
    AmazonMusicalInstrumentsLoader,
    AmazonVideoGamesLoader,
    MovieLens1MLoader,
)

LOADERS = {
    "ml1m": MovieLens1MLoader,
    "video_games": AmazonVideoGamesLoader,
    "arts_crafts": AmazonArtsCraftsLoader,
    "musical_instruments": AmazonMusicalInstrumentsLoader,
}

ATTACKS = {
    "random": RandomAttack,
    "bandwagon": BandwagonAttack,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject a shilling attack into a clean dataset")
    parser.add_argument("--dataset", required=True, choices=list(LOADERS))
    parser.add_argument("--attack", default="random", choices=list(ATTACKS))
    parser.add_argument("--injection-rate", type=float, default=0.01,
                         help="Fraction of real users to add as fake attackers")
    parser.add_argument("--budget-pct", type=float, default=0.05,
                         help="Fraction of the item catalog each fake user rates")
    parser.add_argument("--max-filler-items", type=int, default=200,
                         help="Hard cap on filler items per fake user, regardless "
                              "of catalog size (prevents budget-pct from blowing up "
                              "injected-review counts on large catalogs). "
                              "Pass -1 to disable the cap.")
    parser.add_argument("--n-targets", type=int, default=3,
                         help="Number of target items to push (ignored if --target-items given)")
    parser.add_argument("--target-items", type=int, nargs="*", default=None,
                         help="Explicit item global indices to push to max rating")
    parser.add_argument("--min-interactions", type=int, default=5, metavar="K")
    parser.add_argument("--raw-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--cache-dir", default="data/processed", metavar="DIR",
                         help="Unused by injection; accepted for CLI parity with train.py")
    parser.add_argument("--output-dir", default="data/attacked", metavar="DIR")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.dataset}...")
    loader = LOADERS[args.dataset](raw_dir=args.raw_dir)
    dataset = loader.load(
        min_user_interactions=args.min_interactions,
        min_item_interactions=args.min_interactions,
    )

    n_fake = max(1, round(dataset.num_users * args.injection_rate))
    print(f"Injecting {n_fake} fake users ({args.injection_rate * 100:.1f}% of {dataset.num_users} users)...")

    if args.target_items is not None:
        target_items = args.target_items
    else:
        n_targets = min(args.n_targets, dataset.num_items)
        target_items = rng.choice(dataset.num_items, size=n_targets, replace=False).tolist()
    print(f"Target items: {target_items}")

    max_filler_items = None if args.max_filler_items < 0 else args.max_filler_items
    attack = ATTACKS[args.attack](dataset, seed=args.seed, max_filler_items=max_filler_items)
    profiles = attack.generate_profiles(
        n_fake_users=n_fake,
        budget_pct=args.budget_pct,
        target_items=target_items,
    )
    attacked_dataset = attack.inject(dataset, profiles)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.attack}_{args.injection_rate}.pt")
    torch.save(attacked_dataset, out_path)

    n_fake_actual = int(attacked_dataset.labels.sum().item())
    n_real_actual = int((attacked_dataset.labels == 0).sum().item())
    print(f"Attacked dataset saved to {out_path}")
    print(f"Labels: {n_fake_actual} fake, {n_real_actual} real")


if __name__ == "__main__":
    main()
