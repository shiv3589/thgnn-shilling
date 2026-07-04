#!/bin/bash
set -e
mkdir -p logs checkpoints

echo "$(date) — Pair 1: ml1m + musical_instruments"
python scripts/train.py --dataset ml1m --seed 42 --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir data/raw --cache-dir data/processed \
  2>&1 | tee logs/train_ml1m_seed42.log &
PID1=$!

python scripts/train.py --dataset musical_instruments --seed 42 --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir data/raw --cache-dir data/processed \
  2>&1 | tee logs/train_musical_instruments_seed42.log &
PID2=$!

wait $PID1 $PID2
echo "$(date) — Pair 1 done"

echo "$(date) — Pair 2: video_games + arts_crafts"
python scripts/train.py --dataset video_games --seed 42 --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir data/raw --cache-dir data/processed \
  2>&1 | tee logs/train_video_games_seed42.log &
PID3=$!

python scripts/train.py --dataset arts_crafts --seed 42 --device cuda \
  --use-aaa --epochs 30 --batch-size 256 \
  --raw-dir data/raw --cache-dir data/processed \
  2>&1 | tee logs/train_arts_crafts_seed42.log &
PID4=$!

wait $PID3 $PID4
echo "$(date) — Pair 2 done"
echo "All 4 training jobs complete."
