#!/bin/bash
set -e
DATASETS=(ml1m video_games arts_crafts musical_instruments)
SEED=42
INJECTION_RATE=0.01
mkdir -p logs checkpoints /data/attacked

# Step 1: inject attacks for all datasets
echo "$(date) — Injecting attacks..."
for ds in "${DATASETS[@]}"; do
    echo "Injecting random attack into $ds..."
    python scripts/run_attack.py \
        --dataset $ds \
        --attack random \
        --injection-rate $INJECTION_RATE \
        --raw-dir /data/raw \
        --cache-dir /data/processed \
        --output-dir /data/attacked \
        --seed $SEED \
        2>&1 | tee logs/attack_${ds}.log
done
echo "$(date) — All attacks injected"

# Step 2: train in parallel pairs on attacked datasets
echo "$(date) — Pair 1: ml1m + musical_instruments"
python scripts/train.py --dataset ml1m --seed $SEED --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir /data/raw --cache-dir /data/processed \
  --attacked-dir /data/attacked \
  --tc-alpha 0.01 \
  2>&1 | tee logs/train_ml1m_seed${SEED}.log &
PID1=$!

python scripts/train.py --dataset musical_instruments --seed $SEED --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir /data/raw --cache-dir /data/processed \
  --attacked-dir /data/attacked \
  --tc-alpha 0.01 \
  2>&1 | tee logs/train_musical_instruments_seed${SEED}.log &
PID2=$!

wait $PID1 $PID2
echo "$(date) — Pair 1 done"

echo "$(date) — Pair 2: video_games + arts_crafts"
python scripts/train.py --dataset video_games --seed $SEED --device cuda \
  --use-aaa --epochs 30 --batch-size 512 \
  --raw-dir /data/raw --cache-dir /data/processed \
  --attacked-dir /data/attacked \
  --tc-alpha 0.01 \
  2>&1 | tee logs/train_video_games_seed${SEED}.log &
PID3=$!

python scripts/train.py --dataset arts_crafts --seed $SEED --device cuda \
  --use-aaa --epochs 30 --batch-size 256 \
  --raw-dir /data/raw --cache-dir /data/processed \
  --attacked-dir /data/attacked \
  --tc-alpha 0.01 \
  2>&1 | tee logs/train_arts_crafts_seed${SEED}.log &
PID4=$!

wait $PID3 $PID4
echo "$(date) — Pair 2 done"
echo "All 4 training jobs complete."
