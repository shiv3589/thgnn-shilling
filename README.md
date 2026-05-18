# thgnn-shilling

Temporal Heterogeneous Graph Neural Network for shilling attack detection in recommender systems.

## Overview

<!-- Briefly describe the problem, approach, and key contributions. -->

## Setup

```bash
git clone <repo-url>
cd thgnn-shilling
pip install -e .
```

## Data

<!-- Describe datasets used, sources, and how to obtain them. -->

Raw data goes in `data/raw/`, preprocessing outputs to `data/processed/`, and train/val/test splits to `data/splits/`.

```bash
# Example preprocessing
python scripts/preprocess.py --config configs/data_config.yaml
```

## Training

```bash
python scripts/train.py --config configs/model_config.yaml
```

Key options:
- `--config`: path to model config YAML
- `--wandb`: enable W&B logging

## Evaluation

```bash
python scripts/evaluate.py --config configs/model_config.yaml --checkpoint <path>
```

## Attacks

Shilling attack simulation and robustness evaluation:

```bash
python scripts/run_attack.py --config configs/attack_config.yaml
```

Supported attack types are defined in `configs/attack_config.yaml`.
