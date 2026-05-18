"""Training entry point."""
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train THGNN model")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.wandb:
        cfg["logging"]["wandb"] = True

    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
