"""Evaluation entry point."""
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate THGNN model")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raise NotImplementedError("Evaluation loop not yet implemented")


if __name__ == "__main__":
    main()
