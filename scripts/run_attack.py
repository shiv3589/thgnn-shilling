"""Shilling attack simulation entry point."""
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run shilling attack simulation")
    parser.add_argument("--config", default="configs/attack_config.yaml")
    parser.add_argument("--attack-type", default=None, help="Override attack type from config")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.attack_type is not None:
        cfg["attack"]["type"] = args.attack_type

    raise NotImplementedError("Attack simulation not yet implemented")


if __name__ == "__main__":
    main()
