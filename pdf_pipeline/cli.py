from __future__ import annotations
import argparse
from .config import load_config
from .runner import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="ArXiv PDF parsing pipeline")
    ap.add_argument("--config", type=str, default="configs/example.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = run_pipeline(cfg)
    print(res)


if __name__ == "__main__":
    main()

