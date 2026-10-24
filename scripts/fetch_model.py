"""Usage: python -m scripts.fetch_model [--config models/model_config.json]."""

import argparse
from pathlib import Path

from src.artifacts import ensure_artifact
from src.model_config import ModelError, load_model_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch or verify the configured model artifact.")
    parser.add_argument("--config", type=Path, default=Path("models/model_config.json"))
    args = parser.parse_args()
    try:
        path = ensure_artifact(load_model_config(args.config))
    except ModelError as exc:
        parser.exit(1, f"Model unavailable: {exc}\n")
    print(f"Verified model artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
