"""Download and checksum-verify upstream SAM and optional DINO training weights."""

import argparse
import json
import urllib.request
from pathlib import Path

from src.dataset import file_sha256


def retrieve(url: str, target: Path, checksum: str):
    if target.is_file() and file_sha256(target) == checksum:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
    if file_sha256(temporary) != checksum:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Upstream model checksum mismatch: {target.name}")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--for-training", action="store_true")
    args = parser.parse_args()
    models = Path(__file__).resolve().parents[1] / "models"
    config = json.loads((models / "sam_config.json").read_text())
    retrieve(config["source_url"], models / config["checkpoint"], config["sha256"])
    print("Verified official MobileSAM weights.")
    if args.for_training:
        config = json.loads((models / "dino_source.json").read_text())
        retrieve(config["url"], models / config["checkpoint"], config["sha256"])
        print("Verified official DINOv2 training weights.")


if __name__ == "__main__":
    main()
