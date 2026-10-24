"""Reproduce the balanced SAM/tiled Dinomaly/LoRA integration from source."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    commands = []
    if args.download:
        commands.append(["scripts.download_dataset", "--instruments", "instrument12", "instrument19"])
    for instrument in ("instrument12", "instrument19"):
        commands.append(["scripts.prepare_balanced_dataset", "--instrument", instrument,
                         "--output", f"data/processed/{instrument}"])
    commands.extend([["scripts.setup_research_models", "--for-training"], ["scripts.prepare_sam"],
                     ["scripts.train_dinomaly", "--steps", "1000", "--batch-size", "4", "--validate-every", "200"],
                     ["scripts.evaluate_dinomaly"]])
    for command in commands:
        subprocess.run([sys.executable, "-m", *command], check=True)


if __name__ == "__main__":
    main()
