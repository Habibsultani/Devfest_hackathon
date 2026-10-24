"""Run the saved model on an image from a new process without rebuilding its bank."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference import DEFAULT_CONFIG, inspect_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--instrument", default=None, help="Choose a declared supported instrument class."
    )
    parser.add_argument(
        "--map-output", type=Path, help="Optional .npy output for Person 2's overlay."
    )
    args = parser.parse_args()
    with Image.open(args.image) as image:
        result = inspect_image(
            image,
            config_path=None if args.instrument else args.config,
            instrument_class=args.instrument,
        )
    anomaly_map = result.pop("anomaly_map")
    result["anomaly_map_shape"] = list(anomaly_map.shape)
    result["anomaly_map_range"] = [float(anomaly_map.min()), float(anomaly_map.max())]
    if args.map_output:
        args.map_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.map_output, anomaly_map)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
