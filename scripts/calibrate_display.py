"""Save fixed heatmap display bounds from validation images only; never fit thresholds."""

import json
import math
from pathlib import Path

from PIL import Image

from src.dataset import file_sha256, load_manifest, write_json
from src.inference import get_inspector


def calibrate_display(config_path: Path, manifest_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = file_sha256(manifest_path)
    if digest != config["manifest_sha256"]:
        raise ValueError("Display calibration manifest differs from the trained model.")
    rows = [row for row in load_manifest(manifest_path) if row["split"] == "validation"]
    if not rows or any(row["instrument_class"] != config["instrument_class"] for row in rows):
        raise ValueError("Matching validation images are required for display calibration.")
    inspector = get_inspector(config_path)
    maximum = 0.0
    for row in rows:
        with Image.open(row["image_path"]) as image:
            result = inspector.inspect(image, instrument_class=config["instrument_class"])
        maximum = max(maximum, float(result["anomaly_map"].max()))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Validation maps cannot establish a positive finite display maximum.")
    scale = {
        "min": 0.0,
        "max": maximum,
        "method": "maximum of raw maps on validation images only",
        "validation_images": len(rows),
        "manifest_sha256": digest,
    }
    config["heatmap_scale"] = scale
    write_json(config_path, config)
    return scale


def main() -> None:
    for instrument in ("instrument12", "instrument19"):
        scale = calibrate_display(
            Path("models") / instrument / "dinomaly_config.json",
            Path("data/processed") / instrument / "manifest.csv",
        )
        print(f"{instrument}: validation-only heatmap scale {scale}", flush=True)


if __name__ == "__main__":
    main()
