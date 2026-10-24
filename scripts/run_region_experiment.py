"""Record the optional background experiment without promoting on test scores."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_model import evaluate_model
from scripts.train_model import train_model
from src.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse", action="store_true", help="Evaluate already-trained region artifacts."
    )
    args = parser.parse_args()
    results = {
        "status": "development comparison after baseline test inspection",
        "method": "Fixed RGB blue-background suppression; labels never generate region masks.",
        "promotion": "No automatic promotion; active registry retains the balanced baseline.",
        "per_instrument": {},
    }
    for instrument in ("instrument12", "instrument19"):
        manifest = Path("data/processed") / instrument / "manifest.csv"
        config = Path("models") / instrument / "region/model_config.json"
        if not args.reuse:
            train_model(manifest, config, region_mask=True)
        candidate = json.loads((config.parent / "training_record.json").read_text())[
            "validation_metrics"
        ]
        baseline = json.loads((config.parent.parent / "training_record.json").read_text())[
            "validation_metrics"
        ]
        # Record the validation-only comparison before evaluating test images.
        results["per_instrument"][instrument] = {
            "baseline_validation": baseline,
            "region_validation": candidate,
            "validation_improved": candidate["balanced_accuracy"] > baseline["balanced_accuracy"],
        }
        write_json(Path("reports/region_experiment.json"), results)
        results["per_instrument"][instrument]["test"] = evaluate_model(
            manifest, config, Path("reports/region") / instrument
        )
    write_json(Path("reports/region_experiment.json"), results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
