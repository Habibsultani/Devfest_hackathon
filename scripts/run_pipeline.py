"""Freeze and reproduce the balanced two-instrument PatchCore experiment."""

import argparse
import json
from pathlib import Path
import shutil

from scripts.download_dataset import download_dataset
from scripts.prepare_balanced_dataset import prepare_balanced
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.test_robustness import test_robustness
from src.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", nargs="+", default=["instrument12", "instrument19"])
    parser.add_argument("--download", action="store_true", help="Download/resume selected pinned classes first.")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    if set(args.instruments) != {"instrument12", "instrument19"}:
        parser.error("The balanced experiment uses instrument12 and instrument19 together.")
    summary = {}
    protocol = {"seed": 42, "model": "PatchCore resnet18, layers2/3, 256, 5% coreset",
                "selection": "Instrument classes selected using counts and defect coverage only, before model inference.",
                "calibration": "Maximize validation balanced accuracy; ties lower FPR, then smaller threshold.",
                "cohorts": {}}
    for instrument in args.instruments:
        if args.download:
            download_dataset(Path("data/raw"), instrument)
        processed = Path("data/processed") / instrument
        cohort = prepare_balanced(Path("data/raw"), processed, instrument)
        protocol["cohorts"][instrument] = {"counts": cohort["counts"], "manifest_sha256": cohort["manifest_sha256"]}
    cohorts = list(protocol["cohorts"].values())
    if any(c["counts"] != cohorts[0]["counts"] for c in cohorts):
        raise ValueError("Instrument cohorts are not balanced.")
    # Written before fitting either model or viewing test predictions.
    write_json(Path("reports/protocol.json"), protocol)
    for instrument in args.instruments:
        processed = Path("data/processed") / instrument
        config = Path("models") / instrument / "model_config.json"
        report = Path("reports") / instrument
        train_model(processed / "manifest.csv", config)
        summary[instrument] = evaluate_model(processed / "manifest.csv", config, report)
        if not args.skip_robustness:
            test_robustness(processed / "manifest.csv", config, report)
        print(f"Completed {instrument}", flush=True)
    macro = {key: sum(summary[c][key] for c in summary) / len(summary)
             if all(summary[c][key] is not None for c in summary) else None
             for key in ("recall", "precision", "f1", "false_positive_rate", "balanced_accuracy", "image_auroc", "pixel_auroc")}
    write_json(Path("reports/balanced_metrics.json"), {"per_instrument": summary, "macro_average": macro})
    if Path("data/raw/inventory.json").exists():
        shutil.copyfile("data/raw/inventory.json", "reports/dataset_inventory.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
