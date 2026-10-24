"""Promote on validation evidence, then report the reserved development comparison."""

import json
from pathlib import Path

from scripts.evaluate_model import evaluate_model
from src.dataset import write_json


def main():
    record = json.loads(Path("models/dinomaly/training_record.json").read_text())
    selected = record["selected_validation"]
    # Choose the replacement using validation only, before reading its test scores.
    promote = all(
        selected[c]["metrics"]["f1"] > 0 and selected[c]["metrics"]["false_positive_rate"] == 0
        for c in ("instrument12", "instrument19")
    )
    if promote:
        write_json(
            Path("models/model_registry.json"), {c: f"{c}/dinomaly_config.json" for c in selected}
        )
    results = {}
    for instrument in selected:
        results[instrument] = evaluate_model(
            Path("data/processed") / instrument / "manifest.csv",
            Path("models") / instrument / "dinomaly_config.json",
            Path("reports/dinomaly") / instrument,
        )
        results[instrument]["evaluation_status"] = (
            "Development comparison; these test images were previously inspected for PatchCore."
        )
        write_json(Path("reports/dinomaly") / instrument / "metrics.json", results[instrument])
    keys = (
        "recall",
        "precision",
        "f1",
        "false_positive_rate",
        "balanced_accuracy",
        "image_auroc",
        "pixel_auroc",
    )
    macro = {
        key: sum(m[key] for m in results.values()) / len(results)
        if all(m[key] is not None for m in results.values())
        else None
        for key in keys
    }
    write_json(
        Path("reports/dinomaly_metrics.json"),
        {
            "promoted_from_validation": promote,
            "selected_step": record["selected_step"],
            "per_instrument": results,
            "macro_average": macro,
        },
    )
    print(
        json.dumps(
            {"promoted": promote, "per_instrument": results, "macro_average": macro}, indent=2
        )
    )


if __name__ == "__main__":
    main()
