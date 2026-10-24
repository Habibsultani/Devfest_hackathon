"""Compare real inference with browser analysis, operator review and downloaded CSV.

Requires the verified weights, original dataset files and Playwright. No model is mocked.
"""

import csv
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.smoke_app import stop_process_tree
from src.inference import inspect_image
from src.model_config import load_model_configs
from src.model_gateway import load_gateway
from src.uploads import decode_upload

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ("instrument12", "test/good/IMG_1017.jpg", "Accept"),
    ("instrument12", "test/blood/0013_IMG_1020.png", "Flag for review"),
    ("instrument19", "train/good/IMG_4732.png", "Accept"),
    ("instrument19", "test/blood/0022_IMG_1042.png", "Flag for review"),
    ("instrument19", "test/corrosion/0027_IMG_1044.png", "Inconclusive"),
]


def main() -> None:
    output = ROOT / "artifacts/integration"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.pop("MEDINSPECT_MODEL_CONFIG", None)
    configs = load_model_configs()
    cases = []
    for index, (instrument, relative, decision) in enumerate(SAMPLES):
        config = configs[instrument]
        path = ROOT / "data/raw" / instrument / relative
        content = path.read_bytes()
        image = decode_upload(content, path.name)
        start = time.perf_counter()
        direct = inspect_image(image, instrument_class=instrument, config_path=config.config_path)
        result = load_gateway(config).inspect(image, instrument)
        np.testing.assert_allclose(result.anomaly_map, direct["anomaly_map"], atol=1e-6)
        assert abs(result.anomaly_score - direct["anomaly_score"]) < 1e-6
        assert result.requires_review == direct["requires_review"]
        assert result.anomaly_map.shape == (image.height, image.width)
        case = {
            "instrument_id": f"REAL-{index + 1:03d}",
            "instrument_class": instrument,
            "path": str(path),
            "source_path": f"{instrument}/{relative}",
            "image_sha256": hashlib.sha256(content).hexdigest(),
            "size": list(image.size),
            "anomaly_score": result.anomaly_score,
            "threshold": config.threshold,
            "requires_review": result.requires_review,
            "model_version": config.model_version,
            "artifact_sha256": config.artifact_sha256,
            "config_sha256": config.config_sha256,
            "operator_decision": decision,
            "operator_note": "Real model integration check; human assessment remains required.",
        }
        cases.append(case)
        print(
            f"Direct/gateway verified: {case['source_path']} "
            f"score={result.anomaly_score:.9f} review={result.requires_review} "
            f"({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    # This sample caught the previous default-class routing bug. Preserve the known miss.
    regression = cases[-1]
    assert configs["instrument12"].threshold < regression["anomaly_score"]
    assert regression["anomaly_score"] < configs["instrument19"].threshold
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = os.environ.copy()
    environment["MEDINSPECT_BASE_URL"] = f"http://127.0.0.1:{port}"
    with (output / "streamlit.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--server.headless=true",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("App exited. See artifacts/integration/streamlit.log.")
                try:
                    with urllib.request.urlopen(
                        environment["MEDINSPECT_BASE_URL"] + "/_stcore/health", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("Streamlit did not become healthy.")
            subprocess.run(
                [
                    environment.get("MEDINSPECT_RUNTIME_NODE", "node"),
                    "scripts/check_real_browser.cjs",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
        finally:
            stop_process_tree(process)
    with (output / "inspections.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == len(cases)
    for case, record in zip(cases, records, strict=True):
        for key in (
            "instrument_id",
            "instrument_class",
            "model_version",
            "artifact_sha256",
            "config_sha256",
            "image_sha256",
            "operator_decision",
            "operator_note",
        ):
            assert record[key] == case[key], (key, record[key], case[key])
        assert abs(float(record["anomaly_score"]) - case["anomaly_score"]) < 1e-6
        assert float(record["threshold"]) == case["threshold"]
        assert record["requires_review"] == str(case["requires_review"])
        assert record["source_filename"] == Path(case["path"]).name
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "result": "passed",
        "scope": "Real CPU inference, native-size maps, browser upload, review, CSV parity",
        "mocked_inference": False,
        "class_switch_invalidates_result": True,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "anomalib", "streamlit", "numpy", "Pillow", "pandas")
        },
        "cases": [{key: value for key, value in case.items() if key != "path"} for case in cases],
        "limitations": (
            "Integration acceptance, not an independent accuracy evaluation. Includes the "
            "known instrument19 corrosion false negative. Existing segmentation cache may be warm."
        ),
    }
    report_path = ROOT / "reports/integration_smoke.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Real browser/CSV acceptance passed for {len(cases)} images. {report_path}")


if __name__ == "__main__":
    main()
