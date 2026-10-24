"""Read-only application readiness check. Model availability is reported separately."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.artifacts import sha256_file
from src.model_config import PROJECT_ROOT, ModelError, load_model_config, load_model_configs


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    scope: str = "application"


def collect_checks(root: Path = PROJECT_ROOT, config_path: Path | None = None) -> list[Check]:
    checks = [
        Check(
            "Python",
            "ok" if sys.version_info[:2] == (3, 12) else "error",
            f"{sys.version.split()[0]}; the combined model/application lock uses Python 3.12.",
        )
    ]
    modules = {
        "streamlit": "streamlit",
        "numpy": "numpy",
        "Pillow": "PIL",
        "torch": "torch",
        "torchvision": "torchvision",
        "anomalib": "anomalib",
        "timm": "timm",
        "pandas": "pandas",
        "mobile_sam": "mobile_sam",
    }
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        package, expected = line.strip().split("==", 1)
        if package not in modules:
            continue
        try:
            actual = importlib.metadata.version(package)
            importlib.import_module(modules[package])
            status = "ok" if actual.split("+")[0] == expected.split("+")[0] else "error"
            detail = f"Installed {actual}; required {expected}."
        except (ImportError, OSError) as exc:
            status, detail = (
                "error",
                f"Unavailable: {type(exc).__name__}. Install requirements.txt.",
            )
        checks.append(Check(package, status, detail))
    for name in ("app.py", "assets/styles.css", ".streamlit/config.toml", "src/ui.py"):
        present = (root / name).is_file()
        checks.append(
            Check(name, "ok" if present else "error", "Present." if present else "Missing.")
        )
    try:
        configs = [load_model_config(config_path)] if config_path else load_model_configs().values()
        for config in configs:
            checks.append(Check(config.instrument_class, "ok", "Metadata is valid.", "model"))
            for artifact in config.artifacts:
                if artifact.path.is_file():
                    matches = sha256_file(artifact.path) == artifact.sha256
                    status = "ok" if matches else "error"
                    detail = "SHA-256 verified." if matches else "Checksum mismatch."
                else:
                    status, detail = "pending", "No local artifact. No download attempted."
                checks.append(Check(artifact.path.name, status, detail, "model"))
    except (ModelError, OSError) as exc:
        checks.append(Check("Model configuration", "pending", str(exc), "model"))
    if (root / "src/inference.py").is_file():
        checks.append(
            Check(
                "Inference API",
                "ok",
                "Module present; real inference still needs acceptance testing.",
                "model",
            )
        )
    else:
        checks.append(Check("Inference API", "pending", "src/inference.py is missing.", "model"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-model", action="store_true", help="Fail if model checks are pending."
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    checks = collect_checks(config_path=args.config)
    app_ready = all(item.status == "ok" for item in checks if item.scope == "application")
    model_ready = all(item.status == "ok" for item in checks if item.scope == "model")
    if args.json:
        print(
            json.dumps(
                {
                    "application_ready": app_ready,
                    "model_files_ready": model_ready,
                    "real_inference_verified": False,
                    "checks": [asdict(item) for item in checks],
                },
                indent=2,
            )
        )
    else:
        for item in checks:
            print(f"[{item.status.upper()}] {item.name}: {item.detail}")
        print("\nThis check does not run inference or establish model accuracy.")
    return 0 if app_ready and (model_ready or not args.require_model) else 1


if __name__ == "__main__":
    raise SystemExit(main())
