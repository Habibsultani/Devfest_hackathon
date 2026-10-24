"""Retrieve a team model artifact and verify its configured SHA-256 before use."""

import argparse
import json
import urllib.request
from pathlib import Path

from src.artifact_io import SafeAssetRedirect as SafeAssetRedirect
from src.artifact_io import github_token
from src.dataset import file_sha256


def fetch_model(config_path: Path, authenticate: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config_path.parent / config["artifact_path"]
    if target.exists() and file_sha256(target) == config["artifact_sha256"]:
        return target
    url = config.get("artifact_api_url") if authenticate else config.get("artifact_download_url")
    if not url:
        raise ValueError(
            "No hosted artifact URL configured. Train locally or obtain the "
            "team's matching artifact."
        )
    headers = {"User-Agent": "MedInspect-model-retrieval", "Accept": "application/octet-stream"}
    if authenticate:
        if not url.startswith("https://api.github.com/"):
            raise ValueError(
                "Authenticated downloads must use the configured GitHub API asset URL."
            )
        headers["Authorization"] = "Bearer " + github_token()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    opener = urllib.request.build_opener(SafeAssetRedirect())
    with (
        opener.open(urllib.request.Request(url, headers=headers), timeout=120) as response,
        temporary.open("wb") as stream,
    ):
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if file_sha256(temporary) != config["artifact_sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError("Downloaded artifact checksum mismatch; model was not installed.")
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="instrument12")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--include-region",
        action="store_true",
        help="Also retrieve the optional region experiment artifacts.",
    )
    parser.add_argument(
        "--github-auth",
        action="store_true",
        help="Use GITHUB_TOKEN or existing Git credentials for this private repository.",
    )
    args = parser.parse_args()
    models = Path(__file__).resolve().parents[1] / "models"
    registry = json.loads((models / "model_registry.json").read_text(encoding="utf-8"))
    selected = list(registry) if args.all else [args.instrument]
    for instrument in selected:
        if instrument not in registry:
            raise ValueError(f"Unsupported instrument class: {instrument}")
        print(
            f"Verified {instrument}: {fetch_model(models / registry[instrument], args.github_auth)}"
        )
        if args.include_region:
            region_config = (models / registry[instrument]).parent / "region/model_config.json"
            print(
                f"Verified {instrument} region experiment: "
                f"{fetch_model(region_config, args.github_auth)}"
            )


if __name__ == "__main__":
    main()
