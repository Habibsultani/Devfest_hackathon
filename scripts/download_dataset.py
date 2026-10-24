"""Download Purdue classes at a fixed revision, verifying source checksums."""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "purdueviperlab/Surgical_Instruments_Unsupervised_Defect_Detection"
REVISION = "e791b82e9616889e356b4cf7d40d6b56ba87b0c5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_dataset(output: Path, instrument: str, revision: str = REVISION) -> dict:
    if not re.fullmatch(r"instrument(?:[1-9]|1[0-9]|20)", instrument):
        raise ValueError("Select instrument1 through instrument20.")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character dataset revision.")
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{instrument}?recursive=true&limit=1000"
    items = []
    while api:
        with urllib.request.urlopen(api, timeout=60) as response:
            items.extend(json.load(response))
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>; rel="next"', link)
        api = match.group(1) if match else None
    files = [item for item in items if item["type"] == "file"]
    if not files:
        raise ValueError("The selected class contains no files.")
    output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != instrument:
            raise ValueError("Unexpected path in dataset listing.")
        target = output / relative
        expected = item.get("lfs", {}).get("oid")
        if target.exists() and expected and sha256(target) == expected:
            return {"path": item["path"], "sha256": expected, "bytes": target.stat().st_size}
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(item['path'])}"
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    temporary.open("wb") as stream,
                ):
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                digest = sha256(temporary)
                if temporary.stat().st_size != item["size"] or (expected and digest != expected):
                    raise ValueError(f"Checksum or size mismatch: {item['path']}")
                temporary.replace(target)
                return {"path": item["path"], "sha256": digest, "bytes": target.stat().st_size}
            except (OSError, ValueError):
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, files))
    card_url = f"https://huggingface.co/datasets/{REPO}/raw/{revision}/README.md"
    with urllib.request.urlopen(card_url, timeout=60) as response:
        (output / instrument / "DATASET_CARD.md").write_bytes(response.read())
    provenance = {
        "repository": REPO,
        "revision": revision,
        "instrument_class": instrument,
        "license": "CC-BY-NC-ND-4.0",
        "files": records,
    }
    (output / f"{instrument}_source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--instrument", default="instrument1")
    parser.add_argument("--instruments", nargs="+", help="Download a specific list of classes.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 20 classes, with bounded parallel downloads.",
    )
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    instruments = (
        [f"instrument{i}" for i in range(1, 21)]
        if args.all
        else args.instruments or [args.instrument]
    )

    def fetch_class(instrument):
        result = download_dataset(args.output, instrument, args.revision)
        print(
            f"Verified {len(result['files'])} files for {instrument} at {args.revision}.",
            flush=True,
        )
        return {
            "instrument_class": instrument,
            "files": len(result["files"]),
            "bytes": sum(r["bytes"] for r in result["files"]),
            "folder_counts": dict(
                Counter(str(Path(r["path"]).parent).replace("\\", "/") for r in result["files"])
            ),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        inventory = list(pool.map(fetch_class, instruments))
    (args.output / "inventory.json").write_text(
        json.dumps({"revision": args.revision, "classes": inventory}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
