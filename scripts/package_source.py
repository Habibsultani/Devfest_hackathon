"""Build a verified source ZIP with a file manifest and no local data or secrets."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".aws",
    ".codex",
    ".agents",
    ".idea",
    ".vscode",
    "data",
    "artifacts",
    "dist",
    "build",
}
EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".safetensors",
    ".pyc",
    ".log",
    ".zip",
}


def eligible_source(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in name:
        return False
    if set(part.casefold() for part in path.parts) & EXCLUDED_PARTS:
        return False
    if any(part.casefold().startswith((".venv-", "venv-")) for part in path.parts):
        return False
    if path.name == "secrets.toml" or path.suffix.casefold() in EXCLUDED_SUFFIXES:
        return False
    if path.name.startswith(".env") and path.name != ".env.example":
        return False
    if path.name.startswith("~$"):
        return False
    if path.match("inspection-*.csv") or path.as_posix() == "SOURCE_MANIFEST.json":
        return False
    return True


def git_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted(set(item for item in result.stdout.decode("utf-8").split("\0") if item))


def build_archive(
    root: Path,
    output: Path,
    names: list[str],
    *,
    revision: str,
    dirty: bool,
    force: bool = False,
    max_size_mb: float | None = None,
) -> dict:
    root = root.resolve()
    output = output.resolve()
    if output.exists() and not force:
        raise ValueError("The output already exists. Choose a new path or explicitly use --force.")
    if max_size_mb is not None and (not math.isfinite(max_size_mb) or max_size_mb <= 0):
        raise ValueError("The size limit must be greater than zero.")
    source = {}
    for name in sorted(set(names)):
        if not eligible_source(name):
            continue
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing a source path outside the project: {name}")
        if path.is_file() and path.resolve() != output:
            source[PurePosixPath(name.replace("\\", "/")).as_posix()] = path.read_bytes()
    if not source:
        raise ValueError("No eligible source files were found.")
    manifest = {
        "schema_version": 1,
        "git_revision": revision,
        "working_tree_has_changes": dirty,
        "file_count": len(source),
        "files": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in source.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, content in {
                **source,
                "SOURCE_MANIFEST.json": json.dumps(manifest, indent=2).encode("utf-8"),
            }.items():
                entry = zipfile.ZipInfo(f"medinspect-ai/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                # Unix executable permission for the optional shell launcher.
                entry.create_system = 3
                entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                archive.writestr(entry, content)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("The archive did not pass CRC verification.")
        if max_size_mb is not None and temporary.stat().st_size > max_size_mb * 1024 * 1024:
            raise ValueError("The source archive exceeds the requested upload-size limit.")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, output)
        temporary = None
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{digest}  {output.name}\n", encoding="ascii"
        )
        return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest, **manifest}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-size-mb", type=float, help="Use the event's actual form size limit.")
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
        suffix = "-working" if dirty else ""
        output = args.output or ROOT / "dist" / f"MedInspect_AI_Source_{revision[:8]}{suffix}.zip"
        result = build_archive(
            ROOT,
            output,
            git_files(ROOT),
            revision=revision,
            dirty=dirty,
            force=args.force,
            max_size_mb=args.max_size_mb,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
    print(
        f"Source ZIP: {result['path']}\nFiles: {result['file_count']}\n"
        f"Bytes: {result['bytes']}\nSHA-256: {result['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
