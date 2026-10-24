"""Start a temporary real app, check HTTP/browser behavior, and stop its process tree."""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    logs = ROOT / "artifacts/smoke"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medinspect-smoke-") as temporary:
        environment = os.environ.copy()
        environment["MEDINSPECT_MODEL_CONFIG"] = str(Path(temporary) / "model-unavailable.json")
        environment["MEDINSPECT_BASE_URL"] = f"http://127.0.0.1:{port}"
        with (logs / "streamlit.log").open("w", encoding="utf-8") as handle:
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
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Streamlit exited early; see artifacts/smoke/streamlit.log."
                        )
                    try:
                        with urllib.request.urlopen(
                            environment["MEDINSPECT_BASE_URL"] + "/_stcore/health", timeout=1
                        ) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("The app did not become healthy within 30 seconds.")
                print(
                    "HTTP startup check passed. Model configuration is intentionally unavailable."
                )
                if args.browser:
                    subprocess.run(
                        [
                            environment.get("MEDINSPECT_RUNTIME_NODE", "node"),
                            "scripts/check_browser.cjs",
                        ],
                        cwd=ROOT,
                        env=environment,
                        check=True,
                    )
            finally:
                stop_process_tree(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
