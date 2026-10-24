# Application operations and delivery

The application and trained model are integrated. Use the combined dependency lock and verify both model artifacts before analysis. Public deployment, final slides and submission recording remain team tasks.

## Start and check

On Windows, run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1`. The script reuses `.venv`, or creates it when Python 3.12 is available, checks application requirements, and runs the local server. Use `-Setup` to install dependencies, `-CheckOnly` for diagnostics, or `-Port 8503` for a second preview. Pass `-Python 'C:\path\to\python.exe'` if Python is not on PATH.

On macOS/Linux, use `bash scripts/start.sh`, optionally with `--setup`, `--check`, or `--port 8503`. Set `PYTHON` to a compatible executable when creating a new environment. These launchers change no global Python packages.

Run `python -m scripts.doctor` inside the environment for a readable readiness report. `--json` produces machine-readable status. Default success means the application environment is ready, even if model files are pending. `--require-model` additionally requires valid metadata, matching local Dinomaly and MobileSAM artifact checksums, and an inference module. It performs no artifact download, model fitting, or inference. Even a fully successful report must be followed by real unseen-image testing.

`python -m scripts.smoke_app` launches a separate temporary app on a free loopback port, checks its HTTP health, then stops the process it created. The smoke environment deliberately uses a missing configuration, so it cannot download or invoke a model. `--browser` additionally runs `scripts/check_browser.cjs` using the configured Playwright/Node tooling. Logs stay under ignored `artifacts/smoke/`. This test leaves the normal preview on port 8501 alone.

Run `python -m scripts.check_real_app` for real inference, original-size map, browser review and CSV parity checks. It requires both model artifacts, the two downloaded original dataset classes, and Playwright. See README for browser runtime variables. This is separate from the unavailable-model smoke check above.

## Review records

Save the operator decision before selecting **Start new inspection**. That action retains the batch and supported class, clears the image and instrument ID, and keeps all saved records. Changing an image or metadata still invalidates an unsaved active analysis.

Session report supports text search across IDs, class, model, source filename, and notes, plus an operator-decision filter. The table lists the latest matching record first. **Download inspection CSV** always includes every saved record. Search filters never remove records or restrict the export. Download before leaving or restarting because history is session-only.

## Source package

Run `python -m scripts.package_source` from the environment. It reads Git's tracked and eligible untracked source list, excludes private/generated content, and writes a new ZIP with a companion SHA-256 file. Each archive has one `medinspect-ai/` root and a `SOURCE_MANIFEST.json` containing the commit, working-tree status, file sizes, and digests.

The packager rejects paths outside the project, symlinks, checkpoint extensions, environment files, credentials, local datasets, and inspection exports. It verifies ZIP integrity before installing the output. Fixed entry dates and sorted files make identical source produce identical ZIP content. Existing outputs require a new filename or an explicit `--force`. Use `--max-size-mb NUMBER` only with the form's actual published limit. Do not assume the generated ZIP meets unknown event rules.

GitHub Actions uploads the source package as a workflow artifact. Packaging a working tree marks its manifest as changed; package a clean, reviewed commit for the final handoff. Verify the reported archive SHA-256 after transferring it.

## Container and release

See [deployment](deployment.md) for the Docker commands and Streamlit Community Cloud steps. The container recipe and HTTP check can be verified before the model exists. A healthy UI process does not establish a ready AI backend. The integrated PR can be reviewed after the real-inference/browser gate passes.

If the model configuration, artifact, or dependencies change, restart the app and perform the agreed unseen-image checks. If a release fails, return to the previous reviewed image or Git revision through the normal review process. Never replace a failed inference with a fabricated result.
