# MedInspect AI

AI-assisted visible-defect screening for surgical instruments, with human review and session CSV export. Research prototype: results do **not** certify medical-device safety, quality, or sterility.

## Current status

Person Two's application is implemented on `feature/web-app`: image upload validation, the inference integration boundary, fixed-scale anomaly overlays, independent operator decisions, session inspection records, CSV export, automated tests, and deployment documentation.

**Real model integration and public deployment are pending Person One's deliverables.** There is no trained checkpoint, inference implementation, dataset, or measured model accuracy in this branch. The app launches without them and clearly disables analysis. Test fixtures exist only under `tests/` and never serve predictions in the app. No public demo URL has been verified yet.

## Run locally

Use Python 3.11 or 3.12. The application dependencies were verified on Python 3.12.14. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Until the trained model is integrated, you can inspect the interface and upload validation, but cannot generate an AI result or save an inspection.

## Connect Person One's model

1. Person One supplies `src/inference.py` with `inspect_image(image)`, their preprocessing code, and exact compatible inference dependencies. Merge those dependencies into `requirements.txt` and rerun the integration checks. The UI layer itself does not install or depend on Anomalib.
2. Copy `models/model_config.example.json` to `models/model_config.json`. Replace every placeholder and `null` numeric value with Person One's real model metadata. The example intentionally fails validation.
3. Supply the trained artifact at the configured path. For deployment, set `artifact.url` to a stable HTTPS download URL and set `artifact.sha256` to the actual digest. The app downloads a missing artifact, verifies its checksum, then allows Person One's loader to use it. Existing files with a wrong checksum are rejected.
4. Verify retrieval independently:

   ```powershell
   .\.venv\Scripts\python.exe -m scripts.fetch_model
   ```

5. Start the app. Enter batch and instrument IDs, explicitly confirm the supported class, upload an unseen image, then select **Analyze instrument**. Review the score and overlay, choose an operator decision, save the inspection, and download the CSV from **Session report**.

The required return value is:

```python
{
    "anomaly_score": float,          # Finite raw model score, not a probability
    "anomaly_map": numpy.ndarray,   # Finite H x W array aligned to the supplied image
    "requires_review": bool,        # score >= model_config threshold
    "model_version": str,           # Matches model_config exactly
}
```

The app sends one EXIF-oriented PIL RGB image at its original size. **Person One alone performs model resizing, tensor transforms, and normalization.** They must cache the heavyweight model inside their inference module. The app caches the verified backend and serializes calls to its shared model. See the complete [integration contract](docs/inference_contract.md) and [handoff checklist](docs/person_two_status.md).

The default configuration is `models/model_config.json`. Set `MEDINSPECT_MODEL_CONFIG` to a different trusted JSON path if needed. Artifact paths in JSON are relative to the repository root, regardless of the configuration file's location.

## Review and report behavior

- Supports real JPEG/PNG content, up to 10 MB and 12 megapixels. Damaged files, animation, and unsupported formats fail clearly.
- Requires batch ID, instrument ID, and operator confirmation of the model's supported instrument class. The app does not infer instrument class from the image.
- Uses a fixed, model-specific heatmap scale from Person One. It never stretches colors separately for each image. Display clipping and overlay opacity do not change the score or review threshold.
- Keeps **Accept**, **Flag for review**, and **Inconclusive** separate from the AI flag. An inconclusive decision requires a note. Each analysis can be saved once.
- Clears the active result when the image, IDs, instrument class, or model configuration changes. A failed analysis cannot leave a saveable previous result.
- Exports IDs, UTC timestamps, image digest, model/config/artifact identifiers, raw score, threshold, AI flag, latency, operator decision, and note. CSV uses UTF-8 with a BOM. Text starting with spreadsheet formula characters receives a leading apostrophe.
- Stores images and inspection records in server memory for the current Streamlit session. There is no database or durable multi-user history. Export before leaving or restarting.

## Test

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions runs the same checks on Python 3.11 and 3.12. Tests cover uploads, checksum verification, contract rejection, heatmap alignment, review validation, CSV integrity, and Streamlit session behavior. UI tests inject isolated fixtures and **do not establish PatchCore accuracy or real model compatibility**. Real unseen-image acceptance remains a joint integration gate with Person One.

## Deployment and collaboration

Follow [deployment instructions](docs/deployment.md) for Streamlit Community Cloud. The selected repository is [`Habibsultani/Devfest_hackathon`](https://github.com/Habibsultani/Devfest_hackathon), entry point `app.py`, target branch `main` after review. Publish a real demo URL here only after the full unseen-image workflow works in a private browser window.

Work on `feature/web-app`; Person One uses `feature/ai-model`. Open small pull requests, run checks, and have the other teammate review before merging. Never commit datasets, model blobs, secrets, or inspection exports. The development model configuration is ignored; commit a reviewed, nonsecret release configuration explicitly when preparing cloud deployment.

## Project map

```text
app.py                          Streamlit upload, results, review, and session report
src/uploads.py                  File decoding and limits, no model transforms
src/model_config.py             Trained-model metadata validation
src/artifacts.py                HTTPS retrieval and SHA-256 verification
src/model_gateway.py            Person One's API integration and output checks
src/visualization.py            Fixed-scale heatmaps and overlays
src/reporting.py                Human review records and CSV
scripts/fetch_model.py          Artifact retrieval / verification command
models/model_config.example.json Intentionally incomplete configuration reference
tests/                          Unit and Streamlit integration tests
docs/architecture.md            Components, boundaries, and session behavior
docs/inference_contract.md      Exact handoff for Person One
docs/deployment.md              Hosting and public-demo acceptance steps
docs/demo_script.md             Actual-demo recording and submission preparation
docs/person_two_status.md       Completed work and remaining integration gates
docs/MedInspect_AI_Application.pptx Seven-slide application-stage presentation
docs/presentation.md            Presentation status and optional rebuild tooling
```

Person One owns `src/inference.py`, `src/preprocessing.py`, training/data/evaluation scripts, `docs/dataset.md`, `docs/evaluation.md`, model artifacts, and measured results. Those files have deliberately not been implemented here.

## Limitations and dependencies

Background, lighting, viewpoint, and out-of-distribution instruments can cause false alarms or missed defects. Scores do not identify named defect types. A low score cannot guarantee safety or sterility. Dataset provenance, simulated-defect caveats, threshold calibration, and measured performance must accompany Person One's model before submission.

Application libraries: [Streamlit](https://docs.streamlit.io/) (Apache-2.0), [NumPy](https://numpy.org/) (BSD-3-Clause), and [Pillow](https://pillow.readthedocs.io/) (MIT-CMU). Verify dataset usage restrictions and inference-library/model licenses with Person One before publishing.
