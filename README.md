# MedInspect AI

Integrated Streamlit application and real model pipeline for the Devfest hackathon.
Select **instrument12 or instrument19**, upload an image, review the score and
aligned heatmap, record an independent operator decision, and export the session CSV.
The model uses equal training counts and balanced normal/defect evaluation cohorts.

The active model is **Dinomaly + MobileSAM + high-resolution patches + rank-8
LoRA**, reusing the published approach and official implementation components.
It achieved **14/16 correct, macro F1 0.875**, with one false alarm and one missed
defect on the small balanced development comparison. Earlier failed PatchCore
experiments are retained rather than replaced with claimed success.

This research prototype screens visible anomalies. It does not certify device
quality, safety or sterility. Application and inference branches are integrated;
public hosting and final submission materials remain team tasks.

## Setup

Verified on Windows with Python 3.12.10 and CPU PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m scripts.setup_research_models
python -m scripts.fetch_model --all --github-auth
python -m scripts.doctor --require-model
python -m streamlit run app.py
```

`requirements.txt` pins direct dependencies. `requirements-lock.txt` records the
complete tested environment. Anomalib's imported video/CLIP dependencies and the
Hub API version expected by timm are included.

Install Git before setup: MobileSAM is pinned to an upstream Git revision.
The private draft release requires `GITHUB_TOKEN` with repository access or your
existing Git credentials. Tokens stay outside source/config files. The app can
also retrieve missing artifacts on first analysis; both Dinomaly and MobileSAM
are checked against their saved SHA-256 before loading. A corrupt local file is
rejected rather than silently replaced. The class registry supplies both configs,
their distinct thresholds and validation-derived fixed heatmap display scales.

Windows/macOS/Linux launchers and Docker instructions are in
[operations](docs/operations.md) and [deployment](docs/deployment.md).
Uploads accept JPEG/PNG up to 20 MB and 40 million pixels, preserving native size.
Select the instrument class manually; changing class or input clears stale results.

## Reproduce the model

```powershell
python -m scripts.download_dataset --instruments instrument12 instrument19
python -m scripts.run_research_pipeline
python -m pytest -q
```

The pipeline freezes both source-group manifests and the protocol before fitting
models. Each model uses normal training images only. Thresholds use balanced
validation labels; reserved test images never select the threshold.

| Split per instrument | Normal | Defect |
| --- | ---: | ---: |
| Training | 12 | 0 |
| Validation | 4 | 4 |
| Test | 4 | 4 |

Counts match across instrument types. Sampling uses no replacement and related
source groups stay in one fold. Four variants of a source photograph are not four
independent captures; this remains a small research benchmark.

Training is recommended on NVIDIA CUDA: create a separate environment and install
`requirements-gpu.txt`, or the complete `requirements-gpu-lock.txt`. The tested
run used an RTX 3060 6 GB for 1,000 steps; validation selected step 200. CPU
inference is verified. Saved artifacts contain the encoder, decoder and adapters;
only the separate MobileSAM checkpoint is also required for inference. Dataset images, masks, weights and local overlays are
excluded from Git.

## Inference and artifacts

```powershell
python -m scripts.setup_research_models
python -m scripts.fetch_model --all --github-auth
python -m scripts.inspect_image path\to\new_image.png --instrument instrument19
```

Artifact downloads verify the configured SHA-256. For this private repository,
`--github-auth` uses `GITHUB_TOKEN` or existing Git credentials without storing a
token in the project. If no hosted artifact is available, reproduce it locally.

```python
from PIL import Image
from src.inference import inspect_image

result = inspect_image(Image.open('new_image.png'), instrument_class='instrument12')
score = result['anomaly_score']
heatmap = result['anomaly_map']  # original-size NumPy float32 array
review = result['requires_review']
```

The operator declares the instrument class. Models do not recognize instrument
types or classify named defects. Scores are distances, not probabilities.
Missing models or inference failures raise clear errors; no fabricated result
is substituted. Read [the integration contract](docs/inference_contract.md).

## Evidence

- [Current Dinomaly results](reports/dinomaly_metrics.json) contains per-instrument and equal-weight macro metrics.
- [Person 2 handoff](docs/PERSON2_HANDOFF.md) lists integration changes and exact setup commands.
- [Research sources](docs/research_sources.md) credits reused code and records deviations from the paper.
- [Model protocol](reports/dinomaly_protocol.json) records selection and calibration rules.
- `reports/dinomaly/instrument12/` and `reports/dinomaly/instrument19/` contain current predictions, confusion/ROC plots and audits.
- `reports/instrument12/`, `reports/instrument19/` and `reports/balanced_metrics.json` preserve the failed balanced PatchCore baseline and its stress tests.
- [Dataset notes](docs/dataset.md) describe provenance, masks, balance and restrictions.
- [Evaluation notes](docs/evaluation.md) explain calibration and limitations.
- `reports/baseline/` preserves the first unbalanced experiment.

| Current Dinomaly model | TP | FP | TN | FN | Image AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| instrument12 | 4 | 0 | 4 | 0 | 1.000 |
| instrument19 | 3 | 1 | 3 | 1 | 0.8125 |

The comparison has eight normal and eight defect images. Several defects share
source captures, and these test images were viewed during earlier experiments;
this is development evidence, not independent clinical validation. Pixel AUROC
was 0.9941 and 0.9785. More independent data is needed for final validation.

`python -m scripts.run_pipeline` reproduces the older PatchCore baseline;
`python -m scripts.run_region_experiment` reproduces its color-mask experiment.
Neither is the current default model. See the archived reports for negative results.

Tests cover balance quotas, leakage rejection, preprocessing/map alignment,
calibration, input errors, artifact integrity, dynamic predictions and fresh-process
reload. Integration tests skip on a source-only checkout without data and models.

For actual application acceptance, after downloading the two original dataset
classes, run `python -m scripts.check_real_app`. Set `MEDINSPECT_NODE_MODULES` to
the directory containing Playwright and optionally `MEDINSPECT_RUNTIME_NODE` to
the Node executable. Use `MEDINSPECT_BROWSER_CHANNEL=chromium` for Playwright's
installed Chromium; Chrome is the local default. This starts a fresh Streamlit
process, compares direct inference with the gateway, uploads five real images,
checks class switching, saves human decisions and verifies the downloaded CSV.
The instrument19 corrosion sample intentionally retains the known model miss.
Screenshots and CSV stay in ignored `artifacts/integration/`; numerical evidence
is written to `reports/integration_smoke.json`. This checks integration, not new accuracy.

Source: [Purdue VIPER dataset](https://huggingface.co/datasets/purdueviperlab/Surgical_Instruments_Unsupervised_Defect_Detection),
revision `e791b82e9616889e356b4cf7d40d6b56ba87b0c5`, author's declared
CC-BY-NC-ND-4.0. Method: [the dataset paper](https://arxiv.org/html/2509.21561v2),
[official Dinomaly](https://github.com/guojiajeremy/Dinomaly),
[Anomalib](https://github.com/open-edge-platform/anomalib/tree/v2.2.0/src/anomalib/models/image/dinomaly)
and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM). Defects are simulated;
dataset performance does not establish clinical effectiveness.
