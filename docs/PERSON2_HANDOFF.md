# Person 2 model integration handoff

Person 1 changed the experiment to instrument12 and instrument19. Each has 12
normal training images, four normal plus four defect validation images, and four
normal plus four defect test images. Both instrument types therefore have equal
counts, and both evaluation folds are balanced by label. Related source-image
groups stay together; no defect mask enters model inference.

The original whole-image PatchCore results are preserved, including failures.
The active replacement reuses Anomalib's official Dinomaly components,
MobileSAM instrument segmentation, native-resolution image tiles and rank-8
encoder LoRA adapters. Check `models/model_registry.json` for the currently active
configs; do not select a backend based on a filename from an earlier message.
The active version is `medinspect-dinomaly-sam-lora-v4`. Training completed 1,000
steps; validation selected step 200. All 34 CPU/unit/integration checks passed,
including fresh-process reload. Our 16-image balanced development comparison
produced 14 correct decisions, one false alarm and one missed defect (macro F1
0.875). Instrument12 was 8/8 correct; instrument19 was 6/8. These are our measured
results, not the paper's reported performance.

## Application interface

Keep using `from src.inference import inspect_image, supported_classes`.
Ask the operator to select a value returned by `supported_classes()`, then call:

```python
result = inspect_image(image, instrument_class=selected_class)
```

The dictionary still contains anomaly_score, anomaly_map, requires_review,
model_version, threshold and instrument_class. The map is a float32 NumPy array
aligned to the EXIF-oriented original image. Do not resize the image before
calling the model, normalize it again, hardcode a threshold, or treat a raw score
as a probability. Display the EXIF-oriented original alongside its heatmap.

## Exact setup and retrieval

```powershell
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m scripts.setup_research_models
python -m scripts.fetch_model --all --github-auth
python -m scripts.inspect_image path\to\new_image.png --instrument instrument19
```

The unified Dinomaly weights are about 150.5 MB; MobileSAM is about 40.7 MB.
Both declared classes share the Dinomaly weights, with distinct validation
thresholds in their configs. `fetch_model` retrieves the shared artifact only once.
The GitHub artifact release is a private team draft; your Git account or
GITHUB_TOKEN must have repository access. Do not commit credentials.

## Model and dependency changes

Use the tested dependency lock and retrieve matching, checksum-verified artifacts.
The tiled backend needs both its saved Dinomaly artifact and the separately
configured MobileSAM checkpoint in `models/sam_config.json`. Large weights remain
outside Git. A CPU build supports inference; the optional CUDA environment is
for training/faster inference on an NVIDIA GPU. Version-specific loading and
artifact checks happen in the adapter.

Keep heavy model loading cached. Native-resolution tiling can take substantially
longer than the previous single 256-pixel forward pass. Show a progress/loading
state and measure deployment latency/memory using real images before selecting
a cloud instance. SAM masks are computed from the input image and cached locally;
they are not ground-truth defect masks.

## Workflow and claims

Missing/incompatible weights, unsupported declared classes, segmentation failure
or inference errors must remain visible errors. Never substitute a normal result.
Models are configured for the selected instrument classes photographed on
blue/purple cloth. They do not automatically recognize other tools or reject all
unrelated photographs.

Keep the human decision and note separate from the AI flag. CSV records should
include declared class, model version, score, threshold, human decision and
timestamp. Use suspicious-region wording; no named defect diagnosis or guarantee
of device safety/sterility is supported. This is a research demonstration.

The same test cohorts were already viewed for earlier models. Any replacement
comparison on them is development evidence; it is not a new untouched final test.
Consult `docs/evaluation.md`, the current model reports and
`docs/research_sources.md` when preparing slides and deployment notes.
