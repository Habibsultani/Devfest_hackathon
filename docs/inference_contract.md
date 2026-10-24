# Person One integration contract

Person Two calls only `src.inference.inspect_image(image)`. Person One owns the implementation, model preparation, preprocessing, calibration, and evaluation. This document defines the application boundary so those parts can be developed independently.

## Input

`image` is a fully decoded `PIL.Image.Image`, mode `RGB`, with EXIF orientation already applied. Transparent uploads are composited on white. The image keeps its original oriented dimensions and 8-bit pixel values. Application limits are 10 MB compressed and 12 million pixels. Ground-truth masks never enter the application or inference input.

Person One alone applies model-specific resize, cropping/padding, tensor conversion, and normalization exactly once. They must document the actual model input size and all transforms in their model documentation. If they crop/pad, they must undo the spatial transformation when returning the map.

## Output

Return a dictionary containing all four fields:

| Field | Type | Required meaning |
| --- | --- | --- |
| `anomaly_score` | finite Python or NumPy numeric scalar | Raw image-level model score. Boolean, string, NaN, and infinity are invalid. |
| `anomaly_map` | `numpy.ndarray` | Exactly `(image.height, image.width)`, finite real numbers, aligned to input coordinates. No batch/channel dimensions. |
| `requires_review` | Python/NumPy boolean | Exactly `anomaly_score >= configured_threshold`. |
| `model_version` | string | Exact match for the selected configuration's `model_version`. |

The app does not treat the score as a probability, infer named defect types, choose thresholds, or recalibrate maps. The returned map must fit finite float32 values for display. Missing fields, wrong shapes, nonfinite values, mismatched versions, and inconsistent flags fail closed with no saveable inspection.

## Required metadata

Person One fills `models/model_config.json` using the checked-in `.example.json` structure:

- `schema_version`: integer `1`.
- `model_version`: the exact release identifier returned by inference.
- `instrument_class`: the single evaluated instrument class, selected explicitly by the operator.
- `threshold`: a finite numeric threshold selected using validation data only.
- `artifact.path`: local file path, relative to the repository root unless absolute. Person One's loader must use this same artifact.
- `artifact.sha256`: actual SHA-256 of the saved model, including its normal memory bank.
- `artifact.url`: stable HTTPS direct download URL for a missing artifact, or `null` for a locally supplied model.
- `heatmap_scale.min` / `.max`: finite, increasing fixed display bounds suitable for the actual map output. Use bounds based on model semantics or validation data and document the choice. Do not estimate them from each uploaded image.

Compute an artifact digest in PowerShell with `Get-FileHash -Algorithm SHA256 models\your-model.ckpt`. The example intentionally contains invalid placeholders. It does not imply a selected model, class, threshold, or map range.

The app uses `MEDINSPECT_MODEL_CONFIG` if set. Person One's loader must honor the same environment setting, or read metadata through `src.model_config.load_model_config()`. No UI code modifies the checkpoint or threshold.

## Loading and exceptions

Person One must load the heavyweight model exactly once per release per process, for example with `functools.lru_cache` around a private loader keyed by model version/configuration. `inspect_image` must reuse the saved memory bank and must work after a restart without fitting again. The application caches its verified gateway with `st.cache_resource` and serializes calls, but caching a function reference cannot prevent a backend from reloading internally on each call.

Raise a useful exception if loading, preprocessing, or inference fails. The app catches it, logs the traceback on the server, and displays an explicit failure without converting it into a normal or defective result. No synthetic fallback belongs in `src/inference.py`.

## Handoff and joint acceptance

Person One supplies the API module, transforms, trained artifact and checksum/URL, completed metadata, exact package versions, dataset documentation, held-out evaluation, and unseen normal/defect examples with their expected ground truth. Verify together:

1. A fresh process reloads the real memory bank without fitting or downloading training data.
2. Multiple unseen images yield genuine model results and maps aligned to the originals.
3. The application shows the same score, threshold flag, and map as a direct API call.
4. Invalid input and unsupported instrument classes produce explicit errors.
5. Human decisions and CSV fields reflect the selected image, model, IDs, and timestamps.
6. The public deployment can repeat that entire flow after restart.

UI fixture tests establish application behavior only. Person One's evaluated evidence establishes model performance.
