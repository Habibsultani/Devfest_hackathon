# Inference interface for Person 2

Person 1 supplies preprocessing, saved models, thresholds and evaluation. Person 2
supplies Streamlit, overlays, human review, CSV records and deployment. The
supported classes are instrument12 and instrument19. They share the unified
Dinomaly artifact and use class-specific thresholds.

```python
from PIL import Image, ImageOps
from src.inference import inspect_image, supported_classes

classes = supported_classes()
image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert('RGB')
result = inspect_image(image, instrument_class='instrument19')
```

Ask the operator to declare a supported class. Models do not identify instrument
types. Omitting a class selects instrument12. `config_path` or the environment
variable `MEDINSPECT_MODEL_CONFIG` selects a fixed artifact; a conflicting declared
class is rejected. Loaded models are cached, with invalidation when config contents
or artifact file stats change. Both supported models fit in the cache.

| Field | Meaning |
| --- | --- |
| anomaly_score | Python float, raw model-specific distance, not probability. |
| anomaly_map | NumPy float32 H by W, EXIF-oriented original-image coordinates, no fixed upper bound. |
| requires_review | Python bool, score >= threshold; separate from human review. |
| model_version | Saved artifact version. |
| threshold | Validation-calibrated raw image-score threshold. |
| instrument_class | Supported class from saved config. |

Input is one PIL image, at least 8 pixels on each side and at most 40 million
pixels total. Grayscale/palette inputs convert to RGB and transparency composites
over white. Animated/multipage inputs are rejected. The app must enforce its
upload-byte limit before decoding.

The active backend applies EXIF orientation, RGB conversion and independent
MobileSAM instrument segmentation. It preserves native image resolution while
extracting 256-pixel tiles, resizes each to 252 for DINO's 14-pixel stride, zeros
background and applies ImageNet normalization. The model computes real cosine
reconstruction-error maps, stitches them in original coordinates and reapplies
the instrument mask. The score averages the highest 0.1% of instrument-region
map values. It is not a probability or a named-defect classification.

Do not shrink images before calling this API. Display the EXIF-oriented original
for an aligned overlay. Use a fixed validation-derived color scale; the image
threshold is not a calibrated pixel segmentation threshold. The older PatchCore
backend remains available only through explicit legacy config paths and applies
its saved letterboxing pipeline internally.

ModelLoadError describes missing/incompatible config or dependencies, missing
weights, checksum failure or an invalid bank. TypeError/ValueError describe bad
inputs or classes; RuntimeError describes inference failure. Display errors
without replacing them with invented normal/defective predictions.

Record batch/instrument IDs, declared class, model version, score, threshold,
timestamp and a separate human decision/note. An unflagged image does not
guarantee quality, safety or sterility.
