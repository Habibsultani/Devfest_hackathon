"""Fixed-scale anomaly visualization; display colors never select a review threshold."""

import math

import numpy as np
from PIL import Image

# Viridis anchors: dark purple -> blue -> teal -> green -> yellow.
# A fixed table avoids adding matplotlib/OpenCV to the application runtime.
_POSITIONS = np.linspace(0, 1, 5)
_COLORS = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    dtype=np.float32,
)


def colorize_map(
    anomaly_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    scale_min: float,
    scale_max: float,
) -> Image.Image:
    """Bilinearly align a scalar map, then apply a model-wide fixed display scale.

    No per-image min/max stretching: a constant map remains constant. Values
    beyond the scale clip for color display only; raw scores remain unchanged.
    The production gateway requires original-image alignment already; resizing
    here also supports standalone visualization of lower-resolution maps.
    """
    if not math.isfinite(scale_min) or not math.isfinite(scale_max) or scale_max <= scale_min:
        raise ValueError("The heatmap display scale must have finite increasing bounds.")
    data = np.asarray(anomaly_map)
    if data.ndim != 2 or not data.size or data.dtype.kind not in "uif":
        raise ValueError("The anomaly map must be a non-empty two-dimensional numeric array.")
    if not np.isfinite(data).all():
        raise ValueError("The anomaly map must contain only finite values.")
    aligned = Image.fromarray(data.astype(np.float32))
    if aligned.size != image_size:
        aligned = aligned.resize(image_size, Image.Resampling.BILINEAR)
    normalized = np.clip((np.asarray(aligned) - scale_min) / (scale_max - scale_min), 0, 1)
    channels = [np.interp(normalized, _POSITIONS, _COLORS[:, channel]) for channel in range(3)]
    return Image.fromarray(np.stack(channels, axis=-1).round().astype(np.uint8))


def create_overlay(
    image: Image.Image,
    anomaly_map: np.ndarray,
    *,
    scale_min: float,
    scale_max: float,
    opacity: float = 0.45,
) -> tuple[Image.Image, Image.Image]:
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between zero and one.")
    heatmap = colorize_map(anomaly_map, image.size, scale_min=scale_min, scale_max=scale_max)
    return heatmap, Image.blend(image.convert("RGB"), heatmap, opacity)


def color_legend() -> Image.Image:
    return colorize_map(np.linspace(0, 1, 256).reshape(1, 256), (512, 20), scale_min=0, scale_max=1)
