"""Independent color heuristic for instruments photographed on blue cloth."""

import cv2
import numpy as np
from PIL import Image


def instrument_region(image: Image.Image) -> np.ndarray:
    """Return a coarse metal/foreground mask without using defect annotations.

    Blue/purple cloth has more blue than red/green; neutral metal and red dye do
    not. The heuristic assumes this acquisition setup and is not a trained
    instrument segmenter. Closing/dilation retain narrow instrument boundaries.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    mask = ((blue <= 1.12 * np.maximum(red, green)) & (pixels.max(axis=2) > 0.22)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for label in range(1, components):
        if stats[label, cv2.CC_STAT_AREA] >= 12:
            cleaned[labels == label] = 1
    return cv2.dilate(cleaned, kernel, iterations=1).astype(bool)


def suppress_blue_background(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGB")).copy()
    mask = instrument_region(image)
    if not mask.any():
        raise ValueError("No instrument region detected on the expected blue background.")
    pixels[~mask] = 255
    return Image.fromarray(pixels)
