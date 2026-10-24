"""One explicit transform shared by training, calibration and inference."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

MAX_PIXELS = 40_000_000
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Geometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def validate_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("inspect_image expects a PIL image.")
    if image.width < 8 or image.height < 8:
        raise ValueError("Image dimensions must each be at least 8 pixels.")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"Image exceeds {MAX_PIXELS:,} pixels.")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Animated or multipage images are unsupported.")
    image.load()
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return oriented.convert("RGB")


def preprocess_image(image: Image.Image, input_size: int = 256, region_mask: bool = False) -> tuple[torch.Tensor, Geometry]:
    """RGB -> aspect-preserving resize -> white letterbox -> ImageNet normalize."""
    if not isinstance(input_size, int) or input_size < 32 or input_size % 8:
        raise ValueError("Input size must be a multiple of 8 and at least 32.")
    rgb = validate_image(image)
    scale = input_size / max(rgb.size)
    width, height = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
    left, top = (input_size - width) // 2, (input_size - height) // 2
    resized = TF.resize(rgb, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if region_mask:
        from src.region import suppress_blue_background
        resized = suppress_blue_background(resized)
    canvas = Image.new("RGB", (input_size, input_size), "white")
    canvas.paste(resized, (left, top))
    tensor = TF.normalize(TF.to_tensor(canvas), MEAN, STD)
    return tensor, Geometry(rgb.width, rgb.height, width, height, left, top)


def align_anomaly_map(anomaly_map: torch.Tensor, geometry: Geometry) -> np.ndarray:
    """Undo letterboxing. Coordinates follow the EXIF-oriented input image."""
    if anomaly_map.ndim != 2 or not torch.isfinite(anomaly_map).all():
        raise ValueError("Model returned an invalid anomaly map.")
    crop = anomaly_map[geometry.top:geometry.top + geometry.resized_height,
                       geometry.left:geometry.left + geometry.resized_width]
    if crop.shape != (geometry.resized_height, geometry.resized_width):
        raise ValueError("Anomaly-map shape does not match model preprocessing.")
    aligned = torch.nn.functional.interpolate(
        crop[None, None].float(), size=(geometry.original_height, geometry.original_width),
        mode="bilinear", align_corners=False,
    )[0, 0]
    return aligned.cpu().numpy().astype(np.float32, copy=False)
