"""SAM instrument masks from the official MobileSAM implementation, no GT input."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image
import torch

from src.dataset import file_sha256
from src.preprocessing import validate_image

ROOT = Path(__file__).resolve().parents[1]


class InstrumentSegmenter:
    def __init__(self, device: str = "cpu"):
        from mobile_sam import sam_model_registry, SamPredictor
        self.config = json.loads((ROOT / "models/sam_config.json").read_text())
        if self.config.get("segmentation_version") != "rgb-automatic-prompts-v3":
            raise ValueError("Unsupported SAM preprocessing version.")
        checkpoint = ROOT / "models" / self.config["checkpoint"]
        if file_sha256(checkpoint) != self.config["sha256"]:
            raise ValueError("MobileSAM checksum mismatch.")
        model = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(device).eval()
        self.predictor = SamPredictor(model)
        self.lock = Lock()
        self.cache = ROOT / "data/cache/sam"
        self.cache.mkdir(parents=True, exist_ok=True)

    def segment(self, image: Image.Image) -> np.ndarray:
        rgb = validate_image(image)
        if max(rgb.size) / min(rgb.size) > 8:
            raise ValueError("Image aspect ratio exceeds the supported 8:1 limit.")
        pixels_digest = rgb.tobytes() + self.config["sha256"].encode()
        key = hashlib.sha256(pixels_digest + b"rgb-automatic-prompts-v3").hexdigest()
        coarse_key = hashlib.sha256(pixels_digest + b"rgb-automatic-prompts-v2").hexdigest()
        cache_file = self.cache / f"{key}.npz"
        if cache_file.is_file():
            with np.load(cache_file) as saved:
                return saved["mask"].astype(bool)
        scale = 512 / rgb.width
        small = np.asarray(rgb.resize((512, max(1, round(rgb.height * scale))), Image.Resampling.BILINEAR))
        pixels = small.astype(np.float32)
        neutral = ((pixels[..., 2] <= 1.20 * np.maximum(pixels[..., 0], pixels[..., 1])) & (pixels.max(2) > 40)).astype(np.uint8)
        neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(neutral)
        seed = np.zeros_like(neutral, dtype=bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= 12:
                seed |= labels == label
        coordinates = np.argwhere(seed)
        if len(coordinates) < 32:
            raise ValueError("No automatic instrument prompts on the expected blue/purple cloth.")
        # Deterministic spatially spread positive prompts derived from RGB only.
        candidates = coordinates[::max(1, len(coordinates) // 4096)]
        selected_points = [candidates[len(candidates) // 2]]
        distances = np.full(len(candidates), np.inf)
        for _ in range(15):
            distances = np.minimum(distances, ((candidates - selected_points[-1]) ** 2).sum(1))
            selected_points.append(candidates[np.argmax(distances)])
        points = np.asarray(selected_points, dtype=np.float32)[:, ::-1]
        minimum, maximum = coordinates.min(0), coordinates.max(0)
        box = np.array([minimum[1], minimum[0], maximum[1], maximum[0]], dtype=np.float32)
        coarse_cache = self.cache / f"{coarse_key}.npz"
        if coarse_cache.is_file():
            with np.load(coarse_cache) as saved:
                coarse = saved["mask"]
            masks = np.asarray([cv2.resize(coarse.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)])
            confidence = [1.0]
        else:
            with self.lock, torch.inference_mode():
                self.predictor.set_image(small)
                masks, confidence, _ = self.predictor.predict(point_coords=points, point_labels=np.ones(len(points), dtype=np.int32), box=box, multimask_output=True)
        def mask_score(index):
            mask = masks[index]
            overlap = float((mask & seed).sum())
            return 2 * overlap / max(1, mask.sum() + seed.sum()) + 0.25 * overlap / seed.sum() + 0.05 * confidence[index]
        selected = masks[max(range(len(masks)), key=mask_score)]
        # Remove SAM's occasional towel islands while retaining a broad border
        # around neutral metal/red foreground. Uses input colors only.
        neighborhood = cv2.dilate(seed.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        selected = selected & neighborhood
        if (selected & seed).sum() / seed.sum() < 0.5:
            raise ValueError("SAM segmentation excluded too much of the detected instrument.")
        full = cv2.resize(selected.astype(np.uint8), rgb.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        temporary = cache_file.with_suffix(".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, mask=full)
        temporary.replace(cache_file)
        return full


@lru_cache(maxsize=2)
def get_segmenter(device: str = "cpu") -> InstrumentSegmenter:
    return InstrumentSegmenter(device)
