"""Native-resolution patch extraction and spatial reconstruction."""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from src.preprocessing import MEAN, STD

TILE_SIZE = 256
MODEL_SIZE = 252  # divisible by the DINOv2 14-pixel patch stride


def extract_tiles(image: Image.Image, mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    pixels = np.asarray(image.convert("RGB")).copy()
    if mask.shape != pixels.shape[:2]:
        raise ValueError("Instrument mask does not match image dimensions.")
    pixels[~mask] = 0
    height, width = mask.shape
    tiles = []
    for top in range(0, height, TILE_SIZE):
        for left in range(0, width, TILE_SIZE):
            bottom, right = min(top + TILE_SIZE, height), min(left + TILE_SIZE, width)
            if mask[top:bottom, left:right].sum() < 32:
                continue
            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            tile[:bottom - top, :right - left] = pixels[top:bottom, left:right]
            tiles.append(((left, top, right, bottom), tile))
    if not tiles:
        raise ValueError("No instrument patches available for inference.")
    return tiles


def tile_tensor(tile: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(tile, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    return TF.normalize(TF.to_tensor(resized), MEAN, STD)


def stitch_tile_maps(tiles, maps, shape: tuple[int, int], mask: np.ndarray) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for (box, _), anomaly_map in zip(tiles, maps, strict=True):
        left, top, right, bottom = box
        resized = cv2.resize(anomaly_map, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
        result[top:bottom, left:right] = resized[:bottom - top, :right - left]
    result = cv2.GaussianBlur(result, (5, 5), 1.0)
    result[~mask] = 0
    return np.maximum(result, 0)


def instrument_score(anomaly_map: np.ndarray, mask: np.ndarray) -> float:
    values = anomaly_map[mask]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite instrument-region anomaly scores.")
    count = max(1, int(np.ceil(len(values) * 0.001)))
    return float(np.partition(values, -count)[-count:].mean())
