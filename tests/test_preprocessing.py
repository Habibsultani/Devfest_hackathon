import numpy as np
from PIL import Image
import pytest
import torch

from src.preprocessing import align_anomaly_map, preprocess_image, validate_image


def test_letterbox_and_inverse_remove_padding():
    tensor, geometry = preprocess_image(Image.new("RGB", (80, 40), "black"), 64)
    assert tensor.shape == (3, 64, 64)
    assert (geometry.left, geometry.top, geometry.resized_width, geometry.resized_height) == (0, 16, 64, 32)
    raw_map = torch.full((64, 64), 90.0)
    raw_map[16:48] = 3
    aligned = align_anomaly_map(raw_map, geometry)
    assert aligned.shape == (40, 80)
    assert aligned.dtype == np.float32
    np.testing.assert_allclose(aligned, 3, atol=1e-6)


def test_exif_orientation_is_shared_with_output_coordinates():
    image = Image.new("RGB", (40, 80), "white")
    image.getexif()[274] = 6
    rgb = validate_image(image)
    assert rgb.size == (80, 40)
    _, geometry = preprocess_image(image, 64)
    assert (geometry.original_width, geometry.original_height) == rgb.size


def test_transparency_is_composited_on_white():
    rgb = validate_image(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
    assert rgb.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("size", [(1, 1), (7, 10)])
def test_tiny_images_rejected(size):
    with pytest.raises(ValueError, match="at least 8"):
        validate_image(Image.new("RGB", size))


def test_non_image_and_invalid_map_rejected():
    with pytest.raises(TypeError):
        preprocess_image(np.zeros((10, 10, 3)))
    _, geometry = preprocess_image(Image.new("RGB", (16, 16)), 32)
    with pytest.raises(ValueError, match="invalid anomaly map"):
        align_anomaly_map(torch.full((32, 32), float("nan")), geometry)
    with pytest.raises(ValueError, match="does not match"):
        align_anomaly_map(torch.zeros(8, 8), geometry)
