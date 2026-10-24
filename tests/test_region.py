import numpy as np
import pytest
from PIL import Image

from src.region import instrument_region, suppress_blue_background


def test_region_uses_image_colors_and_preserves_coordinates():
    pixels = np.full((64, 64, 3), (40, 50, 150), dtype=np.uint8)
    pixels[20:44, 28:36] = (180, 180, 180)
    image = Image.fromarray(pixels)
    mask = instrument_region(image)
    assert mask.shape == (64, 64)
    assert mask[32, 32] and not mask[0, 0]
    result = np.asarray(suppress_blue_background(image))
    np.testing.assert_array_equal(result[32, 32], pixels[32, 32])
    np.testing.assert_array_equal(result[0, 0], [255, 255, 255])


def test_missing_foreground_is_an_explicit_error():
    with pytest.raises(ValueError, match="No instrument region"):
        suppress_blue_background(Image.new("RGB", (32, 32), (40, 50, 150)))
