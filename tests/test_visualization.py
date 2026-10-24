import numpy as np
import pytest
from PIL import Image

from src.visualization import colorize_map, create_overlay


def test_fixed_scale_does_not_stretch_each_image():
    first = colorize_map(np.array([[1, 2]]), (2, 1), scale_min=0, scale_max=4)
    second = colorize_map(np.array([[1, 4]]), (2, 1), scale_min=0, scale_max=4)
    assert first.getpixel((0, 0)) == second.getpixel((0, 0))
    assert first.getpixel((1, 0)) != second.getpixel((1, 0))


def test_constant_zero_map_and_opacity():
    original = Image.new("RGB", (9, 5), "white")
    heatmap, overlay = create_overlay(
        original, np.zeros((2, 3)), scale_min=0, scale_max=1, opacity=0
    )
    assert heatmap.size == overlay.size == original.size
    assert len(set(heatmap.getdata())) == 1
    assert overlay.tobytes() == original.tobytes()


def test_non_square_map_keeps_spatial_alignment():
    anomaly_map = np.zeros((3, 7))
    anomaly_map[2, 6] = 1
    heatmap = colorize_map(anomaly_map, (7, 3), scale_min=0, scale_max=1)
    assert heatmap.getpixel((6, 2)) == (253, 231, 37)
    assert heatmap.getpixel((0, 0)) == (68, 1, 84)


@pytest.mark.parametrize("bounds", [(0, 0), (2, 1), (0, float("inf"))])
def test_bad_display_scale(bounds):
    with pytest.raises(ValueError):
        colorize_map(np.zeros((2, 3)), (3, 2), scale_min=bounds[0], scale_max=bounds[1])
