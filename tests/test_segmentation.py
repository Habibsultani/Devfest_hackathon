import pytest
from PIL import Image

from src.segmentation import InstrumentSegmenter


def test_extreme_aspect_ratio_is_rejected_before_resizing_or_model_loading():
    segmenter = object.__new__(InstrumentSegmenter)
    with pytest.raises(ValueError, match="aspect ratio"):
        segmenter.segment(Image.new("RGB", (8, 80)))
