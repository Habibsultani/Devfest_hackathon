import io

import pytest
from PIL import Image

from src.uploads import MAX_UPLOAD_BYTES, UploadError, decode_upload


def image_bytes(image, image_format="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=image_format, **kwargs)
    return stream.getvalue()


def test_rgb_size_and_pixels_are_preserved():
    original = Image.new("RGB", (7, 3), (20, 30, 40))
    decoded = decode_upload(image_bytes(original), "sample.PNG")
    assert decoded.mode == "RGB"
    assert decoded.size == original.size
    assert decoded.tobytes() == original.tobytes()


def test_transparency_is_composited_on_white():
    decoded = decode_upload(image_bytes(Image.new("RGBA", (2, 2), (255, 0, 0, 0))), "a.png")
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_is_applied_before_model_input():
    image = Image.new("RGB", (7, 3))
    exif = image.getexif()
    exif[274] = 6
    decoded = decode_upload(image_bytes(image, "JPEG", exif=exif), "rotated.jpg")
    assert decoded.size == (3, 7)
    assert decoded.getexif().get(274) is None


@pytest.mark.parametrize(
    ("content", "filename"),
    [(b"", "a.png"), (b"not an image", "a.jpg"), (b"anything", "a.svg")],
)
def test_invalid_files_are_rejected(content, filename):
    with pytest.raises(UploadError):
        decode_upload(content, filename)


def test_format_spoofing_is_rejected():
    content = image_bytes(Image.new("RGB", (2, 2)), "GIF")
    with pytest.raises(UploadError, match="real JPEG or PNG"):
        decode_upload(content, "spoofed.png")


def test_byte_limit():
    with pytest.raises(UploadError, match="20 MB"):
        decode_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "large.png")


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr("src.uploads.MAX_IMAGE_PIXELS", 4)
    with pytest.raises(UploadError, match="megapixel"):
        decode_upload(image_bytes(Image.new("RGB", (3, 2))), "large.png")


def test_animated_png_is_rejected():
    first = Image.new("RGB", (3, 2), "red")
    content = image_bytes(
        first, save_all=True, append_images=[Image.new("RGB", (3, 2), "blue")], duration=100
    )
    with pytest.raises(UploadError, match="Animated"):
        decode_upload(content, "animated.png")
