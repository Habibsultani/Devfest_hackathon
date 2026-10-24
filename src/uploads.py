"""Decode uploads for display without performing model preprocessing."""

import io
import warnings
from pathlib import PurePath

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class UploadError(ValueError):
    """An upload cannot safely be used for an inspection."""


def decode_upload(content: bytes, filename: str) -> Image.Image:
    """Return a fully decoded, EXIF-oriented RGB image; never resize or normalize."""
    if not content:
        raise UploadError("Choose a non-empty JPEG or PNG image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadError("The image exceeds the 10 MB upload limit.")
    if PurePath(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type. Upload a JPEG or PNG image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as probe:
                if probe.format not in {"JPEG", "PNG"}:
                    raise UploadError("The file contents must be a real JPEG or PNG image.")
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise UploadError("The image exceeds the 12 megapixel limit.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise UploadError("Animated images are unsupported. Upload one still image.")
                probe.verify()
            with Image.open(io.BytesIO(content)) as original:
                oriented = ImageOps.exif_transpose(original)
                if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    oriented = Image.alpha_composite(background, rgba)
                return oriented.convert("RGB").copy()
    except UploadError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise UploadError("The image is unreadable, damaged, or too large to decode.") from exc
