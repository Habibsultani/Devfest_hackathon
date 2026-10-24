"""Cache SAM instrument masks for all selected folds without reading defect masks."""

from pathlib import Path
from PIL import Image
import torch

from src.dataset import load_manifest
from src.segmentation import get_segmenter


def main():
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = get_segmenter(device)
    for instrument in ("instrument12", "instrument19"):
        rows = load_manifest(Path("data/processed") / instrument / "manifest.csv")
        for index, row in enumerate(rows):
            with Image.open(row["image_path"]) as image:
                mask = segmenter.segment(image)
            print(f"{instrument} {index+1}/{len(rows)} {row['source_path']} foreground={mask.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
