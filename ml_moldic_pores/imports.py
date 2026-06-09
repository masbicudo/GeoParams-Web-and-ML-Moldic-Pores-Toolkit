"""
Import data needed by the machine-learning subproject.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "out"
DATASETS_DIR = REPO_DIR / "datasets"


def resize_image_file(input_file: Path, output_file: Path, scale_percent: float) -> None:
    if output_file.exists():
        return

    image = cv2.imread(str(input_file))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_file}")

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_file), resized, [cv2.IMWRITE_JPEG_QUALITY, 99])


def import_params() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = REPO_DIR / "geo_params_web" / "exports" / "c_min_k_max_params.csv"
    target = DATA_DIR / "c_min_k_max_params.csv"

    if source.exists():
        shutil.copy(source, target)
        print(f"Copied {source} to {target}")
        return

    print("Warning: c_min_k_max_params.csv was not found.")
    print("Run this first:")
    print("  cd ../geo_params_web")
    print("  pdm run python exports.py")


def prepare_pore_type_training_images() -> None:
    source_dir = DATASETS_DIR / "pore_type_training"
    target_dir = OUT_DIR / "pore_type_training"

    if not source_dir.exists():
        print(f"Warning: dataset folder was not found: {source_dir}")
        print("Download the public dataset from Google Drive into ./datasets.")
        return

    for image_file in sorted(source_dir.glob("*.jpg")):
        target_file = target_dir / f"{image_file.stem}_25.jpg"
        resize_image_file(image_file, target_file, 25)
        print(f"Prepared {target_file}")


def main() -> int:
    import_params()
    prepare_pore_type_training_images()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
