"""
Smoke test for the user-parameter porosity subproject.

The test uses one article thin-section image and the default no-border crop
metadata. It validates the default data layout and writes a small output table.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from imports import main as import_params
from measure_porosity_from_params import run_analysis


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
IMAGES = REPO_DIR / "datasets" / "article_thin_sections"
CROP_METADATA = (
    REPO_DIR
    / "geo_params_web"
    / "static"
    / "imgs_sections"
    / "cuts"
    / "no-border"
    / "cut-metadata.json"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "quick_test"


def first_image() -> Path:
    if CROP_METADATA.exists():
        with CROP_METADATA.open("r", encoding="utf-8") as fp:
            crop_data = json.load(fp)
        for filename in crop_data:
            image = IMAGES / filename
            if image.exists():
                return image

    raise FileNotFoundError(
        "No image files were found for the quick test.\n\n"
        f"Expected at least one image listed in:\n  {CROP_METADATA}\n\n"
        f"Image directory:\n  {IMAGES}\n\n"
        "Download the public dataset from Google Drive and place its folders "
        "under the repository's datasets directory."
    )


def main() -> int:
    print("Importing collected user parameters...")
    import_params()

    image = first_image()
    print(f"Running quick porosity test on: {image.name}")
    summary_file, _ = run_analysis(
        image_inputs=[str(image)],
        params_path=PROJECT_DIR / "data" / "c_min_k_max_params.csv",
        crop_metadata_path=CROP_METADATA,
        output_dir=OUTPUT_DIR,
        write_mean_images=False,
    )

    df = pd.read_csv(summary_file)
    row = df.iloc[0]
    print("\nQuick test summary")
    print(f"  image: {row['image']}")
    print(f"  crop applied: {row['crop_applied']}")
    print(f"  parameters applied: {int(row['params_total'])}")
    print(f"  valid superposition samples: {int(row['number_of_samples'])}")
    print(f"  manuscript porosity (porosity_20p): {100 * row['porosity_20p']:.2f}%")
    print(f"\nOutput written to: {OUTPUT_DIR}")
    print("Quick test completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
