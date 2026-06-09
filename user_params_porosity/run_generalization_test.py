"""
Run porosity measurements for the generalization-test thin sections.
"""

from __future__ import annotations

from pathlib import Path

from imports import main as import_params
from measure_porosity_from_params import run_analysis


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

IMAGES = REPO_DIR / "datasets" / "generalization_test_thin_sections"
CROP_METADATA = PROJECT_DIR / "data" / "cut-metadata.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "generalization_test_thin_sections"


def main() -> int:
    import_params()
    run_analysis(
        image_inputs=[str(IMAGES)],
        params_path=PROJECT_DIR / "data" / "c_min_k_max_params.csv",
        crop_metadata_path=CROP_METADATA,
        output_dir=OUTPUT_DIR,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
