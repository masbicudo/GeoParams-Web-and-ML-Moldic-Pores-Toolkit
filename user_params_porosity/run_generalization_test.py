"""
Run porosity measurements for the generalization-test thin sections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imports import main as import_params
from measure_porosity_from_params import run_analysis


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

IMAGES = REPO_DIR / "datasets" / "generalization_test_thin_sections"
CROP_METADATA = PROJECT_DIR / "data" / "cut-metadata.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "generalization_test_thin_sections"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run porosity measurements for the generalization-test thin sections."
    )
    parser.add_argument(
        "--write-cropped-images",
        action="store_true",
        help=(
            "Write the cropped input images used for measurement to "
            "data/output/generalization_test_thin_sections/cropped_inputs."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap replicates for uncertainty estimates. Default: 0.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed used by bootstrap resampling. Default: 42.",
    )
    parser.add_argument(
        "--bootstrap-chunk-pixels",
        type=int,
        default=250_000,
        help="Pixels processed per bootstrap chunk. Default: 250000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import_params()
    run_analysis(
        image_inputs=[str(IMAGES)],
        params_path=PROJECT_DIR / "data" / "c_min_k_max_params.csv",
        crop_metadata_path=CROP_METADATA,
        output_dir=OUTPUT_DIR,
        write_cropped_images=args.write_cropped_images,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_chunk_pixels=args.bootstrap_chunk_pixels,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
