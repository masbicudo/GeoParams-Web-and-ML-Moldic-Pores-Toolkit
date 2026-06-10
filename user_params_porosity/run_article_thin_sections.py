"""
Run porosity measurements for the article thin sections using default paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "article_thin_sections"
SCALES = {
    "100%": 1.0,
    "50%": 0.5,
    "25%": 0.25,
    "12.5%": 0.125,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manuscript porosity measurements for the article thin sections."
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


def cropped_article_images() -> list[str]:
    with CROP_METADATA.open("r", encoding="utf-8") as fp:
        crop_data = json.load(fp)
    return [str(IMAGES / filename) for filename in crop_data]


def main() -> int:
    args = parse_args()
    import_params()
    summary_file, _ = run_analysis(
        image_inputs=cropped_article_images(),
        params_path=PROJECT_DIR / "data" / "c_min_k_max_params.csv",
        crop_metadata_path=CROP_METADATA,
        output_dir=OUTPUT_DIR,
        scales=SCALES,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_chunk_pixels=args.bootstrap_chunk_pixels,
    )

    summary_df = pd.read_csv(summary_file)
    table = summary_df.pivot(index="image", columns="scale", values="porosity_20p")
    table = table[[scale for scale in SCALES if scale in table.columns]]
    table_file = OUTPUT_DIR / "article_porosity_table_20p.csv"
    table.to_csv(table_file, float_format="%.15f")
    print(f"Wrote article porosity table to {table_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
