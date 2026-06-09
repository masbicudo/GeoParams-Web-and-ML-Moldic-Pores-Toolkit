"""
Measure porosity in new thin-section images using collected C/K parameters.

This script applies the original threshold-based parametrization to new
images. It does not collect new user data, filter out null detections, or train
machine-learning models.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.05, 0.075, 0.10, 0.20, 0.30, 0.40)
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply collected C/K segmentation parameters to new images and "
            "export porosity measurements."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        help=(
            "Image files, directories, or glob patterns. Directories are "
            "searched non-recursively for jpg/jpeg/png/tif/tiff files."
        ),
    )
    parser.add_argument(
        "--params",
        default="data/c_min_k_max_params.csv",
        help="CSV with clicked_x and clicked_y columns. Default: data/c_min_k_max_params.csv",
    )
    parser.add_argument(
        "--crop-metadata",
        help=(
            "Optional JSON file mapping image filenames to x, y, width, and "
            "height crop rectangles. Measurements are made on cropped pixels."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/porosity_from_params",
        help="Directory for CSV outputs and mean masks.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Consensus thresholds for the mean mask, expressed from 0 to 1.",
    )
    parser.add_argument(
        "--no-mean-images",
        action="store_true",
        help="Do not write mean-mask PNG files.",
    )
    parser.add_argument(
        "--write-cropped-images",
        action="store_true",
        help="Write cropped input images used for measurement.",
    )
    return parser.parse_args()


def friendly_missing_path_message(path: Path, role: str) -> str:
    return (
        f"{role} was not found:\n"
        f"  {path}\n\n"
        "Please make sure the public dataset has been downloaded from Google "
        "Drive and placed in the repository's datasets directory. See the "
        "repository README.md for the expected layout."
    )


def expand_image_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []

    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for extension in IMAGE_EXTENSIONS:
                files.extend(path.glob(extension))
            continue

        matches = [Path(p) for p in glob.glob(item)]
        if matches:
            files.extend(matches)
        else:
            files.append(path)

    return sorted({p.resolve() for p in files if p.is_file()})


def load_crop_metadata(metadata_file: str | None) -> dict[str, dict[str, int]]:
    if metadata_file is None:
        return {}

    path = Path(metadata_file)
    if not path.exists():
        raise FileNotFoundError(friendly_missing_path_message(path, "Crop metadata"))

    with path.open("r", encoding="utf-8") as fp:
        raw_data = json.load(fp)

    crops: dict[str, dict[str, int]] = {}
    for key, value in raw_data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Crop metadata for {key} must be an object.")
        if {"x", "y", "width", "height"} <= set(value):
            crops[key] = {field: int(value[field]) for field in ("x", "y", "width", "height")}
            continue
        raise ValueError(
            f"Crop metadata for {key} must contain x, y, width, and height."
        )
    return crops


def find_crop_for_image(
    image_path: Path,
    crop_metadata: dict[str, dict[str, int]],
) -> dict[str, int] | None:
    return crop_metadata.get(image_path.name) or crop_metadata.get(str(image_path))


def crop_image(
    image: np.ndarray,
    crop: dict[str, int] | None,
    image_name: str,
) -> tuple[np.ndarray, dict[str, int] | None]:
    if crop is None:
        return image, None

    height, width = image.shape[:2]
    x = max(0, int(crop["x"]))
    y = max(0, int(crop["y"]))
    crop_width = max(0, int(crop["width"]))
    crop_height = max(0, int(crop["height"]))
    x2 = min(width, x + crop_width)
    y2 = min(height, y + crop_height)

    if x >= x2 or y >= y2:
        raise ValueError(f"Invalid crop rectangle for {image_name}: {crop}")

    applied_crop = {
        "x": x,
        "y": y,
        "width": x2 - x,
        "height": y2 - y,
    }
    return image[y:y2, x:x2], applied_crop


def bgr_to_cmyk(bgr_img: np.ndarray) -> np.ndarray:
    bgr = bgr_img.astype(np.float32)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    j = np.maximum.reduce([r, g, b])
    k = 255 - j

    nonzero = j > 0
    c = np.zeros_like(j)
    m = np.zeros_like(j)
    y = np.zeros_like(j)
    c[nonzero] = 255 * (j[nonzero] - r[nonzero]) / j[nonzero]
    m[nonzero] = 255 * (j[nonzero] - g[nonzero]) / j[nonzero]
    y[nonzero] = 255 * (j[nonzero] - b[nonzero]) / j[nonzero]

    return np.stack([c, m, y, k], axis=2).clip(0, 255).astype(np.uint8)


def compute_components(binary_image: np.ndarray) -> int:
    num_labels, _ = cv2.connectedComponents(binary_image)
    return int(num_labels - 1)


def clean_stem(path: Path) -> str:
    safe_chars = []
    for char in path.stem:
        safe_chars.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(safe_chars)


def measure_image(
    image_path: Path,
    params_df: pd.DataFrame,
    crop_metadata: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_height, original_width = image.shape[:2]
    crop = find_crop_for_image(image_path, crop_metadata)
    image, applied_crop = crop_image(image, crop, image_path.name)

    img_cmyk = bgr_to_cmyk(image)
    height, width = image.shape[:2]
    image_area = height * width

    mean_image = np.zeros((height, width), dtype=np.float32)
    per_param_rows: list[dict[str, Any]] = []

    for idx, row in params_df.iterrows():
        c_min = int(row["clicked_x"])
        k_max = int(row["clicked_y"])

        binary_image = cv2.inRange(
            img_cmyk,
            np.array([c_min, 0, 0, 0], dtype=np.uint8),
            np.array([255, 255, 64, k_max], dtype=np.uint8),
        )

        pore_pixels = int(np.count_nonzero(binary_image))
        porosity = pore_pixels / image_area if image_area else 0.0
        component_count = compute_components(binary_image)

        per_param_rows.append(
            {
                "image": image_path.name,
                "image_path": str(image_path),
                "param_index": idx,
                "source_param_image": row.get("filename"),
                "experience": row.get("experience"),
                "min_pore_size": row.get("min_pore_size"),
                "c_min": c_min,
                "k_max": k_max,
                "pore_pixels": pore_pixels,
                "porosity": porosity,
                "component_count": component_count,
                "crop_applied": applied_crop is not None,
                "crop_x": None if applied_crop is None else applied_crop["x"],
                "crop_y": None if applied_crop is None else applied_crop["y"],
                "crop_width": None if applied_crop is None else applied_crop["width"],
                "crop_height": None if applied_crop is None else applied_crop["height"],
            }
        )

        mean_image += binary_image

    if len(params_df) > 0:
        mean_image /= len(params_df)

    porosities = [float(row["porosity"]) for row in per_param_rows]
    components = [int(row["component_count"]) for row in per_param_rows]

    summary: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "original_height": original_height,
        "original_width": original_width,
        "height": height,
        "width": width,
        "area_pixels": image_area,
        "crop_applied": applied_crop is not None,
        "crop_x": None if applied_crop is None else applied_crop["x"],
        "crop_y": None if applied_crop is None else applied_crop["y"],
        "crop_width": None if applied_crop is None else applied_crop["width"],
        "crop_height": None if applied_crop is None else applied_crop["height"],
        "params_total": len(per_param_rows),
        "porosity_mean_by_param": np.mean(porosities) if porosities else np.nan,
        "porosity_median_by_param": np.median(porosities) if porosities else np.nan,
        "porosity_std_by_param": np.std(porosities, ddof=1) if len(porosities) > 1 else np.nan,
        "component_count_mean_by_param": np.mean(components) if components else np.nan,
        "component_count_median_by_param": np.median(components) if components else np.nan,
    }

    return summary, per_param_rows, mean_image, image


def run_analysis(
    image_inputs: list[str],
    params_path: str | Path = "data/c_min_k_max_params.csv",
    crop_metadata_path: str | Path | None = None,
    output_dir: str | Path = "data/output/porosity_from_params",
    thresholds: list[float] | tuple[float, ...] = DEFAULT_THRESHOLDS,
    write_mean_images: bool = True,
    write_cropped_images: bool = False,
) -> tuple[Path, Path]:
    params_path = Path(params_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not params_path.exists():
        raise FileNotFoundError(friendly_missing_path_message(params_path, "Parameter CSV"))

    params_df = pd.read_csv(params_path)
    required_columns = {"clicked_x", "clicked_y"}
    missing = required_columns - set(params_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {params_path}: {sorted(missing)}")

    crop_metadata = load_crop_metadata(None if crop_metadata_path is None else str(crop_metadata_path))
    image_paths = expand_image_inputs(image_inputs)
    if not image_paths:
        joined_inputs = "\n  ".join(image_inputs)
        raise FileNotFoundError(
            "No image files were found from these inputs:\n"
            f"  {joined_inputs}\n\n"
            "Please make sure the public dataset has been downloaded from "
            "Google Drive and placed in the repository's datasets directory."
        )

    summary_rows: list[dict[str, Any]] = []
    all_per_param_rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        print(f"Processing {image_path}")
        summary, per_param_rows, mean_image, measured_image = measure_image(
            image_path,
            params_df,
            crop_metadata,
        )

        for threshold in thresholds:
            key = f"porosity_consensus_{threshold:g}"
            summary[key] = float(np.mean(mean_image >= threshold * 255))

        summary_rows.append(summary)
        all_per_param_rows.extend(per_param_rows)

        if write_mean_images:
            out_image = output_dir / f"mean_mask_{clean_stem(image_path)}.png"
            cv2.imwrite(str(out_image), mean_image.astype(np.uint8))

        if write_cropped_images:
            out_image = output_dir / f"cropped_input_{clean_stem(image_path)}.png"
            cv2.imwrite(str(out_image), measured_image)

    summary_df = pd.DataFrame(summary_rows)
    per_param_df = pd.DataFrame(all_per_param_rows)

    summary_file = output_dir / "porosity_summary.csv"
    per_param_file = output_dir / "porosity_by_parameter.csv"
    summary_df.to_csv(summary_file, index=False, float_format="%.15f")
    per_param_df.to_csv(per_param_file, index=False, float_format="%.15f")

    print(f"Wrote {summary_file}")
    print(f"Wrote {per_param_file}")
    if write_mean_images:
        print(f"Wrote mean masks to {output_dir}")

    return summary_file, per_param_file


def main() -> int:
    args = parse_args()
    run_analysis(
        image_inputs=args.images,
        params_path=args.params,
        crop_metadata_path=args.crop_metadata,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        write_mean_images=not args.no_mean_images,
        write_cropped_images=args.write_cropped_images,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
