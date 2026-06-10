"""
Measure porosity in new thin-section images using collected C/K parameters.

This script applies the threshold-based parametrization to new images. The
default summary reproduces the manuscript superposition procedure: masks with
near-null detections are skipped, remaining masks are averaged, the mean mask is
normalized by its top decile, and porosity is measured at fixed normalized
thresholds. A conservative sample-level guard prevents sparse false positives
from being normalized into nonzero porosity estimates.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.05, 0.075, 0.10, 0.20, 0.30, 0.40)
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEFAULT_NULL_PORE_PIXEL_THRESHOLD = 1000
DEFAULT_MIN_VALID_SAMPLE_FRACTION = 0.20
DEFAULT_BOOTSTRAP_CHUNK_PIXELS = 250_000


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
        help="Normalized superposition thresholds, expressed from 0 to 1.",
    )
    parser.add_argument(
        "--null-pore-pixel-threshold",
        type=int,
        default=DEFAULT_NULL_PORE_PIXEL_THRESHOLD,
        help=(
            "Skip masks with fewer pore pixels than this value when building "
            "the manuscript superposition. Default: 1000."
        ),
    )
    parser.add_argument(
        "--min-valid-sample-fraction",
        type=float,
        default=DEFAULT_MIN_VALID_SAMPLE_FRACTION,
        help=(
            "Minimum fraction of all parameter masks that must pass "
            "--null-pore-pixel-threshold before the image is treated as "
            "containing detectable pores. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Number of bootstrap replicates for uncertainty estimates. "
            "Default: 0."
        ),
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
        default=DEFAULT_BOOTSTRAP_CHUNK_PIXELS,
        help=(
            "Number of pixels processed per bootstrap chunk. Lower values "
            "use less memory and more time. Default: 250000."
        ),
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
    parser.add_argument(
        "--cropped-images-dir",
        help=(
            "Optional directory for cropped input images. Defaults to "
            "<output-dir>/cropped_inputs when --write-cropped-images is used."
        ),
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


def available_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError):
            return None
    return None


def format_bytes(size: float | int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_bootstrap_resource_warning(
    image_name: str,
    area_pixels: int,
    param_count: int,
    bootstrap_replicates: int,
    chunk_pixels: int,
    output_dir: Path,
) -> None:
    if bootstrap_replicates <= 0:
        return

    chunk_pixels = max(1, min(chunk_pixels, area_pixels))
    mask_chunk_bytes = param_count * chunk_pixels
    replicate_chunk_bytes = bootstrap_replicates * chunk_pixels * np.dtype(np.uint16).itemsize
    expected_chunk_bytes = mask_chunk_bytes + replicate_chunk_bytes
    full_mask_cache_bytes = param_count * area_pixels
    available_memory = available_memory_bytes()
    free_disk = shutil.disk_usage(output_dir).free

    print(
        "Bootstrap resource estimate for "
        f"{image_name}: chunk memory about {format_bytes(expected_chunk_bytes)}; "
        f"a full mask cache would be about {format_bytes(full_mask_cache_bytes)}."
    )
    if available_memory is not None and expected_chunk_bytes > available_memory * 0.60:
        print(
            "  Warning: available RAM appears low for this chunk size "
            f"({format_bytes(available_memory)} free). Reduce "
            "--bootstrap-chunk-pixels if this run becomes slow or unstable."
        )
    if full_mask_cache_bytes > free_disk * 0.60:
        print(
            "  Note: a persistent full-image mask cache would not be a good "
            f"default here ({format_bytes(free_disk)} free on the output disk), "
            "so this script uses chunked in-memory masks."
        )


def threshold_column_name(threshold: float) -> str:
    percent = threshold * 100
    if float(percent).is_integer():
        return f"porosity_{int(percent):02d}p"
    integer_part = int(percent)
    decimal_part = str(percent).split(".", maxsplit=1)[1].rstrip("0")
    return f"porosity_{integer_part:02d}_{decimal_part}p"


def normalized_superposition_porosities(
    mean_image: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[dict[str, float], float]:
    if mean_image.size == 0 or float(np.max(mean_image)) <= 0:
        return {threshold_column_name(threshold): 0.0 for threshold in thresholds}, 0.0

    top_decile_cutoff = np.percentile(mean_image, 90)
    top_decile_pixels = mean_image[mean_image >= top_decile_cutoff]
    normalizer = float(np.mean(top_decile_pixels))
    if normalizer <= 0:
        return {threshold_column_name(threshold): 0.0 for threshold in thresholds}, normalizer

    normalized = mean_image / normalizer
    porosities = {
        threshold_column_name(threshold): float(np.mean(normalized >= threshold))
        for threshold in thresholds
    }
    return porosities, normalizer


def zero_superposition_porosities(
    thresholds: list[float] | tuple[float, ...],
) -> dict[str, float]:
    return {threshold_column_name(threshold): 0.0 for threshold in thresholds}


def porosities_from_count_histogram(
    histogram: np.ndarray,
    area_pixels: int,
    thresholds: list[float] | tuple[float, ...],
) -> dict[str, float]:
    if area_pixels <= 0 or histogram.sum() == 0:
        return zero_superposition_porosities(thresholds)

    cumulative = np.cumsum(histogram)
    top_decile_start = int(np.searchsorted(cumulative, area_pixels * 0.90, side="left"))
    count_values = np.arange(histogram.size)
    top_histogram = histogram[top_decile_start:]
    top_counts = count_values[top_decile_start:]
    top_pixels = int(top_histogram.sum())
    if top_pixels <= 0:
        return zero_superposition_porosities(thresholds)

    top_mean = float(np.sum(top_counts * top_histogram) / top_pixels)
    if top_mean <= 0:
        return zero_superposition_porosities(thresholds)

    porosities: dict[str, float] = {}
    for threshold in thresholds:
        min_count = int(np.ceil(threshold * top_mean))
        min_count = max(0, min(min_count, histogram.size - 1))
        porosities[threshold_column_name(threshold)] = float(
            histogram[min_count:].sum() / area_pixels
        )
    return porosities


def bootstrap_superposition_porosities(
    measured_image: np.ndarray,
    params_df: pd.DataFrame,
    thresholds: list[float] | tuple[float, ...],
    null_pore_pixel_threshold: int,
    min_valid_sample_fraction: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    chunk_pixels: int,
    image_name: str,
    output_dir: Path,
) -> dict[str, float]:
    if bootstrap_replicates < 0:
        raise ValueError("Bootstrap replicates must be zero or positive.")
    if bootstrap_replicates == 0:
        return {}
    if chunk_pixels <= 0:
        raise ValueError("Bootstrap chunk pixels must be positive.")

    img_cmyk = bgr_to_cmyk(measured_image)
    c_channel = img_cmyk[:, :, 0].reshape(-1)
    y_channel = img_cmyk[:, :, 2].reshape(-1)
    k_channel = img_cmyk[:, :, 3].reshape(-1)
    area_pixels = int(c_channel.size)
    param_count = len(params_df)
    if area_pixels == 0 or param_count == 0:
        return {}

    print_bootstrap_resource_warning(
        image_name,
        area_pixels,
        param_count,
        bootstrap_replicates,
        chunk_pixels,
        output_dir,
    )

    c_mins = params_df["clicked_x"].astype(np.uint8).to_numpy()
    k_maxs = params_df["clicked_y"].astype(np.uint8).to_numpy()
    pore_pixel_counts = np.zeros(param_count, dtype=np.int64)

    chunk_pixels = min(chunk_pixels, area_pixels)
    for start in range(0, area_pixels, chunk_pixels):
        end = min(start + chunk_pixels, area_pixels)
        masks = (
            (c_channel[start:end][None, :] >= c_mins[:, None])
            & (y_channel[start:end][None, :] <= 64)
            & (k_channel[start:end][None, :] <= k_maxs[:, None])
        )
        pore_pixel_counts += masks.sum(axis=1)

    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0,
        param_count,
        size=(bootstrap_replicates, param_count),
        endpoint=False,
    )
    sampled_valid = pore_pixel_counts[sample_indices] >= null_pore_pixel_threshold
    valid_counts = sampled_valid.sum(axis=1)
    valid_sample_fractions = valid_counts / param_count
    has_detectable_pores = valid_sample_fractions >= min_valid_sample_fraction

    max_valid_count = int(valid_counts.max(initial=0))
    histograms = np.zeros((bootstrap_replicates, max_valid_count + 1), dtype=np.int64)
    replicate_weights = np.zeros((bootstrap_replicates, param_count), dtype=np.uint16)
    for replicate_index in range(bootstrap_replicates):
        if not has_detectable_pores[replicate_index]:
            continue
        valid_indices = sample_indices[replicate_index, sampled_valid[replicate_index]]
        replicate_weights[replicate_index] = np.bincount(
            valid_indices,
            minlength=param_count,
        ).astype(np.uint16)

    active_replicates = np.flatnonzero(has_detectable_pores)
    for start in range(0, area_pixels, chunk_pixels):
        end = min(start + chunk_pixels, area_pixels)
        masks = (
            (c_channel[start:end][None, :] >= c_mins[:, None])
            & (y_channel[start:end][None, :] <= 64)
            & (k_channel[start:end][None, :] <= k_maxs[:, None])
        ).astype(np.uint16)

        for replicate_index in active_replicates:
            counts = replicate_weights[replicate_index] @ masks
            histograms[replicate_index] += np.bincount(
                counts,
                minlength=max_valid_count + 1,
            )

    by_threshold: dict[str, list[float]] = {
        threshold_column_name(threshold): []
        for threshold in thresholds
    }
    for replicate_index in range(bootstrap_replicates):
        if not has_detectable_pores[replicate_index]:
            replicate_porosities = zero_superposition_porosities(thresholds)
        else:
            replicate_porosities = porosities_from_count_histogram(
                histograms[replicate_index],
                area_pixels,
                thresholds,
            )
        for key, value in replicate_porosities.items():
            by_threshold[key].append(value)

    result: dict[str, float] = {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_valid_sample_fraction_mean": float(np.mean(valid_sample_fractions)),
        "bootstrap_detectable_pore_fraction": float(np.mean(has_detectable_pores)),
    }
    for key, values in by_threshold.items():
        values_array = np.array(values, dtype=np.float64)
        result[f"{key}_bootstrap_mean"] = float(np.mean(values_array))
        result[f"{key}_bootstrap_std"] = float(
            np.std(values_array, ddof=1) if len(values_array) > 1 else 0.0
        )
        result[f"{key}_bootstrap_p025"] = float(np.quantile(values_array, 0.025))
        result[f"{key}_bootstrap_p975"] = float(np.quantile(values_array, 0.975))

    return result


def clean_stem(path: Path) -> str:
    safe_chars = []
    for char in path.stem:
        safe_chars.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(safe_chars)


def measure_image(
    image_path: Path,
    params_df: pd.DataFrame,
    crop_metadata: dict[str, dict[str, int]],
    thresholds: list[float] | tuple[float, ...],
    null_pore_pixel_threshold: int = DEFAULT_NULL_PORE_PIXEL_THRESHOLD,
    min_valid_sample_fraction: float = DEFAULT_MIN_VALID_SAMPLE_FRACTION,
    scale_label: str = "100%",
    scale_factor: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_height, original_width = image.shape[:2]
    crop = find_crop_for_image(image_path, crop_metadata)
    image, applied_crop = crop_image(image, crop, image_path.name)
    crop_height, crop_width = image.shape[:2]

    if scale_factor <= 0:
        raise ValueError(f"Scale factor must be positive for {image_path.name}.")
    if not 0 <= min_valid_sample_fraction <= 1:
        raise ValueError("Minimum valid sample fraction must be between 0 and 1.")
    if scale_factor != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )

    img_cmyk = bgr_to_cmyk(image)
    height, width = image.shape[:2]
    image_area = height * width

    raw_mean_image = np.zeros((height, width), dtype=np.float32)
    superposition_mean_image = np.zeros((height, width), dtype=np.float32)
    per_param_rows: list[dict[str, Any]] = []
    valid_count = 0

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
        valid_for_superposition = pore_pixels >= null_pore_pixel_threshold

        per_param_rows.append(
            {
                "image": image_path.name,
                "image_path": str(image_path),
                "scale": scale_label,
                "scale_factor": scale_factor,
                "param_index": idx,
                "source_param_image": row.get("filename"),
                "experience": row.get("experience"),
                "min_pore_size": row.get("min_pore_size"),
                "c_min": c_min,
                "k_max": k_max,
                "pore_pixels": pore_pixels,
                "porosity": porosity,
                "component_count": component_count,
                "valid_for_superposition": valid_for_superposition,
                "crop_applied": applied_crop is not None,
                "crop_x": None if applied_crop is None else applied_crop["x"],
                "crop_y": None if applied_crop is None else applied_crop["y"],
                "crop_width": None if applied_crop is None else applied_crop["width"],
                "crop_height": None if applied_crop is None else applied_crop["height"],
            }
        )

        raw_mean_image += binary_image
        if valid_for_superposition:
            superposition_mean_image += binary_image
            valid_count += 1

    if len(params_df) > 0:
        raw_mean_image /= len(params_df)
    if valid_count > 0:
        superposition_mean_image /= valid_count

    porosities = [float(row["porosity"]) for row in per_param_rows]
    components = [int(row["component_count"]) for row in per_param_rows]
    valid_sample_fraction = valid_count / len(per_param_rows) if per_param_rows else 0.0
    has_detectable_pores = valid_sample_fraction >= min_valid_sample_fraction
    if has_detectable_pores:
        superposition_porosities, top_decile_normalizer = normalized_superposition_porosities(
            superposition_mean_image,
            thresholds,
        )
    else:
        superposition_mean_image[:] = 0
        superposition_porosities = zero_superposition_porosities(thresholds)
        top_decile_normalizer = 0.0

    summary: dict[str, Any] = {
        "image": image_path.name,
        "image_path": str(image_path),
        "scale": scale_label,
        "scale_factor": scale_factor,
        "original_height": original_height,
        "original_width": original_width,
        "crop_source_height": crop_height,
        "crop_source_width": crop_width,
        "height": height,
        "width": width,
        "area_pixels": image_area,
        "crop_applied": applied_crop is not None,
        "crop_x": None if applied_crop is None else applied_crop["x"],
        "crop_y": None if applied_crop is None else applied_crop["y"],
        "crop_width": None if applied_crop is None else applied_crop["width"],
        "crop_height": None if applied_crop is None else applied_crop["height"],
        "params_total": len(per_param_rows),
        "number_of_samples": valid_count,
        "valid_sample_fraction": valid_sample_fraction,
        "min_valid_sample_fraction": min_valid_sample_fraction,
        "has_detectable_pores": has_detectable_pores,
        "null_pore_pixel_threshold": null_pore_pixel_threshold,
        "top_decile_normalizer": top_decile_normalizer,
        "porosity_mean_by_param": np.mean(porosities) if porosities else np.nan,
        "porosity_median_by_param": np.median(porosities) if porosities else np.nan,
        "porosity_std_by_param": np.std(porosities, ddof=1) if len(porosities) > 1 else np.nan,
        "component_count_mean_by_param": np.mean(components) if components else np.nan,
        "component_count_median_by_param": np.median(components) if components else np.nan,
    }
    summary.update(superposition_porosities)

    return summary, per_param_rows, superposition_mean_image, image


def run_analysis(
    image_inputs: list[str],
    params_path: str | Path = "data/c_min_k_max_params.csv",
    crop_metadata_path: str | Path | None = None,
    output_dir: str | Path = "data/output/porosity_from_params",
    thresholds: list[float] | tuple[float, ...] = DEFAULT_THRESHOLDS,
    null_pore_pixel_threshold: int = DEFAULT_NULL_PORE_PIXEL_THRESHOLD,
    min_valid_sample_fraction: float = DEFAULT_MIN_VALID_SAMPLE_FRACTION,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_chunk_pixels: int = DEFAULT_BOOTSTRAP_CHUNK_PIXELS,
    scales: dict[str, float] | None = None,
    write_mean_images: bool = True,
    write_cropped_images: bool = False,
    cropped_images_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    params_path = Path(params_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_images_dir = (
        output_dir / "cropped_inputs"
        if cropped_images_dir is None
        else Path(cropped_images_dir)
    )

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
    scales = {"100%": 1.0} if scales is None else scales

    for image_path in image_paths:
        for scale_label, scale_factor in scales.items():
            print(f"Processing {image_path} at {scale_label}")
            summary, per_param_rows, mean_image, measured_image = measure_image(
                image_path,
                params_df,
                crop_metadata,
                thresholds,
                null_pore_pixel_threshold=null_pore_pixel_threshold,
                min_valid_sample_fraction=min_valid_sample_fraction,
                scale_label=scale_label,
                scale_factor=scale_factor,
            )
            summary.update(
                bootstrap_superposition_porosities(
                    measured_image,
                    params_df,
                    thresholds,
                    null_pore_pixel_threshold,
                    min_valid_sample_fraction,
                    bootstrap_replicates,
                    bootstrap_seed,
                    bootstrap_chunk_pixels,
                    f"{image_path.name} at {scale_label}",
                    output_dir,
                )
            )

            summary_rows.append(summary)
            all_per_param_rows.extend(per_param_rows)

            suffix = f"{clean_stem(image_path)}_{scale_label.replace('%', 'pct').replace('.', '_')}"
            if write_mean_images:
                out_image = output_dir / f"superposition_mean_mask_{suffix}.png"
                cv2.imwrite(str(out_image), mean_image.astype(np.uint8))

            if write_cropped_images and scale_factor == 1.0:
                cropped_images_dir.mkdir(parents=True, exist_ok=True)
                out_image = cropped_images_dir / f"{clean_stem(image_path)}_cropped.png"
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
    if write_cropped_images:
        print(f"Wrote cropped input images to {cropped_images_dir}")

    return summary_file, per_param_file


def main() -> int:
    args = parse_args()
    run_analysis(
        image_inputs=args.images,
        params_path=args.params,
        crop_metadata_path=args.crop_metadata,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        null_pore_pixel_threshold=args.null_pore_pixel_threshold,
        min_valid_sample_fraction=args.min_valid_sample_fraction,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_chunk_pixels=args.bootstrap_chunk_pixels,
        write_mean_images=not args.no_mean_images,
        write_cropped_images=args.write_cropped_images,
        cropped_images_dir=args.cropped_images_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
