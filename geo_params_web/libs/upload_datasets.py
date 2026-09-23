"""Persistent custom thin-section datasets stored outside the container image."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import cv2
from werkzeug.datastructures import FileStorage


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DATASET_ID_RE = re.compile(r"^[0-9a-f]{32}$")
IMAGE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
PREVIEW_PERCENTAGE = 12.5
UNIT_TO_MICROMETERS = {"um": 1.0, "mm": 1000.0, "cm": 10000.0}


class DatasetError(ValueError):
    """Raised when dataset input is invalid or unsafe."""


def uploads_root() -> Path:
    configured = os.getenv("GEO_PARAMS_UPLOADS_DIR")
    if configured:
        return Path(configured).resolve()
    return (Path(__file__).resolve().parents[1] / "data" / "uploads").resolve()


def ensure_uploads_root() -> Path:
    root = uploads_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_dir(dataset_id: str) -> Path:
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise DatasetError("Invalid dataset identifier.")
    root = ensure_uploads_root()
    path = (root / dataset_id).resolve()
    if path.parent != root:
        raise DatasetError("Dataset path is outside the upload directory.")
    return path


def _metadata_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "dataset.json"


def _write_metadata(dataset: dict) -> None:
    path = _metadata_path(dataset["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dataset, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def load_dataset(dataset_id: str) -> dict:
    path = _metadata_path(dataset_id)
    if not path.is_file():
        raise DatasetError("Dataset not found.")
    try:
        with path.open("r", encoding="utf-8") as stream:
            dataset = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError("Dataset metadata could not be read.") from exc
    if dataset.get("id") != dataset_id or not isinstance(dataset.get("images"), list):
        raise DatasetError("Dataset metadata is invalid.")
    return dataset


def list_datasets() -> list[dict]:
    datasets: list[dict] = []
    root = ensure_uploads_root()
    for path in root.iterdir():
        if not path.is_dir() or not DATASET_ID_RE.fullmatch(path.name):
            continue
        try:
            dataset = load_dataset(path.name)
        except DatasetError:
            continue
        datasets.append(dataset)
    return sorted(datasets, key=lambda item: item.get("created_at", ""), reverse=True)


def _validated_dataset_name(name: str) -> str:
    name = " ".join(name.strip().split())
    if not name:
        raise DatasetError("Enter a name for the dataset.")
    if len(name) > 120:
        raise DatasetError("Dataset names must contain at most 120 characters.")
    return name


def _extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DatasetError(f"Unsupported image format. Allowed formats: {allowed}.")
    return extension


def _decode_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DatasetError(f"{path.name} is not a readable image.")
    return image


def _save_preview(image, output_path: Path) -> float:
    height, width = image.shape[:2]
    preview_width = max(1, round(width * PREVIEW_PERCENTAGE / 100.0))
    preview_height = max(1, round(height * PREVIEW_PERCENTAGE / 100.0))
    resized = cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise DatasetError("The image preview could not be written.")
    return preview_width / width


def create_dataset(name: str, files: list[FileStorage]) -> dict:
    name = _validated_dataset_name(name)
    uploads = [item for item in files if item and item.filename]
    if not uploads:
        raise DatasetError("Select at least one thin-section image.")

    dataset_id = uuid.uuid4().hex
    directory = _dataset_dir(dataset_id)
    originals = directory / "originals"
    previews = directory / "derived" / "12.5"
    originals.mkdir(parents=True)
    previews.mkdir(parents=True)

    dataset = {
        "version": 1,
        "id": dataset_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images": [],
    }

    try:
        for upload in uploads:
            original_filename = Path(upload.filename or "image").name
            extension = _extension(original_filename)
            image_id = uuid.uuid4().hex[:16]
            stored_filename = f"{image_id}{extension}"
            preview_filename = f"{image_id}.jpg"
            original_path = originals / stored_filename
            upload.save(original_path)

            image = _decode_image(original_path)
            height, width = image.shape[:2]
            preview_scale = _save_preview(image, previews / preview_filename)
            dataset["images"].append(
                {
                    "id": image_id,
                    "original_filename": original_filename,
                    "display_name": Path(original_filename).stem,
                    "stored_filename": stored_filename,
                    "preview_filename": preview_filename,
                    "width_px": width,
                    "height_px": height,
                    "preview_scale": preview_scale,
                    "calibration": None,
                }
            )
        _write_metadata(dataset)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return dataset


def find_image(dataset: dict, image_id: str) -> dict:
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise DatasetError("Invalid image identifier.")
    for image in dataset["images"]:
        if image.get("id") == image_id:
            return image
    raise DatasetError("Image not found in this dataset.")


def preview_path(dataset_id: str, image: dict) -> Path:
    filename = image.get("preview_filename", "")
    if Path(filename).name != filename:
        raise DatasetError("Invalid preview filename.")
    path = (_dataset_dir(dataset_id) / "derived" / "12.5" / filename).resolve()
    if not path.is_file():
        raise DatasetError("Image preview not found.")
    return path


def update_image_metadata(
    dataset_id: str,
    image_id: str,
    display_name: str,
    pixel_distance_preview: float | None,
    physical_distance: float | None,
    unit: str | None,
) -> dict:
    dataset = load_dataset(dataset_id)
    image = find_image(dataset, image_id)
    display_name = " ".join(display_name.strip().split())
    if not display_name:
        raise DatasetError("Enter a display name for the image.")
    if len(display_name) > 160:
        raise DatasetError("Image names must contain at most 160 characters.")
    image["display_name"] = display_name

    if physical_distance is None and unit is None:
        image["calibration"] = None
    else:
        if pixel_distance_preview is None or pixel_distance_preview <= 0:
            raise DatasetError("Mark both endpoints of the scale bar.")
        if physical_distance is None or physical_distance <= 0:
            raise DatasetError("Enter the physical length of the scale bar.")
        if unit not in UNIT_TO_MICROMETERS:
            raise DatasetError("Select a valid physical unit.")
        preview_scale = float(image["preview_scale"])
        pixel_distance_original = pixel_distance_preview / preview_scale
        micrometers = physical_distance * UNIT_TO_MICROMETERS[unit]
        image["calibration"] = {
            "pixel_distance_original": pixel_distance_original,
            "physical_distance": physical_distance,
            "unit": unit,
            "micrometers_per_pixel": micrometers / pixel_distance_original,
        }
    _write_metadata(dataset)
    return image


def delete_dataset(dataset_id: str) -> None:
    directory = _dataset_dir(dataset_id)
    if not directory.is_dir():
        raise DatasetError("Dataset not found.")
    shutil.rmtree(directory)
