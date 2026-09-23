"""Web integration for the scientific user-parameter porosity method."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from werkzeug.datastructures import FileStorage

from libs.upload_datasets import ALLOWED_EXTENSIONS, DatasetError, list_datasets, uploads_root


RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PRIMARY_POROSITY_KEY = "porosity_20p"
ProgressCallback = Callable[[float, str], None]


class PorosityToolError(ValueError):
    """Raised when an interactive porosity analysis cannot be performed."""


@lru_cache(maxsize=1)
def _scientific_module():
    """Load the existing scientific implementation as the single method source."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "user_params_porosity"
        / "measure_porosity_from_params.py"
    )
    if not module_path.is_file():
        raise PorosityToolError(
            "The scientific porosity implementation is not available in this installation."
        )
    spec = importlib.util.spec_from_file_location("geo_params_porosity_method", module_path)
    if spec is None or spec.loader is None:
        raise PorosityToolError("The scientific porosity implementation could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collected_output_root() -> Path:
    configured = os.getenv("GEO_PARAMS_COLLECTED_OUTPUT_DIR")
    if configured:
        return Path(configured).resolve()
    return (Path(__file__).resolve().parents[1] / "static" / "output").resolve()


def analyses_root() -> Path:
    root = uploads_root() / "porosity_analyses"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parameter_records(dataset_id: str) -> list[dict]:
    records: list[dict] = []
    for options_path in collected_output_root().glob("*/*/options.json"):
        try:
            with options_path.open("r", encoding="utf-8") as stream:
                options = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue

        if options.get("params_select.state") == "Cancel":
            continue
        source_dataset = options.get("image_select.dataset_id", "builtin")
        if source_dataset != dataset_id:
            continue

        points = list(options.get("params_select.clicked_points", []))
        points.extend(options.get("region_select.clicked_points", []))
        for point in points:
            try:
                clicked_x = int(point["x"]) * 8
                clicked_y = int(point["y"]) * 8
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= clicked_x < 256 and 0 <= clicked_y < 256):
                continue
            records.append(
                {
                    "clicked_x": clicked_x,
                    "clicked_y": clicked_y,
                    "filename": options.get("image_select.filename"),
                    "experience": options.get("user", {}).get("experience"),
                    "min_pore_size": options.get("initial_image_setup.min_pore_size"),
                }
            )
    return records


def load_dataset_parameters(dataset_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        _parameter_records(dataset_id),
        columns=["clicked_x", "clicked_y", "filename", "experience", "min_pore_size"],
    )


def parameter_sets() -> list[dict]:
    sets = [
        {
            "id": "builtin",
            "name": "Publication thin sections",
            "parameter_count": len(_parameter_records("builtin")),
        }
    ]
    for dataset in list_datasets():
        sets.append(
            {
                "id": dataset["id"],
                "name": dataset["name"],
                "parameter_count": len(_parameter_records(dataset["id"])),
            }
        )
    return sets


def _safe_run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise PorosityToolError("Invalid analysis identifier.")
    root = analyses_root()
    path = (root / run_id).resolve()
    if path.parent != root:
        raise PorosityToolError("Analysis path is outside the results directory.")
    return path


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_upload(
    dataset_id: str,
    upload: FileStorage,
    bootstrap: bool,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    def progress(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    progress(0.01, "Validating image and parameter dataset")
    available = {item["id"]: item for item in parameter_sets()}
    selected = available.get(dataset_id)
    if selected is None:
        raise PorosityToolError("Select a valid named parameter dataset.")

    params_df = load_dataset_parameters(dataset_id)
    if params_df.empty:
        raise PorosityToolError(
            f"The dataset '{selected['name']}' does not contain completed parameter measurements."
        )
    if not upload or not upload.filename:
        raise PorosityToolError("Select a thin-section image to analyze.")

    original_name = Path(upload.filename).name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise PorosityToolError(f"Unsupported image format. Allowed formats: {allowed}.")

    run_id = uuid.uuid4().hex
    run_dir = _safe_run_dir(run_id)
    run_dir.mkdir(parents=True)
    input_path = run_dir / f"input{extension}"
    upload.save(input_path)
    if cv2.imread(str(input_path), cv2.IMREAD_COLOR) is None:
        input_path.unlink(missing_ok=True)
        run_dir.rmdir()
        raise PorosityToolError("The uploaded file is not a readable image.")

    try:
        method = _scientific_module()
        measure_end = 0.58 if bootstrap else 0.92
        summary, per_parameter, mean_mask, measured_image = method.measure_image(
            input_path,
            params_df,
            {},
            method.DEFAULT_THRESHOLDS,
            progress_callback=lambda fraction, message: progress(
                0.05 + (measure_end - 0.05) * fraction,
                message,
            ),
        )

        mask_path = run_dir / "superposition_mean_mask.png"
        if not cv2.imwrite(str(mask_path), mean_mask.astype(np.uint8)):
            raise PorosityToolError("The superposition mask could not be saved.")

        if bootstrap:
            replicates = max(1, int(os.getenv("POROSITY_BOOTSTRAP_REPLICATES", "200")))
            summary.update(
                method.bootstrap_superposition_porosities(
                    measured_image,
                    params_df,
                    method.DEFAULT_THRESHOLDS,
                    method.DEFAULT_NULL_PORE_PIXEL_THRESHOLD,
                    method.DEFAULT_MIN_VALID_SAMPLE_FRACTION,
                    replicates,
                    42,
                    method.DEFAULT_BOOTSTRAP_CHUNK_PIXELS,
                    original_name,
                    run_dir,
                    progress_callback=lambda fraction, message: progress(
                        0.60 + 0.36 * fraction,
                        message,
                    ),
                )
            )

        progress(0.97, "Saving analysis results")
        clean_summary = {key: _json_value(value) for key, value in summary.items()}
        result = {
            "version": 1,
            "id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "dataset_name": selected["name"],
            "original_filename": original_name,
            "input_filename": input_path.name,
            "mask_filename": mask_path.name,
            "bootstrap_requested": bootstrap,
            "summary": clean_summary,
            "threshold_results": [
                {
                    "threshold": threshold,
                    "key": method.threshold_column_name(threshold),
                    "value": clean_summary[method.threshold_column_name(threshold)],
                }
                for threshold in method.DEFAULT_THRESHOLDS
            ],
            "per_parameter_count": len(per_parameter),
        }
        with (run_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        progress(1.0, "Analysis complete")
        return result
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        if isinstance(exc, PorosityToolError):
            raise
        raise PorosityToolError(
            "The porosity calculation could not be completed for this image."
        ) from exc


def load_result(run_id: str) -> dict:
    result_path = _safe_run_dir(run_id) / "result.json"
    if not result_path.is_file():
        raise PorosityToolError("Analysis result not found.")
    try:
        with result_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PorosityToolError("Analysis result could not be read.") from exc


def result_image_path(run_id: str, kind: str) -> Path:
    result = load_result(run_id)
    filenames = {
        "input": result["input_filename"],
        "mask": result["mask_filename"],
    }
    if kind not in filenames:
        raise PorosityToolError("Invalid analysis image type.")
    filename = filenames[kind]
    if Path(filename).name != filename:
        raise PorosityToolError("Invalid analysis image filename.")
    path = (_safe_run_dir(run_id) / filename).resolve()
    if not path.is_file() or path.parent != _safe_run_dir(run_id):
        raise PorosityToolError("Analysis image not found.")
    return path
