"""CSV export for selected persisted porosity results."""

from __future__ import annotations

import csv
from io import StringIO

from libs.porosity_jobs import get_porosity_job
from libs.porosity_tool import PorosityToolError, load_result


BASE_COLUMNS = [
    "analysis_id",
    "created_at",
    "dataset_id",
    "dataset_name",
    "image_name",
    "bootstrap_requested",
]
SUMMARY_PRIORITY = [
    "porosity_20p",
    "porosity_20p_bootstrap_p025",
    "porosity_20p_bootstrap_p975",
    "porosity_20p_bootstrap_std",
    "params_total",
    "number_of_samples",
    "valid_sample_fraction",
    "has_detectable_pores",
    "width",
    "height",
    "area_pixels",
]


def _safe_text(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def build_results_csv(job_ids: list[str]) -> str:
    unique_ids = list(dict.fromkeys(job_ids))
    if not unique_ids:
        raise PorosityToolError("Select at least one completed analysis to export.")

    rows: list[dict] = []
    summary_columns: set[str] = set()
    for job_id in unique_ids:
        job = get_porosity_job(job_id)
        if job.get("status") != "done":
            raise PorosityToolError(
                "CSV export only supports completed analyses."
            )
        result = load_result(job.get("result_id") or job_id)
        summary = result.get("summary", {})
        summary_columns.update(summary)
        row = {
            "analysis_id": job_id,
            "created_at": result.get("created_at"),
            "dataset_id": result.get("dataset_id"),
            "dataset_name": _safe_text(result.get("dataset_name")),
            "image_name": _safe_text(result.get("original_filename")),
            "bootstrap_requested": bool(result.get("bootstrap_requested")),
        }
        row.update(summary)
        rows.append(row)

    ordered_summary = [name for name in SUMMARY_PRIORITY if name in summary_columns]
    ordered_summary.extend(sorted(summary_columns - set(ordered_summary)))
    columns = BASE_COLUMNS + ordered_summary
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
