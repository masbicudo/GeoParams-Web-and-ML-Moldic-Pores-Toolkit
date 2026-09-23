"""Background job runner for interactive porosity analyses."""

from __future__ import annotations

from pathlib import Path
import re
import threading
import uuid

from werkzeug.datastructures import FileStorage

from libs.porosity_tool import PorosityToolError, analyses_root, analyze_upload
from libs.upload_datasets import ALLOWED_EXTENSIONS


JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _public_job(job: dict) -> dict:
    return dict(job)


def _update(job_id: str, **values) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(values)


def submit_porosity_job(
    dataset_id: str,
    upload: FileStorage | None,
    bootstrap: bool,
) -> dict:
    if not upload or not upload.filename:
        raise PorosityToolError("Select a thin-section image to analyze.")
    original_name = Path(upload.filename).name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise PorosityToolError(f"Unsupported image format. Allowed formats: {allowed}.")

    job_id = uuid.uuid4().hex
    incoming_path = analyses_root() / f".incoming-{job_id}{extension}"
    upload.save(incoming_path)
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Waiting to start",
        "bootstrap": bootstrap,
        "result_id": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job

    def run() -> None:
        _update(job_id, status="running", progress=1, message="Starting analysis")

        def report(fraction: float, message: str) -> None:
            _update(
                job_id,
                progress=max(0, min(99, round(100 * fraction))),
                message=message,
            )

        try:
            with incoming_path.open("rb") as stream:
                result = analyze_upload(
                    dataset_id,
                    FileStorage(stream=stream, filename=original_name),
                    bootstrap,
                    progress_callback=report,
                )
            _update(
                job_id,
                status="done",
                progress=100,
                message="Analysis complete",
                result_id=result["id"],
            )
        except Exception as exc:
            message = str(exc) if isinstance(exc, PorosityToolError) else (
                "The porosity calculation could not be completed for this image."
            )
            _update(job_id, status="error", message=message, error=message)
        finally:
            incoming_path.unlink(missing_ok=True)

    threading.Thread(
        target=run,
        name=f"porosity-{job_id[:8]}",
        daemon=True,
    ).start()
    return _public_job(job)


def get_porosity_job(job_id: str) -> dict:
    if not JOB_ID_RE.fullmatch(job_id):
        raise PorosityToolError("Invalid analysis job identifier.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise PorosityToolError("Analysis job not found. It may have been interrupted.")
        return _public_job(job)
