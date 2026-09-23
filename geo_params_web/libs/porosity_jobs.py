"""Persistent, serialized background jobs for porosity analyses."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid

import pandas as pd
from werkzeug.datastructures import FileStorage

from libs.porosity_tool import (
    PorosityToolError,
    analyses_root,
    analysis_identity,
    analyze_saved_image,
    load_result,
    selected_parameter_set,
)
from libs.upload_datasets import ALLOWED_EXTENSIONS


JOB_ID_RE = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{64})$")
HASH_JOB_ID_RE = re.compile(r"^[0-9a-f]{64}$")
ACTIVE_STATUSES = {"queued", "running"}

_condition = threading.Condition(threading.RLock())
_queue: deque[str] = deque()
_active_job_id: str | None = None
_worker_started = False
_initialized = False
_manager_root: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise PorosityToolError("Invalid analysis job identifier.")
    root = analyses_root()
    path = (root / job_id).resolve()
    if path.parent != root:
        raise PorosityToolError("Analysis job path is outside the results directory.")
    return path


def _job_file(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _write_job(job: dict) -> None:
    path = _job_file(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(job, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _read_persistent_job(job_id: str) -> dict | None:
    path = _job_file(job_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            job = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PorosityToolError("Analysis job metadata could not be read.") from exc
    if job.get("id") != job_id:
        raise PorosityToolError("Analysis job metadata is invalid.")
    return job


def _legacy_job(job_id: str) -> dict | None:
    try:
        result = load_result(job_id)
    except PorosityToolError:
        return None
    return {
        "version": 1,
        "id": job_id,
        "analysis_hash": None,
        "status": "done",
        "progress": 100,
        "message": "Analysis complete",
        "error": None,
        "bootstrap": bool(result.get("bootstrap_requested")),
        "dataset_id": result.get("dataset_id"),
        "dataset_name": result.get("dataset_name", "Unknown dataset"),
        "original_filename": result.get("original_filename", "Unknown image"),
        "created_at": result.get("created_at"),
        "updated_at": result.get("created_at"),
        "result_id": job_id,
        "legacy": True,
    }


def _decorate_job(job: dict) -> dict:
    public = dict(job)
    public["short_id"] = job["id"][:12]
    public["legacy"] = bool(job.get("legacy", False))
    public["porosity_20p"] = None
    if job.get("status") == "done":
        try:
            result = load_result(job.get("result_id") or job["id"])
            public["porosity_20p"] = result.get("summary", {}).get("porosity_20p")
        except PorosityToolError:
            public["status"] = "error"
            public["message"] = "The saved result is missing or unreadable"
            public["error"] = public["message"]
    return public


def _refresh_queue_messages_locked() -> None:
    for position, job_id in enumerate(_queue, start=1):
        job = _read_persistent_job(job_id)
        if job is None or job.get("status") != "queued":
            continue
        ahead = position - 1 + (1 if _active_job_id else 0)
        job["queue_position"] = position
        job["message"] = (
            f"Waiting for the processing slot ({ahead} job(s) ahead)"
            if ahead
            else "Waiting for the processing slot"
        )
        job["updated_at"] = _utc_now()
        _write_job(job)


def _initialize_locked() -> None:
    global _initialized, _manager_root
    root = analyses_root().resolve()
    if _manager_root != root:
        if _active_job_id is not None:
            raise RuntimeError("Cannot change the analysis root while a job is running.")
        _queue.clear()
        _initialized = False
        _manager_root = root
    if _initialized:
        return

    resumable: list[dict] = []
    for directory in root.iterdir():
        if not directory.is_dir() or not HASH_JOB_ID_RE.fullmatch(directory.name):
            continue
        job = _read_persistent_job(directory.name)
        if job is None or job.get("status") not in ACTIVE_STATUSES:
            continue
        job["status"] = "queued"
        job["progress"] = 0
        job["message"] = "Waiting to resume after application restart"
        job["updated_at"] = _utc_now()
        _write_job(job)
        resumable.append(job)
    for job in sorted(resumable, key=lambda item: item.get("created_at", "")):
        _queue.append(job["id"])
    _initialized = True
    _refresh_queue_messages_locked()


def _ensure_worker_locked() -> None:
    global _worker_started
    if _worker_started:
        return
    threading.Thread(
        target=_worker_loop,
        name="porosity-job-worker",
        daemon=True,
    ).start()
    _worker_started = True


def _update_job(job_id: str, **values) -> dict:
    with _condition:
        job = _read_persistent_job(job_id)
        if job is None:
            raise PorosityToolError("Analysis job not found.")
        changed = any(job.get(key) != value for key, value in values.items())
        if changed:
            job.update(values)
            job["updated_at"] = _utc_now()
            _write_job(job)
        return job


def _worker_loop() -> None:
    global _active_job_id
    while True:
        with _condition:
            while not _queue:
                _condition.wait()
            job_id = _queue.popleft()
            job = _read_persistent_job(job_id)
            if job is None or job.get("status") != "queued":
                continue
            _active_job_id = job_id
            job.update(
                status="running",
                progress=1,
                message="Starting analysis",
                queue_position=0,
                updated_at=_utc_now(),
            )
            _write_job(job)
            _refresh_queue_messages_locked()

        def report(fraction: float, message: str) -> None:
            _update_job(
                job_id,
                progress=max(0, min(99, round(100 * fraction))),
                message=message,
            )

        try:
            job_dir = _job_dir(job_id)
            input_path = (job_dir / job["input_filename"]).resolve()
            params_path = (job_dir / job["parameters_filename"]).resolve()
            if input_path.parent != job_dir or params_path.parent != job_dir:
                raise PorosityToolError("Analysis job contains an invalid file path.")
            params_df = pd.read_csv(params_path)
            result = analyze_saved_image(
                job["dataset_id"],
                job["dataset_name"],
                params_df,
                input_path,
                job["original_filename"],
                bool(job["bootstrap"]),
                job_id,
                progress_callback=report,
            )
            _update_job(
                job_id,
                status="done",
                progress=100,
                message="Analysis complete",
                result_id=result["id"],
                error=None,
            )
        except Exception as exc:
            message = str(exc) if isinstance(exc, PorosityToolError) else (
                "The porosity calculation could not be completed for this image."
            )
            _update_job(
                job_id,
                status="error",
                message=message,
                error=message,
            )
        finally:
            with _condition:
                _active_job_id = None
                _refresh_queue_messages_locked()
                _condition.notify_all()


def _clean_original_name(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def submit_porosity_job(
    dataset_id: str,
    upload: FileStorage | None,
    bootstrap: bool,
) -> dict:
    if not upload or not upload.filename:
        raise PorosityToolError("Select a thin-section image to analyze.")
    original_name = _clean_original_name(upload.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise PorosityToolError(f"Unsupported image format. Allowed formats: {allowed}.")

    selected, params_df = selected_parameter_set(dataset_id)
    incoming_path = analyses_root() / f".incoming-{uuid.uuid4().hex}{extension}"
    upload.save(incoming_path)
    try:
        job_id, image_sha256, parameters_sha256 = analysis_identity(
            dataset_id,
            params_df,
            incoming_path,
        )
        with _condition:
            _initialize_locked()
            _ensure_worker_locked()
            existing = _read_persistent_job(job_id)
            if existing is not None or (_job_dir(job_id) / "result.json").is_file():
                incoming_path.unlink(missing_ok=True)
                duplicate = _decorate_job(existing or _legacy_job(job_id))
                duplicate["duplicate"] = True
                duplicate["requested_bootstrap"] = bootstrap
                return duplicate

            job_dir = _job_dir(job_id)
            job_dir.mkdir(parents=True)
            input_filename = f"input{extension}"
            os.replace(incoming_path, job_dir / input_filename)
            parameters_filename = "parameters.csv"
            params_df.to_csv(job_dir / parameters_filename, index=False)
            now = _utc_now()
            job = {
                "version": 2,
                "id": job_id,
                "analysis_hash": job_id,
                "image_sha256": image_sha256,
                "parameters_sha256": parameters_sha256,
                "dataset_id": dataset_id,
                "dataset_name": selected["name"],
                "original_filename": original_name,
                "input_filename": input_filename,
                "parameters_filename": parameters_filename,
                "bootstrap": bootstrap,
                "status": "queued",
                "progress": 0,
                "message": "Waiting for the processing slot",
                "error": None,
                "result_id": None,
                "run_count": 1,
                "created_at": now,
                "updated_at": now,
                "queue_position": len(_queue) + 1,
            }
            _write_job(job)
            _queue.append(job_id)
            _refresh_queue_messages_locked()
            _condition.notify_all()
            public = _decorate_job(job)
            public["duplicate"] = False
            return public
    except Exception:
        incoming_path.unlink(missing_ok=True)
        raise


def get_porosity_job(job_id: str) -> dict:
    with _condition:
        _initialize_locked()
        _ensure_worker_locked()
        job = _read_persistent_job(job_id) or _legacy_job(job_id)
        if job is None:
            raise PorosityToolError("Analysis job not found.")
        return _decorate_job(job)


def list_porosity_jobs() -> list[dict]:
    with _condition:
        _initialize_locked()
        _ensure_worker_locked()
        jobs: list[dict] = []
        for directory in analyses_root().iterdir():
            if not directory.is_dir() or not JOB_ID_RE.fullmatch(directory.name):
                continue
            job = _read_persistent_job(directory.name) or _legacy_job(directory.name)
            if job is not None:
                jobs.append(_decorate_job(job))
        return sorted(
            jobs,
            key=lambda item: item.get("updated_at") or item.get("created_at") or "",
            reverse=True,
        )


def recalculate_porosity_job(job_id: str, bootstrap: bool) -> dict:
    with _condition:
        _initialize_locked()
        _ensure_worker_locked()
        job = _read_persistent_job(job_id)
        if job is None:
            raise PorosityToolError(
                "This older result cannot be recalculated in place. Upload the image again."
            )
        if job.get("status") in ACTIVE_STATUSES or job_id == _active_job_id:
            raise PorosityToolError("This analysis is already queued or running.")

        job_dir = _job_dir(job_id)
        for filename in ("result.json", "superposition_mean_mask.png"):
            (job_dir / filename).unlink(missing_ok=True)
        now = _utc_now()
        job.update(
            bootstrap=bootstrap,
            status="queued",
            progress=0,
            message="Waiting for the processing slot",
            error=None,
            result_id=None,
            run_count=int(job.get("run_count", 1)) + 1,
            updated_at=now,
            queue_position=len(_queue) + 1,
        )
        _write_job(job)
        _queue.append(job_id)
        _refresh_queue_messages_locked()
        _condition.notify_all()
        return _decorate_job(job)


def delete_porosity_jobs(job_ids: list[str]) -> int:
    unique_ids = list(dict.fromkeys(job_ids))
    if not unique_ids:
        raise PorosityToolError("Select at least one analysis to delete.")
    with _condition:
        _initialize_locked()
        for job_id in unique_ids:
            job = _read_persistent_job(job_id) or _legacy_job(job_id)
            if job is None:
                raise PorosityToolError(f"Analysis {job_id[:12]} was not found.")
            if job_id == _active_job_id or job.get("status") == "running":
                raise PorosityToolError(
                    "A running analysis cannot be deleted. Wait for it to finish first."
                )
        for job_id in unique_ids:
            while job_id in _queue:
                _queue.remove(job_id)
            directory = _job_dir(job_id)
            if directory.is_dir():
                shutil.rmtree(directory)
        _refresh_queue_messages_locked()
        return len(unique_ids)
