"""Persistent task records for the human-in-the-loop parameter workflow."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

from libs.upload_datasets import uploads_root


COLLECTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ACTIVE_STATUSES = {"awaiting_input", "queued", "running"}
TERMINAL_STATUSES = {"done", "canceled", "error"}

STAGE_LABELS = {
    "user_information": "Ready for background information",
    "image_selection": "Ready for image and region selection",
    "image_processing": "Preparing the parameter space",
    "parameter_selection": "Ready for parameter selection",
    "completed": "Collection complete",
    "canceled": "Collection discarded",
    "error": "Processing failed",
}

_lock = threading.RLock()


class ParameterCollectionError(ValueError):
    """Raised when a collection task is missing or invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collections_root() -> Path:
    root = uploads_root() / "parameter_collections"
    root.mkdir(parents=True, exist_ok=True)
    return root


def collected_output_root() -> Path:
    configured = os.getenv("GEO_PARAMS_COLLECTED_OUTPUT_DIR")
    if configured:
        return Path(configured).resolve()
    return (Path(__file__).resolve().parents[1] / "static" / "output").resolve()


def _collection_dir(collection_id: str) -> Path:
    if not COLLECTION_ID_RE.fullmatch(collection_id):
        raise ParameterCollectionError("Invalid parameter collection identifier.")
    root = collections_root().resolve()
    path = (root / collection_id).resolve()
    if path.parent != root:
        raise ParameterCollectionError("Collection path is outside the data directory.")
    return path


def _record_path(collection_id: str) -> Path:
    return _collection_dir(collection_id) / "collection.json"


def _session_path(collection_id: str) -> Path:
    return _collection_dir(collection_id) / "session.json"


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _session_snapshot(session: dict) -> dict:
    """Return restart-safe state without runtime-only thread objects."""
    snapshot = {}
    for key, value in session.items():
        if key != "tasks":
            snapshot[key] = deepcopy(value)
            continue
        snapshot["tasks"] = {}
        for task_name, task in value.items():
            if not isinstance(task, dict):
                continue
            safe_task = {
                task_key: deepcopy(task_value)
                for task_key, task_value in task.items()
                if task_key not in {"alive_tag", "cache"}
            }
            snapshot["tasks"][task_name] = safe_task
    return snapshot


def _restore_runtime_fields(session: dict) -> dict:
    restored = deepcopy(session)
    for task in restored.get("tasks", {}).values():
        if not isinstance(task, dict):
            continue
        task.setdefault("alive_tag", None)
        task.setdefault("cache", {})
    return restored


def save_collection_session(collection_id: str, session: dict) -> None:
    with _lock:
        _atomic_json_write(_session_path(collection_id), _session_snapshot(session))


def load_collection_session(collection_id: str) -> dict:
    path = _session_path(collection_id)
    if not path.is_file():
        raise ParameterCollectionError("The saved collection state is unavailable.")
    try:
        with path.open("r", encoding="utf-8") as stream:
            session = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterCollectionError("The saved collection state could not be read.") from exc
    if not isinstance(session, dict):
        raise ParameterCollectionError("The saved collection state is invalid.")
    return _restore_runtime_fields(session)


def _write_record(record: dict) -> None:
    _atomic_json_write(_record_path(record["id"]), record)


def _read_record(collection_id: str) -> dict | None:
    path = _record_path(collection_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterCollectionError("Collection metadata could not be read.") from exc
    if record.get("id") != collection_id:
        raise ParameterCollectionError("Collection metadata is invalid.")
    return record


def create_parameter_collection(session_id: str, counter: int, session: dict) -> dict:
    collection_id = uuid.uuid4().hex
    session["parameter_collection_id"] = collection_id
    now = _utc_now()
    record = {
        "version": 1,
        "id": collection_id,
        "session_id": session_id,
        "counter": int(counter),
        "status": "awaiting_input",
        "stage": "user_information",
        "progress": 0,
        "message": "Ready for background information",
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    with _lock:
        _write_record(record)
        save_collection_session(collection_id, session)
    return _decorate(record)


def update_parameter_collection(
    collection_id: str,
    *,
    session: dict | None = None,
    **values,
) -> dict:
    with _lock:
        record = _read_record(collection_id)
        if record is None:
            raise ParameterCollectionError("Parameter collection not found.")
        if record.get("status") in TERMINAL_STATUSES and any(
            key in values for key in ("status", "stage", "progress", "message")
        ):
            raise ParameterCollectionError("Completed collections are read-only.")
        record.update(values)
        record["updated_at"] = _utc_now()
        _write_record(record)
        if session is not None:
            save_collection_session(collection_id, session)
        return _decorate(record)


def _legacy_id(relative_path: Path) -> str:
    identity = "legacy-parameter-collection:" + relative_path.as_posix()
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _load_options(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _technical_summary(options: dict) -> dict:
    points = []
    for point in options.get("params_select.clicked_points", []):
        try:
            x = int(point["x"])
            y = int(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        points.append({"x": x, "y": y, "black_max": x * 8, "cyan_min": y * 8})

    dataset_id = options.get("image_select.dataset_id", "builtin")
    return {
        "source_kind": "Publication dataset" if dataset_id == "builtin" else "Custom dataset",
        "parameter_count": len(points),
        "parameters": points,
        "minimum_pore_area_px": options.get("initial_image_setup.min_pore_size"),
        "priority": options.get("params_select.priority"),
        "has_physical_scale": options.get("image_select.mm_per_pixel") is not None,
    }


def _legacy_records(excluded_outputs: set[tuple[str, int]]) -> list[dict]:
    root = collected_output_root()
    if not root.is_dir():
        return []
    records = []
    for path in root.glob("**/options.json"):
        options = _load_options(path)
        if options is None:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        session_id = parts[0] if parts else ""
        try:
            counter = int(parts[1]) if len(parts) >= 3 else 0
        except ValueError:
            counter = 0
        if (session_id, counter) in excluded_outputs:
            continue
        state = options.get("params_select.state")
        if state not in {"Done", "Cancel"}:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        records.append(
            {
                "version": 0,
                "id": _legacy_id(relative),
                "session_id": session_id,
                "counter": counter,
                "status": "done" if state == "Done" else "canceled",
                "stage": "completed" if state == "Done" else "canceled",
                "progress": 100,
                "message": "Collection complete" if state == "Done" else "Collection discarded",
                "error": None,
                "created_at": modified,
                "updated_at": modified,
                "legacy": True,
                "summary": _technical_summary(options),
            }
        )
    return records


def _decorate(record: dict) -> dict:
    public = dict(record)
    public["short_id"] = record["id"][:12]
    public["legacy"] = bool(record.get("legacy", False))
    public["title"] = f"Collection {public['short_id']}"
    public["detail"] = STAGE_LABELS.get(record.get("stage"), record.get("message", ""))
    if "summary" not in public:
        try:
            state = load_collection_session(record["id"])
            public["summary"] = _technical_summary(state.get("options", {}))
        except ParameterCollectionError:
            public["summary"] = _technical_summary({})
    return public


def list_parameter_collections() -> list[dict]:
    records = []
    excluded_outputs: set[tuple[str, int]] = set()
    with _lock:
        for directory in collections_root().iterdir():
            if not directory.is_dir() or not COLLECTION_ID_RE.fullmatch(directory.name):
                continue
            try:
                record = _read_record(directory.name)
            except ParameterCollectionError:
                continue
            if record is None:
                continue
            records.append(_decorate(record))
            excluded_outputs.add((record.get("session_id", ""), int(record.get("counter", 0))))
    records.extend(_legacy_records(excluded_outputs))
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def get_parameter_collection(collection_id: str) -> dict:
    with _lock:
        record = _read_record(collection_id)
        if record is not None:
            return _decorate(record)
    for legacy in _legacy_records(set()):
        if legacy["id"] == collection_id:
            return legacy
    raise ParameterCollectionError("Parameter collection not found.")
