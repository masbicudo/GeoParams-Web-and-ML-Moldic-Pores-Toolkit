"""Registry that presents active workflows from independent application tools."""

from __future__ import annotations

from collections.abc import Callable
import threading


_providers: dict[str, dict] = {}
_providers_lock = threading.Lock()


def register_job_provider(
    key: str,
    label: str,
    list_jobs: Callable[[], list[dict]],
    tool_endpoint: str,
    job_endpoint: str,
) -> None:
    """Register or replace a tool-specific job provider."""
    with _providers_lock:
        _providers[key] = {
            "key": key,
            "label": label,
            "list_jobs": list_jobs,
            "tool_endpoint": tool_endpoint,
            "job_endpoint": job_endpoint,
        }


def list_active_jobs() -> list[dict]:
    """Return nonterminal workflows using a tool-neutral presentation schema."""
    with _providers_lock:
        providers = list(_providers.values())

    active: list[dict] = []
    for provider in providers:
        for job in provider["list_jobs"]():
            if job.get("status") not in {"queued", "running", "awaiting_input"}:
                continue
            active.append(
                {
                    "id": job["id"],
                    "short_id": job.get("short_id", job["id"][:12]),
                    "status": job["status"],
                    "progress": job.get("progress", 0),
                    "message": job.get("message", ""),
                    "title": job.get("title") or job.get(
                        "original_filename", f"Workflow {job['id'][:12]}"
                    ),
                    "detail": job.get("detail") or job.get("dataset_name", ""),
                    "created_at": job.get("created_at"),
                    "tool_key": provider["key"],
                    "tool_name": provider["label"],
                    "tool_endpoint": provider["tool_endpoint"],
                    "job_endpoint": provider["job_endpoint"],
                }
            )
    return sorted(
        active,
        key=lambda item: item.get("created_at") or "",
    )


def list_active_workflows() -> list[dict]:
    """Preferred terminology for the global nonterminal item list."""
    return list_active_jobs()
