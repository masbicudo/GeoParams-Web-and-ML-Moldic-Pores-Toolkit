from __future__ import annotations

import unittest
import uuid

from libs.job_registry import (
    list_active_jobs,
    list_active_workflows,
    register_job_provider,
    unregister_job_provider,
)


class JobRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider_keys = []

    def tearDown(self) -> None:
        for key in self.provider_keys:
            unregister_job_provider(key)

    def test_only_exposes_active_jobs_with_tool_origin(self) -> None:
        key = f"test-{uuid.uuid4().hex}"
        self.provider_keys.append(key)
        register_job_provider(
            key=key,
            label="Example Tool",
            list_jobs=lambda: [
                {
                    "id": "a" * 64,
                    "status": "queued",
                    "progress": 0,
                    "message": "Waiting",
                    "original_filename": "sample.png",
                    "dataset_name": "Example dataset",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "id": "b" * 64,
                    "status": "done",
                    "progress": 100,
                },
            ],
            tool_endpoint="example_tool",
            job_endpoint="example_job",
        )
        jobs = [job for job in list_active_jobs() if job["tool_key"] == key]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["tool_name"], "Example Tool")
        self.assertEqual(jobs[0]["title"], "sample.png")
        self.assertEqual(jobs[0]["status"], "queued")

    def test_human_input_workflow_is_active_without_being_a_compute_job(self) -> None:
        key = f"test-{uuid.uuid4().hex}"
        self.provider_keys.append(key)
        register_job_provider(
            key=key,
            label="Parameter Collection",
            list_jobs=lambda: [
                {
                    "id": "c" * 32,
                    "status": "awaiting_input",
                    "progress": 100,
                    "message": "Ready for parameter selection",
                    "title": "Collection cccccccccccc",
                    "detail": "Ready for parameter selection",
                }
            ],
            tool_endpoint="collection_tool",
            job_endpoint="collection_item",
        )

        workflows = [item for item in list_active_workflows() if item["tool_key"] == key]
        self.assertEqual(len(workflows), 1)
        self.assertEqual(workflows[0]["status"], "awaiting_input")
        self.assertEqual(workflows[0]["title"], "Collection cccccccccccc")


if __name__ == "__main__":
    unittest.main()
