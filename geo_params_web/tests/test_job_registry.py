from __future__ import annotations

import unittest
import uuid

from libs.job_registry import list_active_jobs, register_job_provider


class JobRegistryTests(unittest.TestCase):
    def test_only_exposes_active_jobs_with_tool_origin(self) -> None:
        key = f"test-{uuid.uuid4().hex}"
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


if __name__ == "__main__":
    unittest.main()
