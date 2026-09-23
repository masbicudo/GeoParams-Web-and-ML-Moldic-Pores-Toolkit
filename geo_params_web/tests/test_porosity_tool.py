from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from libs.porosity_tool import (
    analysis_identity,
    analyze_upload,
    load_dataset_parameters,
    load_result,
    parameter_sets,
    result_image_path,
)
from libs.porosity_jobs import (
    delete_porosity_jobs,
    get_porosity_job,
    list_porosity_jobs,
    recalculate_porosity_job,
    submit_porosity_job,
)
from libs.porosity_tool import analyses_root
from libs.porosity_exports import build_results_csv
from libs.upload_datasets import create_dataset


def blue_image_upload(
    filename: str = "new-thin-section.png",
    blue_value: int = 255,
) -> FileStorage:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 0] = blue_value
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    return FileStorage(stream=BytesIO(encoded.tobytes()), filename=filename)


class PorosityToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uploads = tempfile.TemporaryDirectory()
        self.collected = tempfile.TemporaryDirectory()
        self.previous_uploads = os.environ.get("GEO_PARAMS_UPLOADS_DIR")
        self.previous_collected = os.environ.get("GEO_PARAMS_COLLECTED_OUTPUT_DIR")
        self.previous_replicates = os.environ.get("POROSITY_BOOTSTRAP_REPLICATES")
        os.environ["GEO_PARAMS_UPLOADS_DIR"] = self.uploads.name
        os.environ["GEO_PARAMS_COLLECTED_OUTPUT_DIR"] = self.collected.name
        os.environ["POROSITY_BOOTSTRAP_REPLICATES"] = "3"

    def tearDown(self) -> None:
        for key, previous in (
            ("GEO_PARAMS_UPLOADS_DIR", self.previous_uploads),
            ("GEO_PARAMS_COLLECTED_OUTPUT_DIR", self.previous_collected),
            ("POROSITY_BOOTSTRAP_REPLICATES", self.previous_replicates),
        ):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self.collected.cleanup()
        self.uploads.cleanup()

    def _dataset_with_measurement(self) -> dict:
        dataset = create_dataset("Named measurements", [blue_image_upload("source.png")])
        output = Path(self.collected.name) / "session" / "1"
        output.mkdir(parents=True)
        with (output / "options.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "image_select.dataset_id": dataset["id"],
                    "image_select.filename": "Source image",
                    "params_select.state": "Done",
                    "params_select.clicked_points": [{"x": 0, "y": 31}],
                    "initial_image_setup.min_pore_size": 480,
                    "user": {"experience": 3},
                },
                stream,
            )
        return dataset

    def test_lists_named_dataset_with_measurement_count(self) -> None:
        dataset = self._dataset_with_measurement()
        item = next(item for item in parameter_sets() if item["id"] == dataset["id"])
        self.assertEqual(item["name"], "Named measurements")
        self.assertEqual(item["parameter_count"], 1)

    def test_identity_uses_image_and_exact_parameter_values(self) -> None:
        dataset = self._dataset_with_measurement()
        image_path = Path(self.uploads.name) / dataset["id"] / "originals" / dataset["images"][0]["stored_filename"]
        params = load_dataset_parameters(dataset["id"])
        first, image_hash, params_hash = analysis_identity(dataset["id"], params, image_path)
        repeated, _, _ = analysis_identity(dataset["id"], params, image_path)
        changed_params = params.copy()
        changed_params.loc[0, "clicked_x"] = 8
        changed, _, changed_params_hash = analysis_identity(
            dataset["id"], changed_params, image_path
        )
        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(image_hash), 64)
        self.assertNotEqual(params_hash, changed_params_hash)
        self.assertNotEqual(first, changed)

    def test_analyzes_upload_with_existing_scientific_method(self) -> None:
        dataset = self._dataset_with_measurement()
        updates = []
        result = analyze_upload(
            dataset["id"],
            blue_image_upload(),
            bootstrap=False,
            progress_callback=lambda fraction, message: updates.append((fraction, message)),
        )
        self.assertAlmostEqual(result["summary"]["porosity_20p"], 1.0)
        self.assertFalse(result["bootstrap_requested"])
        self.assertTrue(result_image_path(result["id"], "mask").is_file())
        self.assertEqual(load_result(result["id"])["dataset_name"], "Named measurements")
        self.assertGreater(len(updates), 3)
        self.assertEqual(updates[-1], (1.0, "Analysis complete"))
        self.assertEqual(sorted(value for value, _ in updates), [value for value, _ in updates])

    def test_optional_bootstrap_adds_confidence_interval(self) -> None:
        dataset = self._dataset_with_measurement()
        result = analyze_upload(dataset["id"], blue_image_upload(), bootstrap=True)
        self.assertEqual(result["summary"]["bootstrap_replicates"], 3)
        self.assertAlmostEqual(
            result["summary"]["porosity_20p_bootstrap_p025"],
            1.0,
        )
        self.assertAlmostEqual(
            result["summary"]["porosity_20p_bootstrap_p975"],
            1.0,
        )

    def test_background_job_finishes_and_exposes_progress(self) -> None:
        dataset = self._dataset_with_measurement()
        job = submit_porosity_job(dataset["id"], blue_image_upload(), bootstrap=False)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = get_porosity_job(job["id"])
            if job["status"] in {"done", "error"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["progress"], 100)
        self.assertTrue(load_result(job["result_id"]))

    def test_duplicate_reuses_hash_and_can_recalculate_or_delete(self) -> None:
        dataset = self._dataset_with_measurement()
        job = submit_porosity_job(dataset["id"], blue_image_upload(), bootstrap=False)
        job = self._wait_for_job(job["id"])
        duplicate = submit_porosity_job(
            dataset["id"], blue_image_upload(), bootstrap=True
        )
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["id"], job["id"])
        self.assertTrue(any(item["id"] == job["id"] for item in list_porosity_jobs()))

        recalculated = recalculate_porosity_job(job["id"], bootstrap=True)
        self.assertEqual(recalculated["id"], job["id"])
        recalculated = self._wait_for_job(job["id"])
        self.assertEqual(recalculated["status"], "done")
        self.assertEqual(recalculated["run_count"], 2)

        csv_text = build_results_csv([job["id"]])
        self.assertIn("analysis_id", csv_text.splitlines()[0])
        self.assertIn("porosity_20p", csv_text.splitlines()[0])
        self.assertIn(job["id"], csv_text)

        self.assertEqual(delete_porosity_jobs([job["id"]]), 1)
        self.assertFalse(any(item["id"] == job["id"] for item in list_porosity_jobs()))

    def test_jobs_are_serialized_and_second_job_reports_waiting(self) -> None:
        dataset = self._dataset_with_measurement()
        started = threading.Event()
        release = threading.Event()
        concurrency_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_analysis(*args, **kwargs):
            nonlocal active, maximum_active
            run_id = args[6]
            with concurrency_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            started.set()
            release.wait(2)
            result = {
                "id": run_id,
                "dataset_id": args[0],
                "dataset_name": args[1],
                "original_filename": args[4],
                "bootstrap_requested": args[5],
                "created_at": "2026-01-01T00:00:00+00:00",
                "summary": {"porosity_20p": 0.5},
            }
            with (analyses_root() / run_id / "result.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(result, stream)
            with concurrency_lock:
                active -= 1
            return result

        with patch("libs.porosity_jobs.analyze_saved_image", side_effect=fake_analysis):
            first = submit_porosity_job(
                dataset["id"], blue_image_upload("first.png", 255), bootstrap=False
            )
            self.assertTrue(started.wait(1))
            second = submit_porosity_job(
                dataset["id"], blue_image_upload("second.png", 254), bootstrap=False
            )
            waiting = get_porosity_job(second["id"])
            self.assertEqual(waiting["status"], "queued")
            self.assertIn("ahead", waiting["message"])
            release.set()
            self.assertEqual(self._wait_for_job(first["id"])["status"], "done")
            self.assertEqual(self._wait_for_job(second["id"])["status"], "done")
        self.assertEqual(maximum_active, 1)

    def _wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + 5
        job = get_porosity_job(job_id)
        while time.monotonic() < deadline and job["status"] not in {"done", "error"}:
            time.sleep(0.01)
            job = get_porosity_job(job_id)
        return job


if __name__ == "__main__":
    unittest.main()
