from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from libs.porosity_tool import (
    analyze_upload,
    load_result,
    parameter_sets,
    result_image_path,
)
from libs.porosity_jobs import get_porosity_job, submit_porosity_job
from libs.upload_datasets import create_dataset


def blue_image_upload(filename: str = "new-thin-section.png") -> FileStorage:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 0] = 255
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


if __name__ == "__main__":
    unittest.main()
