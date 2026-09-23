from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from libs.upload_datasets import (
    DatasetError,
    create_dataset,
    delete_dataset,
    list_datasets,
    load_dataset,
    preview_path,
    update_image_metadata,
)


def image_upload(filename: str = "thin-section.png") -> FileStorage:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 0] = 180
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    return FileStorage(stream=BytesIO(encoded.tobytes()), filename=filename)


class UploadDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_root = os.environ.get("GEO_PARAMS_UPLOADS_DIR")
        os.environ["GEO_PARAMS_UPLOADS_DIR"] = self.temporary.name

    def tearDown(self) -> None:
        if self.previous_root is None:
            os.environ.pop("GEO_PARAMS_UPLOADS_DIR", None)
        else:
            os.environ["GEO_PARAMS_UPLOADS_DIR"] = self.previous_root
        self.temporary.cleanup()

    def test_create_calibrate_reload_and_delete_dataset(self) -> None:
        dataset = create_dataset("Research samples", [image_upload()])
        self.assertEqual(dataset["name"], "Research samples")
        self.assertEqual(len(dataset["images"]), 1)
        image = dataset["images"][0]
        self.assertTrue(preview_path(dataset["id"], image).is_file())
        self.assertEqual(len(list_datasets()), 1)

        updated = update_image_metadata(
            dataset["id"],
            image["id"],
            "Sample A",
            pixel_distance_preview=10.0,
            physical_distance=500.0,
            unit="um",
        )
        self.assertEqual(updated["display_name"], "Sample A")
        self.assertAlmostEqual(updated["calibration"]["micrometers_per_pixel"], 6.25)
        reloaded = load_dataset(dataset["id"])
        self.assertEqual(reloaded["images"][0]["display_name"], "Sample A")

        delete_dataset(dataset["id"])
        self.assertEqual(list_datasets(), [])

    def test_rejects_unsupported_files(self) -> None:
        with self.assertRaises(DatasetError):
            create_dataset("Unsafe", [image_upload("notes.svg")])

    def test_rejects_unsafe_dataset_identifier(self) -> None:
        with self.assertRaises(DatasetError):
            load_dataset("../outside")


if __name__ == "__main__":
    unittest.main()
