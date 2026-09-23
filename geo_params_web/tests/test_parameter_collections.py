from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from libs.parameter_collections import (
    ParameterCollectionError,
    create_parameter_collection,
    get_parameter_collection,
    list_parameter_collections,
    load_collection_session,
    update_parameter_collection,
)


class ParameterCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.uploads = root / "uploads"
        self.outputs = root / "outputs"
        self.environment = patch.dict(
            os.environ,
            {
                "GEO_PARAMS_UPLOADS_DIR": str(self.uploads),
                "GEO_PARAMS_COLLECTED_OUTPUT_DIR": str(self.outputs),
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_persists_human_and_automated_task_state(self) -> None:
        session = {"counter": 0, "options": {"user": {"experience": 2}}}
        created = create_parameter_collection("session-one", 0, session)
        running = update_parameter_collection(
            created["id"],
            session=session,
            status="running",
            stage="image_processing",
            progress=42,
            message="Preparing parameter space",
        )

        self.assertEqual(running["status"], "running")
        self.assertEqual(load_collection_session(created["id"])["options"], session["options"])
        self.assertEqual(list_parameter_collections()[0]["progress"], 42)

    def test_completed_collection_is_read_only(self) -> None:
        created = create_parameter_collection("session-two", 0, {"counter": 0})
        update_parameter_collection(created["id"], status="done", stage="completed")

        with self.assertRaises(ParameterCollectionError):
            update_parameter_collection(created["id"], status="awaiting_input")

    def test_legacy_summary_omits_personal_and_file_names(self) -> None:
        output = self.outputs / "old-session" / "0"
        output.mkdir(parents=True)
        with (output / "options.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "user": {"name": "Private name", "email": "private@example.test"},
                    "image_select.filename": "person-name-in-file.jpg",
                    "params_select.state": "Done",
                    "params_select.clicked_points": [{"x": 2, "y": 3}],
                    "params_select.priority": "shape",
                },
                stream,
            )

        collection = list_parameter_collections()[0]
        serialized = json.dumps(collection)
        self.assertNotIn("Private name", serialized)
        self.assertNotIn("private@example.test", serialized)
        self.assertNotIn("person-name-in-file.jpg", serialized)
        self.assertEqual(collection["summary"]["parameter_count"], 1)
        self.assertEqual(get_parameter_collection(collection["id"])["status"], "done")


if __name__ == "__main__":
    unittest.main()
