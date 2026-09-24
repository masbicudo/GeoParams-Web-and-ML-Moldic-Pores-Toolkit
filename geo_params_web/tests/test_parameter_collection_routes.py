from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")

import app as web_app
from libs.parameter_collections import (
    list_parameter_collections,
    save_collection_session,
    update_parameter_collection,
)
from libs.web_helpers import get_session, session_store


class ParameterCollectionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.environment = patch.dict(
            os.environ,
            {
                "GEO_PARAMS_UPLOADS_DIR": str(root / "uploads"),
                "GEO_PARAMS_COLLECTED_OUTPUT_DIR": str(root / "outputs"),
            },
        )
        self.environment.start()
        session_store.clear()
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def tearDown(self) -> None:
        session_store.clear()
        self.environment.stop()
        self.temporary.cleanup()

    def _start_collection(self) -> tuple[dict, str]:
        response = self.client.post("/parameter-collections/start")
        self.assertEqual(response.status_code, 302)
        collection = list_parameter_collections()[0]
        return collection, collection["session_id"]

    def test_tool_landing_precedes_first_input_screen(self) -> None:
        home = self.client.get("/")
        self.assertIn(b"/parameter-collections", home.data)
        self.assertNotIn(b"href=\"/userinfo", home.data)

        collection, session_id = self._start_collection()
        first_screen = self.client.get(f"/userinfo?session_id={session_id}")
        self.assertEqual(first_screen.status_code, 200)
        self.assertEqual(collection["stage"], "user_information")

    def test_completed_collection_redirects_to_private_safe_summary(self) -> None:
        collection, session_id = self._start_collection()
        session = get_session(session_id)
        session["options"] = {
            "user": {"name": "Private person", "email": "private@example.test"},
            "image_select.filename": "person-name-in-image.jpg",
            "params_select.clicked_points": [{"x": 2, "y": 3}],
            "params_select.priority": "shape",
            "params_select.state": "Done",
        }
        save_collection_session(collection["id"], session)
        update_parameter_collection(
            collection["id"],
            session=session,
            status="done",
            stage="completed",
            progress=100,
            message="Collection complete",
        )

        edit_attempt = self.client.get(f"/params_select?session_id={session_id}")
        self.assertEqual(edit_attempt.status_code, 302)
        self.assertIn(collection["id"], edit_attempt.headers["Location"])

        summary = self.client.get(f"/parameter-collections/{collection['id']}")
        self.assertEqual(summary.status_code, 200)
        self.assertIn(b"read-only", summary.data)
        self.assertNotIn(b"Private person", summary.data)
        self.assertNotIn(b"private@example.test", summary.data)
        self.assertNotIn(b"person-name-in-image.jpg", summary.data)

    def test_waiting_for_input_appears_as_workflow_not_processing(self) -> None:
        collection, _ = self._start_collection()
        overview = self.client.get("/jobs")
        self.assertEqual(overview.status_code, 200)
        self.assertIn(b"Active workflows", overview.data)
        self.assertIn(collection["short_id"].encode(), overview.data)
        self.assertIn(b"Input needed", overview.data)
        self.assertNotIn(b"window.setTimeout", overview.data)

    def test_status_repairs_collection_after_processing_finished(self) -> None:
        collection, session_id = self._start_collection()
        session = get_session(session_id)
        session["tasks"] = {
            "initial_image_setup": {
                "state": "Done",
                "result": {"tile_shape": [10, 12]},
                "alive_tag": None,
                "cache": None,
            }
        }
        update_parameter_collection(
            collection["id"],
            session=session,
            status="running",
            stage="image_processing",
            progress=0,
            message="Preparing parameter space",
        )

        response = self.client.get(
            f"/parameter-collections/{collection['id']}/status"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "awaiting_input")
        self.assertEqual(payload["stage"], "parameter_selection")
        self.assertEqual(payload["progress"], 100)
        self.assertIn("/params_select", payload["continue_url"])

    def test_concurrent_status_requests_launch_only_one_worker(self) -> None:
        collection, session_id = self._start_collection()
        session = get_session(session_id)
        session["tasks"] = {
            "initial_image_setup": {
                "arguments": [480],
                "cache": {},
                "alive_tag": None,
                "state": "Requested",
                "timeout": None,
                "progress": None,
                "result": None,
                "error": None,
            }
        }
        update_parameter_collection(
            collection["id"],
            session=session,
            status="running",
            stage="image_processing",
            progress=0,
            message="Preparing parameter space",
        )
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        callers = 8
        barrier = threading.Barrier(callers)
        calls = []
        responses = []
        failures = []
        result_lock = threading.Lock()

        def delayed_task(_session_id, _count_timeouts, _launch_token):
            with result_lock:
                calls.append(_session_id)
            started.set()
            release.wait(timeout=5)
            finished.set()

        def request_status() -> None:
            try:
                barrier.wait(timeout=5)
                with web_app.app.test_client() as client:
                    response = client.get(
                        f"/parameter-collections/{collection['id']}/status"
                    )
                with result_lock:
                    responses.append(response.status_code)
            except Exception as exc:
                with result_lock:
                    failures.append(exc)

        threads = [threading.Thread(target=request_status) for _ in range(callers)]
        try:
            with patch.object(web_app, "initial_image_setup", delayed_task):
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
                self.assertTrue(started.wait(timeout=2))
        finally:
            release.set()
            self.assertTrue(finished.wait(timeout=2))

        self.assertFalse(failures)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(responses, [200] * callers)
        self.assertEqual(calls, [session_id])

    def test_obsolete_worker_cannot_overwrite_current_task(self) -> None:
        _collection, session_id = self._start_collection()
        session = get_session(session_id)
        stale_token = web_app.ProcessToken("Queued")
        current_token = web_app.ProcessToken("Queued")
        session["tasks"] = {
            "initial_image_setup": {
                "state": "Requested",
                "alive_tag": current_token,
            }
        }

        with patch.object(web_app, "_sync_collection") as sync_collection:
            web_app.initial_image_setup(session_id, 0, stale_token)

        sync_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
