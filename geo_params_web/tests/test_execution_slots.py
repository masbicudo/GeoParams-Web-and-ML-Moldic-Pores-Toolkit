from __future__ import annotations

import threading
import unittest

from libs.execution_slots import processing_slot


class ExecutionSlotTests(unittest.TestCase):
    def test_expensive_steps_from_different_tools_do_not_overlap(self) -> None:
        first_entered = threading.Event()
        second_waiting = threading.Event()
        release_first = threading.Event()
        order = []

        def first_tool() -> None:
            with processing_slot("first-tool"):
                order.append("first-start")
                first_entered.set()
                release_first.wait(timeout=2)
                order.append("first-end")

        def second_tool() -> None:
            first_entered.wait(timeout=2)
            with processing_slot(
                "second-tool",
                on_wait=lambda _ahead: second_waiting.set(),
            ):
                order.append("second-start")

        first = threading.Thread(target=first_tool)
        second = threading.Thread(target=second_tool)
        first.start()
        second.start()
        self.assertTrue(first_entered.wait(timeout=2))
        self.assertTrue(second_waiting.wait(timeout=2))
        self.assertEqual(order, ["first-start"])
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertEqual(order, ["first-start", "first-end", "second-start"])


if __name__ == "__main__":
    unittest.main()
