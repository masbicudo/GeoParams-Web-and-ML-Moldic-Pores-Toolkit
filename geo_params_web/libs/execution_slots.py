"""A shared FIFO slot for CPU- and memory-intensive application work."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import threading
import uuid
from collections.abc import Callable, Iterator


WaitCallback = Callable[[int], None]

_condition = threading.Condition(threading.RLock())
_queue: deque[tuple[str, str]] = deque()
_active_token: str | None = None


@contextmanager
def processing_slot(
    label: str,
    on_wait: WaitCallback | None = None,
) -> Iterator[None]:
    """Serialize expensive work across tools without reserving human wait time."""
    global _active_token
    token = uuid.uuid4().hex
    with _condition:
        _queue.append((token, label))
        while _active_token is not None or _queue[0][0] != token:
            if on_wait is not None:
                position = next(
                    index for index, queued in enumerate(_queue) if queued[0] == token
                )
                ahead = position + (1 if _active_token is not None else 0)
                on_wait(ahead)
            _condition.wait()
        _queue.popleft()
        _active_token = token

    try:
        yield
    finally:
        with _condition:
            if _active_token == token:
                _active_token = None
            else:
                _queue_copy = deque(item for item in _queue if item[0] != token)
                _queue.clear()
                _queue.extend(_queue_copy)
            _condition.notify_all()

