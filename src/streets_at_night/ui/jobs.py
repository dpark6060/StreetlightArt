"""Single-worker background job runner for the tabbed UI.

Long-running work (OSM loading, verify, render, MP4 record) runs on one daemon
worker thread. Job functions receive a ``progress`` callable and return plain
data; the UI drains the event queue once per frame on the main thread. The
worker must never touch pygame or pygame-gui.
"""

import queue
import threading
import typing as t
from dataclasses import dataclass

ProgressFn = t.Callable[[str, float], None]  # (message, fraction 0..1; -1 = indeterminate)


@dataclass
class JobEvent:
    """One progress/completion event from the worker thread.

    Attributes:
        kind: "progress" | "done" | "error".
        message: Human-readable progress or error message.
        fraction: Progress fraction 0..1, or -1 for indeterminate.
        result: Return value, set when kind == "done".
        error: Exception instance, set when kind == "error".
    """

    kind: str
    message: str = ""
    fraction: float = -1.0
    result: t.Any = None
    error: t.Optional[BaseException] = None


class JobRunner:
    """Runs one background job at a time and queues its progress events."""

    def __init__(self) -> None:
        """Initialise an idle job runner with an empty event queue."""
        self._queue: "queue.Queue[JobEvent]" = queue.Queue()
        self._busy = threading.Event()

    @property
    def busy(self) -> bool:
        """Return True while a job is running (until its done/error is emitted)."""
        return self._busy.is_set()

    def submit(self, fn: t.Callable[[ProgressFn], t.Any]) -> bool:
        """Start fn on the worker thread.

        Args:
            fn (Callable[[ProgressFn], Any]): Work function; receives a progress
                callable and returns any plain-data result.

        Returns:
            bool: True if the job was started, False if a job is already running.
        """
        if self._busy.is_set():
            return False
        self._busy.set()
        thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        thread.start()
        return True

    def poll(self) -> t.List[JobEvent]:
        """Drain and return all pending events. Called once per frame by the app.

        Returns:
            list[JobEvent]: Every event enqueued since the last poll, in order.
        """
        events: t.List[JobEvent] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self, fn: t.Callable[[ProgressFn], t.Any]) -> None:
        """Execute fn on the worker thread, translating outcomes into events.

        Args:
            fn (Callable[[ProgressFn], Any]): The submitted work function.
        """
        try:
            result = fn(self._emit_progress)
        except BaseException as exc:  # noqa: BLE001 - report every failure to the UI
            self._queue.put(JobEvent("error", message=str(exc), error=exc))
        else:
            self._queue.put(JobEvent("done", result=result))
        finally:
            self._busy.clear()

    def _emit_progress(self, message: str, fraction: float) -> None:
        """Enqueue a progress event from the worker thread.

        Args:
            message (str): Progress message.
            fraction (float): Progress fraction 0..1, or -1 for indeterminate.
        """
        self._queue.put(JobEvent("progress", message=message, fraction=fraction))
