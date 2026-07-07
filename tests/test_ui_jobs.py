# Radical
"""Unit tests for streets_at_night.ui.jobs.JobRunner using real threads."""

import threading
import time
import typing as t

import pytest

from streets_at_night.ui.jobs import JobEvent, JobRunner, ProgressFn

_TIMEOUT_SECONDS = 5.0


def _drain_until(runner: JobRunner, kind: str) -> t.List[JobEvent]:
    """Poll the runner until an event of the given kind arrives or timeout.

    Args:
        runner (JobRunner): The runner to poll.
        kind (str): Event kind to wait for ("done" or "error").

    Returns:
        list[JobEvent]: All events collected up to and including the target.
    """
    collected: t.List[JobEvent] = []
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        collected.extend(runner.poll())
        if any(e.kind == kind for e in collected):
            return collected
        time.sleep(0.01)
    raise AssertionError(f"No {kind!r} event within {_TIMEOUT_SECONDS}s: {collected}")


@pytest.fixture
def runner() -> JobRunner:
    """Return a fresh JobRunner."""
    return JobRunner()


def test_submit_runs_the_function(runner: JobRunner):
    # Arrange
    ran: t.List[bool] = []

    def job(progress: ProgressFn) -> str:
        ran.append(True)
        return "value"

    # Act
    started = runner.submit(job)
    _drain_until(runner, "done")

    # Assert
    assert started is True
    assert ran == [True]


def test_poll_yields_progress_then_done_with_result(runner: JobRunner):
    # Arrange
    def job(progress: ProgressFn) -> str:
        progress("halfway", 0.5)
        return "result-value"

    # Act
    runner.submit(job)
    events = _drain_until(runner, "done")

    # Assert
    progress_events = [e for e in events if e.kind == "progress"]
    done_events = [e for e in events if e.kind == "done"]
    assert progress_events[0].message == "halfway"
    assert progress_events[0].fraction == 0.5
    assert done_events[0].result == "result-value"


def test_raising_job_yields_error_event_with_exception(runner: JobRunner):
    # Arrange
    boom = ValueError("boom")

    def job(progress: ProgressFn) -> None:
        raise boom

    # Act
    runner.submit(job)
    events = _drain_until(runner, "error")

    # Assert
    error_events = [e for e in events if e.kind == "error"]
    assert error_events[0].error is boom
    assert error_events[0].message == "boom"


def test_submit_while_busy_returns_false_and_does_not_run(runner: JobRunner):
    # Arrange — first job blocks until released so the runner stays busy.
    release = _make_gate()
    second_ran: t.List[bool] = []

    def blocking_job(progress: ProgressFn) -> None:
        release.wait(_TIMEOUT_SECONDS)

    def second_job(progress: ProgressFn) -> None:
        second_ran.append(True)

    # Act
    runner.submit(blocking_job)
    _wait_until(lambda: runner.busy)
    started_second = runner.submit(second_job)
    release.set()
    _drain_until(runner, "done")

    # Assert
    assert started_second is False
    assert second_ran == []


def test_busy_is_false_after_completion(runner: JobRunner):
    # Arrange
    def job(progress: ProgressFn) -> None:
        return None

    # Act
    runner.submit(job)
    _drain_until(runner, "done")
    _wait_until(lambda: not runner.busy)

    # Assert
    assert runner.busy is False


# ---------------------------------------------------------------------------
# Local threading helpers
# ---------------------------------------------------------------------------


def _make_gate() -> threading.Event:
    """Return a threading.Event used to gate a worker until the test releases it."""
    return threading.Event()


def _wait_until(predicate: t.Callable[[], bool]) -> None:
    """Spin until predicate is true or the timeout elapses.

    Args:
        predicate (Callable[[], bool]): Condition to wait for.
    """
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition not met within timeout")
