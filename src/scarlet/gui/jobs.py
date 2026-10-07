from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any, Callable

from PySide6 import QtCore


@dataclass(frozen=True)
class JobFailure:
    """Structured information about an exception raised by a background job."""

    exception: Exception
    traceback_text: str

    @property
    def message(self) -> str:
        return str(self.exception) or type(self.exception).__name__


class _WorkerSignals(QtCore.QObject):
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)
    completed = QtCore.Signal()


class _FunctionWorker(QtCore.QRunnable):
    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._operation = operation
        self.signals = _WorkerSignals()
        # JobController releases the runnable after its queued completion signal.
        self.setAutoDelete(False)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._operation()
        except Exception as exc:  # The exception is forwarded to the GUI thread.
            self.signals.failed.emit(
                JobFailure(
                    exception=exc,
                    traceback_text="".join(traceback.format_exception(exc)),
                )
            )
        else:
            self.signals.succeeded.emit(result)
        finally:
            self.signals.completed.emit()


class JobController(QtCore.QObject):
    """Run one blocking operation at a time outside the Qt GUI thread."""

    started = QtCore.Signal(str)
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, object)
    finished = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._worker: _FunctionWorker | None = None
        self._kind: str | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None

    def start(self, kind: str, operation: Callable[[], Any]) -> None:
        if self.is_running:
            raise RuntimeError("A SCARLET background job is already running")

        worker = _FunctionWorker(operation)
        worker.signals.succeeded.connect(self._on_succeeded)
        worker.signals.failed.connect(self._on_failed)
        worker.signals.completed.connect(self._on_worker_completed)

        self._worker = worker
        self._kind = kind
        self.started.emit(kind)
        self._pool.start(worker)

    @QtCore.Slot(object)
    def _on_succeeded(self, result: object) -> None:
        if self._kind is not None:
            self.succeeded.emit(self._kind, result)

    @QtCore.Slot(object)
    def _on_failed(self, failure: object) -> None:
        if self._kind is not None:
            self.failed.emit(self._kind, failure)

    @QtCore.Slot()
    def _on_worker_completed(self) -> None:
        kind = self._kind
        self._worker = None
        self._kind = None
        if kind is not None:
            self.finished.emit(kind)
