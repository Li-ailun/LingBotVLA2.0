from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Callable, Protocol

from .config import RobotActionLayout
from .protocol import RTCRequest, RTCResponse, validate_response


class WorkerError(RuntimeError):
    pass


class PolicyTransport(Protocol):
    def infer(self, payload: dict, timeout_s: float) -> dict: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class WorkerResult:
    request: RTCRequest
    response: RTCResponse | None = None
    error: BaseException | None = None


class AsyncPolicyWorker:
    def __init__(self, transport_factory: Callable[[], PolicyTransport],
                 layout: RobotActionLayout, request_timeout_s: float):
        self._factory = transport_factory
        self._layout = layout
        self._timeout = float(request_timeout_s)
        self._requests: queue.Queue[RTCRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._active_episode = -1

    def start(self) -> None:
        if self._thread is not None:
            raise WorkerError("worker already started")
        self._thread = threading.Thread(target=self._run, name="rtc-policy-worker", daemon=True)
        self._thread.start()

    def reset_episode(self, episode_id: int) -> None:
        with self._lock:
            self._active_episode = int(episode_id)
        self._drain_results()

    def submit(self, request: RTCRequest) -> None:
        with self._lock:
            if request.episode_id != self._active_episode:
                raise WorkerError("request episode is not active")
            if self._busy:
                raise WorkerError("one policy request is already pending")
            self._busy = True
        try:
            self._requests.put_nowait(request)
        except queue.Full as error:
            with self._lock:
                self._busy = False
            raise WorkerError("policy request queue is full") from error

    def poll(self) -> WorkerResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            active = self._active_episode
        return result if result.request.episode_id == active else None

    def wait(self, timeout_s: float) -> WorkerResult:
        result = self._results.get(timeout=float(timeout_s))
        with self._lock:
            active = self._active_episode
        if result.request.episode_id != active:
            raise WorkerError("received stale episode result")
        return result

    def close(self, timeout_s: float) -> None:
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(float(timeout_s))
            if self._thread.is_alive():
                raise WorkerError("worker did not stop before timeout")
            self._thread = None

    def _run(self) -> None:
        transport = self._factory()
        try:
            while not self._stop.is_set():
                try:
                    request = self._requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                if request is None:
                    break
                try:
                    payload = transport.infer(request.to_payload(), self._timeout)
                    response = validate_response(payload, request.episode_id,
                                                 request.request_id, self._layout)
                    result = WorkerResult(request, response=response)
                except BaseException as error:
                    result = WorkerResult(request, error=error)
                with self._lock:
                    self._busy = False
                    active = self._active_episode
                if request.episode_id == active:
                    self._put_latest(result)
        finally:
            transport.close()

    def _put_latest(self, result: WorkerResult) -> None:
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            self._results.put_nowait(result)

    def _drain_results(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                return
