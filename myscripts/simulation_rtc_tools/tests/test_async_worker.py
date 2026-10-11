import threading
import time

import numpy as np
import pytest

from simulation_rtc.async_worker import AsyncPolicyWorker, WorkerError
from simulation_rtc.config import RobotActionLayout
from simulation_rtc.protocol import PROTOCOL_VERSION, build_bootstrap_request


class FakeTransport:
    def __init__(self, *, gate=None, fail=False):
        self.owner = threading.get_ident()
        self.gate = gate
        self.fail = fail
        self.closed_by = None

    def infer(self, payload, timeout_s):
        assert threading.get_ident() == self.owner
        if self.gate is not None and not self.gate.wait(timeout_s):
            raise TimeoutError("fake timeout")
        if self.fail:
            raise RuntimeError("network down")
        return {"protocol_version": PROTOCOL_VERSION,
                "episode_id": payload["episode_id"], "request_id": payload["request_id"],
                "action": np.zeros((50, 14), np.float32), "server_timing": {}, "rtc_ack": {}}

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed_by = threading.get_ident()


def wait_for(worker, timeout=1.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("worker produced no result")


def test_worker_owns_transport_and_returns_validated_response():
    made = []
    def factory():
        transport = FakeTransport()
        made.append(transport)
        return transport
    worker = AsyncPolicyWorker(factory, RobotActionLayout(14, (6, 13)), request_timeout_s=1.0)
    worker.start()
    worker.reset_episode(2)
    worker.submit(build_bootstrap_request({"task": "x"}, episode_id=2, request_id=0))
    result = wait_for(worker)
    worker.close(1.0)
    assert result.response is not None and result.response.actions.shape == (50, 14)
    assert made[0].owner != threading.get_ident()
    assert made[0].closed_by == made[0].owner


def test_queue_rejects_more_than_one_pending_request():
    gate = threading.Event()
    worker = AsyncPolicyWorker(lambda: FakeTransport(gate=gate), RobotActionLayout(14, (6, 13)), 1.0)
    worker.start(); worker.reset_episode(2)
    worker.submit(build_bootstrap_request({}, episode_id=2, request_id=0))
    with pytest.raises(WorkerError):
        worker.submit(build_bootstrap_request({}, episode_id=2, request_id=1))
    gate.set(); wait_for(worker); worker.close(1.0)


def test_reset_discards_stale_response_and_error_is_propagated():
    gate = threading.Event()
    worker = AsyncPolicyWorker(lambda: FakeTransport(gate=gate), RobotActionLayout(14, (6, 13)), 1.0)
    worker.start(); worker.reset_episode(2)
    worker.submit(build_bootstrap_request({}, episode_id=2, request_id=0))
    worker.reset_episode(3)
    gate.set(); time.sleep(0.03)
    assert worker.poll() is None
    worker.close(1.0)

    failed = AsyncPolicyWorker(lambda: FakeTransport(fail=True), RobotActionLayout(14, (6, 13)), 1.0)
    failed.start(); failed.reset_episode(4)
    failed.submit(build_bootstrap_request({}, episode_id=4, request_id=0))
    result = wait_for(failed); failed.close(1.0)
    assert result.error is not None and "network down" in str(result.error)


def test_timeout_allows_bounded_shutdown():
    worker = AsyncPolicyWorker(lambda: FakeTransport(gate=threading.Event()),
                               RobotActionLayout(14, (6, 13)), request_timeout_s=0.03)
    worker.start(); worker.reset_episode(2)
    worker.submit(build_bootstrap_request({}, episode_id=2, request_id=0))
    result = wait_for(worker)
    assert isinstance(result.error, TimeoutError)
    worker.close(0.5)
