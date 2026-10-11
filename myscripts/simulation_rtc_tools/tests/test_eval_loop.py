from collections import deque

import numpy as np
import pytest

from simulation_rtc.async_worker import WorkerResult
from simulation_rtc.config import RTCConfig, RobotActionLayout
from simulation_rtc.eval_loop import run_rtc_episode
from simulation_rtc.protocol import RTCResponse


CONFIG = RTCConfig(50, 15, 5, 10, 4, 0.36, 10.0, "exp", 1.0)
LAYOUT = RobotActionLayout(14, (6, 13))


class FakeEnv:
    def __init__(self, limit=47):
        self.take_action_cnt = 0
        self.step_lim = limit
        self.eval_success = False
        self.actions = []
        self.obs_calls = []

    def get_obs(self):
        self.obs_calls.append(self.take_action_cnt)
        return {"observation.state": np.zeros(14, np.float32), "task": "lift pot"}

    def take_action(self, action):
        self.actions.append(np.asarray(action).copy())
        self.take_action_cnt += 1


class DelayedWorker:
    def __init__(self, bad=False):
        self.results = deque()
        self.pending = None
        self.wait_calls = 0
        self.bad = bad

    def reset_episode(self, episode_id):
        self.episode_id = episode_id

    def submit(self, request):
        actions = np.zeros((50, 14), np.float32)
        if request.request_id == 0:
            self.results.append(WorkerResult(request, response=RTCResponse(
                request.episode_id, request.request_id, actions, {"infer_ms": 10}, {"enabled": False})))
        else:
            if self.bad:
                actions[0, 0] = np.nan
            self.pending = WorkerResult(request, response=RTCResponse(
                request.episode_id, request.request_id, actions, {"infer_ms": 280}, {"enabled": True}))

    def poll(self):
        return self.results.popleft() if self.results else None

    def wait(self, timeout_s):
        self.wait_calls += 1
        result, self.pending = self.pending, None
        return result


class MemoryTrace:
    def __init__(self): self.records = []
    def write(self, event, **fields): self.records.append(dict(event=event, **fields))


def test_episode_requests_at_15_and_pauses_at_fixed_step_20_without_consuming_step():
    env, worker, trace = FakeEnv(), DelayedWorker(), MemoryTrace()
    result = run_rtc_episode(env, worker, CONFIG, LAYOUT, trace,
                             episode_id=9, robo_name="robotwin")
    assert env.take_action_cnt == 47 and len(env.actions) == 47
    assert env.obs_calls[:2] == [0, 15]
    assert worker.wait_calls == 2
    request = next(item for item in trace.records if item["event"] == "request")
    promotion = next(item for item in trace.records if item["event"] == "promotion")
    wait = next(item for item in trace.records if item["event"] == "wait")
    assert request["policy_step"] == 15
    assert request["eligible_policy_step"] == 20
    assert wait["policy_step"] == 20
    assert promotion["policy_step"] == 20
    assert promotion["crop_steps"] == 5
    assert result.action_count == 47 and result.success is False


def test_success_after_prefetch_drains_inflight_result_without_more_actions():
    env, worker, trace = FakeEnv(limit=100), DelayedWorker(), MemoryTrace()
    original = env.take_action
    def succeed(action):
        original(action)
        if env.take_action_cnt == 17:
            env.eval_success = True
    env.take_action = succeed
    result = run_rtc_episode(env, worker, CONFIG, LAYOUT, trace, episode_id=11, robo_name="robotwin")
    assert result.success and result.action_count == 17
    assert worker.pending is None
    assert worker.wait_calls == 1


def test_malformed_response_fails_before_actuation_of_new_chunk():
    env, worker, trace = FakeEnv(), DelayedWorker(bad=True), MemoryTrace()
    with pytest.raises(ValueError):
        run_rtc_episode(env, worker, CONFIG, LAYOUT, trace,
                        episode_id=9, robo_name="robotwin")
    assert env.take_action_cnt == 20


def test_success_stops_immediately_and_does_not_request_after_finish():
    env, worker, trace = FakeEnv(limit=100), DelayedWorker(), MemoryTrace()
    original = env.take_action
    def succeed(action):
        original(action)
        if env.take_action_cnt == 3:
            env.eval_success = True
    env.take_action = succeed
    result = run_rtc_episode(env, worker, CONFIG, LAYOUT, trace,
                             episode_id=10, robo_name="robotwin")
    assert result.success and result.action_count == 3
    assert not any(item["event"] == "request" for item in trace.records)
    assert worker.pending is None
