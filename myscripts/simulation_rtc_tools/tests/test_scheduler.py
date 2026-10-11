import numpy as np

from simulation_rtc.config import RTCConfig
from simulation_rtc.protocol import RTCResponse
from simulation_rtc.scheduler import RTCScheduler, SchedulerState


CONFIG = RTCConfig(50, 15, 5, 10, 4, 0.36, 10.0, "exp", 30.0)


def chunk(value):
    return np.full((50, 14), value, np.float32)


def response(request_id=1, episode_id=7, value=2):
    return RTCResponse(episode_id, request_id, chunk(value), {}, {"enabled": True})


def advance(scheduler, count):
    decisions = []
    for _ in range(count):
        decision = scheduler.before_action()
        assert not decision.should_wait
        decisions.append(decision)
        scheduler.after_action()
    return decisions


def test_requests_once_at_index_15_and_does_not_promote_before_step_20():
    scheduler = RTCScheduler(CONFIG)
    scheduler.start_episode(7)
    scheduler.install_bootstrap(chunk(1))
    advance(scheduler, 15)
    decision = scheduler.before_action()
    assert decision.request_due and decision.chunk_index == 15
    scheduler.mark_request_submitted(1)
    assert scheduler.accept_response(response())
    for expected_index in range(15, 20):
        decision = scheduler.before_action()
        assert decision.chunk_index == expected_index
        assert not decision.promoted and not decision.should_wait
        scheduler.after_action()


def test_promotes_ready_response_at_fifth_elapsed_step():
    scheduler = RTCScheduler(CONFIG)
    scheduler.start_episode(7)
    scheduler.install_bootstrap(chunk(1))
    advance(scheduler, 15)
    scheduler.mark_request_submitted(1)
    assert scheduler.accept_response(response())
    advance(scheduler, 5)
    decision = scheduler.before_action()
    assert decision.promoted and decision.chunk_index == 0
    assert np.all(decision.action == 2)


def test_promotion_crops_five_elapsed_new_actions_and_pads_back_to_50():
    scheduler = RTCScheduler(CONFIG)
    scheduler.start_episode(7)
    scheduler.install_bootstrap(chunk(1))
    advance(scheduler, 15)
    scheduler.mark_request_submitted(1)
    ramp = np.repeat(np.arange(50, dtype=np.float32)[:, None], 14, axis=1)
    assert scheduler.accept_response(RTCResponse(7, 1, ramp, {}, {"enabled": True}))
    advance(scheduler, 5)
    decision = scheduler.before_action()
    assert decision.promoted
    assert np.all(decision.action == 5)
    assert scheduler.current_chunk.shape == (50, 14)
    assert np.all(scheduler.current_chunk[-1] == 49)


def test_waiting_does_not_advance_index_until_response_arrives():
    scheduler = RTCScheduler(CONFIG)
    scheduler.start_episode(7)
    scheduler.install_bootstrap(chunk(1))
    advance(scheduler, 15)
    scheduler.mark_request_submitted(1)
    advance(scheduler, 5)
    first = scheduler.before_action()
    second = scheduler.before_action()
    assert first.should_wait and second.should_wait
    assert first.chunk_index == second.chunk_index == 20
    assert scheduler.state is SchedulerState.WAITING_RESULT
    assert scheduler.accept_response(response())
    promoted = scheduler.before_action()
    assert promoted.promoted and promoted.chunk_index == 0


def test_stale_response_is_rejected_and_done_clears_pending():
    scheduler = RTCScheduler(CONFIG)
    scheduler.start_episode(7)
    scheduler.install_bootstrap(chunk(1))
    advance(scheduler, 15)
    scheduler.mark_request_submitted(1)
    assert not scheduler.accept_response(response(episode_id=6))
    scheduler.finish_episode()
    assert not scheduler.accept_response(response())
    assert scheduler.state is SchedulerState.EPISODE_DONE
