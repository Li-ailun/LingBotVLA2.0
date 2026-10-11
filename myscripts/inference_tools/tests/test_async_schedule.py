from myscripts.inference_tools.async_schedule import AsyncChunkSchedule


def test_q16_prefetch_and_k20_promotion():
    schedule = AsyncChunkSchedule(prefetch_step=16, requery_step=20, horizon=50)
    assert not schedule.should_prefetch(15, inflight=False, ready=False)
    assert schedule.should_prefetch(16, inflight=False, ready=False)
    assert not schedule.should_promote(19, ready=True)
    assert schedule.should_promote(20, ready=True)


def test_only_one_request_and_horizon_holds():
    schedule = AsyncChunkSchedule()
    assert not schedule.should_prefetch(16, inflight=True, ready=False)
    assert schedule.should_hold(50, ready=False)
