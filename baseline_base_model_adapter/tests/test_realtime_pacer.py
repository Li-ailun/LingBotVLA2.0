from RoboTwin_lingbot.envs.utils.realtime_pacer import RealTimePacer


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_prevents_simulation_running_faster_than_wall_clock():
    clock = FakeClock()
    pacer = RealTimePacer(0.004, enabled=True, monotonic=clock.monotonic, sleep=clock.sleep)
    pacer.after_step()
    pacer.after_step()
    assert clock.sleeps == [0.004, 0.004]


def test_disabled_pacer_never_sleeps():
    clock = FakeClock()
    pacer = RealTimePacer(0.004, enabled=False, monotonic=clock.monotonic, sleep=clock.sleep)
    pacer.after_step()
    assert clock.sleeps == []
