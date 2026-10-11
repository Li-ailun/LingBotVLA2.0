from myscripts.video_tools.fresh_frame_recording import install_fresh_frame_recording


class FakeEnv:
    def __init__(self, enabled=True):
        self.eval_video_path = "/tmp/videos" if enabled else None
        self.now_obs = {"frame": 0}
        self.refreshes = 0
        self.recorded = []

    def get_obs(self):
        self.refreshes += 1
        self.now_obs = {"frame": self.refreshes}
        return self.now_obs

    def take_action(self, action):
        self.recorded.append((self.now_obs["frame"], action))


def test_each_recorded_action_uses_a_fresh_observation():
    env = FakeEnv(enabled=True)
    install_fresh_frame_recording(env)
    install_fresh_frame_recording(env)
    env.take_action("a")
    env.take_action("b")
    assert env.refreshes == 2
    assert env.recorded == [(1, "a"), (2, "b")]


def test_non_video_evaluation_does_not_refresh_extra_observations():
    env = FakeEnv(enabled=False)
    install_fresh_frame_recording(env)
    env.take_action("a")
    assert env.refreshes == 0
    assert env.recorded == [(0, "a")]
