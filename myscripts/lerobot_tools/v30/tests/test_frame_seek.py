"""Frame selection must stay paused and never decode extra episode frames."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import urllib.request
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'replay_lerobot_v30_videos_keyboard.py'
spec = importlib.util.spec_from_file_location('rbt_frame_seek', SCRIPT)
player = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = player
spec.loader.exec_module(player)
CAM = 'observation.images.cam_high'


class DummyEpisodeReader:
    calls = []
    def __init__(self, ep, video_fps=15):
        self.ep = ep

    def get(self, idx):
        self.calls.append(idx)
        if idx >= self.ep.segments[CAM].length:
            return {CAM: None}, {CAM: 'MISSING TAIL FRAME'}
        return {CAM: np.full((16, 16, 3), idx, np.uint8)}, {}

    def close(self):
        pass


class DummyViewer:
    def __init__(self, commands, selector):
        self.commands = commands
        self.selector = selector
        self.statuses = []
    def show(self, frame):
        pass
    def set_status(self, status):
        self.statuses.append(status)
        self.selector(self.commands, self.statuses)
    def poll(self, milliseconds):
        return None
    def close(self):
        pass


class TestFrameSeek(TestCase):
    def setUp(self):
        seg = player.Segment(CAM, Path('synthetic.mp4'), 0, 3, 0, 3/15)
        self.ep = player.Episode(0, 0, 4, {CAM: seg}, 'test.parquet')
        DummyEpisodeReader.calls = []

    def args(self, frame=None, paused=False):
        return SimpleNamespace(episode_index=0, frame=frame, start_paused=paused,
                               rate=1.0, loop=False)

    def run_player(self, args, selector):
        captured = {}
        def fake_open_viewer(_args, commands):
            captured['viewer'] = DummyViewer(commands, selector)
            return captured['viewer']
        # input_thread exits immediately so stdin is not touched.
        with patch.object(player, 'EpisodeReader', DummyEpisodeReader), \
             patch.object(player, 'open_viewer', fake_open_viewer), \
             patch.object(player, 'input_thread', lambda commands: None):
            player.display(args, [self.ep], 15, [CAM], 15)
        return captured['viewer'].statuses

    def test_command_frame_and_prev_next_clamp(self):
        s = {'frame': 0, 'stopped': False, 'paused': False, 'rate': 1.0}
        self.assertEqual(player.handle_command('f 3', s, [self.ep], 0), ('seek', 3))
        self.assertTrue(s['paused'])
        self.assertIsNone(player.handle_command('f 4', s, [self.ep], 0))
        self.assertEqual(player.handle_command('prevframe', s, [self.ep], 0), ('seek', 0))
        s['frame'] = 3
        self.assertEqual(player.handle_command('nextframe', s, [self.ep], 0), ('seek', 3))

    def test_start_paused_decodes_only_first_frame(self):
        def on_status(commands, statuses):
            commands.put('q')
        statuses = self.run_player(self.args(paused=True), on_status)
        self.assertEqual(DummyEpisodeReader.calls, [0])
        self.assertEqual(statuses[0]['status'], 'PAUSED')
        self.assertEqual(statuses[0]['frame_index'], 0)

    def test_manual_seek_from_playback_pauses_and_skips_intermediate_frames(self):
        def on_status(commands, statuses):
            latest = statuses[-1]
            if len(statuses) == 1:
                commands.put('f 3')
            elif latest['status'] == 'PAUSED' and latest['frame_index'] == 3:
                commands.put('q')
        statuses = self.run_player(self.args(), on_status)
        self.assertEqual(DummyEpisodeReader.calls, [0, 3])
        self.assertEqual([(s['frame_index'],s['status']) for s in statuses],
                         [(0,'PLAYING'),(3,'PAUSED')])

    def test_initial_frame_is_paused_even_when_video_tail_missing(self):
        def on_status(commands, statuses):
            commands.put('q')
        statuses = self.run_player(self.args(frame=3), on_status)
        self.assertEqual(DummyEpisodeReader.calls, [3])
        self.assertEqual(statuses[0]['frame_index'], 3)
        self.assertEqual(statuses[0]['status'], 'PAUSED')

    def test_browser_get_status_and_frame_command(self):
        commands = queue.Queue()
        args = SimpleNamespace(backend='browser', browser_port=0, no_browser=True,
                               window_name='unused', window_scale=1)
        viewer = player.BrowserViewer(args, commands)
        try:
            viewer.set_status({'episode_index':17,'length':147,
                               'frame_index':120,'status':'PAUSED'})
            with urllib.request.urlopen(viewer.url + 'status', timeout=3) as response:
                data=json.loads(response.read().decode())
            self.assertEqual(data['frame_index'],120)
            with urllib.request.urlopen(viewer.url, timeout=3) as response:
                html=response.read().decode()
            self.assertIn('frameSlider', html)
            self.assertIn('查看该帧并暂停', html)
            request=urllib.request.Request(viewer.url+'command',
                        data=json.dumps({'command':'f 120'}).encode(),
                        headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(request,timeout=3) as response:
                self.assertEqual(response.status,204)
            self.assertEqual(commands.get_nowait(),'f 120')
        finally:
            viewer.close()

if __name__=='__main__':
    import unittest
    unittest.main()
