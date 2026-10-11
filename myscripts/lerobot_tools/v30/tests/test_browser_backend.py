"""Read-only smoke tests for the no-HighGUI browser viewer."""
from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import queue
import sys
import urllib.request
import json
import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'replay_lerobot_v30_videos_keyboard.py'
spec = importlib.util.spec_from_file_location('rbt_browser_replay', SCRIPT)
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


class TestBrowserBackend(TestCase):
    def test_http_page_mjpeg_and_command(self):
        commands = queue.Queue()
        args = SimpleNamespace(backend='browser', browser_port=0, no_browser=True, window_name='unused', window_scale=1.0)
        viewer = replay.BrowserViewer(args, commands)
        try:
            with urllib.request.urlopen(viewer.url, timeout=3) as response:
                page = response.read().decode('utf-8')
                self.assertIn('LeRobot v3.0', page)
                self.assertIn('/stream', page)
                self.assertEqual(response.status, 200)
            data = json.dumps({'command': 'g 17'}).encode('utf-8')
            request = urllib.request.Request(viewer.url + 'command', data=data, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 204)
            self.assertEqual(commands.get_nowait(), 'g 17')
            viewer.show(np.full((128, 128, 3), 120, np.uint8))
            with urllib.request.urlopen(viewer.url + 'stream', timeout=3) as response:
                self.assertIn('multipart/x-mixed-replace', response.headers['Content-Type'])
                payload = response.read(350)
                self.assertIn(b'Content-Type: image/jpeg', payload)
                self.assertIn(b'\xff\xd8', payload)
        finally:
            viewer.close()

    def test_auto_backend_without_highgui_uses_browser(self):
        commands = queue.Queue()
        args = SimpleNamespace(backend='auto', browser_port=0, no_browser=True, window_name='unused', window_scale=1.0)
        with patch.object(replay.cv2, 'getBuildInformation', return_value='  GUI: NONE\n'):
            viewer = replay.open_viewer(args, commands)
        try:
            self.assertIsInstance(viewer, replay.BrowserViewer)
        finally:
            viewer.close()


if __name__ == '__main__':
    main()
