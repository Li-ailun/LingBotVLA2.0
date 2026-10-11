"""AV1 seek regression test. Makes only a tiny temporary MP4 outside the dataset."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'replay_lerobot_v21_videos_keyboard.py'
spec = importlib.util.spec_from_file_location('robotwin_av1_replay', SCRIPT)
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)


class TestAV1Seek(unittest.TestCase):
    def test_seek_random_episode_offsets_and_no_cross_episode_tail(self):
        if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
            self.skipTest('ffmpeg and ffprobe are required')
        encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True).stdout
        if 'libaom-av1' not in encoders:
            self.skipTest('libaom-av1 encoder not available for synthetic fixture')
        fps = 15
        n = 100
        with tempfile.TemporaryDirectory() as workdir:
            mp4 = Path(workdir) / 'fixture_av1.mp4'
            raw = np.stack([np.full((32, 48, 3), (i * 3) % 255, np.uint8) for i in range(n)])
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '48x32',
                   '-framerate', str(fps), '-i', 'pipe:0', '-c:v', 'libaom-av1',
                   '-b:v', '0', '-crf', '0', '-cpu-used', '8', str(mp4)]
            subprocess.run(cmd, input=raw.tobytes(), check=True, timeout=40, capture_output=True)
            for start in (0, 1, 7, 33, 61, 87):
                length = 3
                segment = replay.Segment(replay.CAMERAS[0], mp4,
                                         start, start + length, start/fps, (start+length)/fps)
                reader = replay.SegmentReader(segment, fps=fps)
                try:
                    for pos in range(length):
                        actual, error = reader.get(pos)
                        self.assertIsNone(error, f'start={start} pos={pos}')
                        self.assertIsNotNone(actual)
                        self.assertEqual(actual.shape, (32, 48, 3))
                        observed = int(np.median(actual[:, :, 0]))
                        expected = (start + pos) * 3 % 255
                        self.assertLessEqual(abs(observed - expected), 1)
                    frame, error = reader.get(length)
                    self.assertIsNone(frame)
                    self.assertEqual(error, 'MISSING TAIL FRAME')
                finally:
                    reader.close()


if __name__ == '__main__':
    unittest.main()
