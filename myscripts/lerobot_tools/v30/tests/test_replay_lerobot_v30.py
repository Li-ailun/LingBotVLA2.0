"""Run with: python -m unittest discover -s myscripts/lerobot_tools/tests -v"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import tempfile
import importlib.util

import cv2
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "replay_lerobot_v30_videos_keyboard.py"
spec = importlib.util.spec_from_file_location("replay_lerobot_v30_videos_keyboard", SCRIPT)
replay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replay
spec.loader.exec_module(replay)

CAM = "observation.images.cam_high"


class TestV3Viewer(TestCase):
    def test_metadata_episode_index_and_short_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metapath = root / "meta/episodes/chunk-000/file-000.parquet"
            metapath.parent.mkdir(parents=True)
            metapath.touch()
            base = f"videos/{CAM}/"
            records = [
                {"episode_index": 0, "length": 4, "data/chunk_index": 0, "data/file_index": 0,
                 base + "chunk_index": 0, base + "file_index": 0,
                 base + "from_timestamp": 0.0, base + "to_timestamp": 3/15},
                {"episode_index": 1, "length": 1, "data/chunk_index": 0, "data/file_index": 0,
                 base + "chunk_index": 0, base + "file_index": 0,
                 base + "from_timestamp": 3/15, base + "to_timestamp": 4/15},
            ]
            with patch.object(replay.pd, "read_parquet", return_value=pd.DataFrame(records)):
                eps = replay.load_episodes(root, [CAM], 15)
            self.assertEqual([e.episode_index for e in eps], [0, 1])
            self.assertEqual(eps[0].segments[CAM].length, 3)
            self.assertEqual(eps[0].length, 4)
            self.assertEqual(eps[1].segments[CAM].first, 3)

    def test_video_hard_fence_blocks_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 64))
            self.assertTrue(writer.isOpened())
            for brightness in [40, 80, 120, 160]:
                writer.write(np.full((64, 64, 3), brightness, np.uint8))
            writer.release()
            seg = replay.Segment(CAM, path, 0, 3, 0, 3/15)
            reader = replay.SegmentReader(seg)
            try:
                for i in range(3):
                    img, error = reader.get(i)
                    self.assertIsNotNone(img)
                    self.assertIsNone(error)
                img, error = reader.get(3)
                self.assertIsNone(img)
                self.assertEqual(error, "MISSING TAIL FRAME")
            finally:
                reader.close()
            next_segment = replay.Segment(CAM, path, 3, 4, 3/15, 4/15)
            reader2 = replay.SegmentReader(next_segment)
            try:
                img, error = reader2.get(0)
                self.assertIsNone(error)
                self.assertGreater(float(np.mean(img)), 145.0)
            finally:
                reader2.close()

    def test_layout_and_controls(self):
        cams = list(replay.CAMERAS)
        ep = replay.Episode(35, 0, 100, {c: replay.Segment(c, Path('x.mp4'), 0, 99, 0, 99/15) for c in cams}, 'fake')
        blank = {c: None for c in cams}
        errors = {cams[0]: "MISSING TAIL FRAME"}
        canvas = replay.compose_canvas(blank, errors, ep, 99, 15, 1, 'PLAYING', cams)
        self.assertEqual(canvas.shape, (840, 1280, 3))
        eps = [ep, replay.Episode(36, 1, 50, ep.segments, 'fake')]
        state = {'stopped': False, 'paused': False, 'rate': 1.0}
        self.assertEqual(replay.handle_command('n', state, eps, 0), ('goto', 1))
        self.assertEqual(replay.handle_command('g 35', state, eps, 1), ('goto', 0))
        self.assertEqual(replay.handle_command('rate 0.5', state, eps, 0), ('replay', 0))
        self.assertEqual(state['rate'], 0.5)
        replay.handle_command('p', state, eps, 0)
        self.assertTrue(state['paused'])

if __name__ == '__main__':
    main()
