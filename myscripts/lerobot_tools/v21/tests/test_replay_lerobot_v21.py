"""Synthetic v2.1 metadata, camera index, missing-tail and browsing tests."""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'replay_lerobot_v21_videos_keyboard.py'
spec=importlib.util.spec_from_file_location('robotwin_v21_player_test', SCRIPT)
player=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=player
spec.loader.exec_module(player)

CAM=player.CAMERAS[0]

class TestV21(unittest.TestCase):
    def create_fixture(self, root):
        (root/'meta').mkdir(parents=True)
        info={
            'codebase_version':'v2.1','fps':15,'chunks_size':1000,
            'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features':{cam:{'dtype':'video'} for cam in player.CAMERAS},
        }
        (root/'meta/info.json').write_text(json.dumps(info))
        with (root/'meta/episodes.jsonl').open('w') as fp:
            for idx, length in [(0,4),(1,5),(1000,2)]:
                fp.write(json.dumps({'episode_index':idx,'length':length,'tasks':['test']})+'\n')
        return info

    def test_metadata_and_chunks(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            self.create_fixture(root)
            eps=player.load_episodes(root,list(player.CAMERAS),15)
            self.assertEqual([ep.episode_index for ep in eps],[0,1,1000])
            self.assertEqual(eps[1].data_file,'data/chunk-000/episode_000001.parquet')
            self.assertEqual(eps[2].data_file,'data/chunk-001/episode_001000.parquet')
            self.assertEqual(str(eps[2].segments[CAM].path.relative_to(root)),
                'videos/chunk-001/observation.images.cam_high/episode_001000.mp4')
            self.assertEqual(eps[0].length,4)

    def test_video_episode_short_tail(self):
        with TemporaryDirectory() as td:
            root=Path(td);self.create_fixture(root)
            eps=player.load_episodes(root,list(player.CAMERAS),15)
            for cam in player.CAMERAS:
                p=eps[0].segments[cam].path
                p.parent.mkdir(parents=True,exist_ok=True)
                raw=np.stack([np.full((32,48,3),q*60,np.uint8) for q in range(3)])
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
                  '-pixel_format','bgr24','-video_size','48x32','-framerate','15','-i','pipe:0',
                  '-c:v','mpeg4','-q:v','2',str(p)],input=raw.tobytes(),check=True,
                  timeout=15,capture_output=True)
            ep=player.prepare_episode(eps[0],15)
            self.assertEqual(ep.length,4)
            self.assertEqual([s.length for s in ep.segments.values()],[3,3,3])
            reader=player.EpisodeReader(ep,video_fps=15)
            try:
                for idx in range(3):
                    images, errors=reader.get(idx)
                    self.assertFalse(errors,errors)
                    self.assertEqual(tuple(images[CAM].shape),(32,48,3))
                images,errors=reader.get(3)
                self.assertEqual(set(errors),set(player.CAMERAS))
                self.assertTrue(all(error=='MISSING TAIL FRAME' for error in errors.values()))
            finally:
                reader.close()

    def test_commands_and_layout(self):
        with TemporaryDirectory() as td:
            root=Path(td);self.create_fixture(root)
            eps=player.load_episodes(root,list(player.CAMERAS),15)
            state={'rate':1.,'frame':0,'paused':False,'stopped':False}
            self.assertEqual(player.handle_command('f 3',state,eps,0),('seek',3))
            self.assertTrue(state['paused'])
            self.assertEqual(player.handle_command('g 1000',state,eps,0),('goto',2))
            self.assertEqual(player.handle_command('nextframe',state,eps,0),('seek',1))
            blank={cam:None for cam in player.CAMERAS}
            frame=player.compose_canvas(blank,{},eps[0],0,15,1,'PAUSED',list(player.CAMERAS))
            self.assertEqual(frame.shape,(840,1280,3))

if __name__=='__main__':
    unittest.main()
