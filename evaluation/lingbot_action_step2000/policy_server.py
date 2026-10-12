"""Official LingBot server adapted to a self-contained exported checkpoint."""

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINGBOT_VLA_ROOT = PROJECT_ROOT / "lingbot-vla-v2"
for path in (PROJECT_ROOT, LINGBOT_VLA_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

import torch
import yaml
from transformers import AutoConfig

from deploy.lingbot_vla_v2_policy import (
    FeatureTransform,
    LingBotVlaV2InferencePolicy,
    LingbotVLAv2Server,
    WebsocketPolicyServer,
    build_processor,
    str2bool,
)
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config


class ExportedCheckpointServer(LingbotVLAv2Server):
    def load_vla(self, path_to_pi_model):
        checkpoint = Path(path_to_pi_model).resolve()
        asset_root = Path(os.environ.get("LINGBOT_MODEL_ASSET_ROOT", checkpoint)).resolve()
        training_config_path = Path(os.environ.get(
            "LINGBOT_TRAINING_CONFIG", asset_root / "training/robotwin_clean_head.yaml"
        )).resolve()
        with training_config_path.open("r", encoding="utf-8") as stream:
            training_config = yaml.safe_load(stream)

        # The data loader expects these entries in their CLI string form.
        data = dict(training_config["data"])
        data["joints"] = [repr(item) if isinstance(item, dict) else item for item in data["joints"]]
        data["norm_type"] = [repr(item) if isinstance(item, dict) else item for item in data["norm_type"]]
        data["norm_stats_file"] = str(Path(os.environ.get(
            "LINGBOT_NORM_STATS", asset_root / "assets/norm_stats/robotwin_clean.json"
        )).resolve())

        model_values = dict(training_config["model"])
        train_values = dict(training_config["train"])
        model_values.update(train_values)
        align = model_values.get("align_params", {})
        models_root = Path(os.environ.get("LINGBOT_MODELS_ROOT", checkpoint.parent)).resolve()
        base_assets = models_root / "lingbot-vla-v2-6b"
        depth_assets = models_root / "depth/moge-2-vitb-normal/model.pt"
        if "depth" in align:
            align["depth"]["moge_path"] = str(depth_assets)
            align["depth"]["morgbd_path"] = str(base_assets / "depth/model.pt")
        if "video" in align:
            align["video"]["ckpt_path"] = str(base_assets / "dino_video/teacher_step_10000.pth")
            align["video"]["config_path"] = str(base_assets / "dino_video/config.yaml")

        config = LingbotVLAV2Config(**model_values)
        for key, value in model_values.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        config.attention_implementation = "eager"
        base_model_path = os.environ["QWEN3VL_PATH"]
        config.tokenizer_path = base_model_path
        self.model_name = "qwen3vl"
        self.config = config
        self.merge_qwen_config(AutoConfig.from_pretrained(base_model_path))
        config.use_cache = True
        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        self.data_config = SimpleNamespace(**data)

        self.vla = LingBotVlaV2InferencePolicy(config, eval=True)
        self.load_model_weights(str(checkpoint), strict=True)
        self.vla.feature_transform = None
        self.vla.model._use_compile_predict_velocity = bool(self.use_compile)
        self.vla.model._compiled_predict_velocity = None
        self.sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            self.vla.model.qwenvl_with_expert = torch.compile(self.vla.model.qwenvl_with_expert)
            self.sample_actions_fn = torch.compile(self.vla.model.sample_actions)
        self.robot_norm_path = data["norm_stats_file"]
        self.exported_robot_config = Path(os.environ.get(
            "LINGBOT_ROBOT_CONFIG", asset_root / "configs/robot_configs/robotwin.yaml"
        )).resolve()
        return self.vla

    def reset(self, robo_name, path_to_pi_model=None):
        if path_to_pi_model is not None:
            raise ValueError("Runtime checkpoint switching is disabled for sealed evaluation")
        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        with self.exported_robot_config.open("r", encoding="utf-8") as stream:
            self.robot_config = yaml.safe_load(stream)
        feature_transform = FeatureTransform(
            str(self.exported_robot_config), self.data_config, self.config,
            self.processor, chunk_size=self.config.chunk_size,
            norm_stats_path=self.robot_norm_path,
        )
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--use_bf16", type=str2bool, default=True)
    parser.add_argument("--use_fp32", type=str2bool, default=False)
    parser.add_argument("--use_compile", type=str2bool, default=True)
    args = parser.parse_args()
    model = ExportedCheckpointServer(
        args.model_path, use_length=args.use_length, chunk_ret=True,
        use_bf16=args.use_bf16, use_fp32=args.use_fp32,
        use_compile=args.use_compile,
    )
    WebsocketPolicyServer(model, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
