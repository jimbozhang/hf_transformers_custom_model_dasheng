# coding=utf-8
# Copyright 2023-2024 Xiaomi Corporation and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
from pathlib import Path

import torch

from dasheng_model.configuration_dasheng import DashengConfig
from dasheng_model.feature_extraction_dasheng import DashengFeatureExtractor
from dasheng_model.modeling_dasheng import DashengModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_keys(state_dict):
    ignore_keys = [key for key in state_dict.keys() if key.startswith("front_end")]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    return f"encoder.{name}"


@torch.no_grad()
def convert_ced_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    r"""
    Convert a Dasheng checkpoint from the original repository to a 🤗 Transformers checkpoint.
    """

    config = DashengConfig(model_name)

    model_name_to_path = {
        "dasheng-base": (Path(os.environ["LEMONSTORE_BASEPATH"]) / "ssl/mae/dasheng_01b.pt"),
        "dasheng-0.6B": (Path(os.environ["LEMONSTORE_BASEPATH"]) / "ssl/mae/dasheng_06b.pt"),
        "dasheng-1.2B": (Path(os.environ["LEMONSTORE_BASEPATH"]) / "ssl/mae/dasheng_12b.pt"),
    }

    state_dict = torch.load(model_name_to_path[model_name], map_location="cpu")["model"]
    remove_keys(state_dict)
    new_state_dict = {rename_key(key): val for key, val in state_dict.items()}

    model = DashengModel(config)
    model.load_state_dict(new_state_dict)

    feature_extractor = DashengFeatureExtractor(return_attention_mask=False)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        logger.info(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"mispeech/{model_name}")
        feature_extractor.push_to_hub(f"mispeech/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dasheng-base",
        type=str,
        help="Name of the Dasheng model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_ced_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
