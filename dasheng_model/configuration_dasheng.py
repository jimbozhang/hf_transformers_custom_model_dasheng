# coding=utf-8
# Copyright 2023-2024 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" Dasheng model configuration"""


from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.utils.hub import cached_file

logger = logging.get_logger(__name__)

DASHENG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mispeech/dasheng-base": "https://huggingface.co/mispeech/dasheng-base/resolve/main/config.json",
}


class DashengConfig(PretrainedConfig):
    model_type = "dasheng"

    r"""
    Configuration class for the Dasheng model.

    Args:
        name (str, optional, *optional*):
            Name of the pre-defined configuration. Can be "dasheng-base", "dasheng-0.6B", or "dasheng-1.2B".
        attn_drop_rate (float, *optional*, defaults to 0.0):
            Dropout probability for attention weights. Default to 0.0.
        depth (int, *optional*, defaults to 12): Number of transformer layers. Default to 12.
        drop_path_rate (float, *optional*, defaults to 0.0): Drop path is taken from timm. Default to 0.0.
        drop_rate (float, *optional*, defaults to 0.0):
            Dropout probability for input embeddings. Default to 0.0.
        embed_dim (int, *optional*, defaults to 768):
            Dimensionality of the audio patch embeddings. Default to 768.
        eval_avg (str, *optional*, defaults to `"mean"`):
            Type of pooling to use for evaluation. Can be "mean", "token", "dm" or "logit". Default to "mean".
        mlp_ratio (float, *optional*, defaults to 4.0):
            Ratio of hidden size in the feedforward layer to the embedding size. Default to 4.0.
        num_heads (int, *optional*, defaults to 12): Number of attention heads. Default to 12.
        outputdim (int, *optional*, defaults to 527): Dimensionality of the output. Default to 527.
        patch_size (int, *optional*, defaults to 16): Size of the patches. Default to 16.
        patch_stride (int, *optional*, defaults to 16): Stride of the patches. Default to 16.
        pooling (str, *optional*, defaults to `"mean"`):
            Type of pooling to use for the output. Can be "mean", "token", "dm" or "logit". Default to "mean".
        qkv_bias (bool, *optional*, defaults to `True`):
            Whether to include bias terms in the query, key and value projections. Default to True.
        target_length (int, *optional*, defaults to 1012): Frames of an audio chunk. Default to 1012.
    """

    def __init__(
        self,
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        encoder_kwargs = dict(
            embed_dim=768, depth=12, num_heads=12, target_length=1012, patch_size=[64, 4], patch_stride=[64, 4]
        )
        encoder_kwargs.update((k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs))
        encoder_kwargs = {**encoder_kwargs, **kwargs}

        if name == "dasheng-1.2B":
            encoder_kwargs["embed_dim"] = 1536
            encoder_kwargs["depth"] = 40
            encoder_kwargs["num_heads"] = 24
        elif name == "dasheng-0.6B":
            encoder_kwargs["embed_dim"] = 1280
            encoder_kwargs["depth"] = 32
            encoder_kwargs["num_heads"] = 16
        elif name == "dasheng-base":
            encoder_kwargs["embed_dim"] = 768
            encoder_kwargs["depth"] = 12
            encoder_kwargs["num_heads"] = 12
        else:
            logger.info("No model name specified for DashengConfig, use default settings.")

        self.name = name
        self.encoder_kwargs = encoder_kwargs
        self.loss = "BCE"
