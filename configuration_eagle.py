# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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
""" Bloom configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class EagleConfig(PretrainedConfig):
    model_type = "EagleModel"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        vision_encoder_model_name_or_path="",
        vision_encoder_model_type="",
        connector_model_name_or_path="",
        connector_model_type="",
        language_model_name_or_path="",
        language_model_type="",
        cross_attention_freq=1,
        vision_hidden_size=768,
        query_length=32,
        **kwargs,
    ):        
        self.use_cache = use_cache
        self.vision_hidden_size = vision_hidden_size
        self.query_length = query_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vision_encoder_model_name_or_path = vision_encoder_model_name_or_path
        self.vision_encoder_model_type = vision_encoder_model_type
        self.connector_model_name_or_path = connector_model_name_or_path
        self.connector_model_type = connector_model_type
        self.cross_attention_freq = cross_attention_freq
        self.language_model_name_or_path = language_model_name_or_path
        self.language_model_type = language_model_type

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        return not self.alibi
