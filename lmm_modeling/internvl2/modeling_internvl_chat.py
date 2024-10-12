# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union, Iterable

import torch.utils.checkpoint
import transformers
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

# vllm adaption
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.attention import AttentionMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.vlm_base import VisionLanguageModelBase

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class LlamaForCausalLMWithEmbed(LlamaForCausalLM):
    def forward(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        input_embeds=None,
    ):
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, input_embeds)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class InternVLForCausalLM(VisionLanguageModelBase):
    def __init__(self, config, vision_language_config, cache_config, quant_config):
        super().__init__(vision_language_config)
        self.model = InternVLChatModel(config, cache_config, quant_config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, **kwargs)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.model.language_model.logits_processor(
            self.model.language_model.lm_head.weight,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.model.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(filter(lambda x: not x[0].startswith('model.language_model.'), self.named_parameters()))
        llama_weights = []
        for name, loaded_weight in weights:
            if name.startswith('language_model.'):
                llama_weights.append((name[len('language_model.'):], loaded_weight))
                continue

            name = "model." + name
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.pop(name)

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        assert len(params_dict) == 0

        self.model.language_model.load_weights(iter(llama_weights))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer']

    def __init__(
        self,
        config: InternVLChatConfig,
        cache_config=None,
        quant_config=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.text_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        self.vision_model = InternVisionModel(config.vision_config)
        if config.text_config.architectures[0] == 'LlamaForCausalLM':
            self.language_model = LlamaForCausalLMWithEmbed(config.text_config, cache_config, quant_config)
        else:
            raise NotImplementedError(f'{config.text_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.img_context_token_id = 64000

    def forward(
        self,
        # pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # image_flags: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = kwargs.pop('pixel_values', None)

        # image_flags = image_flags.squeeze(-1)

        if pixel_values is not None:
            input_embeds = self.language_model.model.get_input_embeddings(input_ids)
            vit_embeds = self.extract_feature(pixel_values)
            # vit_embeds = vit_embeds[image_flags == 1]

            N, C = input_embeds.shape
            selected = (input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.to(dtype=torch.int8).sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

            input_ids = None
        else:
            input_embeds = None
        # vit_batch_size = pixel_values.shape[0]

        # input_embeds = input_embeds.reshape(B, N, C)

        hidden_states = self.language_model(
            input_embeds=input_embeds,
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        return hidden_states

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
