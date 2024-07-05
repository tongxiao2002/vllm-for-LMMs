import torch
from torch import Tensor
from typing import List

from transformers import PretrainedConfig
from vllm.config import CacheConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .visual import VisionTransformer
from vllm.attention import AttentionMetadata
from vllm.model_executor.models.qwen import QWenLMHeadModel, QWenModel


class QwenVLModel(QWenModel):
    def __init__(self, config, cache_config, quant_config):
        super().__init__(config, cache_config, quant_config)
        self.visual = VisionTransformer(**config.visual)
        self.image_start_id = config.visual['image_start_id']
        self.image_end_id = self.image_start_id + 1
        self.image_pad_id = self.image_start_id + 2

    def merge_vision_embeddings(self, input_ids, input_embeds, vision_embeddings):
        img_bos_pos = torch.where(input_ids == self.image_start_id)[0].tolist()
        img_eos_pos = torch.where(input_ids == self.image_end_id)[0].tolist()
        if len(img_bos_pos) != len(img_eos_pos):
            # pass dummy data inputs
            return input_embeds

        mask = torch.zeros_like(input_ids)
        for a, b in zip(img_bos_pos, img_eos_pos):
            mask[a + 1:b] = 1
        mask = mask.to(dtype=torch.bool)
        image_feature_size = vision_embeddings.shape[0] * vision_embeddings.shape[1]
        input_embeds[mask, :] = vision_embeddings.view(image_feature_size, vision_embeddings.shape[-1])
        return input_embeds

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> Tensor:
        hidden_states = self.wte(input_ids)

        images = kwargs.pop("pixel_values", None)
        if images is not None:
            vision_embeddings = self.visual.encode_pixels(images)
            hidden_states = self.merge_vision_embeddings(
                input_ids=input_ids,
                input_embeds=hidden_states,
                vision_embeddings=vision_embeddings,
            )

        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class QwenVLLMHeadModel(QWenLMHeadModel):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None
    ):
        super().__init__(config, cache_config, quant_config)
        self.transformer = QwenVLModel(config, cache_config, quant_config)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs
    ) -> Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, **kwargs)
        return hidden_states
