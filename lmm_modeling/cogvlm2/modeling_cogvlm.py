"""largely copy from llama and adapt for cogvlm"""
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Literal, Dict, Any, Iterable

import math
import torch
from torch import nn
from torchvision import transforms

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.logging import get_logger
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_cogvlm import CogVLMConfig
from .visual import EVA2CLIPModel

# vllm adaption
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    QKVParallelLinear,
    ColumnParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.config import VisionLanguageConfig, CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.vlm_base import VisionLanguageModelBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import get_tensor_model_parallel_world_size

logger = get_logger(__name__)

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated_outputs, _ = self.gate_proj(x)
        up_proj_outputs, _ = self.up_proj(x)
        outputs = self.act_fn(gated_outputs) * up_proj_outputs
        outputs, _ = self.down_proj(outputs)
        return outputs


class VisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def forward(self, hidden_states: "torch.Tensor(B, L, D)", vision_token_mask: "torch.LongTensor(B, L)"):
        language_token_mask = ~vision_token_mask
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


class VisionExpertAttention(nn.Module):
    def __init__(self, config, cache_config, quant_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_attention_heads = config.num_attention_heads
        self.num_attention_heads = self.total_num_attention_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_attention_heads

        self.total_num_multi_query_heads = config.num_multi_query_heads
        self.num_multi_query_heads = max(1, self.total_num_multi_query_heads // tp_size)

        self.stride = [self.num_attention_heads, self.num_multi_query_heads, self.num_multi_query_heads]
        self.qkv_size = (self.num_attention_heads + 2 * self.num_multi_query_heads) * self.head_dim

        self.max_position_embeddings = config.max_position_embeddings
        # self.rotary_emb = FastRotaryEmbedding(dim=self.head_dim, pos_idx_in_fp32=False, base=500000)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=500000,
            rope_scaling=None,
            dtype=torch.bfloat16,
        )
        self.vision_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_attention_heads,
            total_num_kv_heads=self.total_num_multi_query_heads,
            bias=True,
        )
        self.vision_expert_dense = RowParallelLinear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_attention_heads,
            total_num_kv_heads=self.total_num_multi_query_heads,
            bias=False,
        )
        self.language_expert_dense = RowParallelLinear(self.hidden_size, self.hidden_size, bias=False)

        self.attn = Attention(
            num_heads=self.num_attention_heads,
            head_size=self.head_dim,
            scale=1 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_multi_query_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        vision_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        language_token_mask = ~vision_token_mask

        shape = list(hidden_states.shape)

        vision_qkv, _ = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        language_qkv, _ = self.language_expert_query_key_value(hidden_states[language_token_mask])

        mixed_raw_layer = torch.empty(
            shape[0],
            self.vision_expert_query_key_value.output_size_per_partition,
            dtype=vision_qkv.dtype,
            device=vision_qkv.device
        )
        mixed_raw_layer[vision_token_mask] = vision_qkv
        mixed_raw_layer[language_token_mask] = language_qkv

        factor = mixed_raw_layer.size()[-1] // sum(self.stride)
        query_states, key_states, value_states = torch.split(mixed_raw_layer, [factor * x for x in self.stride], dim=-1)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        context_layer = self.attn(
            query_states,
            key_states,
            value_states,
            kv_cache,
            attn_metadata,
        )

        attn_output = torch.empty(
            shape[0],
            self.total_num_attention_heads * self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        vision_attn_output, _ = self.vision_expert_dense(context_layer[vision_token_mask])
        language_attn_output, _ = self.language_expert_dense(context_layer[language_token_mask])
        attn_output[vision_token_mask] = vision_attn_output
        attn_output[language_token_mask] = language_attn_output

        return attn_output


class CogVLMDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config, cache_config, quant_config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        vision_token_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            vision_token_mask=vision_token_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, vision_token_mask=vision_token_mask)
        hidden_states = residual + hidden_states

        return hidden_states  # type: ignore


class CogVLMModel(nn.Module):
    def __init__(self, config, cache_config, quant_config):
        super().__init__()
        self.padding_idx = 128002
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CogVLMDecoderLayer(config, cache_config, quant_config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = EVA2CLIPModel(config)

        self.gradient_checkpointing = False
        # self-defined, use a reserved token as image token
        self.image_token_id = 128002

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images_features = self.vision(images)
        return images_features

    def reposition(self, input_ids: torch.Tensor, positions: torch.Tensor, input_with_image: bool):
        if not input_with_image:
            # images are already in kv_caches, input_ids are single tokens whose positions are added by image tokens
            # so we just substract the number of image tokens from positions
            positions -= (1344 // 14 // 2) ** 2 - 1
            return positions

        prev_is_boi = False
        for idx in range(len(positions)):
            if idx == 0:
                if input_ids[idx].item() == self.image_token_id:
                    prev_is_boi = True
                continue

            if input_ids[idx].item() == self.image_token_id:
                if input_ids[idx - 1].item() == self.image_token_id:
                    # image token and is not boi
                    if input_ids[idx + 1].item() == self.image_token_id:
                        # image token and is not eoi
                        if prev_is_boi:
                            positions[idx] = positions[idx - 1] + 1
                            prev_is_boi = False
                        else:
                            positions[idx] = positions[idx - 1]
                    else:
                        # eoi
                        positions[idx] = positions[idx - 1] + 1
                else:
                    # boi
                    positions[idx] = positions[idx - 1] + 1
                    prev_is_boi = True
            else:
                positions[idx] = positions[idx - 1] + 1
        return positions

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        images = kwargs.pop('pixel_values', None)

        if kv_caches[0] is not None:
            positions = self.reposition(input_ids, positions, input_with_image=False if images is None else True)
        if images is not None:
            # image is not mapped, so manually map embeddings of text tokens and vision tokens into `input_embeds`
            inputs_embeds = self.embed_tokens(input_ids)
            images_features = self.encode_images(images)
            images_features = images_features.view(-1, images_features.shape[-1]).to(dtype=inputs_embeds.dtype)
            vision_token_mask = input_ids == self.image_token_id
            inputs_embeds[vision_token_mask] = images_features
            vision_token_mask[:-1] = vision_token_mask[:-1] & vision_token_mask[1:]
            input_ids = None
        else:
            # generate mode with past_key_values. the image features are already mapped
            vision_token_mask = input_ids == self.image_token_id
            vision_token_mask[:-1] = vision_token_mask[:-1] & vision_token_mask[1:]
            inputs_embeds = None

        return self.llm_forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            vision_token_mask=vision_token_mask,
        )

    def llm_forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.FloatTensor],
        vision_token_mask: Optional[torch.FloatTensor],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for idx in range(len(self.layers)):
            decoder_layer = self.layers[idx]
            hidden_states = decoder_layer(
                hidden_states,
                positions,
                kv_caches[idx],
                attn_metadata,
                vision_token_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class CogVLM2ForCausalLM(VisionLanguageModelBase):
    # _auto_class = "AutoModelForCausalLM"

    def __init__(
        self,
        config,
        vision_language_config: VisionLanguageConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None
    ):
        super().__init__(vision_language_config)
        self.model = CogVLMModel(config, cache_config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            bias=False
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **kwargs
        )
        hidden_states = outputs

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.pop(name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        assert len(params_dict) == 0
