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


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


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
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]
        return outputs


def get_expert_mask(token_type_ids: "torch.LongTensor(B, L)") -> "[torch.BoolTensor(B, L), torch.BoolTensor(B, L)]":
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:] == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


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


def attention_fn(
        query_layer: "torch.tensor(B, H, L, HD)",
        key_layer: "torch.tensor(B, H, L, HD)",
        value_layer: "torch.tensor(B, H, L, HD)",
        attention_mask: "torch.tensor(B, H, L, HD)",
        *,
        scaling_attention_score: bool = True,
        attention_dropout: nn.Module = None
):
    attention_mask_bool = (attention_mask == 0)
    is_low_triangle = (attention_mask_bool == torch.ones_like(attention_mask_bool, dtype=torch.float).tril()).all()
    is_full = (attention_mask_bool > 0).all()
    if not (int(torch.__version__.split('.')[0]) >= 2):
        warnings.warn("It's recommended to use torch2.0 or higher.")
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        return torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=not is_full
        )
    else:
        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        if attention_dropout is not None:
            attention_scores = attention_dropout(attention_scores)
        context_layer = torch.matmul(attention_scores, value_layer)
        return context_layer


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

    # def _transpose_for_scores(self, tensor):
    #     """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
    #     new_tensor_shape = tensor.size()[:-1] + \
    #                        (-1, # flexible for multi-query
    #                         self.head_dim)
    #     tensor = tensor.view(*new_tensor_shape)
    #     return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # token_type_ids: torch.LongTensor,
        # position_ids: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,

        # vllm
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

        # query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        factor = mixed_raw_layer.size()[-1] // sum(self.stride)
        query_states, key_states, value_states = torch.split(mixed_raw_layer, [factor * x for x in self.stride], dim=-1)

        # query_states = self._transpose_for_scores(query_states)  # B, H, L, HD
        # key_states = self._transpose_for_scores(key_states)  # B, H, L, HD
        # value_states = self._transpose_for_scores(value_states)  # B, H, L, HD
        # query_states = query_states.view(shape[0], self.num_attention_heads, -1).unsqueeze(0).transpose(-2, -3)
        # key_states = key_states.view(shape[0], self.num_multi_query_heads, -1).unsqueeze(0).transpose(-2, -3)
        # value_states = value_states.view(shape[0], self.num_multi_query_heads, -1).unsqueeze(0).transpose(-2, -3)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]

        # query_states, key_states = self.rotary_emb(query_states, key_states, position_ids=positions.unsqueeze(0), max_seqlen=positions.unsqueeze(0).max() + 1)

        # query_states = query_states.squeeze(0).transpose(0, 1).reshape(shape[0], -1)
        # key_states = key_states.squeeze(0).transpose(0, 1).reshape(shape[0], -1)
        # value_states = value_states.squeeze(0).transpose(0, 1).reshape(shape[0], -1)
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # if past_key_value is not None:
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None

        # query_states = query_states.view(query_states.shape[0], self.num_attention_heads, -1).transpose(0, 1)
        # key_states = key_states.view(key_states.shape[0], self.num_multi_query_heads, -1).transpose(0, 1)
        # value_states = value_states.view(value_states.shape[0], self.num_multi_query_heads, -1).transpose(0, 1)

        # key_states = key_states.unsqueeze(1).expand(-1, self.num_attention_heads // self.num_multi_query_heads, -1, -1).contiguous().view(
        #     self.num_attention_heads, *key_states.shape[1:])
        # value_states = value_states.unsqueeze(1).expand(-1, self.num_attention_heads // self.num_multi_query_heads, -1,
        #                                                 -1).contiguous().view(self.num_attention_heads, *value_states.shape[1:])

        # context_layer = attention_fn(
        #     query_layer=query_states, key_layer=key_states, value_layer=value_states, attention_mask=attention_mask,
        #     scaling_attention_score=True, attention_dropout=None)
        # if context_layer.size() != (bsz, self.num_attention_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_attention_heads, q_len, self.head_dim)}, but is"
        #         f" {context_layer.size()}"
        #     )
        # context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        # for kv_cache
        context_layer = self.attn(
            query_states,
            key_states,
            value_states,
            kv_cache,
            attn_metadata,
        )

        # query_states = query_states.reshape(shape[0], self.num_attention_heads, -1).transpose(0, 1).unsqueeze(0)
        # key_states = key_states.reshape(shape[0], self.num_multi_query_heads, -1).transpose(0, 1).unsqueeze(0)
        # value_states = value_states.reshape(shape[0], self.num_multi_query_heads, -1).transpose(0, 1).unsqueeze(0)
        # key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_attention_heads // self.num_multi_query_heads, -1, -1).contiguous().view(
        #     1, self.num_attention_heads, *key_states.shape[2:])
        # value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_attention_heads // self.num_multi_query_heads, -1,
        #                                                 -1).contiguous().view(1, self.num_attention_heads, *value_states.shape[2:])

        # context_layer = torch.nn.functional.scaled_dot_product_attention(
        #     query=query_states,
        #     key=key_states,
        #     value=value_states,
        #     attn_mask=None,
        #     is_causal=True,
        #     scale=None,
        # )
        # context_layer = context_layer.transpose(1, 2).contiguous().reshape(1, shape[0], -1).squeeze(0)

        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask=torch.ones(1, shape[0], dtype=torch.long, device=hidden_states.device),
        #     input_shape=(1, shape[0]),
        #     inputs_embeds=hidden_states,
        #     past_key_values_length=0,
        # )

        # query_states = query_states / math.sqrt(query_states.shape[-1])
        # attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        # attention_scores = attention_scores + attention_mask.squeeze(0)
        # attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # context_layer = torch.matmul(attention_scores, value_states)
        # context_layer = context_layer.transpose(0, 1).reshape(shape[0], -1)

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

        # if output_attentions:
        #     warnings.warn("output_attentions is not implemented.")

        # return attn_output, None, past_key_value
        return attn_output

    # def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    #     # create causal mask
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     combined_attention_mask = None
    #     if input_shape[-1] > 1:
    #         combined_attention_mask = _make_causal_mask(
    #             input_shape,
    #             inputs_embeds.dtype,
    #             device=inputs_embeds.device,
    #             past_key_values_length=past_key_values_length,
    #         )

    #     if attention_mask is not None:
    #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
    #             inputs_embeds.device
    #         )
    #         combined_attention_mask = (
    #             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    #         )

    #     return combined_attention_mask


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
        # token_type_ids: torch.LongTensor,
        # position_ids: torch.LongTensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,

        # vllm
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


class CogVLMPreTrainedModel(PreTrainedModel):
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogVLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x: "torch.BoolTensor(B, L)", attention_mask: Optional["torch.BoolTensor(B, L)"] = None) -> "torch.LongTensor(B, L)":
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


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
        # Initialize weights and apply final processing
        # self.post_init()

    # def encode_images(self, images: List[List[torch.Tensor]]) -> torch.Tensor:
    #     images_list, images = images, []

    #     images = []
    #     for image_list in images_list:
    #         for image in image_list:
    #             images.append(image)

    #     images = torch.stack(images)
    #     images_features = self.vision(images)
    #     return images_features

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
        # images: List[List[torch.Tensor]] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

        # vllm
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

        # if past_key_values is not None:
        #     pass  # generate mode with past_key_values. the image features are already mapped
        # else:
        #     # not allow for inputs_embeds, because we want to process image feature
        #     assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
        #     if not is_empty(images):  # multi-modality
        #         assert token_type_ids is not None, f"multi-modality requires `token_type_ids`!"
        #         assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
        #         inputs_embeds = self.embed_tokens(input_ids)
        #         images_features = self.encode_images(images)
        #         images_features = rearrange(images_features, 'b n d -> (b n) d')
        #         images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        #         inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
        #     else:  # single-modality
        #         if token_type_ids is None:
        #             token_type_ids = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
        #         assert not (token_type_ids == VISION_TOKEN_TYPE).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
        #         inputs_embeds = self.embed_tokens(input_ids)

        #     if position_ids is None:
        #         position_ids = build_position_ids(token_type_ids, attention_mask)
        #     input_ids = None
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
        # token_type_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor],
        vision_token_mask: Optional[torch.FloatTensor],
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        # elif input_ids is not None:
        #     batch_size, seq_length = input_ids.shape
        # elif inputs_embeds is not None:
        #     batch_size, seq_length, _ = inputs_embeds.shape
        # else:
        #     raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # seq_length_with_past = seq_length
        # past_key_values_length = 0

        # if past_key_values is not None:
        #     past_key_values_length = past_key_values[0][0].shape[2]
        #     seq_length_with_past = seq_length_with_past + past_key_values_length

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        #     )
        #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        # else:
        #     position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        #     )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        hidden_states = inputs_embeds

        # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None

        for idx in range(len(self.layers)):
            # if output_hidden_states:
            #     all_hidden_states += (hidden_states,)

            # past_key_value = past_key_values[idx] if past_key_values is not None else None

            # def custom(index):
            #     def custom_forward(
            #         hidden_states,
            #         positions,
            #         kv_cache,
            #         attn_metadata,
            #         vision_token_mask,
            #     ):
            #         layer = self.layers[index]
            #         outputs = layer(
            #             hidden_states,
            #             positions=positions,
            #             kv_cache=kv_cache,
            #             attn_metadata=attn_metadata,
            #             vision_token_mask=vision_token_mask,
            #         )
            #         return outputs

            #     return custom_forward
            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     token_type_ids=token_type_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_value=past_key_value,
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            # )
            decoder_layer = self.layers[idx]
            hidden_states = decoder_layer(
                hidden_states,
                positions,
                kv_caches[idx],
                attn_metadata,
                vision_token_mask,
            )

            # if use_cache:
            #     next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)

        # next_cache = next_decoder_cache if use_cache else None
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return hidden_states

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     # noinspection PyMethodMayBeStatic
    # # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    #     # create causal mask
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     combined_attention_mask = None
    #     if input_shape[-1] > 1:
    #         combined_attention_mask = _make_causal_mask(
    #             input_shape,
    #             inputs_embeds.dtype,
    #             device=inputs_embeds.device,
    #             past_key_values_length=past_key_values_length,
    #         )

    #     if attention_mask is not None:
    #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
    #             inputs_embeds.device
    #         )
    #         combined_attention_mask = (
    #             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    #         )

    #     return combined_attention_mask


# def _history_to_prompt(signal_type, history, query):
#     if signal_type == 'base':
#         return query
#     elif signal_type == 'vqa':
#         answer_format = 'Short answer:'
#     elif signal_type == 'chat':
#         answer_format = 'Answer:'
#     else:
#         assert False, f"Unknown signal type {signal_type}"

#     prompt = ''
#     for i, (old_query, response) in enumerate(history):
#         prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
#     prompt += 'Question: {} {}'.format(query, answer_format)
#     return prompt


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

        # Initialize weights and apply final processing
        # self.post_init()

    # def get_input_embeddings(self):
    #     return self.model.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.model.embed_tokens = value

    # def get_output_embeddings(self):
    #     return self.lm_head

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings

    # def set_decoder(self, decoder):
    #     self.model = decoder

    # def get_decoder(self):
    #     return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        # images: List[List[torch.Tensor]] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        # labels: Optional[torch.LongTensor] = None,

        # vllm
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # TODO: split vision and language token through input_ids
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            # images=images,
            # token_type_ids=token_type_ids,
            # attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **kwargs
        )

        hidden_states = outputs
        # logits = self.lm_head(hidden_states)
        # logits = logits.float()

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
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

    # def _prepare_attention_mask_for_generation(
    #         self,
    #         inputs: torch.Tensor,
    #         pad_token_id: Optional[int],
    #         eos_token_id: Optional[Union[int, List[int]]],
    # ) -> torch.LongTensor:
    #     return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)  # type: ignore

    # def prepare_inputs_for_generation(
    #         self, input_ids, token_type_ids, images=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # ):
    #     # build position_ids if needed
    #     position_ids = kwargs.get("position_ids", None)
    #     if position_ids is None:
    #         position_ids = build_position_ids(token_type_ids, attention_mask)

    #     if past_key_values:
    #         input_ids = input_ids[:, -1:]
    #         token_type_ids = token_type_ids[:, -1:]
    #         position_ids = position_ids[:, -1:]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "token_type_ids": token_type_ids,
    #             "images": images,
    #             "position_ids": position_ids,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #             "attention_mask": attention_mask,
    #         }
    #     )
    #     return model_inputs

    # def _update_model_kwargs_for_generation(
    #         self,
    #         outputs: "ModelOutput",
    #         model_kwargs: Dict[str, Any],
    #         is_encoder_decoder: bool = False,
    #         standardize_cache_format: bool = False,
    # ) -> Dict[str, Any]:
    #     # update past_key_values
    #     model_kwargs["past_key_values"] = self._extract_past_from_model_output(
    #         outputs, standardize_cache_format=standardize_cache_format
    #     )
    #     if getattr(outputs, "state", None) is not None:
    #         model_kwargs["state"] = outputs.state

    #     # update token_type_ids with last value
    #     if "token_type_ids" in model_kwargs:
    #         token_type_ids = model_kwargs["token_type_ids"]
    #         new_token_type_ids = torch.ones(size=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype, device=token_type_ids.device) * LANGUAGE_TOKEN_TYPE
    #         model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)

    #     if not is_encoder_decoder:
    #         # update attention mask
    #         if "attention_mask" in model_kwargs:
    #             attention_mask = model_kwargs["attention_mask"]
    #             model_kwargs["attention_mask"] = torch.cat(
    #                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    #             )
    #     else:
    #         # update decoder attention mask
    #         if "decoder_attention_mask" in model_kwargs:
    #             decoder_attention_mask = model_kwargs["decoder_attention_mask"]
    #             model_kwargs["decoder_attention_mask"] = torch.cat(
    #                 [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
    #                 dim=-1,
    #             )

    #     return model_kwargs

    # def _reorder_cache(self, past_key_values, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past_key_values:
    #         reordered_past += (
    #             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
    #         )
    #     return reordered_past

    def build_conversation_input_ids(
            self,
            tokenizer: "PreTrainedTokenizer",
            *,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            template_version: Optional[Literal["base", "chat", "vqa"]] = None,
            answer: str = None,
    ):
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = _history_to_prompt(template_version, history, query)
        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = [transform(images[0])]
            # language
            vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2

            tokenizer.pad_token_id = 128002 # llama3 adapt for cogvlm

            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)

        if answer is not None:
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_ids += [tokenizer.eos_token_id]
            text_ids += answer_ids

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        if answer is not None:
            labels = [-100 for _ in range(len(input_ids) - len(answer_ids))] + answer_ids
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = None

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': images,
            'labels': labels,
        }
