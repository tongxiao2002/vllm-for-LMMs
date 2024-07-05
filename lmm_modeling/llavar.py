import torch
from torch import nn
from typing import Iterable, Tuple
from transformers import LlavaConfig, CLIPVisionModel
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_KEYS_TO_MODIFY_MAPPING = {
    # Prioritize replacing "model.mm_projector" with "mm_projector" rather than language_model.mm_projector
    # The earlier the position in the mapping dictionary, the higher the priority
    "model.mm_projector": "mm_projector",
    "model.": "language_model.",
}


@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class LLavaRForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(
        self,
        config: LlavaConfig,
        vision_language_config: VisionLanguageConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None
    ) -> None:
        super().__init__(config, vision_language_config, cache_config, quant_config)
        # llavar only use one linear layer as multimodal projector, so we first delete mm projector in Llava-v1.5,
        # then add an 1-layer linear layer as mm projector.
        delattr(self, "multi_modal_projector")
        self.mm_projector = nn.Linear(
            in_features=config.vision_config.hidden_size,
            out_features=config.text_config.hidden_size,
            bias=True,
        )
        # load vision tower from scratch, because llavar's checkpoints do not include weights of vision tower.
        self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.bfloat16)

    def _process_image_input(self,
                             image_input) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        return self.mm_projector(image_features)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # update_weights = torch.load("checkpoints/LLaVAR_delta/llavar-13b-pretrain.bin")
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # if name in update_weights:
            #     loaded_weight = update_weights[name]
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
