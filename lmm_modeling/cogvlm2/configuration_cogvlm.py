from typing import Literal
from transformers import PretrainedConfig


class CogVLMConfig(PretrainedConfig):
    _auto_class = "AutoConfig"

    def __init__(
            self,
            vocab_size=128256,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_multi_query_heads=8,
            hidden_act='silu',
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            template_version: Literal["base", "chat"] = "chat",
            bos_token_id=128000,
            eos_token_id=128001,
            tie_word_embeddings=False,
            use_cache=True,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_multi_query_heads = num_multi_query_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.template_version = template_version
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
