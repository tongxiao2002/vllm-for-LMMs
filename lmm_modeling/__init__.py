from .cogvlm2 import CogVLM2ForCausalLM
from .qwen_vl import QwenVLLMHeadModel
from .llavar import LLavaRForConditionalGeneration
from vllm import ModelRegistry

ModelRegistry.register_model("LlavaRForConditionalGeneration", LLavaRForConditionalGeneration)
ModelRegistry.register_model("QwenVLLMHeadModel", QwenVLLMHeadModel)
ModelRegistry.register_model("CogVLM2ForCausalLM", CogVLM2ForCausalLM)
