import os
import json
import argparse
from tret import TretArguments, TretWorkspace
from vllm import LLM, SamplingParams, ModelRegistry
from typing import Dict
from lmm_datasets.dataset_args import DatasetArgs
from lmm_datasets import (
    MultimodalDataset,
    MultimodalDatasetForLlava,
    MultimodalDatasetForLlavaR,
    MultimodalDatasetForLlavaNext,
    MultimodalDatasetForQwenVL,
    MultimodalDatasetForCogVLM2,
    MultimodalDatasetForInterVL2,
)
from lmm_modeling import (
    LLavaRForConditionalGeneration,
    QwenVLLMHeadModel,
    CogVLM2ForCausalLM,
    InternVLForCausalLM,
)

ModelRegistry.register_model("LlavaRForConditionalGeneration", LLavaRForConditionalGeneration)
ModelRegistry.register_model("QwenVLLMHeadModel", QwenVLLMHeadModel)
ModelRegistry.register_model("CogVLM2ForCausalLM", CogVLM2ForCausalLM)
ModelRegistry.register_model("InternVLForCausalLM", InternVLForCausalLM)

dataset_cls_map: Dict[str, MultimodalDataset] = {
    "cogvlm2-llama3-chat-19b": MultimodalDatasetForCogVLM2,
    "internvl2-40b": MultimodalDatasetForInterVL2,
    "qwen-vl-chat": MultimodalDatasetForQwenVL,

    "llava-1.5-7b-hf": MultimodalDatasetForLlava,
    "llavar": MultimodalDatasetForLlavaR,
    "llava-v1.6-mistral-7b-hf": MultimodalDatasetForLlavaNext,
}


def parse_args():
    parser = argparse.ArgumentParser(description="LMM Inference")
    # tret workspace
    parser.add_argument("--workspace_name", type=str, default="test")

    # dataset args
    parser.add_argument("--dataset_name", type=str, default="Geometry3K")
    parser.add_argument("--generate_caption", action="store_true")
    parser.add_argument("--generate_solution", action="store_true")

    # model args
    parser.add_argument("--model_name", type=str, default="InternVL2-40B")

    # running args
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--tp_size", type=int, default=4)
    return parser.parse_args()


def get_output_prefix(dataset_args: DatasetArgs):
    if dataset_args.generate_caption:
        return "caption"
    else:
        return "solution"


def run_inference():
    args = parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    tp_size = args.tp_size

    dataset_args = DatasetArgs(
        dataset_name=args.dataset_name,
        generate_caption=args.generate_caption,
        generate_solution=args.generate_solution,
        prompt_provide_choices=args.prompt_provide_choices,
        perturb_blank_pixels=args.perturb_blank_pixels,
    )

    tret_args = TretArguments(
        workspace_name=args.workspace_name,
        create_directory=True,
    )
    workspace = TretWorkspace(arguments=tret_args)
    output_dir = os.path.join(workspace.workspace_dir, "runs", dataset_args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join("checkpoints", model_name)

    prefix = get_output_prefix(dataset_args)
    output_filepath = os.path.join(output_dir, f"{prefix}-results.json")

    dataset_cls = None
    try:
        for key, cls in dataset_cls_map.items():
            if model_name.lower().startswith(key):
                dataset_cls = cls
                break
    except KeyError:
        raise ValueError("LLM is not supported.")

    dataset = dataset_cls(dataset_args, model_path)
    vllm_args, vllm_data = dataset._prepare_vllm_data_and_args()

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="slow",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        **vllm_args['llm_args'],
    )

    sampling_params = SamplingParams(
        n=1,
        top_p=1,
        temperature=0.0,
        max_tokens=2048,
        **vllm_args['sampling_args'],
    )

    completions = llm.generate(
        sampling_params=sampling_params,
        **vllm_data,
    )

    results = {}
    for dataitem, completion in zip(dataset.data, completions):
        prompt = completion.prompt
        generated_text = completion.outputs[0].text
        results[dataitem["id"]] = {
            "ground-truth": dataitem["ground-truth"],
            "response": generated_text,
            "prompt": prompt,
        }
    json.dump(
        results,
        open(output_filepath, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    workspace.backup()


if __name__ == "__main__":
    run_inference()
