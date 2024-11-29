import os
import json
import argparse
from tret import TretArguments, TretWorkspace
from vllm import LLM, SamplingParams
from typing import Dict
from lmm_datasets.prompts import *
from lmm_datasets.dataset_args import DatasetArgs
from lmm_datasets import (
    MultimodalDataset,
    MultimodalDatasetForLlava,
    MultimodalDatasetForLlavaNext,
    MultimodalDatasetForInterVL2,
    MultimodalDatasetForQwen2VL
)

dataset_cls_map: Dict[str, MultimodalDataset] = {
    "r-cot": MultimodalDatasetForInterVL2,
    "internvl2": MultimodalDatasetForInterVL2,
    "qwen2-vl": MultimodalDatasetForQwen2VL,
    "llava-1.5-7b-hf": MultimodalDatasetForLlava,
    "llava-v1.6-mistral-7b-hf": MultimodalDatasetForLlavaNext,
}


def parse_args():
    parser = argparse.ArgumentParser(description="LMM Inference")
    # tret workspace
    parser.add_argument("--workspace_name", type=str, default="test")

    # dataset args
    parser.add_argument("--dataset_name_or_path", type=str, default="data/MathVerse")
    parser.add_argument("--prompt_name", type=str, default="")

    # model args
    parser.add_argument("--model_name", type=str, default="R-CoT-8B")

    # running args
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--tp_size", type=int, default=4)
    return parser.parse_args()


def run_inference():
    args = parse_args()
    model_name = args.model_name
    tp_size = args.tp_size

    dataset_args = DatasetArgs(
        dataset_name_or_path=args.dataset_name_or_path,
        # prompt=eval(args.prompt_name),
        prompt="{query_cot}",
    )

    tret_args = TretArguments(
        workspace_name=args.workspace_name,
        create_directory=True,
    )
    workspace = TretWorkspace(arguments=tret_args)
    output_dir = os.path.join(workspace.workspace_dir, "runs", dataset_args.dataset_name_or_path.replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join("checkpoints", model_name)

    prefix = f"{model_name}-{args.prompt_name}"
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
        gpu_memory_utilization=0.95,
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

    if "prompts" in vllm_data:
        completions = llm.generate(
            sampling_params=sampling_params,
            **vllm_data,
        )
    elif "messages" in vllm_data:
        completions = llm.chat(
            sampling_params=sampling_params,
            **vllm_data,
        )
    else:
        raise ValueError("Unknown data format.")

    results = []
    for dataitem, completion in zip(dataset.instruction_data, completions):
        generated_text = completion.outputs[0].text
        if "image" in dataitem:
            _ = dataitem.pop("image")

        results.append({
            **dataitem,
            "response": generated_text,
        })
    json.dump(
        results,
        open(output_filepath, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    workspace.backup()


if __name__ == "__main__":
    run_inference()
