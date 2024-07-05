import os
import json
import argparse
from vllm import LLM, SamplingParams
from lmm_datasets import (
    MultimodalDatasetForLlava,
    MultimodalDatasetForLlavaR,
    MultimodalDatasetForLlavaNext,
    MultimodalDatasetForQwenVL,
    MultimodalDatasetForCogVLM2,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LMM Inference")
    parser.add_argument("--dataset_name", type=str, default="Geometry3K")
    parser.add_argument("--model_name", type=str, default="cogvlm2-llama3-chat-19B")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    return parser.parse_args()


def run_inference():
    args = parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    tp_size = args.tp_size

    model_path = os.path.join("checkpoints", model_name)
    output_dir = os.path.join("runs", dataset_name, model_name)
    output_filepath = os.path.join(output_dir, "results.json")

    if os.path.isfile(output_filepath) and not args.regenerate:
        return

    os.makedirs(output_dir, exist_ok=True)
    if "llavar" in model_name.lower():
        dataset_cls = MultimodalDatasetForLlavaR
    elif "llava-v1.6" in model_name.lower():
        dataset_cls = MultimodalDatasetForLlavaNext
    elif "llava" in model_name.lower():
        dataset_cls = MultimodalDatasetForLlava
    elif "qwen" in model_name.lower():
        dataset_cls = MultimodalDatasetForQwenVL
    elif "cogvlm2" in model_name.lower():
        dataset_cls = MultimodalDatasetForCogVLM2
    else:
        raise ValueError("LLM is not supported.")

    dataset = dataset_cls(dataset_name=dataset_name, model_path=model_path)
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


if __name__ == "__main__":
    run_inference()
