#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

dataset_names=("Geometry3K" "GeoQA")
model_names=("llava-1.5-7b-hf" "llava-v1.6-mistral-7b-hf" "LLaVAR" "Qwen-VL-Chat" "cogvlm2-llama3-chat-19B")

for dataset_name in "${dataset_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        echo -e "\033[32mDataset: ${dataset_name}; Running model: ${model_name}.\033[0m"
        python3 vllm-inference.py \
            --dataset_name ${dataset_name} \
            --model_name ${model_name} \
            --tp_size 4
    done
done
