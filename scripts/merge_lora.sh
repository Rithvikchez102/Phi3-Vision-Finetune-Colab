#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

python3 src/merge_lora_weights.py \
    --model-path output/train_vqa_rad_llava_lora \
    --model-base $MODEL_NAME  \
    --save-model-path output/train_vqa_rad_llava_lora_merged \
    --safe-serialization