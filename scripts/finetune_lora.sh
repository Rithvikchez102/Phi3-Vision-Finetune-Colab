#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

deepspeed src/training/train.py \
    --lora_enable True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --use_dora False \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path brain-vqa-rad/train_vqa_rad_llava.json \
    --image_folder brain-vqa-rad/train_images \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/train_vqa_rad_llava_lora \
    --num_crops 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 3 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4