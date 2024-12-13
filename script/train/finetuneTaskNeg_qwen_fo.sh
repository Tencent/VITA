#!/bin/bash

MODEL_TYPE=qwen2p5_fo_instruct
OUTPUT_DIR=$1
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s3-finetune_task_neg
mkdir -p ${OUTPUT_DIR_FT}

deepspeed --include localhost:1  --master_port 25903 vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/temp2_llava-s3-finetune_task_neg \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use Pretrain_video \
    --vision_tower /mnt/cfs/lhj/model_weights/InternViT-300M-448px \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder /mnt/cfs2/lhj/model_weights/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --audio_prompt_finetune True \
    --audio_prompt_num 25 \
    --audio_state_predictor_tuning False \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt && echo "Done."







