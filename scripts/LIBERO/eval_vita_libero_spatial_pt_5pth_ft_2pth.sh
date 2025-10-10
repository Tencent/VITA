#!/bin/bash
pthlist=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28")
for ckpt_id in "${pthlist[@]}"; do
    export GIT_PYTHON_REFRESH=quiet
    vit_checkpoint_path="/path/to/checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
    seer_path="/path/to/checkpoints/seer.pth"
    save_checkpoint_path="checkpoints/"
    save_checkpoint_epoch_path="checkpoints/"     
    ### NEED TO CHANGE the checkpoint path ###
    resume_from_checkpoint="/path/to/experiment/checkpoints/run_name/${ckpt_id}.pth"

    IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
    run_name="${path_parts[-3]}"
    log_name="${path_parts[-1]}"
    log_folder="eval_logs/$run_name"
    mkdir -p "$log_folder"
    log_file="eval_logs/$run_name/evaluate_$log_name.log"
    node=1
    node_num=8

    torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10222 eval_vita.py\
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset "/path/to/calvin/dataset" \
        --workers 8 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 1 \
        --seed 42 \
        --batch_size 1 \
        --precision bf16 \
        --weight_decay 1e-4 \
        --hidden_index -1 \
        --num_resampler_query 6 \
        --run_name ${run_name} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --save_checkpoint_epoch_path ${save_checkpoint_epoch_path} \
        --transformer_layers 24 \
        --hidden_dim 384 \
        --action_decoder_dim 384 \
        --transformer_heads 12 \
        --phase "evaluate" \
        --finetune_type "libero_spatial" \
        --action_pred_steps 3 \
        --sequence_length 7 \
        --future_steps 3 \
        --obs_pred \
        --seer_path ${seer_path} \
        --eval_libero_ensembling \
        --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
done