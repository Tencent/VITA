#!/bin/bash
pthlist=("0" "1") 
 

for ckpt_id in "${pthlist[@]}"; do
    export GIT_PYTHON_REFRESH=quiet
    calvin_dataset_path="/path/to/calvin/dataset/task_ABC_D"
    calvin_conf_path="/path/to/calvin/conf"
    vit_checkpoint_path="/path/to/checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
    seer_path="/path/to/checkpoints/seer_large/Seer-Large-PT-ep5-FT-ep12.pth"
    save_checkpoint_path="checkpoints/" 
    resume_from_checkpoint="/path/to/experiment/checkpoints/run_name_2/${ckpt_id}.pth"

    IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
    run_name="ft_nofrevlm_pt_frevlm_from_2pth_0822_revise"
    log_name="${path_parts[-1]}"
    log_folder="eval_logs/$run_name"
    mkdir -p "$log_folder"
    log_file="eval_logs/$run_name/evaluate_$log_name.log"
    node=1
    node_num=4
 
    torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10215 eval_vita.py\
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset ${calvin_dataset_path} \
        --calvin_conf_path ${calvin_conf_path} \
        --workers 8 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 1 \
        --seed 42 \
        --batch_size 1 \
        --precision bf16 \
        --weight_decay 1e-4 \
        --hidden_index -1 \
        --num_resampler_query 16 \
        --num_obs_token_per_image 16 \
        --run_name ${run_name} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --transformer_layers 24 \
        --hidden_dim 1024 \
        --transformer_heads 16 \
        --phase "evaluate" \
        --finetune_type "calvin" \
        --action_pred_steps 3 \
        --sequence_length 10 \
        --future_steps 3 \
        --window_size 13 \
        --obs_pred \
        --seer_path ${seer_path} \
        --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file} 
done