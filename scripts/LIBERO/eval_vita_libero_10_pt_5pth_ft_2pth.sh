#!/bin/bash
pthlist=("832" "1664" "2496" "3328" "4160" "4992" "5824" "6656" "7488" "8320" )
for ckpt_id in "${pthlist[@]}"; do
    export GIT_PYTHON_REFRESH=quiet
    vit_checkpoint_path="/path/to/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
    seer_path="/path/to/seer_checkpoint.pth"
    save_checkpoint_path="/path/to/save/checkpoints/"
    save_checkpoint_epoch_path="/path/to/save/epoch_checkpoints/"
    ### NEED TO CHANGE the checkpoint path ###
    resume_from_checkpoint="/path/to/experiment/checkpoints/${ckpt_id}.pth"

    IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
    # run_name="vita_libero_10_pt_frevlm_ft_nofrevlm_802"
    run_name="vita_libero_10_802_data_efficient"

    log_name="${path_parts[-1]}"
    log_folder="/path/to/logs/$run_name"
    mkdir -p "$log_folder"
    log_file="$log_folder/evaluate_$log_name.log"
    node=1
    node_num=8

    torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10219 eval_vita.py\
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
        --finetune_type "libero_10" \
        --action_pred_steps 3 \
        --sequence_length 7 \
        --future_steps 3 \
        --obs_pred \
        --seer_path ${seer_path} \
        --eval_libero_ensembling \
        --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file}
done