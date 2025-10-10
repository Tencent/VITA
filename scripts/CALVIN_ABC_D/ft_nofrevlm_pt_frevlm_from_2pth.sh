#!/bin/bash
### need to change to your path ### 


calvin_dataset_path="/run/ti/task_ABC_D"
 
save_checkpoint_path="/path/to/save/"
seer_path="/path/to/seer_large/Seer-Large-PT-ep5-FT-ep12.pth"
vit_checkpoint_path_seer="/path/to/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
vita_path='/path/to/llava_vita_1_5_modelscope'
finetune_from_pretrained_ckpt="/path/to/finetune_vita_Seer_ptbs512_24layers_16heads_hd1024-Large_calvin_abc_d/2.pth"


node=1
node_num=8


torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10215 train_ds.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 8 \
    --bf16_module "vision_encoder" \
    --vita_path ${vita_path} \
    --vit_checkpoint_path ${vit_checkpoint_path_seer} \
    --calvin_dataset ${calvin_dataset_path} \
    --projector_type mlp3x_gelu \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 2600 \
    --num_epochs 4 \
    --seed 42 \
    --batch_size 4 \
    --precision bf16 \
    --learning_rate 1e-4 \
    --warmup_epochs 1 \
    --finetune_type "calvin" \
    --wandb_project ft_pt_seer_0820_revise \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 16 \
    --run_name ft_pt_seer_0820_revise \
    --save_checkpoint_path ${save_checkpoint_path} \
    --seer_path ${seer_path} \
    --transformer_layers 24 \
    --action_decoder_dim 1024 \
    --transformer_heads 16 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --loss_action \
    --save_checkpoint \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --report_to_wandb \
    
    
