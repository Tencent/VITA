### NEED TO CHANGE ###
root_dir="path/to/libero"
libero_path="path/to/libero"
save_checkpoint_path="path/to/checkpoints/vita_libero_goal_pt_frevlm_ft_nofrevlm_717/"
seer_path="path/to/checkpoints/38.pth"
vit_checkpoint_path_seer="path/to/checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
vita_path='path/to/llava-s3-finetune_task_ovsi'
finetune_from_pretrained_ckpt="path/to/checkpoints/vita_libero10_pt_frevlm/13.pth"

node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train_ds.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --workers 8 \
    --lr_scheduler cosine \
    --projector_type mlp3x_gelu \
    --save_every_iter 10000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 4 \
    --precision bf16 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_goal \
    --vita_path ${vita_path} \
    --seer_path ${seer_path} \
    --vit_checkpoint_path ${vit_checkpoint_path_seer} \
    --root_dir ${root_dir} \
    --transformer_layers 24 \
    --action_decoder_dim 384 \
    --wandb_project vita_liberogoal_pt_frevlm_ft_nofrevlm_717 \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name vita_liberogoal_pt_frevlm_ft_nofrevlm_717\
    --save_checkpoint_path ${save_checkpoint_path} \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --obs_pred \
    --loss_action \
    --gripper_width \
    --warmup_epochs 2 \
    --libero_path ${libero_path} \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \