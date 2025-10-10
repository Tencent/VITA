import os
from .seer_model import SeerAgent  # 从当前目录下的 seer_model.py 文件中导入 SeerAgent 类
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch
import wandb   
import random
import numpy as np


# 直接设置随机种子以确保实验的可复现性


def random_seed(seed=42, rank=0):
    """
    设置随机种子以保证代码的可复现性。
    在分布式训练中，为每个进程设置不同的种子（基于 rank）是很重要的，
    这样可以确保例如数据加载时的随机打乱等操作在每个进程上是不同的，但总体上是确定的。

    Args:
        seed (int): 基础随机种子。
        rank (int): 当前进程在分布式训练中的排名（rank）。默认为0，适用于单进程场景。
    """
    # 为 PyTorch 在所有设备（CPU 和 CUDA）上设置随机种子
    torch.manual_seed(seed + rank)
    # 为 NumPy 设置随机种子
    np.random.seed(seed + rank)
    # 为 Python 内置的 random 模块设置随机种子
    random.seed(seed + rank)


# 根据传入的参数构建 seer 模型，这与 seer 原始仓库的实现保持一致


def build_seer(args, clip_device_id):
     
    # 使用 args 中定义的各种超参数来实例化 SeerAgent 模型。
    # 这种方式使得模型的结构和行为可以通过外部配置灵活地改变。
    model = SeerAgent(
        finetune_type=args.finetune_type,
        clip_device=clip_device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
    )
    
    # 设置随机种子，以确保模型初始化（如果有随机部分）和后续操作的可复现性。
    random_seed(args.seed, args.rank)  
    # 从指定的路径加载预训练模型的 checkpoint 文件。
    # map_location="cpu" 是一个很好的实践，它将模型权重首先加载到 CPU 内存中，
    # 避免了因 GPU 设备不匹配或显存不足导致的问题。
    checkpoint = torch.load(args.seer_path, map_location="cpu")
    
    # 获取 checkpoint 中 state_dict 的所有参数名称，并存入一个列表。
    # 这可以用于后续快速检查某个参数是否存在于 checkpoint 中。
    name_list = list(checkpoint['model_state_dict'].keys())
    
    # --- 加载预训练权重 ---
    # 遍历刚刚创建的新模型的所有命名参数（包括权重和偏置）。
    for name, param in model.named_parameters():
        # 检查当前参数的名称加上 'module.' 前缀后，是否存在于 checkpoint 的参数名列表中。
        # 'module.' 前缀是 PyTorch 的 DistributedDataParallel (DDP) 在保存模型时自动添加的。
        # 这种检查方式使得代码能够兼容从 DDP 和非 DDP 模式下保存的 checkpoint。
        if 'module.' + name in name_list:
            # 如果找到了匹配的参数，就使用 .copy_() 方法将 checkpoint 中的权重值复制到新模型的对应参数中。
            # .copy_() 是一个原地操作，效率很高。
            param.data.copy_(checkpoint['model_state_dict']['module.' + name])  
            
    # 返回已经构建好并加载了权重的模型。
    return model
