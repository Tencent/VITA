import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist
from .data_utils import prepare_data_vita
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import swanlab


def l1_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, D) tensor.
    v: (N, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return torch.abs(u - v).sum(dim=-1)

def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T, D) tensor.
    v: (N, T, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    device_id,
    swanlab,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # 全部是torch类型，可以直接使用encode去编码，无需动态分辨率或者什么，图像固定是3*200*200
        vita_image_wrist = batch_calvin[1].to(
            device_id, dtype=cast_dtype, non_blocking=True)[:, :args.sequence_length, :]
        vita_image_gripper = batch_calvin[6].to(
            device_id, dtype=cast_dtype, non_blocking=True)[:, :args.sequence_length, :]
        language = batch_calvin[-1]
        states = batch_calvin[7].to(
            device_id, dtype=cast_dtype, non_blocking=True)[:, :args.sequence_length, :]

        if args.gripper_width:
            states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            states[..., 6:] = (states[..., 6:] + 1) // 2


        # actions, 4是对应的动作
        actions = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        
        batch_vita = prepare_data_vita(vita_image_wrist, vita_image_gripper, 
                                       language, states, model.module.model.tokenizer)
        
        
        seer_images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        
        seer_images_wrist = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
         
        text_tokens = batch_calvin[2].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, 13, 1)
         
         
        states = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True)
            
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2
        
        # self_key_point
        self_keypoints = None
        
        # print("action.shape", actions.shape)
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        input_image_primary = seer_images_primary[:, :args.sequence_length, :]
        input_image_wrist = seer_images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]

        label_actions = torch.cat(
            [actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) 
             for j in range(args.action_pred_steps)], dim=-2) 

        if args.phase == 'pretrain':
            action_features_seer = model.module.seer(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
            )[:, :args.sequence_length, :, :]
        

        action_features_vita = model(
            input_ids = batch_vita['input_ids'],
            images = batch_vita['images'],
            states = batch_vita['states'],
            attention_mask = batch_vita["attention_mask"],
            labels = batch_vita["labels"]
        ).view(label_actions.shape[0], args.sequence_length, args.action_pred_steps, -1)
        # import pdb;pdb.set_trace()
        
        # finetune
        if args.phase == 'finetune':
            action_pred_feature = model.module.seer.action_decoder(action_features_vita) 
            arm_pred_action = model.module.seer.arm_action_decoder(action_pred_feature)
            # print(arm_pred_action.shape, "arm_pred_action.shape")
            gripper_pred_action = model.module.seer.gripper_action_decoder(action_pred_feature)
            # loss_action
            if args.loss_action and args.action_pred_steps:
                loss_arm_action = torch.nn.functional.smooth_l1_loss(
                                arm_pred_action[:, :args.sequence_length-args.atten_goal], 
                                label_actions[:, :args.sequence_length-args.atten_goal, :, :6].detach())
                loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                                gripper_pred_action[:, :args.sequence_length-args.atten_goal], 
                                label_actions[:, :args.sequence_length-args.atten_goal, :, 6:].detach())
            else:
                loss_arm_action = torch.tensor([0.0]).to(device_id)
                loss_gripper_action = torch.tensor([0.0]).to(device_id)
            
            loss_calvin = args.loss_arm_action_ratio * loss_arm_action 
            + args.loss_gripper_action_ratio * loss_gripper_action

            # gradient_accumulation_steps        
            loss = loss_calvin / args.gradient_accumulation_steps 
            
            mv_avg_loss.append(loss.item())

        else: 
            loss_calvin = l2_loss(action_features_seer, action_features_vita).mean()

            # gradient_accumulation_steps        
            loss = loss_calvin / args.gradient_accumulation_steps 

            mv_avg_loss.append(loss.item())
            
            
        model.backward(loss)
        model.step()
         

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )

                swanlab.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        # "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                if args.phase=='finetune':
                    swanlab.log(
                        {
                            "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                            "loss_arm": loss_arm_action.item(),
                            "loss_gripper": loss_gripper_action.item(),
                            "global_step": global_step,
                        },
                    )
                else:
                    swanlab.log(
                        {
                            "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                            "global_step": global_step,
                        },
                    )
                
                
                # 重置计时器
                step_time_m.reset()
                data_time_m.reset()

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item()})
        

        if args.save_every_iter != -1 and args.save_checkpoint 
        and global_step % args.save_every_iter == 0 and global_step > 0:
                
            if args.rank == 0: 
                if not os.path.exists(f"{args.save_checkpoint_path}/exp/{args.run_name}"):
                    os.makedirs(f"{args.save_checkpoint_path}/exp/{args.run_name}")

                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    # "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }

                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(f"{args.save_checkpoint_path}/exp", args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                 
                torch.save(checkpoint_dict, ckpt_path) 

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]
    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        