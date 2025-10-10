import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch 
import swanlab
import clip
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
) 
from models.vita_vla import VITAVLA
from utils.train_utils_ds import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, 
get_droid_dataset, get_libero_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  
from tqdm import tqdm

def load_ddp(model, ddp_ckpts):
    
    if os.path.exists(ddp_ckpts):
        ckpt = torch.load(ddp_ckpts, map_location="cpu")
        state_dict = ckpt['model_state_dict']
        
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]
            else:
                new_key = key
            
            
            new_state_dict[new_key] = value
        
        
        # import pdb; pdb.set_trace()
        model.load_state_dict(new_state_dict, strict=False)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        # if 'seer' not in name:
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(name)
    # import pdb;pdb.set_trace()
    return total_params, trainable_params


import torch.distributed as dist
from collections import Counter
from tqdm import tqdm
import pickle
import os
    
@record
def main(args):
    

    swanlab.login(api_key='') # add your api key here if you want to log in swanlab
    
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    device_id = args.rank % torch.cuda.device_count()
    
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("Seer", f"Seer_ptbs{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}")
    print("run_name:", args.run_name)
     
    model = VITAVLA(args, clip_device_id=device_id) 

    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, seer_image_processor=model.seer.image_processor, 
                                            seer_tokenizer=clip, vita_image_processor=model.model.image_processor, 
                                            vita_tokenizer=model.model.tokenizer, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "droid":
        calvin_dataset = get_droid_dataset(args, model.image_processor, clip, epoch=0)
    elif 'libero' in args.finetune_type:
        calvin_dataset = get_libero_dataset(args, seer_image_processor=model.seer.image_processor, 
                                                     seer_tokenizer=clip, vita_image_processor=model.model.image_processor, 
                                                     vita_tokenizer=model.model.tokenizer, epoch=0)
    # elif args.finetune_type == "real":
    #     calvin_dataset = get_real_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "oxe":
        calvin_dataset = get_oxe_dataset(args, model.image_processor, clip, epoch=0)
    random_seed(args.seed, args.rank)
    
     
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
        swanlab.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    
    if args.freeze_vlm:
        print("Freezing VLM model parameters")
        model.model.vlm.requires_grad_(False)
    else:
        model.model.vlm.model.requires_grad_(True)
        model.model.vlm.model.audio_encoder.requires_grad_(False) 
        model.model.vlm.model.mm_projector.requires_grad_(False)
        model.model.vlm.model.vision_tower.requires_grad_(False)
        model.model.vlm.lm_head.requires_grad_(False)
        
    
    model.seer.requires_grad_(False)
    model.model.action_mapper.requires_grad_(True)
    model.model.action_pred_token.requires_grad_(True)
    model.model.arm_state_encoder.requires_grad_(True)
    model.model.gripper_state_encoder.requires_grad_(True)
    model.model.state_projector.requires_grad_(True)
    
    if args.phase == 'finetune':
        model.seer.action_decoder.requires_grad_(True)
        model.seer.arm_action_decoder.requires_grad_(True)
        model.seer.gripper_action_decoder.requires_grad_(True) 
    
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "train_batch_size": args.batch_size * args.world_size * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer":{
            "type": "AdamW",
            "params":{
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupLR" if args.lr_scheduler == "constant" else
            "WarmupDecayLR" if args.lr_scheduler in ['linear', 'cosine'] else
            "WarmupCosineLR" if args.lr_scheduler == "cosine_restart" else "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": int(calvin_dataset.dataloader.num_batches * args.num_epochs * 0.1),
                "total_num_steps": calvin_dataset.dataloader.num_batches * args.num_epochs
            }
        },
        "fp16":{
            "enabled": args.precision == "fp16"
        },
        "bf16":{
            "enabled": args.precision in ["bf16", "amp_bfloat16", "amp_bf16"] 
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "allgather_partitions": True
        }
    }
    
    
    resume_from_epoch = 0
    if args.finetune_from_pretrained_ckpt is not None:
        # if args.rank == 0: 
        print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")  
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")
        if "module.model.vlm.model.audio_encoder.encoder.global_cmvn.mean" in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"]["module.model.vlm.model.audio_encoder.encoder.global_cmvn.mean"]
        if "module.model.vlm.model.audio_encoder.encoder.global_cmvn.istd" in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"]["module.model.vlm.model.audio_encoder.encoder.global_cmvn.istd"]
        load_ddp(model, args.finetune_from_pretrained_ckpt)
         
    
    model, _, _, _ = deepspeed.initialize(
        model = model,
        model_parameters = [p for p in model.parameters() if p.requires_grad],
        config = ds_config,
        dist_init_required=True
    )
    
    
    if hasattr(model, "module"):
        target = model.module
    else:
        target = model
        
    target.seer.clip_model = target.seer.clip_model.float()
    # import pdb; pdb.set_trace()
    
    
    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    total_params, trainable_params = count_parameters(model)
    
    print("total_params: {} M".format(total_params/1024/1024))
    print("trainable_params: {} M".format(trainable_params/1024/1024))
    
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader
        train_one_epoch_calvin(
            args=args,
            model=model,
            epoch=epoch,
            calvin_loader=calvin_loader,
            device_id=device_id,
            swanlab=swanlab,
        )
        if args.rank == 0 and args.save_checkpoint 
        and epoch % args.save_checkpoint_seq == 0 
        and epoch > args.start_save_checkpoint:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(model)
            }
            
            
            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
             
            torch.save(checkpoint_dict, ckpt_path) 

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    deepspeed.init_distributed()
    main(args)
    