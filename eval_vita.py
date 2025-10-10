import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch 
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
) 
# Import custom modules for the model and utilities
from models.vita_vla import VITAVLA
from utils.arguments_utils import get_parser 
from utils.eval_utils_calvin_vita import eval_one_epoch_calvin_ddp
from utils.eval_utils_libero_vita import eval_one_epoch_libero_ddp
from utils.data_utils import get_calvin_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  
from datetime import datetime


def random_seed(seed=42, rank=0):
    """
    Sets the random seed for reproducibility across different libraries.
    The rank is added to the seed to ensure different processes in distributed training
    have different random states, which can be important for data shuffling.
    """
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def count_parameters(model):
    """
    Calculates the total and trainable parameters of a PyTorch model.
    """
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params


@record # Decorator for recording errors in torch.distributed.elastic
def main(args):
    
    # --- Distributed Setup ---
    # Initialize distributed environment to get local rank, global rank, and world size
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    # Set up the device for the current process
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    
    # Set the random seed for the main process before model initialization
    random_seed(args.seed)  
     
    # --- Model Initialization ---
    # Instantiate the VITA-VLA model with provided arguments
    model = VITAVLA(args, clip_device_id=device_id)
  
    # Re-seed with rank to ensure each DDP process has a different random state
    random_seed(args.seed, args.rank)
    print(f"Start running evaluation on rank {args.rank}.") 
    
    # Determine the specific GPU device ID for this process
    device_id = args.rank % torch.cuda.device_count()
    
    # --- Precision and Model Preparation ---
    # Set the model's precision based on the command-line argument
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16() 
    elif args.precision == "fp16":
        model = model.half() 
    elif args.precision == "fp32":
        model = model.float() 

    # Freeze parts of the model that are not being evaluated or trained
    # This saves memory and computation by preventing gradient calculation
    model.model.vlm.requires_grad_(False)
    model.seer.requires_grad_(False) 
 
    # Move the model to the assigned GPU device
    model = model.to(device_id) 
    model.seer._init_model_type()
    
    # Wrap the model with DistributedDataParallel (DDP) for multi-GPU evaluation
    # `find_unused_parameters=True` is often necessary for complex models where not all
    # parameters are used in the forward pass on every iteration.
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    
    # --- Checkpoint Loading ---
    if args.resume_from_checkpoint is not None:
        # Only print the loading message from the master process (rank 0)
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        # Load the checkpoint onto the CPU first to avoid GPU memory spikes
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        # Load the state dict into the DDP-wrapped model. `strict=False` is used here.
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
    
    # --- Evaluation ---
    # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    ddp_model.eval()
    
    # Define the directory for saving evaluation logs
    eval_log_dir = 'evaluate'
    
    # Get the current time to create a unique timestamped directory for this evaluation run
    current_time = datetime.now()
    cur_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
     
    # Check the evaluation type and call the appropriate evaluation function
    if args.finetune_type == "calvin":
        # Start the evaluation loop for one epoch on the CALVIN dataset
        eval_one_epoch_calvin_ddp(
            args=args,
            model=ddp_model,
            # Pass necessary components from the unwrapped model
            image_processor=ddp_model.module.model.image_processor,
            tokenizer=ddp_model.module.model.tokenizer,
            dataset_path=args.calvin_dataset,
            future_act_len=args.future_act_len,
            eval_log_dir=eval_log_dir,
            debug=args.visualize, # Flag for enabling visualization/debugging
            reset=args.reset,
            diverse_inst=args.diverse_inst,
            cur_time=cur_time # Pass the timestamp for logging
        )
    elif 'libero' in args.finetune_type:
        eval_one_epoch_libero_ddp(
            args=args,
            model=ddp_model,
            image_processor=ddp_model.module.model.image_processor,
            tokenizer=ddp_model.module.model.tokenizer,
            cur_time=cur_time
        )
    else:
        # Raise an error if the evaluation type is not supported
        raise NotImplementedError


if __name__ == "__main__":
    # Set an environment variable for NCCL to prevent blocking waits, which can help avoid timeouts
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    
    # Initialize the argument parser, configured for evaluation
    parser = get_parser(is_eval=True)
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the main function to start the evaluation process
    main(args)