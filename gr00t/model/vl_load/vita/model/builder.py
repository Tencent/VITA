import os
import warnings

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging

from vl_load.vita.constants import GLOBAL_WEIGHTS_PATH
from vl_load.vita.model import *

# Suppress logging and warnings for cleaner output
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_type,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    output_hidden_states=False,
    load_act_tokenizer=False,
    **kwargs,
):
    """
    Load a pretrained VITA model with specified configuration.
    
    Args:
        model_path: Path to the pretrained model directory
        model_base: Base model path (used for certain configurations)
        model_name: Name of the model
        model_type: Type of the model architecture (currently supports "qwen2p5_instruct")
        load_8bit: Whether to load model in 8-bit precision
        load_4bit: Whether to load model in 4-bit precision
        device_map: Device mapping strategy for model placement
        device: Target device for model (default: "cuda")
        output_hidden_states: Whether to output hidden states
        load_act_tokenizer: Whether to load action tokenizer
        **kwargs: Additional keyword arguments
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    # Validate model type
    if model_type not in {"qwen2p5_instruct"}:
        raise ValueError(f"Unknown Model Type {model_type}")

    # Prepare keyword arguments for model loading
    kwargs = {"device_map": device_map, **kwargs}

    # Override device map for non-CUDA devices
    if device != "cuda":
        kwargs["device_map"] = {"": device}

    # Set data type to bfloat16 for efficient computation
    kwargs["torch_dtype"] = torch.bfloat16

    # Load VITA model based on specified type
    if model_type == "qwen2p5_instruct":
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VITAQwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, output_hidden_states=output_hidden_states, **kwargs
        )

    # Resize token embeddings to match tokenizer vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    # Get and load the vision tower component
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    # Calculate and print the number of parameters in vision encoder
    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))

    # Load vision tower weights if unfrozen
    if getattr(model.config, "unfreeze_vision_tower", False):
        if True:
            assert model_base is None
            from safetensors.torch import load_file

            # Collect vision tower weights from all safetensors files
            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith("safetensors"):
                    # Extract weights with "model.vision_tower." prefix and remove prefix
                    vision_weights.update(
                        {
                            k[19:]: v
                            for k, v in load_file(os.path.join(model_path, file_name)).items()
                            if k.startswith("model.vision_tower.")
                        }
                    )
            vision_tower.load_state_dict(vision_weights, strict=True)

    # Convert vision tower to bfloat16 precision
    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    # Determine context length from model configuration
    #import pdb; pdb.set_trace()
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # Set padding token if not defined
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor, context_len
