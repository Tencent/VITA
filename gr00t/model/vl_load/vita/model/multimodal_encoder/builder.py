import os

import yaml
import torch
from transformers.utils.hub import get_file_from_repo

from .internvit.internvit_encoder import InternViTVisionTower
from .whale.init_model import init_model


def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Build and initialize a vision tower based on the provided configuration.
    
    Args:
        vision_tower_cfg: Configuration object containing vision tower settings
        **kwargs: Additional keyword arguments for vision tower initialization
    
    Returns:
        VisionTower: Initialized vision tower instance (currently supports InternViT)
    """
    # Extract vision tower name from configuration (try mm_vision_tower first, then vision_tower)
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    use_s2 = getattr(vision_tower_cfg, "use_s2", False)

    # Build InternViT vision tower
    if "internvit" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for InternViT")
        else:
            print("Using InternViT")
            return InternViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}. This build is limited to InternViT per config.")


def build_audio_encoder(audio_encoder_config, **kwargs):
    """
    Build and initialize an audio encoder from the provided configuration.
    
    Args:
        audio_encoder_config: Configuration object containing audio encoder settings
        **kwargs: Additional keyword arguments for audio encoder initialization
    
    Returns:
        AudioEncoder: Initialized audio encoder model with loaded weights
    """
    # Load training configuration from the audio encoder model repository
    with open(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "train.yaml"), "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Set path to global CMVN (Cepstral Mean and Variance Normalization) file
    configs["cmvn_file"] = get_file_from_repo(audio_encoder_config.mm_audio_encoder, "global_cmvn")

    # Configure freezing settings for encoder and adapter
    configs["model_conf"]["freeze_encoder"] = getattr(
        audio_encoder_config, "freeze_audio_encoder", True
    )
    configs["model_conf"]["freeze_adpter"] = getattr(
        audio_encoder_config, "freeze_audio_encoder_adapter", True
    )
    # Configure audio prompt settings for fine-tuning
    configs["model_conf"]["audio_prompt_finetune"] = getattr(
        audio_encoder_config, "audio_prompt_finetune", False
    )
    configs["model_conf"]["audio_prompt_num"] = getattr(
        audio_encoder_config, "audio_prompt_num", 0
    )

    # Initialize the audio encoder model
    audio_encoder = init_model(configs)

    # Load pretrained checkpoint from the model repository
    checkpoint = torch.load(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "final.pt"), map_location="cpu")
    model_dict = audio_encoder.state_dict()
    
    # Iterate through model parameters and load matching weights from checkpoint
    for key in model_dict.keys():
        if key in checkpoint.keys():
            # Load weight if shapes match
            if model_dict[key].shape == checkpoint[key].shape:
                model_dict[key] = checkpoint[key]
            else:
                # Warn if shapes don't match
                print(
                    "Key {} has different shape, {} VS {}".format(
                        key, model_dict[key].shape, checkpoint[key].shape
                    )
                )
        else:
            # Warn if key is missing in checkpoint
            print("Key {} has not in resume model".format(key))
    
    # Load the updated state dict into the model
    audio_encoder.load_state_dict(model_dict)

    return audio_encoder
