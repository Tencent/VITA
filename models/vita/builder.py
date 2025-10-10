import argparse
from .model.builder import load_pretrained_model 
import os

from .util.utils import disable_torch_init

# build vita model from args, different model types are supported, present supported models are:
# qwen2p5_instruct: Qwen2.5-7B-Instruct model, see https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

def build_vita(args):
    
    model_path = args.vita_path
    model_base = args.vita_base
    
    model_path = os.path.expanduser(model_path) 
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_type=args.vita_type
    )

    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model, image_processor
