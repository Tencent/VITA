import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from vl_load.vita.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vl_load.vita.conversation import SeparatorStyle, conv_templates
from vl_load.vita.model.builder import load_pretrained_model
from vl_load.vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from vl_load.vita.util.utils import disable_torch_init

from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoConfig
from vl_load.vita.model.language_model.vita_qwen2 import VITAQwen2ForCausalLM

class VITAModel(nn.Module):
    def __init__(
        self, 
        model_path=None, 
        model_base=None, 
        model_type="qwen2p5_instruct", 
        conv_mode="qwen2p5_instruct",
        frameCat=False,
        p_num=[10],
        ):
        super().__init__()
        self.model_path = model_path
        self.model_base = model_base
        self.model_type = model_type
        self.conv_mode = conv_mode
        self.frameCat = frameCat
        self.p_num = p_num
        
        config = AutoConfig.from_pretrained(self.model_path)
        config.vocab_size=151665
        config.output_hidden_states = True
        self.model = VITAQwen2ForCausalLM(config=config)
        self.linear = torch.nn.Linear(3584, 1536)

    def init_model(self,
                   device_id=None,
                   tune_visual=False, 
                   tune_llm=False,
                   load_separately=False,
                   ):
        self.tune_visual = tune_visual
        self.tune_llm = tune_llm
        self.device_id = device_id

        if load_separately:
            model_name = get_model_name_from_path(self.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                self.model_path, self.model_base, model_name, self.model_type, output_hidden_states=True, device_map=self.device_id,
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = self.model.to(self.device_id).to(torch.bfloat16)
        
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower = vision_tower.to(torch.bfloat16)
        self.image_processor = vision_tower.image_processor

        audio_encoder = self.model.get_audio_encoder()
        audio_encoder.to(dtype=torch.bfloat16, device=self.device_id)  # unable to delete. RuntimeError: Input type (float) and bias type (c10::Half) should be the same
        audio_processor = audio_encoder.audio_processor

        self.set_trainable_parameters()

        audio = torch.zeros(400, 80)
        audio_length = audio.shape[0]
        audio_for_llm_lens = 60
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        self.audios = dict()
        self.audios["audios"] = audio.to(torch.bfloat16).to(self.device_id)
        self.audios["lengths"] = audio_length.to(torch.bfloat16).to(self.device_id)
        self.audios["lengths_for_llm"] = audio_for_llm_lens.to(self.device_id)

    def set_trainable_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = True
        
        if not self.tune_llm:
            self.model.get_audio_encoder().requires_grad_(False)
            # self.model.get_vision_tower().requires_grad_(False)
            self.model.lm_head.requires_grad_(False)
            self.model.model.mm_projector.requires_grad_(False)
            self.model.model.norm.requires_grad_(False)
            self.model.model.embed_tokens.requires_grad_(False)
            for layer in self.model.model.layers:
                layer.requires_grad_(False)
            if self.tune_visual:
                print("training vision_tower!")
        if not self.tune_visual:
            self.model.get_audio_encoder().requires_grad_(False)
            self.model.get_vision_tower().requires_grad_(False)
            self.model.lm_head.requires_grad_(False)
            self.model.model.mm_projector.requires_grad_(False)
            self.model.model.norm.requires_grad_(False)
            self.model.model.embed_tokens.requires_grad_(False)
            for layer in self.model.model.layers[:-3]:
                layer.requires_grad_(False)
            if self.tune_llm:
                print("training language_model!")
        if self.tune_llm and self.tune_visual:
                self.model.get_audio_encoder().requires_grad_(False)
                # self.model.get_vision_tower().requires_grad_(False)
                self.model.lm_head.requires_grad_(False)
                self.model.model.mm_projector.requires_grad_(False)
                self.model.model.norm.requires_grad_(False)
                self.model.model.embed_tokens.requires_grad_(False)
                for layer in self.model.model.layers[:-3]:
                    layer.requires_grad_(False)
                print("training vision_tower and language_model!")
        if not self.tune_llm and not self.tune_visual: # could be deleted
            for param in self.model.parameters():
                param.requires_grad = False

        if False:
            log_file="trainable_params.log"
            with open(log_file, "w") as f:
                f.write("==== Trainable parameters ====\n")
                for name, param in self.model.named_parameters():
                    f.write(f"{name}: requires_grad={param.requires_grad}\n")
                f.write("==== End ====\n")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_llm:
                self.model.get_audio_encoder().eval()
                # self.model.get_vision_tower().eval()
                self.model.lm_head.eval()
                self.model.model.mm_projector.eval()
                self.model.model.norm.eval()
                self.model.model.embed_tokens.eval()
                for layer in self.model.model.layers:
                    layer.eval()
                # print("training vision_tower")
            if not self.tune_visual:            
                self.model.get_audio_encoder().eval()
                self.model.get_vision_tower().eval()
                self.model.lm_head.eval()
                self.model.model.mm_projector.eval()
                self.model.model.norm.eval()
                self.model.model.embed_tokens.eval()
                for layer in self.model.model.layers[:-3]:
                    layer.eval()
                # print("training language_model")
            
            if self.tune_llm and self.tune_visual:
                self.model.get_audio_encoder().eval()
                # self.model.get_vision_tower().eval()
                self.model.lm_head.eval()
                self.model.model.mm_projector.eval()
                self.model.model.norm.eval()
                self.model.model.embed_tokens.eval()
                for layer in self.model.model.layers[:-3]:
                    layer.eval()
                # print("training vision_tower and language_model")
            if not self.tune_llm and not self.tune_visual: # could be deleted
                self.model.eval()
                    
        if False:
            log_file="trainable_params2.log"
            with open(log_file, "w") as f:
                f.write("==== Trainable parameters ====\n")
                for name, module in self.model.named_modules():
                    f.write(f"{name}: {'train' if module.training else 'eval'}\n")
                f.write("==== End ====\n")

    def get_latent(self,
                   image_tensor=None,
                   input_ids=None, 
                   attention_mask=None,
                   ):
        """
        input: torch.Size([16, 3, 448, 448]) torch.Size([8, 226]) torch.Size([8, 226])
        image: tensor, [batch_size, sequence_length, 3, 224, 224]
        """
        self.set_frozen_modules_to_eval_mode()
        
        B_S, C, H, W = image_tensor.shape
        B = input_ids.shape[0]
        images_per_sample = max(1, B_S // B)
        # print("image: ", H, " ", W, flush=True)

        # audio
        required_audio_instances = B * self.p_num[0]
        if self.audios["audios"].shape[0] != required_audio_instances:
            self.audios["audios"] = self.audios["audios"].repeat(required_audio_instances, 1, 1)
            self.audios["lengths"] = self.audios["lengths"].repeat(required_audio_instances)
            self.audios["lengths_for_llm"] = self.audios["lengths_for_llm"].repeat(required_audio_instances)

        # # language
        input_ids=input_ids.repeat(self.p_num[0], 1)
        attention_mask=attention_mask.repeat(self.p_num[0], 1)

        outputs = self.model(
            input_ids, # [8*10, 6200]
            attention_mask=attention_mask, # [8*10, 6200]
            images=image_tensor,  # [8*10, 3, 448, 448]
            audios=self.audios, # [8*10, 400, 80]
            use_cache=True,
        )

        hidden_states = outputs.hidden_states  # tuple
        return hidden_states[-1] # torch.Size([8, 737, 3584])

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        embeddings = self.get_latent(
            image_tensor=vl_input["pixel_values_vita"],
            input_ids=vl_input["input_ids_vita"],
            attention_mask=vl_input["attention_mask_vita"],
        )
        # print("embeddings vlm:", embeddings[0], embeddings.dtype)
        embeddings = self.linear(embeddings)
        # print("embeddings linear:", embeddings[0], embeddings.dtype)
        return BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": None,
            }
        )  # [B, T2, hidden_size]
    
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
# if __name__ == "__main__":
#     # from utils.distributed_utils import init_distributed_device, world_info_from_env  
#     # local_rank, rank, world_size = world_info_from_env()
#     # device_id = rank % torch.cuda.device_count()
#     device_id=0
#     model = VITAModel()
#     model.init_model(device_id=device_id, freeze=True)
#     while(1):
#         model.get_latent()

#     # torchrun --nnodes=1 --nproc_per_node=8 --master_port=10211 models/vita_model.py
#     # CUDA_VISIBLE_DEVICES=0 python models/vita_model.py 