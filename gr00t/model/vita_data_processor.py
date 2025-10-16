import sys
import os
import base64
from io import BytesIO
from typing import List, Union
import yaml
import json
import warnings

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers import logging
from transformers.utils.hub import get_file_from_repo

import gr00t
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from vl_load.vita.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from vl_load.vita.util.mm_utils import tokenizer_image_token
from vl_load.vita.conversation import conv_templates
from vl_load.vita.model import *
from vl_load.vita.model.multimodal_encoder.whale.init_model import init_model


logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 定义VITA模型的默认路径（使用 HuggingFace 模型 ID）
DEFAULT_VITA_MODEL_NAME = "VITA-MLLM/VITA-1.5"

# 定义VIT（Vision Transformer）模型的默认路径（使用 HuggingFace 模型 ID）
DEFAULT_VIT_MODEL_NAME = "OpenGVLab/InternViT-300M-448px"

# 定义音频编码器的默认路径（使用 HuggingFace 模型 ID）
DEFAULT_AUDIO_ENCODER = "VITA-MLLM/VITA-1.5_AudioEnc"

def load_image(image):
    """加载图像，支持多种格式输入"""
    # 如果是文件路径，直接打开
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    # 如果是字典，根据key判断来源
    elif isinstance(image, dict):
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        elif "base64" in image:
            # 解码base64字符串
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        elif "url" in image:
            # 从URL下载
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        elif "bytes" in image:
            # 从字节流加载
            return Image.open(BytesIO(image["bytes"]))
        elif "np_array" in image:
            # 从numpy数组加载
            return Image.fromarray(image["np_array"])
        else:
            raise ValueError(f"Invalid image: {image}")
    else:
        raise ValueError(f"Invalid image: {image}")

def load_pretrained_model():
    """加载预训练的VITA模型所需的预处理工具"""
    # 从预训练模型路径加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_VITA_MODEL_NAME, use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL_NAME, use_fast=True)
    
    # 从预训练模型路径加载图像处理器
    image_processor = CLIPImageProcessor.from_pretrained(DEFAULT_VIT_MODEL_NAME)
    
    # 加载音频处理器
    audio_processor = load_audio_processor()
    
    return tokenizer, image_processor, audio_processor

def load_audio_processor():
    """加载音频处理器"""
    # 读取音频编码器的配置文件（从 HuggingFace Hub 自动下载）
    with open(get_file_from_repo(DEFAULT_AUDIO_ENCODER, "train.yaml"), "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # 指定全局倒谱均值和方差归一化（CMVN）文件路径（从 HuggingFace Hub 自动下载）
    configs["cmvn_file"] = get_file_from_repo(DEFAULT_AUDIO_ENCODER, "global_cmvn")

    # 初始化音频编码器模型
    audio_encoder = init_model(configs)
    # 获取音频处理器
    audio_processor = audio_encoder.audio_processor
    return audio_processor


class VitaProcessor:
    """VITA模型的数据处理器"""
    def __init__(
        self,
    ):
        """初始化函数，加载所有必要的处理器"""
        self.tokenizer, self.image_processor, self.audio_processor = load_pretrained_model()
        self.eagle_processor = EagleProcessor()

    def prepare_input(self, message):
        """准备单个样本的输入数据"""
        ## 处理单个数据
        # 提取prompt，目前只使用单帧
        prompt=message["prompt"][1:][0] 
        language = prompt["content"]
        images = prompt["image"]
        # 记录当前样本的图像数量，供 text_vita_fn 使用
        self._cur_image_cnt = len(images)
        
        # 处理图像数据
        pixel_values = self.image_vita_fn(images)
        # 处理文本数据
        input_ids = self.text_vita_fn(language)

        # 处理eagle模型的数据
        data_eagle = self.eagle_processor.prepare_input(message)

        # 组合成最终的输入数据字典
        data = {
            "pixel_values": data_eagle["pixel_values"],
            "input_ids": data_eagle["input_ids"],
            "attention_mask": data_eagle["attention_mask"],
            
            "pixel_values_vita": pixel_values,
            "input_ids_vita": input_ids,
            # "attention_mask": attention_mask,
        }
        return data

    def collate_fn(self, all_examples):
        """将一个batch的样本组合成一个批次"""
        # 收集批次中所有样本的VITA图像和文本数据
        pixel_values_list = [ex["pixel_values_vita"] for ex in all_examples]
        input_ids_list = [ex["input_ids_vita"] for ex in all_examples]

        assert isinstance(pixel_values_list, List)
        assert isinstance(input_ids_list, List)

        # 将图像张量在第0维拼接
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # 准备批次化的tokenized输入
        tokenized_batch = {
            "input_ids": [ip[0] for ip in input_ids_list],
        }

        # 使用tokenizer进行填充，使其长度一致
        padded = self.tokenizer.pad(tokenized_batch,
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                return_attention_mask=True,
                                return_tensors=None,
                                )
        # 转换为torch tensor
        input_ids = torch.tensor(padded["input_ids"], dtype=torch.long) # [batch, token]
        attention_mask = torch.tensor(padded["attention_mask"], dtype=torch.long)
        
        # 处理eagle模型的批次数据
        data_eagle = self.eagle_processor.collate_fn(all_examples)
        # 组合最终的批次数据
        data = {
            "pixel_values": data_eagle["pixel_values"],
            "input_ids": data_eagle["input_ids"],
            "attention_mask": data_eagle["attention_mask"],
            
            "pixel_values_vita": pixel_values,
            "input_ids_vita": input_ids,
            "attention_mask_vita": attention_mask,
        }
        # 使用BatchFeature封装数据
        return BatchFeature(data)

    def image_vita_fn(self, sample):
        """处理VITA模型的图像输入"""
        image=[]
        # print("sample: ", len(sample)) # two pic
        for s in sample:
            # 加载图像
            s = load_image(s)
            # 调整大小
            s=s.resize((448, 448))
            # 使用图像处理器转换为张量
            image_tensor = self.image_processor(images=s, return_tensors='pt')["pixel_values"]
            image.append(image_tensor)
        # 将列表中的图像张量拼接成一个批次
        image = torch.cat(image, dim=0)

        return image

    def text_vita_fn(self, sample):
        """处理VITA模型的文本输入"""
        # 定义一个固定的问题前缀，用于引导模型扮演机器人角色并理解任务
        question_prompt = (
            "These two images are views of the same robotic arm from the "
            "front and its end effector position. Play the role of the "
            "robot arm in the picture. Based on the given task instructions, "
            "analyze the color and shape of the objects in front of you, and "
            "understand the relative position between the end effector of the "
            "robot arm and these objects. Provide as much information as "
            "possible to complete the task. Ignore objects that are not "
            "relevant to the task. Task instructions: "
        )
        # 指定对话模式
        conv_mode="qwen2p5_instruct"
        input_ids_list=[]
        # 当前 batch 图像数量，以 prepare_input 中记录的为准，默认为 1
        num_img_tokens = getattr(self, "_cur_image_cnt", 1)
        question = sample
        # 按实际图像数量拼接 <image> token
        qs = DEFAULT_IMAGE_TOKEN * num_img_tokens + "\n" + question_prompt + question
        # 获取对话模板
        conv = conv_templates[conv_mode].copy()
        # 添加用户消息
        conv.append_message(conv.roles[0], qs)
        # 添加一个空回复，让模型生成内容
        conv.append_message(conv.roles[1], None)
        # conv.append_message(conv.roles[1], "<|ACT|>")
        # 获取格式化后的prompt
        prompt = conv.get_prompt("image")
        # print("prompt: ", prompt)
        # 将prompt中的图像token替换为特定索引，并进行tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX) # return list
        input_ids_list.append(input_ids)
        # print(input_ids_list)
        return input_ids_list
    
    def audio_mapping(self):
        """音频数据映射（待完成）"""
        # TODO: need to complete
        # 定义元数据路径
        data_path = os.path.join(
            os.path.dirname(gr00t.__file__), "..", "dataset", "libero_spatial_no_noops_lerobot", "meta"
        )
        # 定义音频波形文件路径
        self.wave_path = os.path.join(
            os.path.dirname(gr00t.__file__), "..", "dataset", "wav_dataset", "libero_spatial"
        )

        # 读取任务jsonl文件，创建任务名到索引的映射
        self.tasks_dict = {}
        with open(data_path+'tasks.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.tasks_dict[data['task']] = data['task_index']
    
    def audio_vita_fn(self, sample):
        """处理VITA模型的音频输入（待完成）"""
        # TODO: need to complete
        question = sample
        # f"{self.tasks_dict[question]}.wav"
