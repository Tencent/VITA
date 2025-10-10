# 导入类型提示相关的模块
from typing import List, Optional
from collections import namedtuple
# 从当前项目的子模块中导入构建器函数
from .seer.builder import build_seer
from .vita.builder import build_vita
# 导入常用的库
import json
import numpy as np
import os
import glob
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl


import re
from typing import List, Optional, Tuple, Union


# 从 transformers 库导入标准的模型输出格式
from transformers.modeling_outputs import CausalLMOutputWithPast


# 从 transformers 库导入分词器
from transformers import AutoTokenizer
# 从当前项目的子模块中导入自定义层
from .layers import ComplexProjector




# 定义 VITA 视觉语言-动作 (VLA) 模型的核心部分
class VITAVLAModel(nn.Module):
    def __init__(self, args):
        """
        初始化函数，用于构建模型的各个组件。
        Args:
            args: 包含所有配置参数的对象。
        """
        super().__init__()
        self.args = args

        # --- 1. 构建视觉语言模型 (VLM) ---
        # 根据配置路径判断 VLM 的类型，目前只实现了 'llava'
        if 'llava' in self.args.vita_path:
            # 使用 build_vita 函数来初始化分词器、VLM模型本身以及图像处理器
            self.tokenizer, self.vlm, self.image_processor = build_vita(self.args)
        else:
            raise NotImplementedError


        # --- 2. 根据配置决定是否冻结 VLM 的参数 ---
        if self.args.freeze_vlm:
            # 如果冻结，则遍历所有 VLM 参数并设置 requires_grad=False
            for param in self.vlm.parameters():
                param.requires_grad = False
        else:
            # 否则，将 VLM 设置为训练模式
            self.vlm.train()

        # 获取 VLM 的隐藏层维度，用于后续模块的维度匹配
        self.hidden_size = self.vlm.config.hidden_size

        # --- 3. 设置动作预测相关的参数 ---
        # 动作预测的时间步长
        self.action_pred_steps = self.args.action_pred_steps
        # 动作解码器的特征维度
        self.action_decoder_dim = self.args.action_decoder_dim


        # vita_action mlp (注释：这是一个标记，说明以下是动作映射器的实现)


        # --- 4. 构建动作映射器 (action_mapper) ---
        # 这个模块将 VLM 的隐藏状态映射到动作解码器所需的维度
        # # MLP method (注释：简单的多层感知机方法)
        if 'complex' not in self.args.projector_type:
            # 使用正则表达式匹配 "mlp<N>x_gelu" 格式的配置
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", self.args.projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1)) # 获取 MLP 的层数
                modules = [nn.Linear(self.hidden_size, self.action_decoder_dim)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(self.action_decoder_dim, self.action_decoder_dim))

                # 将所有层组合成一个序列模块
                self.action_mapper = nn.Sequential(*modules)
        else:
            # complex method (注释：使用更复杂的投影器)
            self.action_mapper = ComplexProjector(
                input_dim=self.hidden_size,
                output_dim=self.action_decoder_dim
            )

        # 定义一个可学习的特殊 token，用于提示模型进行动作预测
        self.action_pred_token = nn.Parameter(torch.randn(1, self.hidden_size))


        # --- 5. 构建机器人状态编码器 ---
        # state encoder
        ARM_STATE_FEATURE_DIM = self.hidden_size
        GRIPPER_STATE_FEATURE_DIM = self.hidden_size

        # 线性层，用于编码手臂的6个自由度状态
        self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        # 线性层，用于编码夹爪的2个自由度状态
        self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        # 线性层，用于将编码后的手臂和夹爪特征融合并投影到 VLM 的隐藏维度
        self.state_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_size)




    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            audios: Optional[dict] = None,
            states: Optional[torch.FloatTensor] = None,
            sf_masks: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        模型的前向传播函数。
        """

        # --- 1. 编码机器人状态 ---
        B, S = states.shape[0], states.shape[1] # 获取批次大小和序列长度
        states = states.flatten(0, 1) # 将批次和序列维度合并，方便批量处理
        arm_state_feature = self.arm_state_encoder(states[:, :6]) # 编码手臂状态

        # 根据配置处理夹爪状态
        if not self.args.gripper_width:
            # 如果不直接使用夹爪宽度，则将其二值化（开/合）并进行 one-hot 编码
            gripper_state_one_hot = torch.nn.functional.one_hot(
              torch.where(states[:, 6:].flatten() < 1, torch.tensor(0), torch.tensor(1)), num_classes=2)
            gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(states))
        else:
            # 直接使用夹爪宽度作为输入进行编码
            gripper_state_feature = self.gripper_state_encoder(states[:, 6:])
        # 将编码后的手臂和夹爪特征拼接，并通过投影层得到最终的状态嵌入
        state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))


        # --- 2. 将所有信息传入 VLM 进行处理 ---
        # VLM 的 forward 方法被扩展，可以接收图像、音频、状态等多种模态的输入
        outputs, action_indices = self.vlm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            audios=audios,
            states=state_embedding, # 传入处理好的状态嵌入
            action_pred_token = self.action_pred_token, # 传入动作预测提示 token
            action_pred_steps = self.action_pred_steps,
            sf_masks=sf_masks,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # --- 3. 提取与动作预测相关的隐藏状态 ---
        # [2, 2879, 3584] (注释：这是一个示例张量形状的笔记)
        # 从 VLM 的输出中获取指定层的隐藏状态
        hidden_states = outputs.hidden_states[self.args.hidden_index]
        # 使用 VLM 返回的 `action_indices` 来精确地从隐藏状态序列中提取出用于动作预测的部分
        action_hidden_states = hidden_states[
          action_indices[0], action_indices[1], :].view(hidden_states.shape[0], 
                                                        S, -1, hidden_states.shape[-1])

        # --- 4. 将提取的隐藏状态映射到动作特征空间 ---
        output = self.action_mapper(action_hidden_states)
        
        return output


# 定义顶层封装模型，集成了 VITAVLAModel 和 SEER 动作解码器
class VITAVLA(nn.Module):
    def __init__(self, args, clip_device_id):
        super().__init__()
        # 初始化核心的 VLA 模型
        self.model = VITAVLAModel(args)
        # 初始化 SEER 模型，它在这里主要用作动作解码器
        self.seer = build_seer(args, clip_device_id)


    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        generate: Optional[bool] = False, # 控制是训练模式还是生成（推理）模式
        audios: Optional[dict] = None,
        sf_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tensor:
        """
        使类的实例可以像函数一样被调用，作为整个模型的入口。
        """
        # 首先，调用内部的 VITAVLAModel 获取动作的中间特征表示
        output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                audios=audios,
                states=states,
                sf_masks=sf_masks,
                return_dict=return_dict,
                cache_position=cache_position)
        
        # 根据 `generate` 标志决定后续操作
        if generate:
            # --- 推理/生成模式 ---
            # import pdb;pdb.set_trace() (注释：这是用于调试的断点)
            # 将中间特征送入 SEER 的动作解码器
            action_pred_feature = self.seer.action_decoder(output)
            # 分别解码出手臂和夹爪的具体动作指令
            arm_pred_action = self.seer.arm_action_decoder(action_pred_feature)
            # print(arm_pred_action.shape, "arm_pred_action.shape") (注释：用于调试打印形状)
            gripper_pred_action = self.seer.gripper_action_decoder(action_pred_feature)

            # 返回解码后的具体动作
            return arm_pred_action, gripper_pred_action
        else:
            # --- 训练模式 ---
            # 直接返回中间特征表示，用于后续计算损失函数
            return output
    