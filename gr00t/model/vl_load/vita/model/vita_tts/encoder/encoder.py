'''
VITA TTS 系统的语音编码器模块

本模块实现了一个灵活的语音编码器，可以根据配置组合各种组件（Transformer、子采样等）。
编码器处理语音特征并生成适用于 TTS 任务的编码表示。

实现说明：
此编码器与 Whale 编码器在结构上有很大重叠。这里的重复是有意为之，目的是为 TTS 流水线
保持稳定的公共接口，而 Whale 侧现在直接重用 TTS 模块。任何算法更改都应首先应用于
TTS 模块，然后通过从 Whale 导入的方式传播，将此文件作为权威定义。
'''

import sys
import time
import numpy as np
import torch
import argparse
import logging
import os

from typing import Tuple, Dict, Optional

from vl_load.vita.model.vita_tts.encoder.transformer import Transformer
from vl_load.vita.model.vita_tts.encoder.subsampling import Subsampling
from vl_load.vita.model.multimodal_encoder.whale.utils import make_pad_mask

def add_encoder_args(group):
    '''
    向参数解析器组添加编码器特定的命令行参数。
    
    此函数配置语音编码器的通用参数，包括层配置、输入/输出维度以及组件特定参数。
    
    参数：
        group: 将添加编码器参数的 argparse 参数组
        
    返回：
        group: 已添加编码器参数的修改后的参数组
    '''
    group.add_argument(
        "--encoder-layer-config", 
        type=str, 
        default="tdnn-dtc", 
        help="Layer config of encoder. Format layername-layername-...",
    )
    group.add_argument(
        "--encoder-input-dim", 
        type=int, 
        default=256, 
        help="Input dim of encoder. Must equal to the input dim of the first Component (default=40)"
    )
    group.add_argument(
        "--encoder-output-dim", 
        type=int, 
        default=256, 
        help="Output dim of encoder. Must enqual to the output dim of the last Component ! (default=256)"
    )
    # 添加组件特定参数
    group=Transformer.add_arguments(group)
    group=Subsampling.add_arguments(group)
    return group

def assign_args_from_dict(args, dict, prefix_key=None):
    '''
    从字典中分配值到 argparse Namespace 对象。
    
    此工具函数使用字典中的值更新 argparse.Namespace 对象的属性，
    并处理键格式转换（连字符转下划线）。
    
    参数：
        args: 要更新的 argparse.Namespace 对象
        dict: 包含要分配的配置值的字典
        prefix_key: 可选的键，用于从 dict 中提取嵌套字典
        
    返回：
        args: 更新后的 argparse.Namespace 对象
    '''
    # 如果提供了 prefix_key，提取嵌套字典
    if prefix_key is not None: 
        dict=dict[prefix_key]
    
    # 遍历字典并更新匹配的属性
    for k, v in dict.items(): 
        # 将连字符格式的键转换为下划线格式（如 'encoder-dim' -> 'encoder_dim'）
        k_args = k.replace('-', '_') 
        # 仅当属性存在于 args 中时才更新
        if hasattr(args, k_args): 
            setattr(args, k_args, dict[k]) 
    return args

class speechEncoder(torch.nn.Module):
    '''
    可由各种编码器组件组成的灵活语音编码器。
    
    此编码器根据配置动态构建处理流水线，支持 Transformer 和子采样层等组件。
    它验证连续组件之间的维度兼容性，并应用可选的全局 CMVN（倒谱均值和方差归一化）。
    
    属性：
        global_cmvn: 应用于输入特征的可选归一化模块
        enc: 包含顺序编码器组件的 ModuleList
        config: 定义编码器架构的组件名称列表
        _output_size: 编码器的最终输出维度
    '''
    def __init__(
            self,
            input_dim,
            overview_conf = None,
            para_conf = None,
            global_cmvn = None):
        '''
        使用灵活的基于组件的架构初始化语音编码器。
        
        参数：
            input_dim: 输入特征维度
            overview_conf: 包含高级编码器配置的字典
            para_conf: 将组件名称映射到其特定参数的字典
            global_cmvn: 可选的全局 CMVN 归一化模块
        '''
        super(speechEncoder, self).__init__()

        # 初始化参数解析器并加载配置
        parser=argparse.ArgumentParser()
        add_encoder_args(parser)
        args, _=parser.parse_known_args()
        assign_args_from_dict(args, overview_conf)

        # 解析编码器架构配置
        self.config = args.encoder_layer_config.split('-')
        encoder_input_dim=args.encoder_input_dim # encoder_input_dim = 256
        encoder_output_dim=args.encoder_output_dim # encoder_output_dim = 256
        prev_output_dim=encoder_input_dim # prev_output_dim = 256
        prev_component_name="encoder" # prev_component_name = "encoder"

        self.global_cmvn = global_cmvn
        self.enc=torch.nn.ModuleList([])
        
        # 根据配置顺序构建编码器组件
        for name in self.config: 
            # 加载组件特定参数
            assign_args_from_dict(args, para_conf[name]) 
            
            # 解析组件名称（处理带后缀的变体，如 "transformer_1"）
            if len(name.split('_'))  == 2:
                name = name.split('_')[0]
            elif len(name.split('_'))  == 1:
                name=name
            else:
                print("WRONG CONFIG! {} is not valid".format("encoder", name))
                sys.exit() 
            
            # 实例化适当的组件类型
            if name=="transformer":
                self.enc.append(Transformer(args)) 
            elif name=="subsampling":
                self.enc.append(Subsampling(args)) 
            else:
                print("{} is not supported now! ".format(name))
                return NotImplemented
                
            # 验证连续组件之间的维度兼容性
            component_input_dim = getattr(args, name+"_input_dim")
            if component_input_dim!=prev_output_dim:
                print("WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-input-dim ({})"
                        .format(prev_component_name, prev_output_dim, name, component_input_dim))
                sys.exit()
            prev_output_dim=getattr(args, name + "_output_dim")
            prev_component_name=name
        
        # 确保最终组件的输出与预期的编码器输出维度匹配
        if (prev_output_dim != encoder_output_dim):
            print("WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-output-dim ({}, the last component)"
                        .format("encoder", encoder_output_dim, name, prev_output_dim))
            sys.exit()
        
        self._output_size=encoder_output_dim

        # 计算并显示总参数数量
        num_params=sum(p.numel() for p in self.parameters())
        print('the number of speech encoder params: {}M'.format(num_params/1024/1024))

    def output_size(self) -> int:
        '''
        获取编码器的输出维度。
        
        返回：
            int: 编码器输出特征的维度
        '''
        return self._output_size
    
    def forward(self, xs, ilens, decoding_chunk_size=None, num_decoding_left_chunks=None):
        '''
        编码器的前向传播。

        参数：
        - xs: torch.Tensor，形状 (batch_size, sequence_length, input_dim)
            包含输入向量序列的输入张量。
            - batch_size: 批次中序列的数量
            - sequence_length: 每个序列的长度
            - input_dim: 每个输入向量的维度

        - ilens: torch.Tensor，形状 (batch_size,)
            批次中每个序列的长度，用于填充掩码

        - decoding_chunk_size: int，可选（默认=None）
            用于解码的分块大小

        - num_decoding_left_chunks: int，可选（默认=None）
            用于解码的左侧分块数量

        返回：
        - xs: torch.Tensor，形状 (batch_size, sequence_length, encoded_dim)
            编码后的输出张量，其中 encoded_dim 是编码表示的维度

        - masks: torch.Tensor，形状 (batch_size, 1, sequence_length)
            填充掩码张量，True 表示有效元素，False 表示填充元素
        '''
        # 如果指定了分块解码参数，则配置（用于流式推理）
        if decoding_chunk_size is not None and num_decoding_left_chunks is not None: 
            for layer in self.enc: 
                # 设置每个处理块的大小
                if hasattr(layer, "chunk_size"): 
                    layer.chunk_size = decoding_chunk_size 
                # 设置要使用的左侧上下文块数量
                if hasattr(layer, "left_chunks"): 
                    layer.left_chunks = num_decoding_left_chunks 
                # 当提供显式块大小时禁用动态分块
                if hasattr(layer, "transformer_dynamic_chunks"): 
                    layer.transformer_dynamic_chunks = False 

        # 验证输入张量形状 (batch_size, sequence_length, feature_dim)
        assert(len(xs.shape)) == 3
        T=xs.size(1)
        
        # 创建填充掩码：True 表示有效位置，False 表示填充位置
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)
        
        # 如果配置了全局倒谱均值和方差归一化，则应用
        if self.global_cmvn is not None: 
            xs=self.global_cmvn(xs)
        
        # 依次通过所有编码器组件进行处理
        for module in self.enc: 
            xs, ilens, masks=module(xs, ilens, masks)
        return xs, masks

    def infer(self, xs_pad, buffer, buffer_index, buffer_out, pe_index):
        '''
        使用缓冲区执行流式推理以进行实时处理。
        
        此方法专为在线/流式场景设计，其中音频是增量处理的。它维护内部缓冲区和索引
        以处理逐块处理，同时保留时间上下文。
        
        参数：
            xs_pad: 带填充的当前块输入张量
            buffer: 包含来自先前块的历史上下文的缓存缓冲区
            buffer_index: 跟踪缓冲区中当前位置的索引
            buffer_out: 用于存储已处理特征的输出缓冲区
            pe_index: 用于维护正确位置信息的位置编码索引
            
        返回：
            tuple: (xs_pad, buffer, buffer_index, buffer_out, pe_index)
                - xs_pad: 当前块的处理输出
                - buffer: 包含新上下文的更新缓冲区
                - buffer_index: 更新的缓冲区位置索引
                - buffer_out: 更新的输出缓冲区
                - pe_index: 更新的位置编码索引
        '''
        # 如果配置了全局归一化，则应用
        if self.global_cmvn is not None: 
            xs_pad=self.global_cmvn(xs_pad)
        
        # 使用缓冲区管理通过每个编码器组件进行处理
        for module in self.enc: 
            xs_pad, buffer, buffer_index, buffer_out, pe_index = module.infer(xs_pad, 
                                            buffer, buffer_index, buffer_out, pe_index)
        return xs_pad, buffer, buffer_index, buffer_out, pe_index
