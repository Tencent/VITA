import argparse
import logging
import sys
import time
from typing import Dict, Optional, Tuple


import numpy as np
import six
import torch


# 导入项目内部定义的模块组件
from ......model.multimodal_encoder.whale.module.component.mamba import MambaSSM
from ......model.multimodal_encoder.whale.module.component.subsampling import Subsampling
from ......model.multimodal_encoder.whale.module.component.transformer import Transformer
from ......model.multimodal_encoder.whale.utils import make_pad_mask




# 编码器相关代码，包括 whaleEncoder 类和相关函数。
# add_encoder_args 函数用于向 argparse 添加编码器相关的参数。
# assign_args_from_dict 函数用于从字典中为 argparse 参数赋值。


def add_encoder_args(group):
    """
    向 argparse 参数组中添加编码器通用的参数。

    Args:
        group: 一个 argparse 参数组对象。

    Returns:
        group: 添加了新参数后的参数组对象。
    """
    group.add_argument(
        "--encoder-layer-config",
        type=str,
        default="tdnn-dtc",
        help="编码器的层配置。格式为 layername-layername-...，例如 'subsampling-transformer-transformer'。",
    )
    group.add_argument(
        "--encoder-input-dim",
        type=int,
        default=256,
        help="编码器的输入维度。必须与第一个组件的输入维度相等。",
    )
    group.add_argument(
        "--encoder-output-dim",
        type=int,
        default=256,
        help="编码器的输出维度。必须与最后一个组件的输出维度相等。",
    )
    # 为所有可能用到的组件（如 Transformer, Subsampling 等）添加它们各自的参数。
    # 这是一个可扩展的设计，如果添加了新的组件类型，只需在此处调用其 add_arguments 方法即可。
    group = Transformer.add_arguments(group)
    group = Subsampling.add_arguments(group)
    group = MambaSSM.add_arguments(group)
    return group


# 从字典为 argparse 参数赋值
def assign_args_from_dict(args, dict, prefix_key=None):
    """
    一个工具函数，用于将字典中的键值对赋值给 argparse.Namespace 对象的属性。
    这常用于从配置文件（如 YAML）加载配置来覆盖默认参数。

    Args:
        args (argparse.Namespace): 要被更新的参数对象。
        dict (dict): 包含配置信息的字典。
        prefix_key (str, optional): 如果提供，则只使用字典中以此为键的子字典。默认为 None。

    Returns:
        argparse.Namespace: 更新后的参数对象。
    """
    if prefix_key is not None:
        dict = dict[prefix_key]
    for k, v in dict.items():
        # 将字典中的 key（通常带连字符'-'）转换为 argparse 属性名（带下划线'_'）
        k_args = k.replace("-", "_")
        # 如果 args 对象中存在该属性，则用字典中的值更新它
        if hasattr(args, k_args):
            setattr(args, k_args, dict[k])
    return args


# whale 编码器类，它是由不同组件堆叠而成的
class whaleEncoder(torch.nn.Module):
    def __init__(self, input_dim, overview_conf=None, para_conf=None, global_cmvn=None):
        """
        初始化 whaleEncoder。

        Args:
            input_dim (int): 编码器的输入维度。
            overview_conf (dict, optional): 包含编码器全局配置的字典。
            para_conf (dict, optional): 包含每个组件具体参数的字典。
            global_cmvn (torch.nn.Module, optional): 全局 CMVN 归一化模块。
        """
        super(whaleEncoder, self).__init__()

        # --- 配置解析与组件构建 ---
        # 1. 创建一个临时的 ArgumentParser 来加载所有可能的默认参数
        parser = argparse.ArgumentParser()
        add_encoder_args(parser)
        args, _ = parser.parse_known_args()

        # 2. 使用传入的配置字典覆盖默认参数
        assign_args_from_dict(args, overview_conf)
        # assign_args_from_dict(args, para_conf) # 此行被注释，参数在循环中按需加载

        # 3. 解析编码器层配置字符串，例如 "subsampling-transformer-transformer"
        self.config = args.encoder_layer_config.split("-")
        encoder_input_dim = args.encoder_input_dim
        encoder_output_dim = args.encoder_output_dim
        
        # 4. 动态构建编码器层
        prev_output_dim = encoder_input_dim
        prev_component_name = "encoder"
        self.enc = torch.nn.ModuleList([])
        for name in self.config:
            # 从 para_conf 中加载当前组件的特定参数
            assign_args_from_dict(args, para_conf[name])
            
            # 处理带编号的组件名，例如 "transformer_1" -> "transformer"
            if len(name.split("_")) == 2:
                name = name.split("_")[0]
            elif len(name.split("_")) == 1:
                name = name
            else:
                logging.error("WRONG CONFIG! {} is not valid".format("encoder", name))
                sys.exit()

            # 根据组件名实例化对应的模块
            if name == "transformer":
                self.enc.append(Transformer(args))
            elif name == "subsampling":
                self.enc.append(Subsampling(args))
            elif name == "mamba":
                self.enc.append(MambaSSM(args))
            else:
                print("{} is not supported now!".format(name))
                return NotImplemented
            
            # --- 维度匹配校验 ---
            # 校验当前组件的输入维度是否与前一个组件的输出维度匹配
            component_input_dim = getattr(args, name + "_input_dim")
            if component_input_dim != prev_output_dim:
                logging.error(
                    "WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-input-dim ({})".format(
                        prev_component_name, prev_output_dim, name, component_input_dim
                    )
                )
                sys.exit()
            # 更新 prev_output_dim 以便下一次循环校验
            prev_output_dim = getattr(args, name + "_output_dim")
            prev_component_name = name

        # 保存全局 CMVN 模块
        self.global_cmvn = global_cmvn
        
        # 校验最后一个组件的输出维度是否与编码器要求的总输出维度匹配
        if prev_output_dim != encoder_output_dim:
            logging.error(
                "WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-output-dim ({}, the last component)".format(
                    "encoder", encoder_output_dim, name, prev_output_dim
                )
            )
            sys.exit()

        # 设置编码器的输出维度
        self._output_size = encoder_output_dim

        # 打印模型参数量
        num_params = sum(p.numel() for p in self.parameters())
        print("the number of whale encoder params: {}M".format(num_params / 1024 / 1024))

    def output_size(self) -> int:
        """返回编码器的输出维度。"""
        return self._output_size

    @torch.jit.unused # TorchScript 编译器会忽略此方法，通常用于定义仅在 Python 中使用的训练逻辑
    def forward(self, xs, ilens, decoding_chunk_size=None, num_decoding_left_chunks=None):
        """
        编码器的标准前向传播函数（用于训练或对完整序列进行推理）。

        Args:
            xs (torch.Tensor): 批量的填充输入序列 (B, Tmax, D)。
            ilens (torch.Tensor): 批量中每个序列的真实长度 (B)。
            decoding_chunk_size (int, optional): 流式解码时的块大小。
            num_decoding_left_chunks (int, optional): 流式解码时使用的左侧上下文块数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 编码后的隐藏状态序列 (B, T_out, D_out) 和对应的掩码。
        """

        # 如果是流式解码模式，为支持该模式的层设置块大小等参数
        if decoding_chunk_size is not None and num_decoding_left_chunks is not None:
            for layer in self.enc:
                if hasattr(layer, "chunk_size"):
                    layer.chunk_size = decoding_chunk_size
                if hasattr(layer, "left_chunks"):
                    layer.left_chunks = num_decoding_left_chunks
                if hasattr(layer, "transformer_dynamic_chunks"):
                    layer.transformer_dynamic_chunks = False

        assert (len(xs.shape)) == 3
        T = xs.size(1)
        # 根据输入长度创建掩码，用于处理填充部分
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        
        # 如果配置了全局 CMVN，则首先进行特征归一化
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
            
        # 依次将数据通过编码器中的每个组件
        for module in self.enc:
            xs, ilens, masks = module(xs, ilens, masks)
        return xs, masks

    @torch.jit.export # 将此方法导出，使其在 TorchScript 编译的模型中可以被调用
    def infer(self, xs_pad, buffer, buffer_index, buffer_out):
        """
        用于流式推理的前向传播函数。

        Args:
            xs_pad (torch.Tensor): 当前输入的特征块。
            buffer (torch.Tensor): 存储所有层历史状态的扁平化张量。
            buffer_index (int): 当前层在 buffer 中开始读取的索引。
            buffer_out (list): 用于收集当前层更新后状态的列表。

        Returns:
            Tuple: 处理后的数据块和更新后的 buffer 状态。
        """
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # 依次调用每个组件的 infer 方法，传递并更新 buffer 状态
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out = module.infer(
                xs_pad, buffer, buffer_index, buffer_out
            )
        return xs_pad, buffer, buffer_index, buffer_out

    @torch.jit.export # 导出此方法
    def infer_hidden(self, xs_pad, buffer, buffer_index, buffer_out, hidden_out):
        """
        与 infer 类似，但额外收集中间层的隐藏状态。
        """
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out, hidden_out = module.infer_hidden(
                xs_pad, buffer, buffer_index, buffer_out, hidden_out
            )
        return xs_pad, buffer, buffer_index, buffer_out, hidden_out

    @torch.jit.ignore(drop=True) # TorchScript 编译时完全忽略此方法
    def get_extra_loss(self) -> Dict[str, torch.Tensor]:
        """
        获取额外的损失，例如用于多任务学习的中间层损失。
        在此处为占位符，返回 None。
        """
        return None
        