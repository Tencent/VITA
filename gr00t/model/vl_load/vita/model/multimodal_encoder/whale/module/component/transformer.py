'''Encoder self-attention layer definition.'''
# 导入必要的库
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Duplication note:
This Whale transformer wraps TTS-provided layers (pos-enc, feed-forward, convs)
and retains Whale-specific attention and streaming paths, reducing duplication.
'''
# 从 TTS (Text-to-Speech) 模型中导入基础模块，以实现代码重用
# PositionalEncoding: 绝对位置编码
# PositionwiseFeedForward: 位置前馈网络
# MultiLayeredConv1d, Conv1dLinear: 一维卷积层，用于前馈网络
from vl_load.vita.model.vita_tts.encoder.attention import (
    PositionalEncoding,
    PositionwiseFeedForward,
    MultiLayeredConv1d,
    Conv1dLinear,
)
# 从 TTS Transformer 中导入基础的 Transformer 结构
from vl_load.vita.model.vita_tts.encoder.transformer import (
    MultiSequential as TTSMultiSequential,
    TransformerLayer as TTSTransformerLayer,
)
# 导入 Whale 项目特有的模块
# MultiHeadedAttention: 多头注意力机制
# RelPositionalEncoding: 相对位置编码，适用于流式处理
from vl_load.vita.model.multimodal_encoder.whale.module.layer.attention import (
    MultiHeadedAttention,
    RelPositionalEncoding,
)

# from vl_load.vita.model.multimodal_encoder.whale.module.component.utils import *
# 导入 Whale 项目的工具函数
from vl_load.vita.model.multimodal_encoder.whale.utils import IGNORE_ID, add_optional_chunk_mask, strtobool


def repeat(N, fn):
    '''
    一个辅助函数，用于重复一个模块 N 次。
    它将 N 个由 fn(ix) 创建的模块封装在一个 Whale 特定的序列容器 `MultiSequential` 中。
    :param int N: 重复的次数。
    :param function fn: 一个函数，接受一个索引 `ix` 并返回一个 torch.nn.Module。
    :return: 包含 N 个模块的 MultiSequential 容器。
    '''
    return MultiSequential(*[fn(ix) for ix in range(N)])


class MultiSequential(TTSMultiSequential):
    '''
    继承自 TTS 的 MultiSequential，并增加了 Whale 独有的流式处理功能。
    这个容器可以像 `torch.nn.Sequential` 一样按顺序执行模块，
    但为其 `infer` 和 `infer_hidden` 方法增加了额外的流式处理参数。
    '''

    @torch.jit.export
    def infer_hidden(self, x, pos_emb, buffer, buffer_index, buffer_out, hidden_out):
        '''
        在流式推断（streaming inference）的同时，收集所有中间层的隐藏状态。
        这对于需要访问内部表示的任务（如知识蒸馏）非常有用。
        :param torch.Tensor x: 输入张量。
        :param torch.Tensor pos_emb: 位置编码张量。
        :param torch.Tensor buffer: 用于存储流式处理历史信息的缓冲区。
        :param int buffer_index: 当前在缓冲区中的索引。
        :param list buffer_out: 用于存储输出缓冲区的列表。
        :param list hidden_out: 用于收集每个子模块输出的隐藏状态的列表。
        :return: A tuple containing the final output tensor, position embedding,
            buffer, buffer index, buffer output list, and the list of hidden states.
        '''
        # 遍历容器中的每个子模块 (例如，每个 TransformerLayer)
        for sub_block in self:
            # 依次通过每个子模块的 infer 方法
            x, pos_emb, buffer, buffer_index, buffer_out = sub_block.infer(
                x, pos_emb, buffer, buffer_index, buffer_out
            )
            # 收集当前子模块的输出作为隐藏状态
            hidden_out.append(x)
        return x, pos_emb, buffer, buffer_index, buffer_out, hidden_out


class TransformerLayer(TTSTransformerLayer):
    '''
    标准的 Transformer 层模块，继承自 TTS 的实现。
    它包含一个自注意力模块 (self-attn) 和一个前馈网络 (feed-forward)。
    这个封装主要是为了与 Whale 的整体架构兼容，没有添加新的功能。
    
    结构:
    - 如果 normalize_before=True:
        x -> LayerNorm -> SelfAttention -> Dropout -> Residual Connection
        -> LayerNorm -> FeedForward -> Dropout -> Residual Connection
    - 如果 normalize_before=False:
        x -> SelfAttention -> Dropout -> Residual Connection -> LayerNorm
        -> FeedForward -> Dropout -> Residual Connection -> LayerNorm
    
    :param int size: 输入和输出的维度。
    :param torch.nn.Module self_attn: 自注意力模块 (例如 MultiHeadedAttention)。
    :param torch.nn.Module feed_forward: 前馈网络模块 (例如 PositionwiseFeedForward)。
    :param float dropout_rate: 应用于残差连接之前的 dropout 比率。
    :param bool normalize_before: 是否在子模块 (自注意力和前馈网络) 之前应用层归一化 (LayerNorm)。
    :param bool concat_after: 是否将注意力层的输入和输出进行拼接。
        如果为 True，将应用一个额外的线性层: x -> x + linear(concat(x, att(x)))。
        如果为 False，则直接相加: x -> x + att(x)。
    '''

    def __init__(
        self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False
    ):
        '''直接调用父类 TTSTransformerLayer 的构造函数，重用其实现。'''
        super().__init__(size, self_attn, feed_forward, dropout_rate, normalize_before, concat_after)


class Transformer(torch.nn.Module):
    '''
    完整的 Transformer 编码器模型。
    这个类集成了输入层、位置编码、多个 Transformer 层以及可选的最终归一化层。
    它同时支持标准的批处理前向传播 (`forward`) 和流式推断 (`infer`)。
    '''
    @staticmethod
    def add_arguments(group):
        '''
        一个静态方法，用于向 argparse 参数组中添加该模型所需的通用命令行参数。
        这使得从配置文件或命令行轻松配置模型成为可能。
        '''
        group.add_argument(
            '--transformer-input-dim', default=256, type=int, help='Transformer 的输入特征维度。'
        )
        group.add_argument(
            '--transformer-output-dim', default=4, type=int, help='Transformer 的输出维度 (如果需要，但在此实现中不直接使用)。'
        )
        group.add_argument(
            '--transformer-attention-dim', default=256, type=int, help='模型内部的工作维度，也是注意力的维度。'
        )
        group.add_argument(
            '--transformer-attention-heads',
            default=4,
            type=int,
            help='多头注意力机制中头的数量。',
        )
        group.add_argument(
            '--transformer-linear-units',
            default=1024,
            type=int,
            help='位置前馈网络 (FFN) 的隐藏层单元数。',
        )
        group.add_argument(
            '--transformer-num-blocks', default=6, type=int, help='堆叠的 Transformer 层的数量。'
        )
        group.add_argument(
            '--transformer-dropout-rate',
            default=0.1,
            type=float,
            help='在各个子层（如FFN）中使用的 Dropout 比率。',
        )
        group.add_argument(
            '--transformer-attention-dropout-rate',
            default=0.0,
            type=float,
            help='在多头注意力内部使用的 Dropout 比率。',
        )
        group.add_argument(
            '--transformer-positional-dropout-rate',
            default=0.1,
            type=float,
            help='将位置编码添加到输入嵌入后应用的 Dropout 比率。',
        )
        group.add_argument(
            '--transformer-input-layer', default='linear', type=str, 
            choices=['linear', 'embed', 'none'], help="输入层的类型：'linear' (线性层), 'embed' (嵌入层), 'none' (无操作)。"
        )
        group.add_argument(
            '--transformer-pos-enc-class', default='abs-enc', type=str, 
            choices=['abs-enc', 'rel-enc'], help="位置编码的类别：'abs-enc' (绝对位置编码), 'rel-enc' (相对位置编码)。"
        )
        group.add_argument(
            '--transformer-normalize-before',
            default=True,
            type=strtobool,
            help='是否在每个 Transformer 层的子模块之前使用层归一化 (Pre-LN)。',
        )
        group.add_argument(
            '--transformer-concat-after',
            default=False,
            type=strtobool,
            help='是否在自注意力之后拼接输入和输出，而不是直接相加。',
        )
        group.add_argument(
            '--transformer-positionwise-layer-type',
            default='linear',
            type=str,
            choices=['linear', 'conv1d', 'conv1d-linear'],
            help='位置前馈层的类型。',
        )
        group.add_argument(
            '--transformer-positionwise-conv-kernel_size',
            default=1,
            type=int,
            help='如果使用卷积作为前馈层，此参数指定其核大小。',
        )
        # 流式处理相关参数
        group.add_argument(
            '--transformer-chunk_size', default=-1, type=int,
            help='流式处理的块大小。-1 表示非流式。'
        )
        group.add_argument(
            '--transformer-left_chunks', default=-1, type=int,
            help='流式处理时，当前块可以看到的左侧历史块的数量。-1 表示看到所有历史。'
        )
        group.add_argument(
            '--transformer-dynamic-chunks', default=True, type=strtobool,
            help='是否在训练时使用动态大小的块进行模拟流式处理。'
        )
        return group

    def __init__(
        self,
        arg, # 可以是一个包含所有参数的命名空间对象
        # 也可单独提供以下所有参数
        input_dim=None,
        output_dim=None,
        attention_dim=None,
        attention_heads=None,
        linear_units=None,
        num_blocks=None,
        dropout_rate=None,
        positional_dropout_rate=None,
        attention_dropout_rate=None,
        input_layer=None,
        pos_enc_class=None,
        normalize_before=None,
        concat_after=None,
        positionwise_layer_type=None,
        positionwise_conv_kernel_size=None,
        chunk_size=None,
        left_chunks=None,
    ):
        '''
        构造一个 Transformer 对象。
        参数可以从一个 `args` 对象中获取，也可以单独指定。
        '''
        super(Transformer, self).__init__()
        # 1. 设置模型配置
        # 根据提供的 args 或单个参数来初始化模型的超参数
        if arg is None:
            # 如果没有提供 args 对象，则逐个使用传入的参数
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.attention_dim = attention_dim
            self.attention_heads = attention_heads
            self.linear_units = linear_units
            self.num_blocks = num_blocks
            self.dropout_rate = dropout_rate
            self.positional_dropout_rate = positional_dropout_rate
            self.attention_dropout_rate = attention_dropout_rate
            self.input_layer = input_layer
            self.pos_enc_class = pos_enc_class
            self.normalize_before = normalize_before
            self.concat_after = concat_after
            self.positionwise_layer_type = positionwise_layer_type
            self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
            self.chunk_size = chunk_size
            self.left_chunks = left_chunks
            # 注意: transformer_dynamic_chunks 在此模式下未被初始化
            self.transformer_dynamic_chunks = True 
        else:
            # 如果提供了 args 对象，则从中提取参数
            self.input_dim = arg.transformer_input_dim
            self.output_dim = arg.transformer_output_dim
            self.attention_dim = arg.transformer_attention_dim
            self.attention_heads = arg.transformer_attention_heads
            self.linear_units = arg.transformer_linear_units
            self.num_blocks = arg.transformer_num_blocks
            self.dropout_rate = arg.transformer_dropout_rate
            self.positional_dropout_rate = arg.transformer_positional_dropout_rate
            self.attention_dropout_rate = arg.transformer_attention_dropout_rate
            self.input_layer = arg.transformer_input_layer
            self.pos_enc_class = arg.transformer_pos_enc_class
            self.normalize_before = arg.transformer_normalize_before
            self.concat_after = arg.transformer_concat_after
            self.positionwise_layer_type = arg.transformer_positionwise_layer_type
            self.positionwise_conv_kernel_size = arg.transformer_positionwise_conv_kernel_size
            self.chunk_size = arg.transformer_chunk_size
            self.left_chunks = arg.transformer_left_chunks
            self.transformer_dynamic_chunks = arg.transformer_dynamic_chunks

        # 2. 构建位置编码模块 (Positional Encoding)
        # 根据配置选择是使用绝对位置编码还是相对位置编码
        if self.pos_enc_class=='abs-enc':
            pos_enc_args = (self.attention_dim,self.positional_dropout_rate)
            pos_enc_class=PositionalEncoding
        elif self.pos_enc_class=='rel-enc':
            # 相对位置编码需要流式处理的参数
            pos_enc_args = (
                self.attention_dim,
                self.positional_dropout_rate,
                self.chunk_size,
                self.left_chunks,
            )
            pos_enc_class=RelPositionalEncoding
        
        # 3. 构建输入层 (Input Layer)
        # 该层负责将输入特征映射到模型的工作维度 (attention_dim)
        if self.input_layer=='linear':
            self.embed = nn.Sequential( 
                nn.Linear(self.input_dim, self.attention_dim), 
                nn.LayerNorm(self.attention_dim), 
                nn.Dropout(self.dropout_rate), 
                nn.ReLU(),
            )
        elif self.input_layer=='embed':
            # 用于处理离散的 token ID 输入
            self.embed=nn.Sequential(
                nn.Embedding(self.input_dim,self.attention_dim,padding_idx=IGNORE_ID)
            )
        elif self.input_layer=='none':
            # 当输入维度已经等于 attention_dim 时使用
            self.embed = nn.Sequential(nn.Identity())
        else:
            raise ValueError('未知的输入层类型: ' + self.input_layer)
        
        # 实例化位置编码模块
        self.pe=pos_enc_class(*pos_enc_args)
        self.embed_layer_num = len(self.embed) # 记录输入层包含的子模块数量

        # 4. 构建位置前馈网络 (Position-wise Feed-Forward Network)
        # 根据配置选择前馈网络的具体实现
        if self.positionwise_layer_type=='linear':
            positionwise_layer=PositionwiseFeedForward
            positionwise_layer_args=(self.attention_dim, self.linear_units, self.dropout_rate)
        elif self.positionwise_layer_type=='conv1d':
            positionwise_layer=MultiLayeredConv1d
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate, 
            )
        elif self.positionwise_layer_type=='conv1d-linear':
            positionwise_layer=Conv1dLinear
            positionwise_layer_args = (
                self.attention_dim, 
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate, 
            )
        else:
            raise NotImplementedError('只支持 linear, conv1d, 或 conv1d-linear。')

        # 5. 构建 Transformer 编码器层栈 (Encoder Stack)
        # 使用 repeat 函数创建 `num_blocks` 个 TransformerLayer
        self.encoders=repeat(
            self.num_blocks, 
            lambda lnum:TransformerLayer(
                self.attention_dim, 
                MultiHeadedAttention( # 自注意力模块
                    self.attention_heads,
                    self.attention_dim,
                    self.attention_dropout_rate,
                    self.chunk_size,
                    self.left_chunks,
                    self.pos_enc_class,
                ),
                positionwise_layer(*positionwise_layer_args), # 前馈网络模块
                self.dropout_rate, 
                self.normalize_before, 
                self.concat_after,
            ),
        )
        # 6. 构建最终的层归一化 (Final LayerNorm)
        # 如果采用 Pre-LN 结构，则在所有层之后再加一个 LayerNorm
        if self.normalize_before:
            self.after_norm=nn.LayerNorm(self.attention_dim)

    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        '''
        标准的前向传播函数，用于训练或非流式推断。
        
        :param torch.Tensor xs: 输入张量，形状为 (batch, time, idim)。
        :param torch.Tensor ilens: 输入序列的长度，形状为 (batch,)。
        :param torch.Tensor masks: 输入的掩码张量，用于处理 padding。
        :return: 编码后的张量和对应的 mask。
        :rtype Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''

        # 1. 创建注意力掩码 (Attention Mask)
        # 根据配置决定是使用动态分块还是固定分块来模拟流式注意力
        if self.transformer_dynamic_chunks and self.training:
            # 在训练时，使用动态块大小来增强模型的鲁棒性
            chunk_masks = add_optional_chunk_mask(xs, masks, True, True, 0, 0, -1)
        else:
            # 在评估或固定配置下，使用预设的块大小
            chunk_masks = add_optional_chunk_mask(
                xs, masks, False, False, self.chunk_size, self.chunk_size, self.left_chunks
            ).to(xs.device)
            
        # 2. 应用输入层和位置编码
        # (batch, time, idim) -> (batch, time, attention_dim)
        xs=self.embed(xs)
        # 添加位置信息
        xs, pos_emb=self.pe(xs)
        
        # 3. 通过编码器层栈
        # 依次通过 num_blocks 个 TransformerLayer
        xs, chunk_masks, pos_emb=self.encoders(xs, chunk_masks, pos_emb)
        
        # 4. 应用最终的层归一化
        if self.normalize_before:
            xs=self.after_norm(xs)
            
        return xs, ilens, masks

    @torch.jit.export
    def infer(self, xs, buffer, buffer_index, buffer_out):
        '''
        流式推断方法，用于一次处理一小块（chunk）数据。
        
        :param torch.Tensor xs: 当前输入的块 (chunk)，形状通常为 (1, chunk_size, idim)。
        :param torch.Tensor buffer: 存储历史信息的缓冲区（例如，之前块的键/值）。
        :param int buffer_index: 当前缓冲区索引。
        :param list buffer_out: 存储输出缓冲区的列表。
        :return: 处理后的块、更新后的缓冲区、索引和输出列表。
        '''
        # 1. 应用输入层
        xs=self.embed(xs)

        
        # 2. 流式应用位置编码
        # `pe.infer` 会根据内部计数器为当前块生成正确的位置编码
        xs, pos_emb, _ = self.pe.infer(xs, 0)
        
        # 3. 流式通过编码器层
        # `encoders.infer` 会利用 buffer 中的历史信息来计算当前块的注意力
        xs, pos_emb, buffer, buffer_index, buffer_out = self.encoders.infer(
            xs, pos_emb, buffer, buffer_index, buffer_out
        )

        # 4. 应用最终的层归一化
        if self.normalize_before:
            xs=self.after_norm(xs)
            
        return xs, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, xs, buffer, buffer_index, buffer_out, hidden_out):
        '''
        流式推断方法，同时返回所有中间层的隐藏状态。
        
        :param torch.Tensor xs: 当前输入的块 (chunk)。
        :param torch.Tensor buffer: 历史信息缓冲区。
        :param int buffer_index: 当前缓冲区索引。
        :param list buffer_out: 存储输出缓冲区的列表。
        :param list hidden_out: 用于收集隐藏状态的列表。
        :return: 处理后的块、更新后的缓冲区、索引、输出列表和隐藏状态列表。
        '''
        # 1. 应用输入层
        xs=self.embed(xs)

        # 2. 流式应用位置编码
        xs, pos_emb, _ = self.pe.infer(xs, 0)
        
        # 3. 流式通过编码器层并收集隐藏状态
        # `encoders.infer_hidden` 在执行推断的同时，会将每个层的输出添加到 hidden_out 列表中
        xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out = self.encoders.infer_hidden(
            xs, pos_emb, buffer, buffer_index, buffer_out, hidden_out
        )

        # 4. 应用最终的层归一化
        if self.normalize_before:
            xs=self.after_norm(xs)
            
        return xs, buffer, buffer_index, buffer_out, hidden_out
