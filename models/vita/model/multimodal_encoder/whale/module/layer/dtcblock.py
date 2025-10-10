import math
import pdb


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个深度可分离时间卷积块 (Depthwise Temporal Convolution Block)。
# 它包含逐点卷积 (pointwise conv)、批归一化 (batchnorm)、ReLU激活、Dropout 和残差连接。
# causal_conv 标志位指示是否使用因果卷积。
# buffer 用于流式推理（streaming inference）。
# buffer_index 是 buffer 中的当前索引位置。
class DTCBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, stride, causal_conv, dilation, dropout_rate
    ):
        """
        初始化 DTCBlock。

        Args:
            input_dim (int): 输入和输出特征的维度 (在此块中，输入和输出维度相同)。
            output_dim (int): 输出特征的维度 (注意：此参数在此实现中未被使用，块的输出维度等于输入维度)。
            kernel_size (int): 卷积核的大小。
            stride (int): 卷积的步长。
            causal_conv (bool): 是否使用因果卷积。
            dilation (int): 卷积的扩张率。
            dropout_rate (float): Dropout 的比率。
        """
        super(DTCBlock, self).__init__()
        # 保存初始化参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # --- 根据是否为因果卷积来设置填充 ---
        if causal_conv:
            # 对于因果卷积，不在 Conv1d 层中自动填充，而是手动在左侧填充
            self.padding = 0
            # 计算所需的左侧上下文长度
            self.lorder = (kernel_size - 1) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)
        else:
            # 对于标准卷积，使用对称填充以保持序列长度（当 stride=1 时）
            assert (kernel_size - 1) % 2 == 0
            self.padding = ((kernel_size - 1) // 2) * self.dilation
            self.lorder = 0 # 非因果卷积不需要额外的左侧上下文
        self.causal_conv = causal_conv

        # --- 定义网络层 ---
        # 1. 深度卷积 (Depthwise Convolution): 对每个输入通道独立进行卷积。
        #    groups=self.input_dim 是实现深度卷积的关键。
        self.depthwise_conv = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups=self.input_dim,
        )
        # 2. 逐点卷积 (Pointwise Convolution): 使用 1x1 卷积核来组合跨通道的特征。
        self.point_conv_1 = nn.Conv1d(self.input_dim, self.input_dim, 1, 1, self.padding)
        self.point_conv_2 = nn.Conv1d(self.input_dim, self.input_dim, 1, 1, self.padding)
        
        # 批归一化层
        self.bn_1 = nn.BatchNorm1d(self.input_dim)
        self.bn_2 = nn.BatchNorm1d(self.input_dim)
        self.bn_3 = nn.BatchNorm1d(self.input_dim)
        
        # Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

        # --- 为流式推理计算缓冲区大小 ---
        # 计算在流式处理时，当前块需要从前一个块继承多少历史信息（左侧上下文）
        self.lorder = (kernel_size - 1) * self.dilation - (self.stride - 1)
        self.buffer_size = 1 * self.input_dim * self.lorder


    @torch.jit.unused # TorchScript 编译器会忽略此方法，用于定义训练逻辑
    def forward(self, x):
        """
        标准的前向传播函数，用于训练或对完整序列进行推理。
        输入 x 的形状应为 (batch, time_steps, features)。
        """
        # 保存原始输入，用于最后的残差连接
        x_in = x
        # Conv1d 需要的输入形状是 (batch, features, time_steps)，因此需要转置
        x_data = x_in.transpose(1, 2)
        
        # 如果是因果卷积，手动在左侧进行填充
        if self.causal_conv:
            x_data_pad = self.left_padding(x_data)
        else:
            x_data_pad = x_data
            
        # --- 卷积块的核心计算流程 ---
        x_depth = self.depthwise_conv(x_data_pad) # 深度卷积
        x_bn_1 = self.bn_1(x_depth)
        x_point_1 = self.point_conv_1(x_bn_1)    # 逐点卷积
        x_bn_2 = self.bn_2(x_point_1)
        x_relu_2 = torch.relu(x_bn_2)
        x_point_2 = self.point_conv_2(x_relu_2)    # 逐点卷积
        x_bn_3 = self.bn_3(x_point_2)
        
        # 将形状转置回 (batch, time_steps, features) 以便进行残差连接
        x_bn_3 = x_bn_3.transpose(1, 2)
        
        # --- 残差连接和最终激活 ---
        # 只有当步长为1时，输出和输入的序列长度才相同，才能进行残差连接
        if self.stride == 1:
            x_relu_3 = torch.relu(x_bn_3 + x_in)
        else:
            x_relu_3 = torch.relu(x_bn_3)
            
        # 应用 Dropout
        x_drop = self.dropout(x_relu_3)
        return x_drop


    @torch.jit.export # 将此方法导出，使其在 TorchScript 编译的模型中可以被调用
    def infer(self, x, buffer, buffer_index, buffer_out):
        """
        用于流式推理的前向传播函数。
        
        Args:
            x (Tensor): 当前输入的数据块。
            buffer (Tensor): 存储所有层历史状态的扁平化张量。
            buffer_index (int): 当前层在 buffer 中开始读取的索引。
            buffer_out (list): 用于收集当前层更新后状态的列表。

        Returns:
            Tuple: (处理后的数据块, buffer, 更新后的 buffer_index, buffer_out)
        """
        # 保存原始输入块，用于残差连接
        x_in = x
        # 转置以匹配 Conv1d 的输入格式
        x = x_in.transpose(1, 2)
        
        # --- 流式处理的上下文管理 ---
        # 从全局 buffer 中提取本层所需的左侧上下文（缓存）
        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.input_dim, self.lorder]
        )
        # 将缓存与当前输入块拼接，形成完整的卷积输入
        x = torch.cat([cnn_buffer, x], dim=2)
        # 将当前输入的末尾部分（即下一个块所需的上下文）存入 buffer_out
        buffer_out.append(x[:, :, -self.lorder :].reshape(-1))
        # 更新全局 buffer 的索引
        buffer_index = buffer_index + self.buffer_size
        
        # --- 执行与 forward 方法相同的计算流程 ---
        x = self.depthwise_conv(x)
        x = self.bn_1(x)
        x = self.point_conv_1(x)
        x = self.bn_2(x)
        x = torch.relu(x)
        x = self.point_conv_2(x)
        x = self.bn_3(x)
        
        # 转置回原始形状
        x = x.transpose(1, 2)
        
        # --- 残差连接 ---
        if self.stride == 1:
            x = torch.relu(x + x_in)
        else:
            x = torch.relu(x)
            
        return x, buffer, buffer_index, buffer_out
    