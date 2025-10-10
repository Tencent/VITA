import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个集成了填充、归一化、激活函数、Dropout 和残差连接的 Conv1d 层。
# causal_conv 标志位指示是否使用因果卷积（causal convolution）。
# buffer 用于流式推理（streaming inference）。
# buffer_index 是 buffer 中的当前索引位置。


class Conv1dLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        causal_conv,
        dilation,
        dropout_rate,
        residual=True,
    ):
        """
        初始化 Conv1dLayer。

        Args:
            input_dim (int): 输入特征的维度。
            output_dim (int): 输出特征的维度。
            kernel_size (int): 卷积核的大小。
            stride (int): 卷积的步长。
            causal_conv (bool): 如果为 True，则使用因果卷积，确保当前时间步的输出
                                不依赖于未来的输入。
            dilation (int): 卷积的扩张率（dilation rate）。
            dropout_rate (float): Dropout 的比率。
            residual (bool): 是否使用残差连接。
        """
        super(Conv1dLayer, self).__init__()
        # 保存初始化参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.causal_conv = causal_conv

        # --- 计算并设置填充 ---
        if causal_conv:
            # 对于因果卷积，我们只在输入的左侧进行填充。
            # 填充的长度 lorder 确保卷积核在计算第一个输出时，其感受野的最后一个元素
            # 恰好对应输入的第一个元素。
            self.lorder = (kernel_size - 1) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)
        else:
            # 对于标准（非因果）卷积，我们进行对称填充以保持特征图长度。
            # 这要求卷积核大小为奇数。
            assert (kernel_size - 1) % 2 == 0
            self.lorder = ((kernel_size - 1) // 2) * self.dilation
            self.left_padding = nn.ConstantPad1d((self.lorder, self.lorder), 0.0)
        
        # --- 定义网络层 ---
        # 1D 卷积层。注意 padding=0，因为我们手动处理了填充。
        self.conv1d = nn.Conv1d(
            self.input_dim, self.output_dim, self.kernel_size, self.stride, 0, self.dilation
        )
        # 批归一化层
        self.bn = nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.99)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # --- 残差连接设置 ---
        self.residual = residual
        # 如果输入和输出维度不同，则无法进行残差连接 (x + output)，因此禁用它。
        if self.input_dim != self.output_dim:
            self.residual = False

        # --- 为流式推理（streaming inference）计算缓冲区大小 ---
        # 这部分是为了在处理实时音频等流式数据时，能够逐块进行推理。
        # 缓冲区（buffer）用于存储上一数据块的末尾部分，作为当前数据块的左侧上下文。
        
        # 计算主卷积路径所需的上下文（缓存）大小
        self.lorder = (kernel_size - 1) * self.dilation - (self.stride - 1)
        self.buffer_size = 1 * self.input_dim * self.lorder
        
        # 计算残差连接路径所需的上下文（缓存）大小
        self.x_data_chache_size = self.lorder
        self.x_data_buffer_size = self.input_dim * self.x_data_chache_size


    @torch.jit.unused # TorchScript 编译器会忽略此方法，通常用于定义仅在 Python 中使用的训练逻辑
    def forward(self, x):
        """
        标准的前向传播函数，用于训练或对完整序列进行推理。
        """
        # 保存原始输入 x，用于残差连接
        x_data = x
        # 应用左填充
        x = self.left_padding(x)
        # 应用卷积
        x = self.conv1d(x)
        # 应用批归一化
        x = self.bn(x)
        # 如果可以进行残差连接（步长为1且启用残差）
        if self.stride == 1 and self.residual:
            # 先相加再激活 (ReLU(x + identity))
            x = self.relu(x + x_data)
        else:
            # 否则直接激活
            x = self.relu(x)
        # 应用 Dropout
        x = self.dropout(x)
        return x


    @torch.jit.export # 将此方法导出，使其在 TorchScript 编译的模型中可以被调用
    def infer(self, x, buffer, buffer_index, buffer_out):
        """
        用于流式推理的前向传播函数。
        它处理一个数据块 `x`，并利用 `buffer` 来管理跨块的上下文依赖。

        Args:
            x (Tensor): 当前输入的数据块。
            buffer (Tensor): 一个扁平化的张量，存储了所有层的历史状态/缓存。
            buffer_index (int): 当前层在 `buffer` 中开始读取的索引。
            buffer_out (list): 一个列表，用于收集当前层更新后的状态/缓存。

        Returns:
            Tuple: (处理后的数据块, buffer, 更新后的 buffer_index, buffer_out)
        """
        # 复制当前输入块，用于可能的残差连接
        x_data = x.clone()

        # --- 主卷积路径的流式处理 ---
        # 从全局 buffer 中提取本层所需的左侧上下文（缓存）
        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.input_dim, self.lorder]
        )
        # 将缓存与当前输入块拼接起来，形成完整的卷积输入
        x = torch.cat([cnn_buffer, x], dim=2)
        # 将当前输入的末尾部分（即下一个块所需的上下文）存入 buffer_out
        buffer_out.append(x[:, :, -self.lorder :].reshape(-1))
        # 更新全局 buffer 的索引
        buffer_index = buffer_index + self.buffer_size

        # 执行卷积和批归一化
        x = self.conv1d(x)
        x = self.bn(x)

        # --- 残差连接路径的流式处理 ---
        if self.stride == 1 and self.residual:
            # 从全局 buffer 中提取残差路径所需的左侧上下文
            x_data_cnn_buffer = buffer[
                buffer_index : buffer_index + self.x_data_buffer_size
            ].reshape([1, self.input_dim, self.x_data_chache_size])
            # 将缓存与原始输入块拼接
            x_data = torch.cat([x_data_cnn_buffer, x_data], dim=2)
            # 将更新后的残差路径缓存存入 buffer_out
            buffer_out.append(x_data[:, :, -self.x_data_chache_size :].reshape(-1))
            # 更新全局 buffer 索引
            buffer_index = buffer_index + self.x_data_buffer_size
            # 裁剪拼接后的 x_data，使其长度与卷积输出 x 匹配，以便相加
            x_data = x_data[:, :, : -self.x_data_chache_size]
            # 执行残差连接和激活
            x = self.relu(x + x_data)
        else:
            # 如果没有残差连接，直接激活
            x = self.relu(x)

        return x, buffer, buffer_index, buffer_out
        