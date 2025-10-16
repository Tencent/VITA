import torch
import torch.nn as nn
import math
import numpy

"""
重复说明：本文件集中定义了 VITA TTS 端的注意力与前馈模块，是各个音视频
编码器的权威实现。Whale 侧通过直接导入这些类来复用实现，消除重复保
养成本。
"""
# 调试导入，在开发过程中保留；保持注释状态以避免 linter 噪音
# import pdb

def project_qkv(linear_q, linear_k, linear_v, query, key, value, n_head, d_k):
    """将查询（query）、键（key）、值（value）投影到多头张量中。返回值形状均为 (batch, head, time, dim)。"""
    n_batch = query.size(0)  # 获取批次大小
    # 线性投影并重塑张量以适应多头注意力机制
    q = linear_q(query).view(n_batch, -1, n_head, d_k).transpose(1, 2)
    k = linear_k(key).view(n_batch, -1, n_head, d_k).transpose(1, 2)
    v = linear_v(value).view(n_batch, -1, n_head, d_k).transpose(1, 2)
    return q, k, v


def apply_attention_mask(scores, mask, min_value):
    """应用注意力 mask；在 Whale 与 TTS 代码之间共享以减少重复。"""
    if mask is not None:
        # unsqueeze(1) 在 head 维度上增加一个维度以进行广播
        mask_bool = mask.unsqueeze(1).eq(0)
        # 使用一个极小值填充被 mask 的位置，以便在 softmax 后变为零
        scores = scores.masked_fill(mask_bool, min_value)
        # 计算 softmax 后，再次将被 mask 的位置填充为 0.0
        return torch.softmax(scores, dim=-1).masked_fill(mask_bool, 0.0)
    # 如果没有 mask，直接计算 softmax
    return torch.softmax(scores, dim=-1)


def merge_attention_output(attn, value, n_head, d_k):
    """合并注意力输出，返回形状 (batch, time, n_head * d_k)。"""
    n_batch = attn.size(0) # 获取批次大小
    # 将注意力权重应用于 value
    x = torch.matmul(attn, value)
    # 转换维度并重塑为最终输出形状
    return x.transpose(1, 2).contiguous().view(n_batch, -1, n_head * d_k)


class PositionalEncoding(torch.nn.Module):
    """位置编码模块。
    :param int d_model: 嵌入维度
    :param float dropout_rate: dropout 比率
    :param int max_len: 最大输入长度
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 1500,
                 reverse: bool = False):
        """构造一个 PositionalEncoding 对象。"""
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.xscale = math.sqrt(self.d_model)  # 缩放因子
        self.dropout = torch.nn.Dropout(p=dropout_rate)  # Dropout 层
        self.max_len = max_len  # 最大序列长度

        # 初始化位置编码矩阵
        self.pe = torch.zeros(self.max_len, self.d_model)
        # 创建位置张量
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        # 计算除法项，用于缩放不同维度的位置
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        # 计算偶数维度的 sin 编码
        self.pe[:, 0::2] = torch.sin(position * div_term)
        # 计算奇数维度的 cos 编码
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个批次维度
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0):
        """添加位置编码。
        Args:
            x (torch.Tensor): 输入张量。形状为 (batch, time, ...)
            offset (int): 位置偏移量
        Returns:
            torch.Tensor: 编码后的张量。形状为 (batch, time, ...)
            torch.Tensor: 为了与 RelPositionalEncoding 兼容
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        # 获取当前输入序列所需的位置编码部分
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        # 将位置编码加到输入张量上
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int):
        """ 用于以流式方式获取编码
        注意!!!!!
        在非流式处理中，我们在整个话语级别上只应用一次 dropout，
        但在流式场景中，会随着输入大小的增加多次调用此函数，
        因此 dropout 会被应用多次。
        Args:
            offset (int): 起始偏移量
            size (int): 所需位置编码的大小
        Returns:
            torch.Tensor: 对应的编码
        """
        assert offset + size < self.max_len
        section = self.pe[:, offset:offset + size]
        return self.dropout(section)

class RelPositionalEncoding(PositionalEncoding):
    """相对位置编码模块。
    参见：https://arxiv.org/abs/1901.02860 中的附录 B
    Args:
        d_model (int): 嵌入维度。
        dropout_rate (float): Dropout 比率。
        max_len (int): 最大输入长度。
    """
    def __init__(self, d_model, dropout_rate, chunk_size, left_chunks, max_len = 5000):
        """初始化类。"""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)
        self.chunk_size = chunk_size  # 块大小
        self.left_chunks = left_chunks  # 左侧块数
        self.full_chunk_size = (self.left_chunks + 1) * self.chunk_size  # 完整块大小

        # 计算除法项
        self.div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        # 重新计算最大长度以适应分块
        self.max_len = self.chunk_size * (max_len // self.chunk_size) - self.full_chunk_size

    def forward(self,
                x: torch.Tensor,
                offset: int = 0):
        """计算位置编码。
        Args:
            x (torch.Tensor): 输入张量 (batch, time, `*`)。
        Returns:
            torch.Tensor: 编码后的张量 (batch, time, `*`)。
            torch.Tensor: 位置嵌入张量 (1, time, `*`)。
        """
        self.pe=self.pe.to(x.device)
        scaled_tokens = x * self.xscale
        # 截取所需的位置编码
        pos_slice = self.pe[:, offset:offset + x.size(1)]
        return self.dropout(scaled_tokens), self.dropout(pos_slice)

    def infer(self, xs, pe_index, pe_length):
        """流式推断方法。"""
        # 使用取模运算处理循环索引
        pe_index = pe_index%self.max_len
        xs = xs*self.xscale

        # 动态生成位置编码
        pe = torch.zeros(pe_length, self.d_model)
        position=torch.arange(
            max(0, pe_index - self.full_chunk_size),
            max(0, pe_index - self.full_chunk_size) + pe_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        # 计算 sin 和 cos 波
        sin_wave = torch.sin(position * self.div_term)
        cos_wave = torch.cos(position * self.div_term)
        pe[:, 0::2] = sin_wave
        pe[:, 1::2] = cos_wave
        pos_emb=pe.unsqueeze(0)

        # 更新位置索引
        pe_index = pe_index+self.chunk_size
        return xs, pos_emb, pe_index

class PositionwiseFeedForward(torch.nn.Module):
    """位置前馈网络层。
    :param int idim: 输入维度
    :param int hidden_units: 隐藏单元数
    :param float dropout_rate: dropout 比率
    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """构造一个 PositionwiseFeedForward 对象。"""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)  # 第一个线性层
        self.w_2 = torch.nn.Linear(hidden_units, idim)  # 第二个线性层
        self.dropout = torch.nn.Dropout(dropout_rate)  # Dropout 层

    def forward(self, x):
        """前向函数。"""
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
    
    def infer(self, xs, buffer, buffer_index, buffer_out):
        """流式推断方法。"""
        # 对于简单的前馈网络，流式推断与常规前向传播相同
        return self.w_2(torch.relu(self.w_1(xs))), buffer, buffer_index, buffer_out

class MultiLayeredConv1d(torch.nn.Module):
    """用于 Transformer 块的多层一维卷积。
    这是一个多层一维卷积模块，旨在替代 Transformer 块中的
    位置前馈网络，该设计在 `FastSpeech` 中被引入。
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """初始化 MultiLayeredConv1d 模块。
        Args:
            in_chans (int): 输入通道数。
            hidden_chans (int): 隐藏通道数。
            kernel_size (int): 一维卷积的核大小。
            dropout_rate (float): Dropout 比率。
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # 保持长度不变的 padding
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """计算前向传播。
        Args:
            x (Tensor): 输入张量的批次 (B, ..., in_chans)。
        Returns:
            Tensor: 输出张量的批次 (B, ..., hidden_chans)。
        """
        # 交换最后两个维度以适应 Conv1d
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)

class Conv1dLinear(torch.nn.Module):
    """用于 Transformer 块的 Conv1D + Linear。
    MultiLayeredConv1d 的一个变体，将第二个卷积层替换为线性层。
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """初始化 Conv1dLinear 模块。
        Args:
            in_chans (int): 输入通道数。
            hidden_chans (int): 隐藏通道数。
            kernel_size (int): 一维卷积的核大小。
            dropout_rate (float): Dropout 比率。
        """
        super(Conv1dLinear, self).__init__()
        self.lorder = (kernel_size - 1)  # 左侧上下文大小
        self.left_padding = nn.ConstantPad1d((self.lorder, 0), 0.0)  # 左侧填充
        self.w_1 = torch.nn.Sequential(
                        torch.nn.Conv1d( # 深度可分离卷积
                            in_chans,
                            in_chans,
                            kernel_size,
                            stride=1,
                            padding=0,
                            groups=in_chans
                        ),
                        torch.nn.Conv1d( # 逐点卷积
                            in_chans, 
                            hidden_chans, 
                            1,
                            padding=0
                        )
                    )
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)  # 第二层是线性层
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.in_chans = in_chans

        # 计算流式推断所需的缓冲区大小
        self.buffer_size = 1 * self.in_chans * self.lorder

    def forward(self, x):
        """计算前向传播。
        Args:
            x (Tensor): 输入张量的批次 (B, ..., in_chans)。
        Returns:
            Tensor: 输出张量的批次 (B, ..., hidden_chans)。
        """
        x = torch.relu(self.w_1(self.left_padding(x.transpose(-1, 1)))).transpose(-1, 1)
        return self.w_2(self.dropout(x))

    def infer(self, x, buffer, buffer_index, buffer_out):
        """流式推断方法。"""
        x = x.transpose(-1, 1)

        # 从缓冲区中获取左侧上下文
        cnn_buffer = buffer[buffer_index: buffer_index + self.buffer_size].reshape([1, self.in_chans, self.lorder])
        # 将上下文与当前输入拼接
        x = torch.cat([cnn_buffer, x], dim=2)
        # 将新的上下文存入输出缓冲区
        buffer_out.append(x[:, :, -self.lorder:].reshape(-1))
        buffer_index = buffer_index + self.buffer_size

        # 执行卷积和线性变换
        x = self.w_1(x)
        x = torch.relu(x).transpose(-1, 1)
        x = self.w_2(x)
        return x, buffer, buffer_index, buffer_out

class MultiHeadedAttention(nn.Module):
    """多头注意力层。
    :param int n_head: 头的数量
    :param int n_feat: 特征的数量
    :param float dropout_rate: dropout 比率
    """
    def __init__(self, n_head_, n_feat_, dropout_rate_, chunk_size_, left_chunks_, pos_enc_class_):
        """构造一个 MultiHeadedAttention 对象。"""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat_ % n_head_==0
        # 我们假设 d_v 总是等于 d_k
        self.d_k = n_feat_//n_head_
        self.h = n_head_
        self.linear_q = nn.Linear(n_feat_,n_feat_)
        self.linear_k = nn.Linear(n_feat_,n_feat_)
        self.linear_v = nn.Linear(n_feat_,n_feat_)
        self.linear_out = nn.Linear(n_feat_,n_feat_)
        self.dropout = nn.Dropout(p = dropout_rate_)
        self.min_value = float(numpy.finfo(torch.tensor(0, dtype=torch.float16).numpy().dtype).min)
        # 分块参数
        if chunk_size_ > 0 and left_chunks_ > 0: # 流式模式
            self.buffersize = chunk_size_*(left_chunks_)
            self.left_chunk_size = chunk_size_*left_chunks_
        else: # 非流式模式
            self.buffersize=1
            self.left_chunk_size=1
        self.chunk_size=chunk_size_

        # 编码设置
        if pos_enc_class_ == "rel-enc": # 相对位置编码
            self.rel_enc=True
            self.linear_pos=nn.Linear(n_feat_, n_feat_, bias=False)
            # 这两个可学习的偏置用于矩阵 c 和矩阵 d
            # 如 https://arxiv.org/abs/1901.02860 第 3.3 节所述
            self.pos_bias_u=nn.Parameter(torch.Tensor(self.h, self.d_k))
            self.pos_bias_v=nn.Parameter(torch.Tensor(self.h, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        else: # 绝对位置编码
            self.rel_enc=False
            self.linear_pos=nn.Identity()
            self.pos_bias_u=torch.tensor([0])
            self.pos_bias_v=torch.tensor([0])
        
        # 缓冲区大小计算
        # key_buffer shape: (1, h, buffersize, d_k)
        self.key_buffer_size = 1 * self.h * self.buffersize * self.d_k # key_buffer
        # value_buffer shape: (1, h, buffersize, d_k)
        self.value_buffer_size = 1 * self.h * self.buffersize * self.d_k # value_buffer

        if self.chunk_size>0:
            # buffer_mask shape: (1, h, chunk_size, buffersize)
            self.buffer_mask_size = (
                1 * self.h * self.chunk_size * self.buffersize
            )
        else:
            self.buffer_mask=torch.ones([1, self.h, 1, 1], dtype=torch.bool)

    def rel_shift(self, x_, zero_triu_: bool = False):
        """计算相对位置编码。
        Args:
            x (torch.Tensor): 输入张量 (batch, time, size)。
            zero_triu (bool): 如果为 true，则返回矩阵的下三角部分。
        Returns:
            torch.Tensor: 输出张量。
        """

        zero_pad_ = torch.zeros((x_.size()[0], x_.size()[1], x_.size()[2], 1),
                               device=x_.device,
                               dtype=x_.dtype)
        x_padded_=torch.cat([zero_pad_, x_], dim=-1)

        x_padded_ = x_padded_.view(x_.size()[0],
                                 x_.size()[1],
                                 x_.size(3) + 1, x_.size(2))
        x_=x_padded_[:, :, 1:].view_as(x_)

        if zero_triu_:
            ones_=torch.ones((x_.size(2), x_.size(3)))
            x_ = x_ * torch.tril(ones_, x_.size(3)-x_.size(2))[None, None, :, :]
        return x_

    def forward(self, query_, key_, value_, mask_=None, pos_emb_=torch.tensor(1.0)):
        """计算'缩放点积注意力'。
        :return torch.Tensor: 经过注意力和变换后的 `value` (batch, time1, d_model)
             由 query 点积 key 的注意力加权 (batch, head, time1, time2)
        """
        q_, k_, v_=project_qkv(self.linear_q, self.linear_k, self.linear_v, query_, key_, value_, self.h, self.d_k)

        if self.rel_enc:
            q_ = q_.transpose(1, 2)
            n_batch_pos_=pos_emb_.size(0)
            p_=self.linear_pos(pos_emb_.to(query_.dtype)).view(n_batch_pos_, -1, self.h, self.d_k)
            p_ = p_.transpose(1, 2)

            q_with_bias_u_ = (q_ + self.pos_bias_u).transpose(1, 2) # (batch, head, time1, d_k)

            q_with_bias_v_ = (q_ + self.pos_bias_v).transpose(1, 2) # (batch, head, time1, d_k)

            # 计算注意力分数
            # 首先计算矩阵 a 和矩阵 c

            matrix_ac_ = torch.matmul(q_with_bias_u_, k_.transpose(-2, -1)) # (batch, head, time1, time2)
            # 计算矩阵 b 和矩阵 d
            
            matrix_bd_ = torch.matmul(q_with_bias_v_, p_.transpose(-2, -1)) # (batch, head, time1, time2)
            # 在语音识别中移除 rel_shift，因为它无用，
            # 并且在流式处理中需要特别注意。

            scores_ = (matrix_ac_ + matrix_bd_) / math.sqrt(self.d_k)
        else:
            scores_ = torch.matmul(q_, k_.transpose(-2, -1) ) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        attn__ = apply_attention_mask(scores_, mask_, self.min_value)

        p_attn_ = self.dropout(attn__)
        X_OK_ = merge_attention_output(p_attn_, v_, self.h, self.d_k)
        return self.linear_out(X_OK_)

    def infer(self, query_, key_, value_, pos_emb_, buffer_, buffer_index_, buffer_out_):
        """流式推断方法。"""
        n_batch_ = query_.size(0)

        q_, k_, v_ = project_qkv(self.linear_q, self.linear_k, self.linear_v, query_, key_, value_, self.h, self.d_k)

        # 从缓冲区获取并更新 key 和 value
        key_value_buffer_ = buffer_[buffer_index_]
        if buffer_[buffer_index_] is None:
            buffer_[buffer_index_] = [None, None]
            key_buffer_ = k_
            value_buffer_ = v_
        else:
            key_buffer_ = torch.cat([key_value_buffer_[0], k_], dim=2)
            value_buffer_ = torch.cat([key_value_buffer_[1], v_], dim=2)
        # 保持缓冲区大小固定
        if key_buffer_.size(2) > self.buffersize:
            buffer_[buffer_index_][0] = key_buffer_[:, :, -self.buffersize:, :]
            buffer_[buffer_index_][1] = value_buffer_[:, :, -self.buffersize:, :]
        else:
            buffer_[buffer_index_] = [key_buffer_, value_buffer_]
        buffer_index_ += 1
        
        if self.rel_enc:
            # 相对位置编码的注意力计算
            q_ = q_.transpose(1, 2)
            n_batch_pos_ = pos_emb_.size(0)
            p_ = self.linear_pos(pos_emb_).view(n_batch_pos_, -1, self.h, self.d_k)
            p_ = p_.transpose(1, 2)
            q_with_bias_u_ = (q_ + self.pos_bias_u).transpose(1, 2) # (batch, head, time1, d_k)
            q_with_bias_v_ = (q_ + self.pos_bias_v).transpose(1, 2) # (batch, head, time1, d_k)
            matrix_ac_ = torch.matmul(q_with_bias_u_, key_buffer_.transpose(-2, -1)) # (batch, head, time1, time2)
            matrix_bd_ = torch.matmul(q_with_bias_v_, p_.transpose(-2, -1)) # (batch, head, time1, time2)
            scores_ = (matrix_ac_ + matrix_bd_) / math.sqrt(self.d_k)
        else:
            # 标准注意力计算
            scores_ = torch.matmul(q_, key_buffer_.transpose(-2, -1) ) / math.sqrt(self.d_k)

        attn__ = torch.softmax(scores_, dim=-1)

        X_OK_ = merge_attention_output(attn__, value_buffer_, self.h, self.d_k)
        return self.linear_out(X_OK_), buffer_, buffer_index_, buffer_out_
