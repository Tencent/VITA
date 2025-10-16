import math
import pdb

import numpy
import torch
from torch.nn import Module, Linear, Dropout, Parameter, Identity, ConstantPad1d

# Reuse TTS implementations and shared helpers to avoid duplication.
from vl_load.vita.model.vita_tts.encoder.attention import (
    PositionalEncoding as TTSPositionalEncoding,
    PositionwiseFeedForward as TTSPositionwiseFeedForward,
    MultiLayeredConv1d as TTSMultiLayeredConv1d,
    Conv1dLinear as TTSConv1dLinear,
    project_qkv,
    apply_attention_mask,
    merge_attention_output,
)


# ============================================================================
# Attention 工具函数
# ============================================================================

def create_causal_mask(seq_len, device=None):
    """创建因果注意力mask，用于自回归模型。
    
    生成一个下三角矩阵，使得每个位置只能关注到自己及之前的位置。
    
    Args:
        seq_len (int): 序列长度
        device (torch.device, optional): 设备类型
    
    Returns:
        torch.Tensor: 因果mask，形状为 (seq_len, seq_len)，dtype为bool
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def create_padding_mask(seq_lengths, max_len):
    """根据序列长度创建padding mask。
    
    Args:
        seq_lengths (torch.Tensor): 每个样本的实际长度，形状为 (batch_size,)
        max_len (int): 序列的最大长度
    
    Returns:
        torch.Tensor: padding mask，形状为 (batch_size, max_len)，dtype为bool
                     True表示有效位置，False表示padding位置
    """
    batch_size = seq_lengths.size(0)
    mask = torch.arange(max_len, device=seq_lengths.device).expand(batch_size, max_len) < seq_lengths.unsqueeze(1)
    return mask


def scaled_dot_product_attention_fn(q, k, v, mask=None, dropout_fn=None):
    """独立的scaled dot product attention实现。
    
    实现标准的缩放点积注意力机制，可以作为standalone函数使用。
    
    Args:
        q (torch.Tensor): Query张量，形状为 (batch, ..., seq_len_q, d_k)
        k (torch.Tensor): Key张量，形状为 (batch, ..., seq_len_k, d_k)
        v (torch.Tensor): Value张量，形状为 (batch, ..., seq_len_k, d_v)
        mask (torch.Tensor, optional): 注意力mask
        dropout_fn (callable, optional): Dropout函数
    
    Returns:
        torch.Tensor: 注意力输出，形状为 (batch, ..., seq_len_q, d_v)
        torch.Tensor: 注意力权重，形状为 (batch, ..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    if dropout_fn is not None:
        attn_weights = dropout_fn(attn_weights)
    
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


def additive_attention(query, keys, values, w_query, w_keys, v_attn):
    """加性注意力机制（Bahdanau风格）。
    
    使用加性（而非点积）方式计算注意力分数，常用于seq2seq模型。
    
    Args:
        query (torch.Tensor): Query张量，形状为 (batch, d_query)
        keys (torch.Tensor): Keys张量，形状为 (batch, seq_len, d_key)
        values (torch.Tensor): Values张量，形状为 (batch, seq_len, d_value)
        w_query (torch.Tensor): Query权重矩阵，形状为 (d_query, d_attn)
        w_keys (torch.Tensor): Keys权重矩阵，形状为 (d_key, d_attn)
        v_attn (torch.Tensor): 注意力向量，形状为 (d_attn,)
    
    Returns:
        torch.Tensor: 上下文向量，形状为 (batch, d_value)
        torch.Tensor: 注意力权重，形状为 (batch, seq_len)
    """
    # 投影query和keys到相同的空间
    query_proj = torch.matmul(query, w_query).unsqueeze(1)  # (batch, 1, d_attn)
    keys_proj = torch.matmul(keys, w_keys)  # (batch, seq_len, d_attn)
    
    # 计算加性注意力分数
    scores = torch.matmul(torch.tanh(query_proj + keys_proj), v_attn)  # (batch, seq_len)
    attn_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
    
    # 加权求和values
    context = torch.matmul(attn_weights.unsqueeze(1), values).squeeze(1)  # (batch, d_value)
    return context, attn_weights


def compute_attention_bias(attention_scores, bias_type='alibi', **kwargs):
    """为attention scores添加各种类型的bias。
    
    支持多种位置偏置方案，如ALiBi、相对位置bias等。
    
    Args:
        attention_scores (torch.Tensor): 原始注意力分数，形状为 (batch, heads, seq_len, seq_len)
        bias_type (str): bias类型，可选 'alibi', 'relative', 'learned'
        **kwargs: 其他参数，如slope（ALiBi斜率）、max_distance（相对位置最大距离）等
    
    Returns:
        torch.Tensor: 添加bias后的注意力分数
    """
    if bias_type == 'alibi':
        # ALiBi: Attention with Linear Biases
        seq_len = attention_scores.size(-1)
        slopes = kwargs.get('slopes', torch.tensor([1.0]))
        
        # 创建距离矩阵
        positions = torch.arange(seq_len, device=attention_scores.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # 应用斜率
        bias = distance.unsqueeze(0).unsqueeze(0) * slopes.view(-1, 1, 1)
        return attention_scores + bias
    
    elif bias_type == 'relative':
        # 简单的相对位置bias
        max_distance = kwargs.get('max_distance', 128)
        bias_table = kwargs.get('bias_table')  # 预定义的bias表
        
        if bias_table is not None:
            seq_len = attention_scores.size(-1)
            positions = torch.arange(seq_len, device=attention_scores.device)
            relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
            relative_pos = torch.clamp(relative_pos, -max_distance, max_distance) + max_distance
            bias = bias_table[relative_pos]
            return attention_scores + bias
        return attention_scores
    
    else:
        return attention_scores


def linear_attention_weights(q, k):
    """线性注意力权重计算（不使用softmax）。
    
    使用kernel方法计算线性复杂度的注意力，避免softmax操作。
    
    Args:
        q (torch.Tensor): Query张量，形状为 (batch, heads, seq_len_q, d_k)
        k (torch.Tensor): Key张量，形状为 (batch, heads, seq_len_k, d_k)
    
    Returns:
        torch.Tensor: 归一化后的注意力权重
    """
    # 使用ELU + 1作为特征映射
    q_prime = torch.nn.functional.elu(q) + 1
    k_prime = torch.nn.functional.elu(k) + 1
    
    # 计算归一化因子
    k_sum = k_prime.sum(dim=-2, keepdim=True)  # (batch, heads, 1, d_k)
    z = 1.0 / (torch.einsum('...nd,...nd->...n', q_prime, k_sum) + 1e-6)  # (batch, heads, seq_len_q)
    
    return q_prime, k_prime, z.unsqueeze(-1)


def local_attention_mask(seq_len, window_size, device=None):
    """创建局部attention mask（仅关注窗口内的token）。
    
    生成一个带状矩阵，使得每个位置只能关注到窗口范围内的其他位置。
    
    Args:
        seq_len (int): 序列长度
        window_size (int): 窗口大小（单侧）
        device (torch.device, optional): 设备类型
    
    Returns:
        torch.Tensor: 局部attention mask，形状为 (seq_len, seq_len)，dtype为bool
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True
    return mask


def multi_query_projection(x, n_heads, n_kv_heads, d_k):
    """Multi-Query Attention的投影辅助函数。
    
    将输入投影为多个query heads但只有少量key/value heads，用于减少KV cache。
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (batch, seq_len, d_model)
        n_heads (int): query heads的数量
        n_kv_heads (int): key/value heads的数量（通常小于n_heads）
        d_k (int): 每个head的维度
    
    Returns:
        torch.Tensor: 投影后的张量，适用于MQA/GQA
    """
    batch_size, seq_len, d_model = x.size()
    
    # 确保n_heads能被n_kv_heads整除
    assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    
    # 重复KV heads以匹配query heads的数量
    repeat_factor = n_heads // n_kv_heads
    
    # 重塑为 (batch, seq_len, n_kv_heads, repeat_factor, d_k)
    x_reshaped = x.view(batch_size, seq_len, n_kv_heads, repeat_factor, d_k)
    
    # 转换为 (batch, n_heads, seq_len, d_k)
    x_output = x_reshaped.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    
    return x_output


# ============================================================================
# 原有类定义
# ============================================================================


class PositionalEncoding(TTSPositionalEncoding):
    pass


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        chunk_size: int,
        left_chunks: int,
        max_len: int = 5000,
    ):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)
        self.chunk_sizes = chunk_size
        self.left_chunk = left_chunks
        self.full_chunk_sizes = (self.left_chunk + 1) * self.chunk_sizes

        self.div_terms = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.max_lens = self.chunk_sizes * (max_len // self.chunk_sizes) - self.full_chunk_sizes

    @torch.jit.export
    def forward(self, x: torch.Tensor, offset: int = 0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)

    @torch.jit.export
    def infer(self, xs, pe_index):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        pe_index = pe_index % self.max_lens
        xs = xs * self.xscale

        pe = torch.zeros(self.full_chunk_sizes, self.d_model)
        position = torch.arange(
            pe_index, pe_index + self.full_chunk_sizes, dtype=torch.float32
        ).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_terms)
        pe[:, 1::2] = torch.cos(position * self.div_terms)
        pos_emb = pe.unsqueeze(0)

        pe_index = pe_index + self.chunk_sizes
        return xs, pos_emb, pe_index


class PositionwiseFeedForward(TTSPositionwiseFeedForward):
    pass


class MultiLayeredConv1d(TTSMultiLayeredConv1d):
    pass


class Conv1dLinear(TTSConv1dLinear):
    pass


class MultiHeadedAttention(Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate, chunk_size, left_chunks, pos_enc_class):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = Linear(n_feat, n_feat)
        self.linear_k = Linear(n_feat, n_feat)
        self.linear_v = Linear(n_feat, n_feat)
        self.linear_out = Linear(n_feat, n_feat)
        self.dropout = Dropout(p=dropout_rate)
        # self.min_value = float(numpy.finfo(torch.tensor(0, dtype=torch.float16).numpy().dtype).min)
        self.min_value = float(torch.finfo(torch.float16).min)
        # chunk par
        if chunk_size > 0 and left_chunks > 0:  # for streaming mode
            self.buffersize = chunk_size * (left_chunks)
            self.left_chunk_size = chunk_size * left_chunks
        else:  # for non-streaming mode
            self.buffersize = 1
            self.left_chunk_size = 1
        self.chunk_size = chunk_size

        # encoding setup
        if pos_enc_class == "rel-enc":
            self.rel_enc = True
            self.linear_pos = Linear(n_feat, n_feat, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = Parameter(torch.Tensor(self.h, self.d_k))
            self.pos_bias_v = Parameter(torch.Tensor(self.h, self.d_k))
            torch.nn.init.xavier_uniform_(self.pos_bias_u)
            torch.nn.init.xavier_uniform_(self.pos_bias_v)
        else:
            self.rel_enc = False
            self.linear_pos = Identity()
            self.pos_bias_u = torch.tensor([0])
            self.pos_bias_v = torch.tensor([0])

        # buffer
        # key_buffer = 1, self.h, self.buffersize, self.d_k
        self.key_buffer_size = 1 * self.h * self.buffersize * self.d_k
        # value_buffer = 1, self.h, self.buffersize, self.d_k
        self.value_buffer_size = 1 * self.h * self.buffersize * self.d_k
        if self.chunk_size > 0:
            # buffer_mask_size = 1, self.h, self.chunk_size, self.buffersize
            self.buffer_mask_size = 1 * self.h * self.chunk_size * self.buffersize
            # self.buffer_mask = torch.ones([1, self.h, self.chunk_size, self.buffersize], dtype=torch.bool)
        else:
            self.buffer_mask = torch.ones([1, self.h, 1, 1], dtype=torch.bool)

    @torch.jit.unused
    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    @torch.jit.export
    def forward(self, query, key, value, mask=None, pos_emb=torch.tensor(1.0)):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Tensor) -> Tensor
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = project_qkv(self.linear_q, self.linear_k, self.linear_v, query, key, value, self.h, self.d_k)

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb.to(query.dtype)).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            # matrix_bd = self.rel_shift(matrix_bd)
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, head, time1, time2)

        attn = apply_attention_mask(scores, mask, self.min_value)
        p_attn = self.dropout(attn)

        x = merge_attention_output(p_attn, v, self.h, self.d_k)
        return self.linear_out(x)  # (batch, time1, d_model)

    @torch.jit.export
    def infer(self, query, key, value, pos_emb, buffer, buffer_index, buffer_out):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        n_batch = query.size(0)

        q, k, v = project_qkv(self.linear_q, self.linear_k, self.linear_v, query, key, value, self.h, self.d_k)

        key_value_buffer = buffer[
            buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
        ].reshape([1, self.h, self.buffersize * 2, self.d_k])
        key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
        value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
        buffer_out.append(
            torch.cat(
                [key_buffer[:, :, self.chunk_size :, :], value_buffer[:, :, self.chunk_size :, :]],
                dim=2,
            ).reshape(-1)
        )
        buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        if self.rel_enc:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)
            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, key_buffer.transpose(-2, -1))
            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # Remove rel_shift since it is useless in speech recognition,
            # and it requires special attention for streaming.
            # matrix_bd = self.rel_shift(matrix_bd)
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        else:
            scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
                self.d_k
            )  # (batch, head, len_q, buffersize)

        attn = torch.softmax(scores, dim=-1)

        x = merge_attention_output(attn, value_buffer, self.h, self.d_k)
        return self.linear_out(x), buffer, buffer_index, buffer_out  # (batch, time1, d_model)

    @torch.jit.export
    def infer_mask(self, query, key, value, mask, buffer, buffer_index, buffer_out, is_static):
        n_batch = query.size(0)

        q, k, v = project_qkv(self.linear_q, self.linear_k, self.linear_v, query, key, value, self.h, self.d_k)

        if is_static:
            key_buffer = k
            value_buffer = v
        else:
            key_value_buffer = buffer[
                buffer_index : buffer_index + self.key_buffer_size + self.value_buffer_size
            ].reshape([1, self.h, self.buffersize * 2, self.d_k])
            key_buffer = torch.cat([key_value_buffer[:, :, : self.buffersize, :], k], dim=2)
            value_buffer = torch.cat([key_value_buffer[:, :, self.buffersize :, :], v], dim=2)
            buffer_out.append(
                torch.cat(
                    [
                        key_buffer[:, :, self.chunk_size :, :],
                        value_buffer[:, :, self.chunk_size :, :],
                    ],
                    dim=2,
                ).reshape(-1)
            )
            buffer_index = buffer_index + self.key_buffer_size + self.value_buffer_size

        scores = torch.matmul(q, key_buffer.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, head, len_q, buffersize)

        attn = apply_attention_mask(scores, mask, self.min_value)

        x = merge_attention_output(attn, value_buffer, self.h, self.d_k)
        return self.linear_out(x), buffer_index, buffer_out  # (batch, time1, d_model)


class SoftAttention(Module):
    def __init__(self, in_dim, hidden_dim):
        super(SoftAttention, self).__init__()
        self.q = Parameter(torch.rand([hidden_dim]), requires_grad=True)
        self.wb = Linear(in_dim, hidden_dim)
        self.min_value = float(numpy.finfo(torch.tensor(0, dtype=torch.float32).numpy().dtype).min)
        # buffer
        self.window_size = 50
        self.buffer_in = torch.zeros([1, self.window_size, in_dim], dtype=torch.float32)
        self.buffer = torch.zeros([1, self.window_size], dtype=torch.float32)
        self.buffer[:, :] = float(
            numpy.finfo(torch.tensor(0, dtype=torch.float32).numpy().dtype).min
        )

    @torch.jit.unused
    def forward(self, x, mask=None):
        hidden = torch.tanh(self.wb(x))  # B T D
        hidden = torch.einsum("btd,d->bt", hidden, self.q)
        score = torch.softmax(hidden, dim=-1)  # B T
        if mask is not None:
            score = score.masked_fill(mask, 0.0)
        output = torch.einsum("bt,btd->bd", score, x)
        return output

    @torch.jit.export
    def infer(self, x):
        # type: (Tensor) -> Tensor
        hidden = torch.tanh(self.wb(x))  # B T D
        hidden = torch.einsum("btd,d->bt", hidden, self.q)
        size = hidden.shape[1]
        output = torch.zeros([size, x.shape[-1]])
        for i in range(size):
            self.buffer = torch.cat([self.buffer, hidden[:, i : i + 1]], dim=-1)
            self.buffer = self.buffer[:, 1:]
            score = torch.softmax(self.buffer, dim=-1)  # B T
            self.buffer_in = torch.cat([self.buffer_in, x[:, i : i + 1, :]], dim=1)
            self.buffer_in = self.buffer_in[:, 1:]
            output[i : i + 1] = torch.einsum("bt,btd->bd", score, self.buffer_in)
        return output
