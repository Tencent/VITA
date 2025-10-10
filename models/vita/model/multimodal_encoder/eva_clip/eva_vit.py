"""
# Adapted from https://github.com/baaivision/EVA/tree/master/EVA-CLIP
"""


import logging


# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from dataclasses import dataclass
from functools import partial
from math import pi
from typing import Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn as nn


import xformers.ops as xops


# 
def broadcat(tensors, dim=-1):
    """
    一个自定义的拼接函数，可以在拼接前自动广播张量以匹配彼此的形状。
    这对于处理具有不同但可广播维度的张量非常有用。

    Args:
        tensors (list[torch.Tensor]): 要拼接的张量列表。
        dim (int): 拼接的维度。

    Returns:
        torch.Tensor: 拼接后的张量。
    """
    num_tensors = len(tensors)
    # 检查所有张量是否具有相同的维度数
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    
    # 处理负数维度索引
    dim = (dim + shape_len) if dim < 0 else dim
    
    # 获取每个张量在各个维度上的大小
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    
    # 找出除了拼接维度之外可以扩展的维度
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    # 确保在这些可扩展维度上，每个张量的大小要么是1，要么是相同的值
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    
    # 计算扩展后的目标维度大小（取最大值）
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    
    # 生成每个张量需要扩展到的最终形状
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    
    # 扩展每个张量到目标形状
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    
    # 沿指定维度拼接张量
    return torch.cat(tensors, dim=dim)




def rotate_half(x):
    """
    旋转输入张量的一半维度。这是旋转位置编码 (RoPE) 的核心操作。
    将特征维度分成两半，交换它们并取反其中一半。

    Args:
        x (torch.Tensor): 输入张量，形状为 `... (d r)`。

    Returns:
        torch.Tensor: 经过旋转操作后的张量。
    """
    # 将最后一个维度拆分为 (d, 2)
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    # 分离出 x1 和 x2
    x1, x2 = x.unbind(dim=-1)
    # 交换并取反其中一半：(-x2, x1)
    x = torch.stack((-x2, x1), dim=-1)
    # 合并回原来的形状
    return rearrange(x, "... d r -> ... (d r)")




class VisionRotaryEmbeddingFast(nn.Module):
    """
    视觉任务的快速旋转位置编码 (RoPE) 实现。
    RoPE 是一种将位置信息注入到 Transformer 注意力机制中的方法，
    它通过旋转查询和键向量来实现，而不是使用绝对或相对位置嵌入。
    """
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        patch_dropout=0.0,
    ):
        super().__init__()
        # 根据 freqs_for 参数计算频率
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            # 语言模型中常用的频率计算方式
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            # 像素空间中使用的频率
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        # 如果没有提供微调时的序列长度，则使用预训练时的长度
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        # 创建一个时间步长序列，用于插值或外推位置
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # 计算每个时间步长的频率
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        # 重复频率以匹配特征维度（因为旋转是成对进行的）
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        # 使用 broadcat 创建 2D 位置的频率（一个用于行，一个用于列）
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        # 预计算余弦和正弦值以提高效率
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        # 将预计算的值注册为 buffer，这样它们会被移动到正确的设备上，但不会被视为模型参数
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, patch_indices_keep=None):
        """
        应用旋转位置编码。

        Args:
            t (torch.Tensor): 输入张量 (查询或键)。
            patch_indices_keep (torch.Tensor, optional): 如果使用了 PatchDropout，
                                                        则为保留的 patch 的索引。默认为 None。

        Returns:
            torch.Tensor: 应用了 RoPE 的张量。
        """
        # 如果使用了 PatchDropout，需要根据保留的索引来选择相应的位置编码
        if patch_indices_keep is not None:
            batch = t.size()[0]
            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            # 扩展频率以匹配批次和序列维度
            freqs_cos = repeat(self.freqs_cos, "i j -> n i m j", n=t.shape[0], m=t.shape[1])
            freqs_sin = repeat(self.freqs_sin, "i j -> n i m j", n=t.shape[0], m=t.shape[1])

            # 根据保留的 patch 索引选择频率
            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, "n i m j -> n m i j")
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, "n i m j -> n m i j")

            # 应用 RoPE: t' = t * cos(theta) + rotate_half(t) * sin(theta)
            return t * freqs_cos + rotate_half(t) * freqs_sin

        # 如果没有 PatchDropout，直接应用预计算的频率
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin




class LayerNorm(nn.LayerNorm):
    """
    torch.nn.LayerNorm 的子类，确保输出的数据类型与输入相同。
    这在混合精度训练中很有用，可以防止不必要的类型转换。
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # 使用 F.layer_norm 进行计算，这通常在 float32 下进行
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 将结果转换回原始数据类型
        return x.to(orig_type)




class PatchDropout(nn.Module):
    """
    PatchDropout 实现，来自论文 "https://arxiv.org/abs/2212.00794"。
    在训练期间随机丢弃一部分图像 patch，以提高模型的泛化能力。
    """
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # 是否排除 CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        # 如果不在训练模式或丢弃概率为0，则不执行任何操作
        if not self.training or self.prob == 0.0:
            return x

        # 如果需要，分离 CLS token
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        # 创建批次索引，用于高级索引
        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        # 计算需要保留的 patch 数量
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        # 生成随机数并选择要保留的 patch 的索引
        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        # 根据索引选择保留的 patch
        x = x[batch_indices, patch_indices_keep]

        # 如果之前分离了 CLS token，现在将其重新拼接回去
        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        # 如果启用了 RoPE，则需要返回保留的索引，以便 RoPE 模块可以选择正确的位置编码
        if self.training and os.getenv("RoPE") == "1":
            return x, patch_indices_keep

        return x




try:
    # 尝试从 timm.models.layers 导入
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    # 如果失败，尝试从 timm.layers 导入（适用于较新版本的 timm）
    from timm.layers import drop_path, to_2tuple, trunc_normal_


# 根据环境变量选择 checkpointing 的实现
if os.getenv("ENV_TYPE") == "deepspeed":
    try:
        # 优先使用 deepspeed 的 checkpointing
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        # 如果 deepspeed 不可用，回退到 torch 的实现
        from torch.utils.checkpoint import checkpoint
else:
    # 默认使用 torch 的 checkpointing
    from torch.utils.checkpoint import checkpoint




class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    随机深度，一种正则化技术，在训练期间随机“丢弃”整个残差块。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 调用 timm 中的 drop_path 实现
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)




class Mlp(nn.Module):
    """
    标准的多层感知机 (MLP) 或前馈网络 (FFN) 模块。
    结构为: Linear -> Activation -> (LayerNorm) -> Linear -> Dropout
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        subln=False, # 是否在激活函数后添加 LayerNorm
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        # subln (Sub-LayerNorm) 选项，在 FFN 内部增加一个归一化层
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x) # 原始 BERT 实现中没有在这里加 dropout
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数的前馈网络实现。
    相比标准 MLP，它使用门控机制，通常能获得更好的性能。
    结构为: (w1(x) * act(w2(x))) -> (LayerNorm) -> w3 -> Dropout
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU, # Swish 激活函数
        drop=0.0,
        norm_layer=nn.LayerNorm,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 两个线性层用于门控
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        # 门控机制：逐元素相乘
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x




class Attention(nn.Module):
    """
    多头注意力模块。
    支持标准自注意力、xformers 高效注意力、旋转位置编码 (RoPE) 和相对位置偏置。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
        xattn=False, # 是否使用 xformers 的 memory_efficient_attention
        rope=None, # 旋转位置编码模块
        subln=False, # 是否使用 Sub-LayerNorm 结构
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # 注意力分数的缩放因子
        self.scale = qk_scale or head_dim**-0.5

        self.subln = subln
        if self.subln:
            # Sub-LN 结构中，Q, K, V 分别有独立的线性投影层
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            # 标准结构中，Q, K, V 合并在一个线性层中
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        # 是否为 Q 和 V 添加偏置项
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # 如果提供了 window_size，则计算并使用相对位置偏置
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            # 相对位置偏置表
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )
            # 特殊处理 CLS token 的相对位置
            # ... (计算相对位置索引)
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        # 在注意力计算后和最终投影前添加的 LayerNorm (subln)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # 最终的线性投影层
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        # --- 1. 计算 Q, K, V ---
        if self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
            # 调整形状以进行多头注意力计算: (B, N, C) -> (B, num_heads, N, head_dim)
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                # K 没有偏置
                qkv_bias = torch.cat(
                    (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
                )
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            # 调整形状并分离 Q, K, V
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 2. 应用旋转位置编码 (RoPE) ---
        if self.rope:
            # RoPE 只应用于 patch token，不应用于 CLS token
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        # --- 3. 计算注意力 ---
        if self.xattn:
            # 使用 xformers 的内存高效注意力实现
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = xops.memory_efficient_attention(q, k, v, p=self.xattn_drop, scale=self.scale)
            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            # 标准的点积注意力
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            # 添加相对位置偏置
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1,
                    -1,
                )
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            # 应用注意力掩码
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

            # 计算 softmax 和 dropout
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # --- 4. 计算输出 ---
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x




class Block(nn.Module):
    """
    一个标准的 Transformer Block。
    包含一个多头注意力模块和一个 MLP 模块。
    支持 pre-norm 和 post-norm 结构，以及 LayerScale。
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None, # LayerScale 的初始值
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        xattn=False,
        rope=None,
        postnorm=False, # 是否使用 post-norm 结构
        subln=False,
        naiveswiglu=False, # 是否使用 SwiGLU FFN
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
            xattn=xattn,
            rope=rope,
            subln=subln,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 根据 naiveswiglu 选择 MLP 类型
        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                subln=subln,
                drop=drop,
            )

        # LayerScale: 为每个残差连接引入一个可学习的缩放因子
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        # 如果没有 LayerScale
        if self.gamma_1 is None:
            if self.postnorm:
                # Post-norm: x -> Layer -> Add -> Norm
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                )
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                # Pre-norm: x -> Norm -> Layer -> Add
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                )
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 如果有 LayerScale
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1
                    * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                )
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1
                    * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                )
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x




class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    将输入的图像通过一个卷积层转换为一系列 patch embedding。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 计算 patch 的数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用一个卷积层实现 patch embedding，步长和核大小都等于 patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # 检查输入图像尺寸是否与模型配置匹配
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 卷积 -> 展平 -> 维度转换
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x




class RelativePositionBias(nn.Module):
    """
    计算相对位置偏置的模块，可用于注意力分数。
    这是一个独立的模块，可以被多个注意力层共享。
    """
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        # 计算相对距离的数量
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 可学习的相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )
        # ... (计算相对位置索引，逻辑与 Attention 类中相同)
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        # 从偏置表中查找对应索引的偏置值
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        # 调整形状以匹配注意力分数的形状 (nH, N, N)
        return relative_position_bias.permute(2, 0, 1).contiguous()




class EVAVisionTransformer(nn.Module):
    """
    EVA 视觉 Transformer 模型。
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        patch_dropout=0.0,
        use_abs_pos_emb=True, # 是否使用绝对位置嵌入
        use_rel_pos_bias=False, # 是否在每个块中使用相对位置偏置
        use_shared_rel_pos_bias=False, # 是否共享相对位置偏置
        rope=False, # 是否使用旋转位置编码
        use_mean_pooling=True, # 是否使用平均池化作为最终输出
        init_scale=0.001,
        grad_checkpointing=False, # 是否使用梯度检查点以节省内存
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
    ):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # 1. Patch Embedding 层
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. CLS Token 和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. 相对位置偏置
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        # 4. 旋转位置编码 (RoPE)
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope = None

        self.naiveswiglu = naiveswiglu

        # 5. Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                    xattn=xattn,
                    rope=self.rope,
                    postnorm=postnorm,
                    subln=subln,
                    naiveswiglu=naiveswiglu,
                )
                for i in range(depth)
            ]
        )
        
        # 6. 输出层
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # 7. Patch Dropout
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        """
        根据层 ID 重新缩放某些层的权重，这是一种初始化技巧。
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        # 获取模型用于计算的数据类型
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        # 初始化线性层和 LayerNorm 层的权重
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        # 冻结模型所有参数
        assert unlocked_groups == 0, "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # 指定哪些参数不应进行权重衰减
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):
        # 1. Patch Embedding
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # 2. 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. 添加位置嵌入
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. 应用 Patch Dropout
        # 如果启用了 RoPE，PatchDropout 会返回保留的索引
        if os.getenv("RoPE") == "1":
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                # 将保留的索引传递给 RoPE 模块
                self.rope.forward = partial(
                    self.rope.forward, patch_indices_keep=patch_indices_keep
                )
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        # 5. 通过 Transformer Blocks
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            # 在最后一个块之前停止，以便可以返回所有特征
            if i == len(self.blocks) - 1 and return_all_features:
                continue
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        # 6. 最终输出处理
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                # 平均池化
                return self.fc_norm(x.mean(1))
            else:
                # 返回 CLS token 的输出
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x




def load_state_dict(
    checkpoint_path: str,
    map_location: str = "cpu",
    model_key: str = "model|module|state_dict",
    is_openai: bool = False,
    skip_list: list = [],
):
    """
    从检查点文件加载状态字典。
    处理不同的检查点格式（例如 OpenAI、DeepSpeed、普通 PyTorch）。
    """
    if is_openai:
        # 加载 OpenAI 发布的 JIT 模型
        model = torch.jit.load(checkpoint_path, map_location="cpu").eval()
        state_dict = model.state_dict()
        # 移除不需要的键
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        # 加载标准的 PyTorch 检查点
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        # 在字典中查找模型的状态字典
        for mk in model_key.split("|"):
            if isinstance(checkpoint, dict) and mk in checkpoint:
                state_dict = checkpoint[mk]
                break
            else:
                state_dict = checkpoint
        # 如果键以 "module." 开头（通常来自 DDP 训练），则移除该前缀
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 跳过指定的键
    for k in skip_list:
        if k in list(state_dict.keys()):
            logging.info(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]

    # 如果启用了 RoPE，则从检查点中删除预训练的 RoPE 频率，因为它们是动态计算的
    if os.getenv("RoPE") == "1":
        for k in list(state_dict.keys()):
            if "freqs_cos" in k or "freqs_sin" in k:
                del state_dict[k]
    return state_dict




def load_clip_visual_state_dict(
    checkpoint_path: str, map_location: str = "cpu", is_openai: bool = False, skip_list: list = []
):
    """
    专门为 CLIP 的视觉部分加载状态字典。
    它会过滤掉所有非视觉部分的键，并调整键名。
    """
    state_dict = load_state_dict(
        checkpoint_path, map_location=map_location, is_openai=is_openai, skip_list=skip_list
    )

    # 删除所有不以 "visual." 开头的键
    for k in list(state_dict.keys()):
        if not k.startswith("visual."):
            del state_dict[k]
    # 将 "visual." 前缀从剩余的键中移除
    for k in list(state_dict.keys()):
        if k.startswith("visual."):
            new_k = k[7:]
            state_dict[new_k] = state_dict[k]
            del state_dict[k]
    return state_dict




try:
    # 尝试导入 Apex 的 FusedLayerNorm 以获得更好的性能
    from apex.normalization import FusedLayerNorm
except:
    # 如果 Apex 不可用，则回退到自定义的 LayerNorm
    FusedLayerNorm = LayerNorm
    print(
        '''Please build and install Nvidia apex package with option 
        '--cuda_ext' according to https://github.com/NVIDIA/apex#from-source .'''
    )




@dataclass
class CLIPVisionCfg:
    """
    用于配置 CLIP 视觉塔的数据类。
    """
    layers: Union[Tuple[int, int, int, int], int] = 12 # Transformer 层数
    width: int = 768 # 嵌入维度
    head_width: int = 64 # 每个注意力头的维度
    mlp_ratio: float = 4.0 # MLP 隐藏层维度与嵌入维度的比率
    patch_size: int = 16 # Patch 大小
    image_size: Union[Tuple[int, int], int] = 224 # 输入图像大小
    ls_init_value: Optional[float] = None  # LayerScale 初始值
    patch_dropout: float = 0.0 # PatchDropout 概率
    global_average_pool: bool = False # 是否使用全局平均池化
    drop_path_rate: Optional[float] = None  # 随机深度概率
    timm_model_name: str = None  # (未使用)
    timm_model_pretrained: bool = False  # (未使用)
    timm_pool: str = "avg"  # (未使用)
    timm_proj: str = "linear"  # (未使用)
    timm_proj_bias: bool = False  # (未使用)
    eva_model_name: str = None  # EVA 模型名称
    qkv_bias: bool = True # QKV 投影是否使用偏置
    fusedLN: bool = False # 是否使用 FusedLayerNorm
    xattn: bool = False # 是否使用 xformers 注意力
    postnorm: bool = False # 是否使用 post-norm
    rope: bool = False # 是否使用 RoPE
    pt_hw_seq_len: int = 16  # 预训练时的 patch 序列边长 (例如 224/14=16)
    intp_freq: bool = False # 是否对 RoPE 频率进行插值
    naiveswiglu: bool = False # 是否使用 SwiGLU
    subln: bool = False # 是否使用 Sub-LayerNorm




def _build_vision_tower(vision_tower_path: str, embed_dim: int, vision_cfg: CLIPVisionCfg):
    """
    构建视觉塔模型的工厂函数。
    """
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm

        # 实例化 EVAVisionTransformer 模型
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(FusedLayerNorm, eps=1e-6)
            if vision_cfg.fusedLN
            else partial(norm_layer, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )

        # 加载预训练权重
        state_dict = load_clip_visual_state_dict(vision_tower_path)
        incompatible_keys = visual.load_state_dict(state_dict, strict=False)
        print("EVA-CLIP incompatible_keys:", incompatible_keys)

    return visual




class Eva2LargePlusEncoder(nn.Module):
    """
    一个封装了特定配置（EVA2-Large-Plus）的视觉编码器。
    """
    def __init__(self, vision_tower_path):
        super(Eva2LargePlusEncoder, self).__init__()
        # 定义 EVA2-Large-Plus 模型的特定配置
        self.config = {
            "embed_dim": 768, # 输出嵌入维度
            "vision_cfg": {
                "image_size": 336,
                "layers": 24,
                "width": 1024,
                "drop_path_rate": 0,
                "head_width": 64,
                "mlp_ratio": 2.6667,
                "patch_size": 14,
                "eva_model_name": "eva-clip-l-14-336",
                "xattn": True,
                "fusedLN": True,
                "rope": True,
                "pt_hw_seq_len": 16,
                "intp_freq": True,
                "naiveswiglu": True,
                "subln": True,
            },
        }

        self.config["vision_tower_path"] = vision_tower_path
        # 使用工厂函数构建模型
        self.model = _build_vision_tower(**self.config)

    def forward(self, image, **kwargs):
        """
        前向传播，返回除 CLS token 外的所有 patch 特征。
        """
        # 获取所有特征，并丢弃第一个 CLS token
        encode = self.model(image, return_all_features=True)[:, 1:, :]
        return encode

    @property
    def dtype(self):
        # 获取模型的数据类型
        return list(self.parameters())[-1].dtype

    @property
    def device(self):
        # 获取模型所在的设备
        return list(self.parameters())[-1].device
