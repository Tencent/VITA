# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# 从 timm 库导入 DropPath，用于实现随机深度 (Stochastic Depth)
from timm.models.layers import DropPath

# 导入模型的配置文件类
from .configuration_intern_vit import InternVisionConfig

# 尝试导入 FlashAttention，这是一个高效的注意力实现，可以加速计算并减少内存占用
try:
    from .flash_attention import FlashAttention

    has_flash_attn = True
except:
    print("FlashAttention is not installed.")
    has_flash_attn = False


# 获取一个日志记录器实例
logger = logging.get_logger(__name__)


class InternRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization 的自定义实现。
    RMSNorm 是 LayerNorm 的一种变体，计算上更简单高效。
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 可学习的缩放参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 一个小的常数，防止除以零
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # 为了计算稳定性，将输入转换为 float32
        hidden_states = hidden_states.to(torch.float32)
        # 计算均方值 (mean of squares)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 归一化：输入除以 (均方根 + epsilon)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 应用可学习的缩放参数，并将数据类型转换回原始类型
        return self.weight * hidden_states.to(input_dtype)


# 尝试使用 NVIDIA Apex 库中更快的 FusedRMSNorm 实现
try:
    from apex.normalization import FusedRMSNorm

    # 如果导入成功，则用 FusedRMSNorm 覆盖自定义的 InternRMSNorm
    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm")
except ImportError:
    # 如果 Apex 未安装，则继续使用我们自定义的 InternRMSNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to InternRMSNorm")
    pass


# 创建一个字典，将字符串名称映射到对应的归一化层类
NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    """
    将图像转换为 patch 嵌入和位置嵌入的序列。
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # [CLS] token 的可学习嵌入，用于整个图像的表征
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        # 使用一个卷积层将图像分割成 patch 并进行线性投影
        # 这是一个高效的实现方式，等价于先分块再用 nn.Linear
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # 计算 patch 的数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 总的位置数量 = patch 数量 + 1个 [CLS] token
        self.num_positions = self.num_patches + 1

        # 可学习的位置嵌入
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        """
        通过双三次插值动态调整位置嵌入的大小，以适应不同分辨率的输入图像。
        """
        target_dtype = pos_embed.dtype
        # 将位置嵌入从 (1, N, C) 变形为 (1, C, H_orig, W_orig) 的 2D 网格形式
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        # 使用 F.interpolate 进行双三次插值，调整到目标 H, W
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        # 通过卷积层得到 patch 嵌入
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [B, C, H, W]
        batch_size, _, height, width = patch_embeds.shape
        # 将 H, W 维度展平，并交换维度顺序，得到 (B, N, C) 的序列形式
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # 扩展 [CLS] token 嵌入以匹配批次大小
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 将 [CLS] token 和 patch 嵌入拼接在一起
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 动态调整位置嵌入大小并与 [CLS] token 的位置嵌入拼接
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        # 将 patch 嵌入和位置嵌入相加
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """多头自注意力机制模块"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print("Warning: Flash Attention is not available, use_flash_attn is set to False.")
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 注意力分数的缩放因子
        self.scale = self.head_dim**-0.5
        # 一个线性层同时计算 Q, K, V，以提高效率
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        # 是否对 Query 和 Key 进行归一化
        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            # 如果使用 FlashAttention，实例化 FlashAttention 层
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        # 最终的输出投影层
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        """标准的、非优化的自注意力实现。"""
        B, N, C = x.shape
        # 计算 Q, K, V 并重塑为多头形式 (3, B, num_heads, N, head_dim)
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        # 分离 Q, K, V
        q, k, v = qkv.unbind(0)

        # 如果配置了 QK 归一化，则对 Q 和 K 进行归一化
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        # 计算注意力分数：(Q * scale) @ K.T
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 将注意力分数应用于 V，并重塑回 (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        """使用 FlashAttention 库的高效实现。"""
        # 计算 Q, K, V
        qkv = self.qkv(x)
        # 使用 einops 重排张量以适应 FlashAttention 的输入格式
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        # 如果配置了 QK 归一化
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        # 调用 FlashAttention
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        # 将输出重塑并应用投影层
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 根据配置选择使用 FlashAttention 还是朴素实现
        x = (
            self._naive_attn(hidden_states)
            if not self.use_flash_attn
            else self._flash_attn(hidden_states)
        )
        return x


class InternMLP(nn.Module):
    """
    Transformer 中的前馈网络 (Feed-Forward Network) 部分。
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # 从 transformers 库中获取指定的激活函数
        self.act = ACT2FN[config.hidden_act]
        # 第一个线性层，将维度从 hidden_size 扩展到 intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个线性层，将维度从 intermediate_size 压缩回 hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    """
    一个完整的 Transformer Encoder 层，包含自注意力、MLP、层归一化和残差连接。
    """
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        # 注意力块之前的层归一化 (Pre-Norm)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        # MLP 块之前的层归一化 (Pre-Norm)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        # LayerScale: 可学习的参数，用于缩放残差分支的输出
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        # DropPath (随机深度): 以一定概率随机丢弃整个残差分支
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        前向传播遵循 Pre-Norm 结构。
        """
        # 第一个残差块：注意力
        # hidden_states = hidden_states + DropPath(LayerScale(Attention(Norm(hidden_states))))
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states)) * self.ls1
        )

        # 第二个残差块：MLP
        # hidden_states = hidden_states + DropPath(LayerScale(MLP(Norm(hidden_states))))
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states)) * self.ls2
        )

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    由多个 InternVisionEncoderLayer 堆叠而成的 Transformer Encoder。
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # 计算每一层的 DropPath 概率，从 0 线性增加到 config.drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 创建一个包含所有 Encoder 层的 ModuleList
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)]
        )
        # 启用梯度检查点，以节省训练时的显存
        self.gradient_checkpointing = True

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 用于存储所有中间层的隐藏状态
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        # 依次通过每一个 Encoder 层
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            # 如果在训练模式下且启用了梯度检查点，则使用 checkpoint
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(encoder_layer, hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        # 返回一个包含最后隐藏状态和所有中间隐藏状态的输出对象
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class InternVisionModel(PreTrainedModel):
    """
    完整的 InternViT 模型，集成了 Embedding 和 Encoder。
    """
    main_input_name = "pixel_values"
    config_class = InternVisionConfig
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """
        当模型在不同分辨率的图像上微调时，调整位置嵌入的大小。
        """
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        # 提取 patch 的位置嵌入并重塑为 2D 网格
        pos_emb = (
            pos_emb[:, 1:, :]
            .reshape(1, old_size // patch_size, old_size // patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        # 使用双三次插值调整大小
        pos_emb = F.interpolate(
            pos_emb.float(), size=new_size // patch_size, mode="bicubic", align_corners=False
        )
        # 转换回序列形式
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        # 与 [CLS] token 的位置嵌入拼接
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        # 更新模型参数
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info("Resized position embeddings from {} to {}".format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        device = next(self.encoder.parameters()).device
        pixel_values = pixel_values.to(device)

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 输入可以是原始像素值 (pixel_values) 或已经计算好的嵌入 (pixel_embeds)
        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
 
        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                # 通过 embedding 层将图像转换为 token 序列
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")
        
        # 将 token 序列输入到 Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        # 池化输出：取 [CLS] token 对应的输出作为整个图像的表征
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回一个包含最后隐藏状态、池化输出和所有中间隐藏状态的输出对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
```好的，这是为您添加了详细注释的代码版本。

```python
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# timm is a popular library for computer vision models, used here for DropPath
from timm.models.layers import DropPath

# Import the configuration class for this model
from .configuration_intern_vit import InternVisionConfig

# Attempt to import FlashAttention for optimized attention calculation
try:
    from .flash_attention import FlashAttention

    has_flash_attn = True
except:
    print("FlashAttention is not installed.")
    has_flash_attn = False


# Get a logger instance from the transformers library
logger = logging.get_logger(__name__)


class InternRMSNorm(nn.Module):
    """
    A custom implementation of Root Mean Square Layer Normalization.
    RMSNorm normalizes the hidden states by their root mean square, which can be
    faster than standard LayerNorm as it avoids computing the mean.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # A small value to prevent division by zero
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # Cast to float32 for higher precision during calculation
        hidden_states = hidden_states.to(torch.float32)
        # Calculate the mean of the squares of the hidden states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Normalize the hidden states
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply the learnable scaling parameter and cast back to the original dtype
        return self.weight * hidden_states.to(input_dtype)


# Try to use the highly optimized FusedRMSNorm from NVIDIA's Apex library if available.
# This replaces the custom implementation for better performance.
try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm")
except ImportError:
    # If Apex is not installed, fall back to the custom implementation.
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to InternRMSNorm")
    pass


# A dictionary to map normalization type names to their corresponding classes.
# This allows for easy configuration of the normalization layer used in the model.
NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    """
    This class handles the conversion of input images into a sequence of embedding vectors.
    It performs three main steps:
    1. Patch Embedding: Divides the image into patches and linearly projects them.
    2. Class Token: Prepends a learnable [CLS] token embedding.
    3. Positional Embedding: Adds learnable positional information to the patch and class tokens.
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # A learnable embedding for the [CLS] token, used for image-level classification tasks.
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        # A 2D convolution layer to perform patch embedding. It acts like a linear projection
        # on flattened image patches.
        self.patch_embedding = nn.Conv2d(
            in_channels=3,  # RGB images
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Calculate the total number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Total number of tokens is number of patches + 1 (for the [CLS] token)
        self.num_positions = self.num_patches + 1

        # Learnable positional embeddings for each token.
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        """
        Interpolates positional embeddings to match the input image resolution.
        This allows the model to handle images of different sizes than it was trained on.
        """
        target_dtype = pos_embed.dtype
        # Reshape and permute to (B, C, H, W) format for interpolation
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        # Use bicubic interpolation to resize the positional embedding grid
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        # Apply patch embedding convolution
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [B, C, H, W]
        batch_size, _, height, width = patch_embeds.shape
        # Flatten the spatial dimensions and transpose to (B, num_patches, C)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # Expand the [CLS] token embedding to match the batch size
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # Concatenate the [CLS] token and patch embeddings
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # Interpolate positional embeddings to match the current input size
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :], # [CLS] token position
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width), # Patch positions
            ],
            dim=1,
        )
        # Add positional embeddings to the token embeddings
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """
    Multi-headed self-attention module.
    Can use either a standard implementation or the optimized FlashAttention.
    Also supports optional Query-Key normalization for improved stability.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print("Warning: Flash Attention is not available, use_flash_attn is set to False.")
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Scaling factor for the dot product
        self.scale = self.head_dim**-0.5
        # Linear layer to project input to Q, K, V
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        # Flag for enabling Query-Key normalization
        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.head_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            # Use the optimized FlashAttention implementation
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        # Final linear projection layer
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        """Standard self-attention implementation."""
        B, N, C = x.shape
        # Project to Q, K, V and reshape for multi-head attention
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # Separate Q, K, V

        # Apply QK normalization if enabled
        if self.qk_normalization:
            # Reshape for normalization, apply norm, and reshape back
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        # Compute attention scores: (Q * scale) @ K.T
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention scores to V and reshape back to original dimensions
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        """Self-attention implementation using FlashAttention."""
        # Project to QKV
        qkv = self.qkv(x)
        # Reshape for FlashAttention's expected input format
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        # Apply QK normalization if enabled
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        # Call the FlashAttention kernel
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        # Project the output
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Choose the attention implementation based on configuration and availability
        x = (
            self._naive_attn(hidden_states)
            if not self.use_flash_attn
            else self._flash_attn(hidden_states)
        )
        return x


class InternMLP(nn.Module):
    """
    The feed-forward network (MLP) part of a Transformer layer.
    It consists of two linear layers with a non-linear activation in between.
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # Get the activation function (e.g., GELU) from the config
        self.act = ACT2FN[config.hidden_act]
        # First linear layer (expands the dimension)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second linear layer (projects back to the original dimension)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    """

    A single layer of the Vision Transformer encoder.
    It follows the Pre-LN (Layer Normalization before attention/MLP) architecture.
    Structure: Input -> Norm1 -> Attention -> Residual -> Norm2 -> MLP -> Residual
    """
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        # Self-attention module
        self.attn = InternAttention(config)
        # MLP (feed-forward) module
        self.mlp = InternMLP(config)
        # Normalization layers
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        # LayerScale parameters: learnable scalars that scale the output of sub-layers
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        # DropPath (stochastic depth): randomly drops a residual connection during training
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Forward pass for the encoder layer.
        """
        # Attention block with pre-normalization, LayerScale, DropPath, and residual connection
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states)) * self.ls1
        )

        # MLP block with pre-normalization, LayerScale, DropPath, and residual connection
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states)) * self.ls2
        )

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    The main Transformer encoder, which consists of a stack of InternVisionEncoderLayer layers.
    """
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # Create a schedule for DropPath rates, increasing from 0 to the specified rate
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # Create a list of encoder layers
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)]
        )
        # Enable gradient checkpointing to save memory during training
        self.gradient_checkpointing = True

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        # Iterate through each layer in the encoder
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # Store the hidden states before this layer if requested
                encoder_states = encoder_states + (hidden_states,)
            
            # Use gradient checkpointing during training to save memory
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(encoder_layer, hidden_states)
            else:
                layer_outputs = encoder_layer(hidden_states)
            
            hidden_states = layer_outputs

        if output_hidden_states:
            # Store the final hidden state
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class InternVisionModel(PreTrainedModel):
    """
    The main Vision Transformer model class, combining the embedding and encoder modules.
    Inherits from `PreTrainedModel` to integrate with the Hugging Face ecosystem.
    """
    main_input_name = "pixel_values"
    config_class = InternVisionConfig
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """
        Utility function to resize positional embeddings when fine-tuning on a different image resolution.
        """
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        # Reshape and interpolate the patch positional embeddings
        pos_emb = (
            pos_emb[:, 1:, :]
            .reshape(1, old_size // patch_size, old_size // patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_emb = F.interpolate(
            pos_emb.float(), size=new_size // patch_size, mode="bicubic", align_corners=False
        )
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        # Concatenate the [CLS] and resized patch embeddings
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        # Update the model's parameters
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info("Resized position embeddings from {} to {}".format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        device = next(self.encoder.parameters()).device
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
 
        # The model can accept either raw pixel values or pre-computed embeddings
        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")
        
        # Pass the embeddings through the encoder
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        # The pooled output is the embedding of the [CLS] token
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        