# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from torch.func import vjp

from .cross_attention_dit import DiT


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """
    # 生成形状为 (B, T, w) 的正弦位置编码
    # 输入时间步形状为 (B, T)

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: 形状 (B, T)
        # 我们将在 T 维度上计算 sin/cos 频率
        timesteps = timesteps.float()  # 确保是浮点数

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # 正弦编码中典型的对数空间频率
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # 将 timesteps 扩展到 (B, T, 1) 然后相乘
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificLinear(nn.Module):
    # 特定类别的线性层
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # 对每个类别，我们有独立的权重和偏置
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # 根据类别ID选择相应的权重和偏置
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # 执行批处理矩阵乘法
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    # 特定类别的多层感知机
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    # 多形态动作编码器
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        # 使用特定类别的线性层
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        # actions: 形状 (B, T, action_dim)
        # timesteps: 形状 (B,) -- 每个批次项一个标量
        # cat_ids: 形状 (B,)
        # 返回: 形状 (B, T, hidden_size)
        B, T, _ = actions.shape

        # 1) 将每个批次的单个标量时间 'tau' 扩展到所有 T 步
        #    使其形状变为 => (B, T)
        #    例如，如果 timesteps 是 (B,)，则在 T 维度上复制
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) 标准的动作MLP步骤，形状变为 => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) 获取正弦编码 (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) 沿最后一个维度连接 => (B, T, 2w), 然后通过 W2 => (B, T, w), 再通过 swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) 最后通过 W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    # 流匹配动作头配置
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"} # 是否添加位置嵌入
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."}) # 模型数据类型
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."} # 扩散模型配置
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."} # 输入嵌入通道维度
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."}) # 输入嵌入维度
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"}) # 最大序列长度
    action_dim: int = field(default=None, metadata={"help": "Action dimension."}) # 动作维度
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."}) # 动作时域
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""}) # 噪声Beta分布的alpha参数
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""}) # 噪声Beta分布的beta参数
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."} # 流匹配噪声Beta分布的s参数
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."} # 时间步离散化桶的数量
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."}, # 噪声扩散的推理步数
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."}) # 最大形态数量
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."}) # 是否微调投影器
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."} # 是否微调扩散模型
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    # 流匹配动作头
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg) # DiT扩散模型
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = CategorySpecificMLP( # 状态编码器
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder( # 动作编码器
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP( # 动作解码器
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim) # 位置嵌入
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta) # Beta分布用于采样时间t
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        # 设置可训练参数
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            # 如果不微调投影器，则冻结相关层
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            # 如果不微调扩散模型，则冻结DiT模型
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # 检查是否还有可训练的参数。如果没有，则打印警告。
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        # Huggingface会在每个训练步骤调用model.train()。为确保
        # dropout、batchnorm等模块的预期行为，我们需要对冻结的模块
        # 调用model.eval()。
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        # 从Beta分布中采样时间t
        try:
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        except Exception as e:
            print("Error in sample_time:")
            print("  batch_size:", batch_size)
            print("  device:", device)
            print("  dtype:", dtype)
            print("  alpha:", self.beta_dist.concentration1)
            print("  beta:", self.beta_dist.concentration0)
            print("  config.noise_s:", self.config.noise_s)
            import traceback
            traceback.print_exc()
            raise e

        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        # 准备输入
        return BatchFeature(data=batch)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # 前向传播（用于训练）
        # 将冻结的模块设置为评估模式
        self.set_frozen_modules_to_eval_mode()

        # 获取视觉和语言嵌入
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # 获取形态ID
        embodiment_id = action_input.embodiment_id

        # 编码状态
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # 编码带噪声的动作轨迹
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # 形状 (B,1,1) 以便广播

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise # 速度场是目标动作和噪声的差

        # 将（连续的）t -> 离散化
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # 可能添加位置嵌入
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # 沿序列维度连接视觉、语言、状态和动作嵌入
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_embs = vl_embeds
        vl_attn_mask = backbone_output.backbone_attention_mask

        # DiT模型前向传播
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
        )
        # 解码得到预测的速度场
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # 只切片出预测和目标的动作部分
        action_mask = action_input.action_mask
        # 计算MSE损失
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # 获取动作（用于推理）
        # 获取视觉和语言嵌入
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # 编码状态
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # 将初始动作设置为采样的噪声
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # 运行去噪步骤（欧拉积分）
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # 例如，从 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # 编码带噪声的动作轨迹
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # 可能添加位置嵌入
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # 沿序列维度连接视觉、语言、状态和动作嵌入
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # 模型前向传播
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # 使用欧拉积分更新动作
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    def get_realtime_action(self, 
                            backbone_output: BatchFeature, 
                            action_input: BatchFeature,
                            prev_action_chunk: torch.Tensor,
                            inference_delay: int,
                            prefix_attention_horizon: int,
                            max_guidance_weight: float = 5.0,
                            ) -> BatchFeature:
        # 获取实时动作（用于需要快速响应的场景）
        # 获取视觉和语言嵌入
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # 编码状态
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # 将初始动作设置为采样的噪声
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # 获取前缀权重，用于指导生成过程
        prefix_weights = get_prefix_weights(inference_delay, 
                                           prefix_attention_horizon, 
                                           self.config.action_horizon)
        prefix_weights = prefix_weights.to(device=device, dtype=vl_embeds.dtype)
        
        # 运行去噪步骤
        for t in range(num_steps):
            t_param = t / float(num_steps)  # 例如，从 0, 1/N, 2/N, ...
            def denoiser(actions: torch.Tensor):
                # 定义去噪函数
                t_discretized = int(t_param * self.num_timestep_buckets)
                # 编码带噪声的动作轨迹
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
                # 可能添加位置嵌入
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                vl_embs = vl_embeds

                # 沿序列维度连接
                sa_embs = torch.cat((state_features, action_features), dim=1)

                # 模型前向传播
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -self.action_horizon :]
                
                v_t = pred_velocity
                # 返回去噪后的动作x_1和速度场v_t
                return actions + v_t * (1.0 - t_param), v_t  # actions=x_t
            
            # 使用vjp计算向量-雅可比积，用于指导
            x_1, vjp_fun, v_t = vjp(denoiser, actions, has_aux=True)
            
            # 计算误差和修正项
            error = (prev_action_chunk - x_1) * prefix_weights.unsqueeze(-1)  # prev_action_chunk=y
            pinv_correction = vjp_fun(error)[0]
            inv_r2 = (t_param**2 + (1 - t_param) ** 2) / ((1 - t_param) ** 2)
            c_param = torch.nan_to_num(
                (torch.tensor(1.0, device=device, dtype=vl_embeds.dtype) - t_param) / t_param,
                posinf=max_guidance_weight
            )
            # 计算指导权重
            guidance_weight = torch.minimum(
                c_param * inv_r2,
                torch.tensor(max_guidance_weight, device=device, dtype=vl_embeds.dtype)
            )
            # 更新速度场
            v_t = v_t + guidance_weight * pinv_correction
            
            # 使用欧拉积分更新动作
            actions = actions + dt * v_t
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        # 获取模型所在的设备
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        # 获取模型的数据类型
        return next(iter(self.parameters())).dtype

import math
def get_prefix_weights(
    start: int, 
    end: int, 
    total: int, 
    schedule: str="exp"
) -> torch.Tensor:
    # 获取前缀权重，用于指导生成
    start = min(start, end)
    arange = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32)
    elif schedule == "zeros":
        w = (arange < start).float()
    elif schedule == "linear" or schedule == "exp":
        # 线性或指数衰减
        denom = (end - start + 1)
        w = ((start - 1 - arange) / denom + 1).clamp(0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    # 在结束点之后权重为0
    w = torch.where(arange >= end, torch.tensor(0.0, dtype=w.dtype), w)
    return w
