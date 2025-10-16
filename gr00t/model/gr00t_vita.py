# 导入必要的库
from dataclasses import dataclass, field # 用于创建数据类，简化类的定义
from typing import Tuple # 用于类型注解，表示元组类型

import numpy as np # 用于数值计算
import torch # PyTorch 深度学习框架
import tree # 用于处理嵌套数据结构（如字典、列表）
from huggingface_hub import snapshot_download # 从 Hugging Face Hub 下载模型文件
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError
)  # Hugging Face Hub 的特定异常
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel
)  # Hugging Face Transformers 库，用于加载和使用预训练模型
from transformers.feature_extraction_utils import (
    BatchFeature
)  # Hugging Face 的一个数据结构，用于封装批处理数据

# 从当前项目的其他模块中导入
from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead, # 动作头模块
    FlowmatchingActionHeadConfig, # 动作头配置
)
from .backbone import EagleBackbone # 另一个可选的骨干网络 (这里没有被使用)
import os # 用于与操作系统交互，如此处用于获取环境变量

from .vita_model import VITAModel # 导入 VITA 模型，作为该策略模型的骨干网络
# from .vita_model_act import VITAModel # 备用导入，已被注释掉


# 定义一些常量键，用于在字典中存取数据
BACKBONE_FEATURE_KEY = "backbone_features" # 骨干网络输出特征的键名
ACTION_KEY = "action_pred" # 预测动作的键名
LOSS_KEY = "loss" # 损失值的键名
ERROR_MSG = "错误：意料之外的输入/输出" # 用于异常信息的字符串
N_COLOR_CHANNELS = 3 # 图像的颜色通道数 (RGB)


# 使用 dataclass 定义模型的配置类，继承自 PretrainedConfig
@dataclass
class GR00T_N1Config(PretrainedConfig):
    """
    GR00T_N1 模型的配置类。
    它定义了模型的所有超参数和设置。
    """
    model_type = "gr00t_n1" # 模型类型标识符
    # backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."}) # 骨干网络配置 (已注释掉)

    # `field` 用于为类属性提供额外信息
    action_head_cfg: dict = field(init=False, metadata={"help": "动作头配置。"})

    action_horizon: int = field(init=False, metadata={"help": "动作序列的长度。"})

    action_dim: int = field(init=False, metadata={"help": "动作的维度。"})
    compute_dtype: str = field(default="float32", metadata={"help": "计算时使用的数据类型。"})

    def __init__(self, **kwargs):
        """
        构造函数。
        :param kwargs: 包含配置参数的关键字参数字典。
        """
        super().__init__(**kwargs) # 调用父类的构造函数
        # 遍历传入的关键字参数，并将它们设置为类的属性
        for key, value in kwargs.items():
            setattr(self, key, value)


# 定义 GR00T_N1 模型类，继承自 PreTrainedModel
class GR00T_N1(PreTrainedModel):
    """
    GR00T_N1 模型的主体。
    这个模型由一个 VITA 骨干网络和一个 Flowmatching 动作头组成。
    它负责接收观测（如视频、文本指令），提取特征，并预测出机器人动作。
    """
    supports_gradient_checkpointing = True # 声明支持梯度检查点，用于节省显存
    config_class = GR00T_N1Config # 将模型与上面的配置类关联起来
    """
    我们期望骨干网络的输出是一个字典，其中包含一个键 'backbone_features'，
    其对应的值的形状为 (batch_size, n, hidden_size)，n 是可变长度。
    我们期望动作头的输出在推理时是一个字典，其中包含一个键 'action_pred'，
    其对应的值的形状为 (batch_size, time, action_dim)。
    我们期望这些输出是 BatchFeature 类型，当然它们也可以包含其他用户自定义的键。
    """

    def __init__(
        self,
        config: GR00T_N1Config, # 模型的配置对象
        local_model_path: str, # 本地模型路径，在 from_pretrained 中传入
    ):
        """
        构造函数。
        :param config: GR00T_N1Config 对象，包含所有模型超参数。
        :param local_model_path: 预训练模型在本地的路径。
        """
        # assert isinstance(config.backbone_cfg, dict)
        # 断言，确保传入的配置是字典类型
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config) # 调用父类的构造函数
        self.local_model_path = local_model_path

        # 实例化 VITA 模型作为骨干网络
        # 注意：这里的模型路径是硬编码的，可能需要根据实际情况修改
        self.vita_model = VITAModel(
            model_path="/root/VITA/checkpoints/vita_vla_finetune_ended/llava-s3-finetune_task_neg",
            p_num=[1]
        )
        # 根据配置实例化 Flowmatching 动作头
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        # 从配置中获取动作相关的维度信息
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        
    def validate_inputs(self, inputs):
        """
        验证输入数据的格式是否正确。
        :param inputs: 包含输入数据的字典。
        """
        # 注意：这部分检查理论上应该由模型内部处理，但为了避免破坏性更改，暂时放在这里。

        detected_error = False # 错误标志
        error_msg = ERROR_MSG # 初始错误信息
        # 检查 'action' 键
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor) # 动作应为 torch.Tensor
            # 检查动作的形状是否为 (batch, action_horizon, action_dim)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n动作类型错误，应为 torch.Tensor，实际为 {type(action)}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n动作形状错误，应为 (B, {self.action_horizon}, {self.action_dim})，实际为 {action.shape}"
                detected_error = True

        # 检查 'video' 键
        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray) # 视频应为 np.ndarray
            dtype_ok = video.dtype == np.uint8 # 数据类型应为 uint8
            # 视频形状应为6维，且颜色通道数为3
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n视频类型错误，应为 np.ndarray，实际为 {type(video)}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n视频数据类型错误，应为 np.uint8，实际为 {video.dtype}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n视频形状错误，应为 (B, T, N, C, H, W)，实际为 {video.shape}"
                detected_error = True

        # 如果检测到任何错误，则抛出 ValueError
        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        """
        验证骨干网络和动作头的输出数据格式是否正确。
        :param action_head_outputs: 动作头的输出。
        :param backbone_outputs: 骨干网络的输出。
        :param is_training: 当前是否处于训练模式。
        """
        # 检查骨干网络的输出
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) # 输出应为 BatchFeature 类型
            or BACKBONE_FEATURE_KEY not in backbone_outputs # 必须包含 'backbone_features'
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n骨干网络输出是否为 BatchFeature: {isinstance(backbone_outputs, BatchFeature)}"
            error_msg += f"\n'{BACKBONE_FEATURE_KEY}' 是否在骨干网络输出中: {BACKBONE_FEATURE_KEY in backbone_outputs}"
            if BACKBONE_FEATURE_KEY in backbone_outputs:
                error_msg += f"\n特征形状: {backbone_outputs[BACKBONE_FEATURE_KEY].shape}"
            raise ValueError(error_msg)

        # 检查动作头的输出
        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training # 训练时必须有 loss
            )  # 训练时可能没有动作预测
            or (
                ACTION_KEY in action_head_outputs # 推理时必须有 action_pred
                # 检查动作预测的形状
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n动作头输出是否为 BatchFeature: {isinstance(action_head_outputs, BatchFeature)}"
            error_msg += f"\n'{LOSS_KEY}' 是否在动作头输出中: {LOSS_KEY in action_head_outputs}"
            if ACTION_KEY in action_head_outputs:
                error_msg += f"\n预测动作的形状: {action_head_outputs[ACTION_KEY].shape}"
            error_msg += f"\n期望的动作序列长度: {self.action_horizon}"
            error_msg += f"\n期望的动作维度: {self.action_dim}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        """
        模型的前向传播函数，主要用于训练。
        :param inputs: 包含输入数据的字典。
        :return: BatchFeature，通常包含损失值。
        """
        # 1. 准备输入数据
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # 2. 通过 VITA 骨干网络提取特征
        backbone_outputs = self.vita_model(vita_inputs)

        # 3. 将特征输入动作头，计算损失
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        # 4. 验证输出格式
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        """
        获取预测的动作，主要用于推理。
        :param inputs: 包含输入数据的字典。
        :return: BatchFeature，包含预测的动作序列。
        """
        # 1. 准备输入数据
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # 2. 通过 VITA 骨干网络提取特征 (推理和训练时骨干网络行为一致)
        backbone_outputs = self.vita_model(vita_inputs)
        # 3. 将特征输入动作头，生成动作
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        # 4. 验证输出格式
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def get_hidden_states(
        self,
        inputs: dict,
    ) -> torch.Tensor:
        """
        从 VITA 模型获取隐藏状态，而不生成动作。
        
        Args:
            inputs (dict): 包含观测值的输入数据。
            
        Returns:
            torch.Tensor: 从 VITA 骨干模型中提取的隐藏状态 (维度为 3584，在最后一个线性投影层之前)。
        """
        # 1. 只准备 VITA 模型需要的输入
        vita_inputs, _ = self.prepare_input(inputs)
        # 2. 调用 VITA 模型的 `get_latent` 方法获取原始隐藏状态 (3584维)
        hidden_states = self.vita_model.get_latent(
            image_tensor=vita_inputs["pixel_values_vita"],
            input_ids=vita_inputs["input_ids_vita"],
            attention_mask=vita_inputs["attention_mask_vita"],
        )
        
        return hidden_states  # 返回形状为 (B, T, 3584) 的张量

    def get_realtime_action(
        self,
        inputs,
        prev_action_chunk,
        inference_delay,
        prefix_attention_horizon
    ) -> BatchFeature:
        """
        获取实时动作，用于需要低延迟响应的场景。
        :param inputs: 当前的观测输入。
        :param prev_action_chunk: 上一个时间步的动作块。
        :param inference_delay: 推理延迟。
        :param prefix_attention_horizon: 注意力前缀范围。
        :return: 包含实时动作预测的 BatchFeature。
        """
        # 1. 准备输入数据
        vita_inputs, action_inputs = self.prepare_input(inputs)
        # 2. 通过 VITA 骨干网络提取特征
        backbone_outputs = self.vita_model(vita_inputs)
        # 3. 调用动作头的实时动作生成方法
        action_head_outputs = self.action_head.get_realtime_action(backbone_outputs, 
                                                                   action_inputs,
                                                                   prev_action_chunk,
                                                                   inference_delay,
                                                                   prefix_attention_horizon)
        # 4. 验证输出格式
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        """
        准备 VITA 模型和动作头的输入数据。
        包括验证、预处理和移动到正确的设备。
        :param inputs: 原始输入字典。
        :return: 一个元组，分别包含为 VITA 模型和动作头准备好的输入。
        """
        # 1. 验证原始输入
        self.validate_inputs(inputs)
        # 2. 分别调用 VITA 模型和动作头的输入准备函数
        vita_inputs = self.vita_model.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # 定义一个辅助函数，用于将数据移动到指定设备并转换数据类型
        def to_device_with_maybe_dtype(x):
            # 只对浮点数张量转换其 dtype
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # 保持原始的 dtype (例如，整数或布尔值)
                return x.to(self.device)

        print("vita_inputs: ", vita_inputs) # 打印 VITA 输入，用于调试
        # 使用 tree.map_structure 递归地将函数应用到嵌套数据结构中的每个元素
        vita_inputs = tree.map_structure(to_device_with_maybe_dtype, vita_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        
        return vita_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        一个类方法，用于从预训练权重加载模型。
        :param pretrained_model_name_or_path: 模型名称 (在 Hugging Face Hub 上) 或本地路径。
        :param kwargs: 其他关键字参数。
        :return: 加载了预训练权重的模型实例。
        """
        # 从 kwargs 中提取用于控制微调的参数
        tune_visual = kwargs.pop("tune_visual", False) # 是否微调视觉部分
        tune_llm = kwargs.pop("tune_llm", False) # 是否微调语言模型部分
        tune_projector = kwargs.pop("tune_projector", False) # 是否微调动作头的投影层
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", False) # 是否微调动作头的 DiT 模型
        load_separately = kwargs.pop("load_separately", True) # 是否分开加载模型的各个部分

        print(f"从 {pretrained_model_name_or_path} 加载预训练的双脑模型")
        print(f"微调骨干网络视觉塔: {tune_visual}")
        print(f"微调骨干网络大语言模型: {tune_llm}")
        print(f"微调动作头投影层: {tune_projector}")
        print(f"微调动作头 DiT: {tune_diffusion_model}")

        # 这段代码被 `if False` 禁用了，它原本用于从 Hugging Face Hub 下载模型
        if False:
            # get the current model path being downloaded
            try:
                # 注意(YL): 这会将模型下载到本地缓存并返回本地路径
                # 默认保存在 ~/.cache/huggingface/hub/
                local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
                # HFValidationError, RepositoryNotFoundError
            except (HFValidationError, RepositoryNotFoundError):
                # 如果在 Hub 上找不到，则尝试作为本地路径加载
                print(
                    f"在 Hugging Face Hub 中找不到模型或模型不可用。从本地路径加载: {pretrained_model_name_or_path}"
                )
                local_model_path = pretrained_model_name_or_path
        else:
            # 当前逻辑：总是将传入的路径视为本地路径
            print(
                f"从本地路径加载: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # 调用父类的 from_pretrained 方法加载模型的配置和基本结构
        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        ) # ignore_mismatched_sizes=True,
        
        # 这段代码被 `if True` 激活，用于删除可能存在的旧的 'backbone' 属性
        if True:
            if hasattr(pretrained_model, "backbone"):
                del pretrained_model.backbone
        
        # 初始化 VITA 编码器
        print("正在加载 VITA 模型...")
        # 获取分布式训练中的本地排名 (local rank)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = torch.device(f"cuda:{local_rank}")
        # 调用 VITA 模型的初始化函数，加载权重并设置微调参数
        pretrained_model.vita_model.init_model(
            device_id=device_id,
            tune_visual=tune_visual,
            tune_llm=tune_llm,
            load_separately=load_separately
        )
        print("VITA 模型加载成功。")
        
        # 设置动作头中哪些参数是可训练的
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# 将自定义模型和配置注册到 Hugging Face 的 AutoClass 中
# 这样就可以使用 AutoModel.from_pretrained("gr00t_n1", ...) 来加载模型了
AutoConfig.register("gr00t_n1", GR00T_N1Config)
AutoModel.register(GR00T_N1Config, GR00T_N1)
