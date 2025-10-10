import torch
import torch.nn as nn


# 导入项目内部定义的模块
from .eva_clip_processors import EvaClipImageTrainProcessor
from .eva_vit import Eva2LargePlusEncoder


# 定义 EVA-CLIP 视觉塔 (Vision Tower)
# 这是一个封装了预训练视觉模型（如 EVA ViT）的模块，
# 主要用于从图像中提取高级特征。
class EvaClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        """
        初始化 EvaClipVisionTower。

        Args:
            vision_tower (str): 预训练视觉模型权重或标识符的路径。
            args: 其他配置参数（在此代码片段中未使用，但可能在完整项目中使用）。
            delay_load (bool): 如果为 True，则延迟加载实际的模型权重，
                               只初始化配置。这对于节省初始化时间和内存很有用。
        """
        super().__init__()

        # 标志位，用于跟踪模型权重是否已加载到内存中
        self.is_loaded = False

        # 保存视觉模型的路径和配置
        self.vision_tower_path = vision_tower
        self.config = VisionTowerConfig()

        # 根据 delay_load 标志决定是否立即加载模型
        if not delay_load:
            self.load_model()
        else:
            # 如果延迟加载，只保存配置信息
            self.cfg_only = self.config

    def load_model(self):
        """
        加载实际的视觉模型权重和图像处理器。
        """
        # 初始化图像处理器，用于对输入图像进行预处理（如调整大小、归一化）
        self.image_processor = EvaClipImageTrainProcessor(self.config.image_size)
        # 从指定路径加载预训练的 EVA ViT 模型
        self.vision_tower = Eva2LargePlusEncoder(self.vision_tower_path)
        # 冻结视觉塔的所有参数，使其在训练过程中不更新。
        # 这是一种常见的做法，将预训练模型用作固定的特征提取器。
        self.vision_tower.requires_grad_(False)

        # 更新标志位，表示模型已成功加载
        self.is_loaded = True

    @torch.no_grad() # 装饰器，表示在此方法下的所有 torch 操作都不会计算梯度，以节省计算资源和内存
    def forward(self, images):
        """
        前向传播函数，用于从输入图像中提取特征。

        Args:
            images (torch.Tensor or list[torch.Tensor]): 
                输入的图像数据。可以是一个批次的张量 (B, C, H, W)，
                也可以是一个图像张量的列表。

        Returns:
            torch.Tensor: 提取出的图像特征。
        """
        # 处理输入是图像列表的情况
        if type(images) is list:
            image_features = []
            for image in images:
                # 对单个图像进行处理：
                # 1. 移动到正确的设备和数据类型
                # 2. 增加一个批次维度 (unsqueeze(0))
                # 3. 通过视觉塔模型
                # 4. 将输出特征的数据类型转换回与输入图像一致
                image_feature = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                ).to(image.dtype)
                image_features.append(image_feature)
        # 处理输入是单个批次张量的情况
        else:
            # 将整个批次移动到正确的设备和数据类型，然后通过视觉塔模型
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(
                images.dtype
            )

        return image_features

    @property # 将方法定义为属性，使其可以像访问变量一样被调用 (e.g., model.dtype)
    def dtype(self):
        """返回底层视觉塔模型的数据类型 (e.g., torch.float16)"""
        return self.vision_tower.dtype

    @property
    def device(self):
        """返回底层视觉塔模型所在的设备 (e.g., 'cuda:0')"""
        return self.vision_tower.device

    @property
    def hidden_size(self):
        """返回模型的隐藏层维度（即输出特征的维度）"""
        return self.config.hidden_size

    @property
    def num_patches(self):
        """计算并返回图像被切分成的 patch 数量"""
        return (self.config.image_size // self.config.patch_size) ** 2


# 视觉塔的配置类
# 这是一个简单的数据类，用于存储与视觉模型相关的超参数。
class VisionTowerConfig:
    def __init__(self):
        """
        初始化默认配置。
        """
        # 输入图像的目标尺寸
        self.image_size = 336
        # Vision Transformer 中每个 patch 的尺寸
        self.patch_size = 14
        # 模型输出特征的维度
        self.hidden_size = 1024
        