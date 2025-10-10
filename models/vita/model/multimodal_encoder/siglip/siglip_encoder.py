import torch
import torch.nn as nn
# 从 transformers 库导入 Siglip 模型的特定组件
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

# 导入一个自定义的 S2（多尺度）前向传播工具函数
from ....util.s2wrapper import forward as multiscale_forward


class SiglipVisionTower(nn.Module):
    """
    一个封装了 Hugging Face SiglipVisionModel 的模块。
    主要作用是作为一个冻结的视觉特征提取器（Vision Tower）。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        # 标志位，用于判断模型权重是否已经加载
        self.is_loaded = False

        # 视觉塔的名称，通常是 Hugging Face Hub 上的模型标识符
        self.vision_tower_name = vision_tower
        # 选择要提取特征的层。-2 表示倒数第二层，这通常是比最后一层更好的特征表示
        self.select_layer = -2

        # delay_load 参数允许延迟加载模型权重，这在某些情况下可以节省初始化时间和内存
        if not delay_load:
            self.load_model()
        else:
            # 如果延迟加载，只加载模型的配置信息，以便访问 hidden_size 等属性
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        """
        加载预训练的 Siglip 图像处理器和视觉模型，并冻结其权重。
        """
        # 加载与模型匹配的图像处理器
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        # 确保 crop_size 与 size 一致
        self.image_processor.crop_size = self.image_processor.size
        # 加载预训练的视觉模型权重
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        # 冻结模型的所有参数，使其在训练中不更新
        self.vision_tower.requires_grad_(False)

        # 更新加载状态
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        从模型的前向传播输出中选择指定的隐藏层作为图像特征。
        """
        # image_forward_outs.hidden_states 是一个包含所有层输出的元组
        # self.select_layer (-2) 选择了倒数第二层的输出
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features

    # @torch.no_grad()  # 装饰器，确保在前向传播期间不计算梯度，节省计算和内存
    # def forward(self, images):
    #     """
    #     对输入的图像进行前向传播，并提取特征。
    #     """
    #     # 处理输入是图像张量列表的情况
    #     if type(images) is list:
    #         image_features = []
    #         for image in images:
    #             # 对单个图像进行前向传播
    #             image_forward_out = self.vision_tower(
    #                 # 将图像移动到正确的设备和数据类型，并增加一个批次维度
    #                 image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
    #                 # 必须设置为 True 才能获取 hidden_states
    #                 output_hidden_states=True,
    #             )
    #             # 提取特征并转换回原始数据类型
    #             image_feature = self.feature_select(image_forward_out).to(image.dtype)
    #             image_features.append(image_feature)
    #     # 处理输入是单个批次张量的情况
    #     else:
    #         image_forward_outs = self.vision_tower(
    #             images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
    #         )
    #         image_features = self.feature_select(image_forward_outs).to(images.dtype)

    #     return image_features

    @property
    def dummy_feature(self):
        """
        返回一个形状正确的零张量，可用作占位符。
        """
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """
        获取视觉塔模型的数据类型（如 torch.float16）。
        """
        return self.vision_tower.dtype

    @property
    def device(self):
        """
        获取视觉塔模型所在的设备（如 'cuda:0'）。
        """
        return self.vision_tower.device

    @property
    def config(self):
        """
        获取模型的配置。如果模型已加载，则返回加载模型的配置；否则返回仅配置对象。
        """
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """
        获取模型的隐藏层维度大小。
        """
        return self.config.hidden_size

    @property
    def num_patches(self):
        """
        计算模型将图像分割成的 patch 数量。
        """
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipVisionTowerS2(SiglipVisionTower):
    """
    SiglipVisionTower 的一个扩展版本，实现了 S2 (Scale-to-Scale) 策略。
    它在多个不同的分辨率下处理图像，并将提取的特征拼接起来，以获得更丰富的多尺度信息。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        # 从参数中获取多尺度的分辨率设置，例如 "384,768,1152"
        self.s2_scales = getattr(args, "s2_scales", "384,768,1152")
        # 将字符串解析为整数列表
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()  # 确保尺度从小到大排序
        # 最小的尺度用作图像分割的大小
        self.s2_split_size = self.s2_scales[0]
        # 最大的尺度是图像预处理的目标尺寸
        self.s2_image_size = self.s2_scales[-1]

        # 调用父类的构造函数
        super().__init__(vision_tower, args, delay_load)

        # 引用外部的多尺度前向传播函数
        self.multiscale_forward = multiscale_forward

        # 如果不是延迟加载，立即更新图像处理器的尺寸以适应 S2 策略
        if not delay_load:
            self.image_processor.size["height"] = self.image_processor.size[
                "width"
            ] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size[
                "width"
            ] = self.s2_image_size

    def load_model(self):
        """
        重写 load_model 方法，在加载模型后，强制更新图像处理器的尺寸为 S2 策略的最大尺寸。
        """
        # 调用父类的加载逻辑
        super().load_model()

        # 将图像处理器的目标尺寸和裁剪尺寸都设置为 S2 的最大尺度
        self.image_processor.size["height"] = self.image_processor.size[
            "width"
        ] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size[
            "width"
        ] = self.s2_image_size

    @torch.no_grad()
    def forward_feature(self, images):
        """
        一个辅助函数，用于在单一尺度下对图像进行特征提取。
        这个函数将被传递给 multiscale_forward 工具函数。
        """
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    # @torch.no_grad()
    # def forward(self, images):
    #     """
    #     重写 forward 方法，使用 multiscale_forward 工具函数执行 S2 多尺度前向传播。
    #     """
    #     # 处理输入是图像张量列表的情况
    #     if type(images) is list:
    #         image_features = []
    #         for image in images:
    #             # 对每个图像调用多尺度前向传播
    #             image_feature = self.multiscale_forward(
    #                 self.forward_feature,  # 传递单尺度特征提取函数
    #                 image.unsqueeze(0),
    #                 img_sizes=self.s2_scales,
    #                 max_split_size=self.s2_split_size,
    #             )
    #             image_features.append(image_feature)
    #     # 处理输入是单个批次张量的情况
    #     else:
    #         image_features = self.multiscale_forward(
    #             self.forward_feature,
    #             images,
    #             img_sizes=self.s2_scales,
    #             max_split_size=self.s2_split_size,
    #         )

    #     return image_features

    @property
    def hidden_size(self):
        """
        重写 hidden_size 属性。
        由于 S2 策略将不同尺度的特征拼接在一起，最终的特征维度是原始 hidden_size 乘以尺度的数量。
        """
        return self.config.hidden_size * len(self.s2_scales)
