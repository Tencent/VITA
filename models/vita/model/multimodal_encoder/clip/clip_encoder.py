import torch
import torch.nn as nn
# 从 transformers 库导入 CLIP 模型相关的类
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
 
# 定义一个封装了 CLIP 视觉模型的模块，通常称为 "Vision Tower"
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        """
        初始化 CLIPVisionTower。

        Args:
            vision_tower (str): Hugging Face Hub 上 CLIP 视觉模型的名称或本地路径。
                                例如 "openai/clip-vit-large-patch14-336"。
            args: 模型的其他配置参数（在此代码段中未使用，但为保持接口一致性而保留）。
            delay_load (bool): 是否延迟加载模型权重。如果为 True，则只加载配置信息，
                               直到显式调用 load_model()，这有助于节省初始化时的内存和时间。
        """
        super().__init__()

        # 标志位，用于跟踪模型权重是否已完全加载
        self.is_loaded = False

        # 存储 CLIP 模型的名称或路径
        self.vision_tower_name = vision_tower
        # 指定要从 CLIP 模型中提取哪一层作为特征。
        # -2 表示倒数第二层，这通常比最后一层（经过池化后）包含更丰富的空间信息，
        # 对于需要细粒度视觉信息的任务（如 VQA）更有效。
        self.select_layer = -2

        # 根据 delay_load 标志决定是立即加载模型还是仅加载配置
        if not delay_load:
            self.load_model()
        else:
            # 如果延迟加载，只从预训练模型中获取配置信息
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        """
        实际加载 CLIP 视觉模型权重和图像处理器。
        """
        # 加载与指定 CLIP 模型配套的图像处理器，用于图像的预处理（如缩放、归一化）
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # 从预训练权重加载 CLIP 视觉模型
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        # 冻结视觉塔的所有参数。这非常重要，因为我们通常将其用作固定的特征提取器，
        # 在下游任务的训练中不更新它的权重，以防止过拟合并保留其强大的通用视觉表示能力。
        self.vision_tower.requires_grad_(False)

        # 更新标志位，表示模型已加载完成
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        从模型的前向传播输出中选择并处理所需的特征。

        Args:
            image_forward_outs: CLIPVisionModel 的原始输出对象。

        Returns:
            torch.Tensor: 提取出的图像块特征。
        """
        # 从模型输出中获取所有隐藏层的状态，并根据 self.select_layer 选择特定层的输出
        image_features = image_forward_outs.hidden_states[self.select_layer]

        # 对于 ViT 架构，输出的第一个 token 是 [CLS] token，用于整个图像的分类表示。
        # 后面的 tokens 对应于图像的各个 patch。
        # 这里我们通过 [:, 1:] 切片操作，移除了 [CLS] token，只保留 patch 特征。
        image_features = image_features[:, 1:]

        return image_features

    @torch.no_grad()  # 装饰器，表示在此方法下的所有 torch 操作都不会计算梯度，用于推理阶段以节省计算资源和内存
    def forward(self, images):
        """
        定义模型的前向传播逻辑。

        Args:
            images (torch.Tensor or list[torch.Tensor]): 输入的图像张量。
                        可以是一个批次的图像 (B, C, H, W)，也可以是一个图像张量的列表。

        Returns:
            torch.Tensor: 从指定层提取的图像特征。
        """
        # 处理输入是图像列表的情况（通常用于非批处理或图像大小不同的情况）
        if type(images) is list:
            image_features = []
            for image in images:
                # 对单个图像进行前向传播
                image_forward_out = self.vision_tower(
                    # 将单张图像移动到模型所在的设备和数据类型，并增加一个批次维度
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    # 必须设置为 True，这样才能在输出中获取 hidden_states
                    output_hidden_states=True,
                )
                # 提取特征并转换回原始图像的数据类型
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # 处理输入是批次张量的情况
            image_forward_outs = self.vision_tower(
                # 将整个批次移动到模型所在的设备和数据类型
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            # 提取整个批次的特征并转换回原始图像的数据类型
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property  # 装饰器，将一个方法变成一个属性，调用时无需加括号
    def dummy_feature(self):
        """
        返回一个形状正确的虚拟特征张量，通常用于初始化或占位。
        """
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """
        返回底层视觉模型的数据类型（如 torch.float16 或 torch.float32）。
        """
        return self.vision_tower.dtype

    @property
    def device(self):
        """
        返回底层视觉模型所在的设备（如 'cuda:0' 或 'cpu'）。
        """
        return self.vision_tower.device

    @property
    def config(self):
        """
        返回模型的配置对象。处理了延迟加载的情况。
        """
        if self.is_loaded:
            # 如果模型已加载，返回完整模型的配置
            return self.vision_tower.config
        else:
            # 如果模型未加载，返回仅加载的配置对象
            return self.cfg_only

    @property
    def hidden_size(self):
        """
        返回模型隐藏层的大小（即特征维度）。
        """
        return self.config.hidden_size

    @property
    def num_patches(self):
        """
        计算并返回图像被分割成的 patch 数量。
        """
        return (self.config.image_size // self.config.patch_size) ** 2
    