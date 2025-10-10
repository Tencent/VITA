import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

# 导入本地或自定义的 InternVisionModel 模型结构
from .modeling_intern_vit import InternVisionModel

class InternViTVisionTower(nn.Module):
    """
    一个视觉编码器模块，封装了 InternViT 模型。
    它负责加载预训练的 InternViT 模型，处理输入图像，并提取特征。
    此外，它还应用了一个自定义的 pixel_shuffle 操作来转换特征图。
    """
    def __init__(self, vision_tower, args, delay_load=False):
        """
        初始化 InternViTVisionTower。

        Args:
            vision_tower (str): 预训练模型在 Hugging Face Hub 上的名称或本地路径。
            args: 其他配置参数 (在此代码段中未使用)。
            delay_load (bool): 如果为 True，则不立即加载模型权重，只加载配置。
                               这有助于节省初始化时间和内存，直到模型真正被使用。
        """
        super().__init__()

        # 标记模型权重是否已加载
        self.is_loaded = False

        # 模型名称或路径
        self.vision_tower_name = vision_tower
        # 选择要从中提取特征的 Transformer 层的索引 (-1 表示最后一层)
        self.select_layer = -1
        # pixel_shuffle 操作的缩放因子，也用于调整特征值
        self.scale_pix_shuffle = 0.5

        if not delay_load:
            # 如果不延迟加载，则立即加载模型
            self.load_model()
        else:
            # 否则，只加载模型的配置信息
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )

    def load_model(self):
        """
        加载预训练的 InternViT 模型和相应的图像处理器。
        """
        # 从预训练模型加载图像处理器 (用于图像预处理，如归一化、调整大小等)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # 加载 InternViT 模型权重
        self.vision_tower = InternVisionModel.from_pretrained(
            self.vision_tower_name, trust_remote_code=True
        )
        # 将模型设置为评估模式，并冻结其参数，因为我们只用它来提取特征
        self.vision_tower.requires_grad_(False)

        # 更新加载状态
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        从模型的前向传播输出中选择指定的隐藏层特征。

        Args:
            image_forward_outs: InternVisionModel 的输出对象。

        Returns:
            torch.Tensor: 提取出的图像块特征。
        """
        # 从所有隐藏状态中选择 `select_layer` 指定的层
        image_features = image_forward_outs.hidden_states[self.select_layer]

        # 忽略第一个 token，它通常是 [CLS] token，我们只需要图像块 (patch) 的特征
        image_features = image_features[:, 1:]

        return image_features

    def pixel_shuffle(self, x, scale_factor=0.5):
        """
        对特征图进行 pixel_shuffle 操作。
        这个操作会降低特征图的空间分辨率，同时增加通道维度。
        这是一种将空间信息重新组织到通道维度的方法。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, W, H, C)。
            scale_factor (float): 缩放因子，这里 0.5 表示空间维度减半。

        Returns:
            torch.Tensor: 经过变换后的特征图。
        """
        n, w, h, c = x.size()
        # 步骤 1: (N, W, H, C) -> (N, W, H*scale, C/scale)
        # 在高度维度上进行重塑，将通道维度的一部分移到高度维度
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # 步骤 2: (N, W, H*scale, C/scale) -> (N, H*scale, W, C/scale)
        # 交换宽度和高度维度
        x = x.permute(0, 2, 1, 3).contiguous()
        # 步骤 3: (N, H*scale, W, C/scale) -> (N, H*scale, W*scale, C/(scale*scale))
        # 在新的宽度维度上再次重塑，将通道维度的另一部分移到宽度维度
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        # 步骤 4: (N, H*scale, W*scale, C/(scale*scale)) -> (N, W*scale, H*scale, C/(scale*scale))
        # 再次交换宽度和高度维度，恢复原始的 W, H 顺序
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    # 图像处理这块，花活太多了，不知道会不会影响对原本像素的理解
    # (保留原作者注释)

    # @torch.no_grad() # 通常特征提取不需要计算梯度
    def forward(self, images):
        """
        模型的前向传播函数。

        Args:
            images (torch.Tensor or list[torch.Tensor]): 输入的图像张量或张量列表。

        Returns:
            torch.Tensor: 处理后的图像特征。
        """
        # 处理输入是图像列表的情况 (例如，视频帧)
        if type(images) is list:
            image_features = []
            for image in images:
                # 将单张图片传递给 vision tower
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), # 增加 batch 维度
                    output_hidden_states=True, # 确保返回所有隐藏层
                )
                # 提取特征
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:  
            # 处理输入是单个批次张量的情况
            images = images.to(device=self.device, dtype=self.dtype)
            # 将整个批次传递给 vision tower
            image_forward_outs = self.vision_tower(
                images, output_hidden_states=True
            )
            # 提取特征
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        # --- 对提取的特征进行后处理 ---
        # 计算原始特征图的高度和宽度 (假设是正方形)
        h = w = int(image_features.shape[1] ** 0.5)
        assert image_features.shape[1] == h * w, "特征数量必须是完美的平方数"
        
        # 将扁平的 patch 序列重塑为 2D 特征图 (N, H, W, C)
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        
        # 应用 pixel_shuffle 操作，并按比例缩放特征值
        image_features = self.pixel_shuffle(image_features * self.scale_pix_shuffle)
        
        # 将处理后的特征图重新扁平化为序列 (N, num_patches_new, C_new)
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )

        return image_features

    @property
    def dummy_feature(self):
        """返回一个形状正确的零张量，可用作占位符。"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """返回底层 vision_tower 模型的数据类型 (如 float16, float32)。"""
        return self.vision_tower.dtype

    @property
    def device(self):
        """返回底层 vision_tower 模型所在的设备 (如 'cuda:0')。"""
        return self.vision_tower.device

    @property
    def config(self):
        """返回模型的配置对象。"""
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """
        返回经过 pixel_shuffle 后的有效隐藏层大小。
        原始隐藏层大小会因为 pixel_shuffle 操作而增加。
        """
        # C_new = C_original / (scale_factor^2)
        return self.config.hidden_size * (int(1 / self.scale_pix_shuffle) ** 2)

    @property
    def num_patches(self):
        """返回原始 vision tower 输出的图像块数量。"""
        return (self.config.image_size // self.config.patch_size) ** 2
