import os


import yaml
import torch
from transformers.utils.hub import get_file_from_repo


# 从当前项目的相对路径导入各种视觉编码器的实现类
from .clip.clip_encoder import CLIPVisionTower
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .internvit.internvit_encoder import InternViTVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .whale.init_model import init_model


def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    一个工厂函数，根据提供的配置对象 (vision_tower_cfg) 来构建并返回一个特定的视觉编码器实例。

    Args:
        vision_tower_cfg: 一个配置对象（通常是 SimpleNamespace 或类似的类） 
        **kwargs: 构造函数的关键字参数。
    """
    # 它首先尝试读取 "mm_vision_tower" 属性，如果不存在，则回退到读取 "vision_tower" 属性。
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    # 从配置中获取 'use_s2' 标志，该标志可能用于选择模型的特定变体（例如 S2 架构）。默认为 False。
    use_s2 = getattr(vision_tower_cfg, "use_s2", False)

 
    if "sig" in vision_tower.lower():  # 如果名称中包含 "sig"，则认为是 SigLIP 模型。
        if use_s2:
            # 如果 use_s2 为 True，则使用 SiglipVisionTowerS2 版本。
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            # 否则，使用标准的 SiglipVisionTower。
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "eva" in vision_tower.lower():  # 如果名称中包含 "eva"，则认为是 EVA-CLIP 模型。
        if use_s2:
            # 当前不支持 EVA-CLIP 的 S2 变体，因此抛出错误。
            raise ValueError(f"Currently not supporting S2 for EVA-CLIP")
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


    elif "clip" in vision_tower.lower():  # 如果名称中包含 "clip"，则认为是标准的 CLIP 模型。
        if use_s2:
            # 当前不支持 CLIP 的 S2 变体，因此抛出错误。
            raise ValueError(f"Currently not supporting S2 for CLIP")
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "internvit" in vision_tower.lower():  # 如果名称中包含 "internvit"，则认为是 InternViT 模型。
        if use_s2:
            # 当前不支持 InternViT 的 S2 变体，因此抛出错误。
            raise ValueError(f"Currently not supporting S2 for InternViT")
        else:
            return InternViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


    else: 
        raise ValueError(f"Unknown vision tower: {vision_tower}")




def build_audio_encoder(audio_encoder_config, **kwargs):
    """
    根据提供的配置对象构建并初始化一个音频编码器模型。
    它会从 Hugging Face Hub 下载配置文件和权重，然后加载它们。

    Args:
        audio_encoder_config: 一个配置对象，包含了音频编码器的仓库ID和其他相关配置。
        **kwargs: 其他保留参数（当前未使用）。

    Returns:
        一个初始化完成并加载了预训练权重的音频编码器模型实例。
    """
    # 从 Hugging Face Hub 上的指定仓库 (audio_encoder_config.mm_audio_encoder) 下载 'train.yaml' 配置文件并读取。
    with open(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "train.yaml"), "r") as fin:
        # 使用 yaml 库解析配置文件内容。
        configs = yaml.load(fin, Loader=yaml.FullLoader)


    # 从 Hub 下载 'global_cmvn' 文件（通常用于音频特征归一化），并将其路径添加到配置字典中。
    configs["cmvn_file"] = get_file_from_repo(audio_encoder_config.mm_audio_encoder, "global_cmvn")


    # 从传入的 audio_encoder_config 对象中读取特定配置，并更新从 yaml 文件加载的 configs 字典。
    # 这允许在运行时覆盖 yaml 文件中的默认设置。
    # 是否冻结主编码器。
    configs["model_conf"]["freeze_encoder"] = getattr(
        audio_encoder_config, "freeze_audio_encoder", True
    )
    # 是否冻结适配器层。
    configs["model_conf"]["freeze_adpter"] = getattr(
        audio_encoder_config, "freeze_audio_encoder_adapter", True
    )
    # 是否对音频提示进行微调。
    configs["model_conf"]["audio_prompt_finetune"] = getattr(
        audio_encoder_config, "audio_prompt_finetune", False
    )
    # 音频提示的数量。
    configs["model_conf"]["audio_prompt_num"] = getattr(
        audio_encoder_config, "audio_prompt_num", 0
    )


    # 使用更新后的配置字典来初始化音频编码器模型。
    audio_encoder = init_model(configs)


    # 从 Hub 下载预训练权重文件 'final.pt'，并加载到 CPU 内存中。
    checkpoint = torch.load(get_file_from_repo(audio_encoder_config.mm_audio_encoder, "final.pt"), map_location="cpu")
    # 获取新初始化的模型的 state_dict。
    model_dict = audio_encoder.state_dict()
    # 遍历新模型的每一层参数。
    for key in model_dict.keys():
        # 检查该参数是否存在于加载的 checkpoint 中。
        if key in checkpoint.keys():
            # 如果存在，再检查它们的形状是否匹配。
            if model_dict[key].shape == checkpoint[key].shape:
                # 如果形状匹配，则将预训练的权重复制到新模型的 state_dict 中。
                model_dict[key] = checkpoint[key]
            else:
                # 如果形状不匹配，打印一条警告信息。
                print(
                    "Key {} has different shape, {} VS {}".format(
                        key, model_dict[key].shape, checkpoint[key].shape
                    )
                )
        else:
            # 如果 checkpoint 中不存在该参数，也打印一条警告信息。
            print("Key {} has not in resume model".format(key))
    # 将填充了预训练权重的 state_dict 加载到模型中。
    audio_encoder.load_state_dict(model_dict)


    # 返回最终构建好的音频编码器。
    return audio_encoder
