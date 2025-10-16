#!/usr/bin/env python3
import os
import sys
import cv2
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as Tv2
from torchvision.transforms import InterpolationMode

# repo_root = os.path.abspath(os.path.dirname(__file__))
# isaac_root = os.path.join(repo_root, "Isaac-GR00T")
# print(isaac_root)
# if os.path.isdir(os.path.join(isaac_root, "gr00t")):
#     sys.path.insert(0, isaac_root)
# else:
#     raise ValueError("Isaac-GR00T not found")

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_vita_action_head import Gr00tActionHeadPolicy
from gr00t.model.vita_data_processor import VitaProcessor
from gr00t.model.vita_model import VITAModel


def load_image_rgb(path: str, fallback_size=(400, 225)) -> np.ndarray:
    img_bgr = cv2.imread("asset/test.png")
    img_bgr = cv2.resize(img_bgr, fallback_size)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def apply_eval_image_transform(
    img_rgb: np.ndarray,
    scale: float = 0.95,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Note: duplicated logic with `vita_e.server_vla_vita:_apply_eval_image_transform` is
    intentionally kept in this helper to avoid cross-module dependency in the
    standalone inference script. Any changes must be mirrored there.
    """
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb
    h, w, _ = img_rgb.shape
    crop_h = max(1, int(h * scale))
    crop_w = max(1, int(w * scale))
    frame_tensor = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    transform_image = Tv2.Compose([
        Tv2.CenterCrop((crop_h, crop_w)),
        Tv2.Resize((target_size[1], target_size[0]), interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    out_tensor = transform_image(frame_tensor)
    out_np = (out_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return out_np


def test_vita_gr00t_integration_vita(model_path_vlm, model_path_policy):
    """使用 VITA -> hidden states -> Gr00tActionHeadPolicy 动作预测。"""
    embodiment_tag = "new_embodiment"
    data_config_name = "real_data_robot_vita_action_head"

    print("=== 初始化 Gr00tActionHeadPolicy ===")
    data_config = DATA_CONFIG_MAP[data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tActionHeadPolicy(
        model_path=model_path_policy,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )

    print("=== 准备测试数据（文本 + 单张图片） ===")
    test_prompt = "Pick up the red toy and place into the basket."
    test_image_path = "test.png"
    frame_rgb = load_image_rgb(test_image_path)
    # 图像变换（中心裁剪0.95 + 224线性缩放）
    frame_rgb = apply_eval_image_transform(frame_rgb, scale=0.95, target_size=(224, 224))

    # 使用 VitaProcessor 构建输入（包含提示词模板与 <image> 占位符展开）
    processor = VitaProcessor()
    message = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_prompt, "image": [{"np_array": frame_rgb}]},
        ]
    }
    single = processor.prepare_input(message)
    batch = processor.collate_fn([single])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = batch["pixel_values_vita"].to(device=device, dtype=torch.bfloat16)
    input_ids_t = batch["input_ids_vita"].to(device=device)
    attn_mask_t = batch["attention_mask_vita"].to(device=device)
    # print(batch)

    # 构建并初始化 VITAModel（用于获取 hidden states）
    print("=== 初始化 VITAModel 并提取 hidden states ===")
    vita_model = VITAModel(model_path=model_path_vlm, p_num=[1])
    vita_model.init_model(device_id=device, tune_visual=False, tune_llm=False, load_separately=True)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden_states = vita_model.get_latent(
            image_tensor=image_tensor,
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
        )
    print(f"shape hidden_states: {tuple(hidden_states.shape)}")

    print("=== 预测机器人动作 ===")
    # 随机模拟机器人状态
    state_hand = np.random.rand(1, 12).astype(np.float32)
    state_robot = np.random.rand(1, 14).astype(np.float32)
    observations = {
        "state.hand": state_hand,
        "state.robot": state_robot,
    }

    action_dict = policy.get_action(hidden_states, observations)

    print("=== 输出结果 ===")
    for key, value in action_dict.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f"Action {key}: shape={getattr(value, 'shape', None)}, dtype={value.dtype}")
            if hasattr(value, "min") and hasattr(value, "max") and "action." in key:
                v_min = float(value.min()) if isinstance(value, np.ndarray) else float(value.min().item())
                v_max = float(value.max()) if isinstance(value, np.ndarray) else float(value.max().item())
                print(f"  数值范围: [{v_min:.4f}, {v_max:.4f}]")
        else:
            print(f"Action {key}: {type(value)}")

    # 验证动作维度
    expected_dims = {"action.hand": 12, "action.robot": 14}
    for action_key, expected_dim in expected_dims.items():
        if action_key in action_dict:
            actual_shape = action_dict[action_key].shape
            if len(actual_shape) >= 2 and actual_shape[-1] == expected_dim:
                print(f"✓ {action_key} 维度正确: {actual_shape}")
            else:
                print(f"✗ {action_key} 维度错误: expected [..., {expected_dim}], got {actual_shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_vlm", type=str, default="checkpoints/vita_vla_finetune")
    parser.add_argument("--model_path_policy", type=str, default="checkpoints/vita_gr00t_robot_head")
    args = parser.parse_args()
    test_vita_gr00t_integration_vita(args.model_path_vlm, args.model_path_policy)
