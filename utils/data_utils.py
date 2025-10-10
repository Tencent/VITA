import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

# data processing and loading related code
# including text and image preprocessing, data collation, dataset definition and data loader definition
# support different types of datasets, e.g., jsonl, hdf5, webdataset

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union
import transformers
from .constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_DATA_RATIO,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    MAX_IMAGE_LENGTH,
    MIN_IMAGE_LENGTH,
)
from . import conversation as conversation_lib
from .mm_utils import tokenizer_image_audio_token, tokenizer_image_token, tokenizer_image_action_token


# preprocess data for qwen2p5_instruct model
def preprocess_qwen2p5_instruct(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    end_tag: bool = True,
    modality: str = "image",
) -> Dict:
    

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv = conversation_lib.conv_templates['qwen2p5_instruct'].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt(modality)) 

    # Tokenize conversations
    if not end_tag:
        conversations[0] = conversations[0][:-10]
    if has_image and not has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_action_token(prompt, tokenizer, return_tensors="pt") 
                for prompt in conversations
            ],
            dim=0,
        )
    elif has_image and has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_audio_token(prompt, tokenizer, return_tensors="pt") 
                for prompt in conversations
            ],
            dim=0,
        )
    elif not has_image and has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_audio_token(prompt, tokenizer, return_tensors="pt") 
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    #print(f'end_tag: {end_tag}')
    #print(conversations)
    #print(input_ids)
    #import pdb; pdb.set_trace()

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.Qwen2p5Instruct

    # Mask targets
    sep = '\n' + conv.sep + conv.roles[1] + "\n"   #\n<|im_start|>assistant\n
    sep2 = '\n' + conv.sep2 + conv.roles[0] + "\n" #\n<|im_start|>user\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        rounds = [rounds[0] + sep2 + rounds[1]] + rounds[2:]
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if i > 0:
                rou = sep2 + rou

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            #import pdb; pdb.set_trace()
            if has_image and not has_audio:
                round_len = len(tokenizer_image_action_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_action_token(parts[0], tokenizer))
            elif has_image and has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer))
            elif not has_image and has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")
                # print(f"YOU NEED GO TO DEBUG THIS DATA ITEM: {conversations}")

    #print(targets)
    #import pdb; pdb.set_trace()
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# prepare data for vita model
def prepare_data_vita(vita_image_wrist, vita_image_gripper, language, states, tokenizer):
    
    # 前三个元素是位置，接下来三个是方向，然后是夹爪宽度，接着七个关节状态，最后是夹爪动作

    batch_size, sequence_length = states.shape[0], states.shape[1]
    sources = []
    
    
    actions = ' '.join(['<action>'] * 3)
    
    # 一批数据的构成
    for i in range(batch_size):
        # 一条数据的构成
        source = []
        for j in range(sequence_length):
            human, gpt = {}, {}
            human["from"] = 'human'
            
            if isinstance(language, str):
                if 'no lang' in language:
                    human["value"] = f'''<image> <image> <state>'''
                else:  
                    human["value"] = f'''<image> <image> {language} <state>'''
            else:
                if 'no lang' in language[i]:
                    human["value"] = f'''<image> <image> <state>'''
                else:  
                    human["value"] = f'''<image> <image> {language[i]} <state>'''
            gpt["from"] = 'gpt'
            gpt["value"] = actions
            
            source.append(human)
            source.append(gpt)
        sources.append(source) 
    
    # import pdb;pdb.set_trace()
    data_dict = [preprocess_qwen2p5_instruct([source], tokenizer, has_image=True, 
                                             has_audio=False) for source in sources]

    input_ids, labels = tuple(
            [instance[key] for instance in data_dict] for key in ("input_ids", "labels")
        )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        for input_id in input_ids:
            input_id[input_id == tokenizer.eos_token_id] = -300

    input_ids = [input_id.squeeze(0) for input_id in input_ids]
    labels = [label.squeeze(0) for label in labels]
 
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )

    input_ids = input_ids[:, :tokenizer.model_max_length]

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    # import pdb; pdb.set_trace()

    labels = labels[:, :tokenizer.model_max_length]

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        for input_id in input_ids:
            input_id[input_id == -300] = tokenizer.eos_token_id

    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )
    
    # batch['image_wrist'] = vita_image_wrist
    # batch['image_gripper'] =vita_image_gripper
    
    
    # batch['images'] = torch.cat((vita_image_wrist, vita_image_gripper), 1).flatten(0, 1)
    
    # tensor = torch.cat((vita_image_wrist, vita_image_gripper), 1).flatten(0, 1)
    
    # import pdb;pdb.set_trace()
    tensor_1 = []
    for batch_id in range(vita_image_wrist.shape[0]):
        for seq_id in range(vita_image_wrist.shape[1]):
            tensor_1.append(vita_image_wrist[batch_id][seq_id])
            tensor_1.append(vita_image_gripper[batch_id][seq_id])
    
    # import pdb;pdb.set_trace()
    tensor_1 = torch.stack(tensor_1)

    # # # 假设 tensor 是形状为 (40, 3, 200, 200) 的张量
    # # tensor = torch.randn(40, 3, 200, 200)  # 示例张量，您可以用实际数据替换

    # # 创建保存图像的目录
    # output_dir = '/utils/output_images_eval_old'
    # os.makedirs(output_dir, exist_ok=True)

    # # 遍历每个图像并保存
    # for i in range(tensor.shape[0]):
    #     # 获取当前图像
    #     img_tensor = tensor[i].to(torch.float32)  # 形状为 (3, 200, 200)

    #     # 将张量转换为 PIL 图像
    #     img = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

    #     # 保存图像
    #     img.save(os.path.join(output_dir, f'image_{i}.png'))
        
    # output_dir = '/utils/output_images_eval_new'
    # os.makedirs(output_dir, exist_ok=True)

    # # 遍历每个图像并保存
    # for i in range(tensor_1.shape[0]):
    #     # 获取当前图像
    #     img_tensor = tensor_1[i].to(torch.float32)  # 形状为 (3, 200, 200)

    #     # 将张量转换为 PIL 图像
    #     img = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

    #     # 保存图像
    #     img.save(os.path.join(output_dir, f'image_{i}.png'))

    # print(f'Saved {tensor.shape[0]} images to {output_dir}')
    
    # import pdb; pdb.set_trace()
    

    batch['states'] = states
    batch['images'] = tensor_1
 
    return batch

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_image_vita(sample, image_processor):
    image = [image_processor.preprocess(s, size={'height': 200, 'width': 200}, 
                                        crop_size={'height': 200, 'width': 200}, 
                                        return_tensors="pt")["pixel_values"][0].unsqueeze(0) 
             for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def preprocess_text_vita(sample, tokenizer):
    text = tokenizer.tokenize(sample)
    return text

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}

def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data


def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]
    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(
                rgb_obs[seq_idx : seq_idx + window_size]
            ).byte()
        
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxHxWxC)
    return {"rgb_obs": seq_rgb_obs_dict}

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    """
    通过重复第一个或最后一个元素来填充张量。

    Args:
        input_tensor: 需要填充的序列张量 (T, ...)，T是时间/序列维度。
        pad_size: 需要填充的帧数。
        head (bool): 如果为 True，在序列开头填充；否则在结尾填充。

    Returns:
        填充后的张量。
    """
    if pad_size == 0:
        return input_tensor

    # 选择要重复的帧：第一帧或最后一帧
    frame_to_repeat = input_tensor[0] if head else input_tensor[-1]
    
    # unsqueeze(0) 增加一个维度以匹配 repeat 的输入要求
    # repeat(pad_size, 1, ..., 1) 在第一个维度上重复 pad_size 次
    padding = frame_to_repeat.unsqueeze(0).repeat(pad_size, *[1] * (input_tensor.dim() - 1))

    if head:
        # 在开头拼接
        padded_tensor = torch.cat([padding, input_tensor], dim=0)
    else:
        # 在结尾拼接
        padded_tensor = torch.cat([input_tensor, padding], dim=0)
        
    return padded_tensor


def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    """
    用零来填充张量。

    Args:
        input_tensor: 需要填充的序列张量 (T, ...)。
        pad_size: 需要填充的帧数。
        head (bool): 如果为 True，在序列开头填充；否则在结尾填充。

    Returns:
        填充后的张量。
    """
    if pad_size == 0:
        return input_tensor

    # 创建一个形状正确的零张量
    # (pad_size, *input_tensor.shape[1:]) 会创建 (pad_size, H, W, C) 或 (pad_size, D) 这样的形状
    zeros_padding = torch.zeros(pad_size, *input_tensor.shape[1:], dtype=input_tensor.dtype, device=input_tensor.device)
    
    if head:
        padded_tensor = torch.cat([zeros_padding, input_tensor], dim=0)
    else:
        padded_tensor = torch.cat([input_tensor, zeros_padding], dim=0)

    return padded_tensor


def pad_sequence(seq: Dict, pad_size: int, relative_actions: bool, head: bool = False) -> Dict:
    """
    对一个包含多模态数据的序列字典进行填充。

    这个函数是填充逻辑的核心协调者，它会根据每个数据项的类型（状态、图像、动作等）
    应用不同的填充策略。

    Args:
        seq: 包含序列数据的字典。
        pad_size: 需要填充的帧数。
        relative_actions: 一个布尔标志，指示动作是否是相对坐标。这会影响动作的填充方式。
        head (bool): 如果为 True，在序列开头填充；否则在结尾填充。

    Returns:
        填充后的序列字典。
    """
    if pad_size <= 0:
        return seq # 如果不需要填充，直接返回

    # 1. 填充机器人本体状态 (robot_obs)，通常重复最后一帧的状态
    seq['robot_obs'] = _pad_with_repetition(seq['robot_obs'], pad_size, head)

    # 2. 填充 RGB 图像观测 (rgb_obs)，对字典中的每个相机视角都进行重复填充
    if 'rgb_obs' in seq:
        seq['rgb_obs'] = {
            k: _pad_with_repetition(v, pad_size, head)
            for k, v in seq['rgb_obs'].items()
        }

    # 3. 填充深度图像观测 (depth_obs)，逻辑同上
    if 'depth_obs' in seq:
        seq['depth_obs'] = {
            k: _pad_with_repetition(v, pad_size, head)
            for k, v in seq['depth_obs'].items()
        }

    # 4. 填充动作 (actions)，这是逻辑最复杂的部分
    if 'actions' in seq:
        if not relative_actions:
            # 对于绝对坐标动作（世界坐标系），通常重复最后一个动作是合理的
            seq['actions'] = _pad_with_repetition(seq['actions'], pad_size, head)
        else:
            # 对于相对坐标动作，填充零通常更安全，因为一个非零的相对动作会持续改变状态。
            # 特殊情况：夹爪动作（通常是最后一维）可能需要重复，因为它代表状态（开/合）而不是位移。
            if head:
                # 在开头填充时，用零填充整个动作
                seq_acts = _pad_with_zeros(seq['actions'], pad_size, head)
            else:
                # 在结尾填充时，将动作分为两部分：
                # a) 位移/旋转部分 (所有维度除了最后一个): 用零填充
                # b) 夹爪部分 (最后一个维度): 重复最后一个状态
                motion_actions = seq['actions'][..., :-1]
                gripper_action = seq['actions'][..., -1:]
                
                padded_motion = _pad_with_zeros(motion_actions, pad_size, head)
                padded_gripper = _pad_with_repetition(gripper_action, pad_size, head)
                
                seq_acts = torch.cat([padded_motion, padded_gripper], dim=-1)
            
            seq['actions'] = seq_acts

    # 5. 填充状态元信息 (state_info)，通常重复最后一帧
    if 'state_info' in seq:
        seq['state_info'] = {
            k: _pad_with_repetition(v, pad_size, head)
            for k, v in seq['state_info'].items()
        }

    return seq

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def _apply_shift(self, x):
        """
        对一个4D张量 (B, C, H, W) 应用随机平移增强。
        这是核心的增强逻辑。
        """
        n, c, h, w = x.size()
        assert h == w, "Input height and width must be equal"

        # 1. 对图像进行padding
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        
        # 更新图像尺寸
        h_padded, w_padded = h + 2 * self.pad, w + 2 * self.pad

        # 2. 创建基础采样网格 (base_grid)
        # 这个网格对应于从padding后的图像中裁剪出原始大小的区域
        eps = 1.0 / h_padded
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_padded, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        # 3. 创建随机位移
        # 注意：原始代码中 forward 和 forward_traj 的 randint 范围不同，
        # 这里统一为 [0, 2*pad]，这通常是更合理的选择，因为它允许不发生位移。
        shift = torch.randint(0, 2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        # 将像素位移转换为 grid_sample 所需的 [-1, 1] 范围内的坐标偏移
        shift = shift * 2.0 / h_padded

        # 4. 将位移应用到基础网格上
        grid = base_grid + shift

        # 5. 使用 grid_sample 进行采样
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward(self, x):
        """
        处理单张图片或一个批次的图片 (n, c, h, w)。
        """
        return self._apply_shift(x)

    def forward_traj(self, x):
        """
        处理一个批次的图片序列 (n, t, c, h, w)。
        """
        # 记录原始维度
        n, t, c, h, w = x.shape
        
        # 1. 将 (n, t, c, h, w) -> (n*t, c, h, w)
        #    将轨迹维度合并到批次维度，以便可以作为一个大批次进行处理
        x_reshaped = x.view(n * t, c, h, w)
        
        # 2. 对这个大批次应用核心增强逻辑
        aug_x = self._apply_shift(x_reshaped)
        
        # 3. 将维度恢复为 (n, t, c, h, w)
        return aug_x.view(n, t, c, h, w)

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

# basecalvindataset, to be inherited by other dataset classes
class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        *args: Any,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        data_in_ceph=False,
        **kwargs: Any,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.except_lang = key == "except_lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.conf_path = '~/petreloss.conf'
            self.client = Client(self.conf_path)
       
        with open('./utils/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        if self.data_in_ceph:
            assert (
                "validation" in self.abs_datasets_dir
                or "training" in self.abs_datasets_dir
            )
            self.validation = "validation" in self.abs_datasets_dir
        else:
            assert (
                "validation" in self.abs_datasets_dir.as_posix()
                or "training" in self.abs_datasets_dir.as_posix()
            )
            self.validation = "validation" in self.abs_datasets_dir.as_posix()
        print(f"loading dataset at {self.abs_datasets_dir}")
        print("finished loading dataset")

    # def process_rgb(
    #     self,
    #     episode: Dict[str, np.ndarray],
    #     observation_space: DictConfig,
    #     transforms: Dict,
    #     seq_idx: int = 0,
    #     window_size: int = 0,
    # ) -> Dict[str, Dict[str, torch.Tensor]]:
    #     rgb_obs_keys = observation_space["rgb_obs"]
    #     seq_rgb_obs_dict = {}
    #     for _, rgb_obs_key in enumerate(rgb_obs_keys):
    #         rgb_obs = episode[rgb_obs_key]
    #         # expand dims for single environment obs
    #         if len(rgb_obs.shape) != 4:
    #             rgb_obs = np.expand_dims(rgb_obs, axis=0)
    #         assert len(rgb_obs.shape) == 4
    #         if window_size == 0 and seq_idx == 0:  # single file loader
    #             # To Square image
    #             seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
    #         else:  # episode loader
    #             seq_rgb_obs_ = torch.from_numpy(
    #                 rgb_obs[seq_idx : seq_idx + window_size]
    #             ).byte()
            
    #         if rgb_obs_key in transforms:
    #             seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
    #         seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    #     # shape: N_rgb_obs x (BxHxWxC)
    #     return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]} if with_lang else {"lang": 'no lang'}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        从数据集中获取一个数据序列。

        Args:
            idx: 序列的索引。可以是单个整数，也可以是 (索引, 窗口大小) 的元组。
            fixed_seed: (未使用，但保留在签名中)

        Returns:
            一个字典，包含加载和处理过的序列数据。
        """
        # --- 1. 确定窗口大小 (Window Size) ---
        # 本方法支持两种调用方式：
        # a) dataset[i]: 传入单个整数索引，窗口大小将根据预设的 min/max 范围随机确定。
        # b) dataset[(i, w)]: 传入 (索引, 窗口大小) 元组，直接使用指定的窗口大小。

        if isinstance(idx, int):
            # --- 情况 a): 索引是单个整数，需要动态确定窗口大小 ---

            if self.min_window_size == self.max_window_size:
                # 优化：如果最小和最大窗口大小相等，说明窗口大小是固定的。
                # 直接使用该值，避免不必要的随机采样。
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                # 标准情况：在一个范围内随机采样一个窗口大小。
                # 这是一种数据增强，让模型学习处理不同长度的序列。
                window_size = self._get_window_size(idx)
            else:
                # 错误检查：确保配置是有效的。
                print(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError("最小窗口大小不能大于最大窗口大小")
        else:
            # --- 情况 b): 索引是一个元组，直接解包 ---
            # 调用者明确指定了要加载的序列索引和窗口大小。
            idx, window_size = idx

        # --- 2. 加载原始序列数据 ---
        # `head` 参数可能用于控制是从序列的开头还是结尾进行采样，这里固定为 False。
        head = False
        # 调用内部辅助函数，根据索引和确定的窗口大小从磁盘或内存中加载原始数据。
        sequence = self._get_sequences(idx, window_size, head=head)

        # --- 3. 对序列进行填充 (Padding) ---
        # 如果启用了填充功能（通常是为了让批次内所有序列长度一致）。
        if self.pad:
            # 首先计算需要填充多少。例如，如果一个序列在数据集的开头，长度可能小于 `window_size`。
            pad_size = self._get_pad_size(sequence)
            # 然后应用填充，通常是重复第一帧或最后一帧的数据。
            sequence = pad_sequence(sequence, pad_size, head=head)

        # --- 4. 将图像张量转换为 PIL Image 对象列表 ---
        # 目的：许多预处理流程（例如 Hugging Face 的 CLIPImageProcessor 或 torchvision.transforms）
        # 期望输入的图像格式是 PIL Image 对象的列表，而不是一个堆叠的 (T, H, W, C) 张量。
        # 这一步就是进行这种格式转换。

        # 注意: 在函数内部导入模块通常不是最佳实践，最好将 `import copy` 移到文件顶部。
        
        # --- 处理静态视图图像 (rgb_static) ---
        new_list = []
        # 从 PyTorch 张量转换为 NumPy 数组。使用 deepcopy 确保不会意外修改原始数据。
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        # 遍历序列中的每一帧图像 (T, H, W, C)
        for i in range(np_rgb.shape[0]):
            # 将单帧 (H, W, C) 的 NumPy 数组转换为 PIL Image 对象。
            # 需要确保数据类型是 uint8，这是图像的典型格式。
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        # 用转换后的 PIL Image 列表替换掉字典中原来的张量。
        sequence["rgb_obs"]["rgb_static"] = new_list

        # --- 处理夹爪视图图像 (rgb_gripper)，逻辑完全相同 ---
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        # --- 5. 返回最终结果 ---
        # 返回处理完成的序列字典，现在它可以被 DataLoader 的 collate_fn 函数进一步处理成一个批次。
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    # def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
    #     """
    #     Pad a sequence by repeating the last frame.

    #     Args:
    #         seq: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded sequence.
    #     """
    #     seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
    #     seq.update(
    #         {
    #             "rgb_obs": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["rgb_obs"].items()
    #             }
    #         }
    #     )
    #     seq.update(
    #         {
    #             "depth_obs": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["depth_obs"].items()
    #             }
    #         }
    #     )

    #     if not self.relative_actions:
    #         if head:
    #             seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
    #         else:
    #             # repeat action for world coordinates action space
    #             seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
    #     else:
    #         # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
    #         if head:
    #             seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
    #         else:
    #             seq_acts = torch.cat(
    #                 [
    #                     self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
    #                     self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
    #                 ],
    #                 dim=-1,
    #             )
    #         seq.update({"actions": seq_acts})
    #     seq.update(
    #         {
    #             "state_info": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["state_info"].items()
    #             }
    #         }
    #     )
    #     return seq

    # @staticmethod
    # def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    #     """
    #     Pad a sequence Tensor by repeating last element pad_size times.

    #     Args:
    #         input_tensor: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded Tensor.
    #     """
    #     if head:
    #         last_repeated = torch.repeat_interleave(
    #             torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
    #         )
    #         padded = torch.vstack((last_repeated, input_tensor))
    #     else:
    #         last_repeated = torch.repeat_interleave(
    #             torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
    #         )
    #         padded = torch.vstack((input_tensor, last_repeated))
    #     return padded

    # @staticmethod
    # def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    #     """
    #     Pad a Tensor with zeros.

    #     Args:
    #         input_tensor: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded Tensor.
    #     """
    #     zeros_repeated = torch.repeat_interleave(
    #         torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
    #         repeats=pad_size,
    #         dim=0,
    #     )
    #     if head:
    #         padded = torch.vstack((zeros_repeated, input_tensor))
    #     else:
    #         padded = torch.vstack((input_tensor, zeros_repeated))
    #     return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        seer_image_fn: Callable,
        seer_text_fn: Callable,
        vita_image_fn: Callable,
        vita_text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.seer_image_fn = seer_image_fn
        self.seer_text_fn = seer_text_fn
        self.vita_image_fn = vita_image_fn
        self.vita_text_fn = vita_text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = self.load_pkl
        elif self.save_format == "npz":
            self.load_file = partial(self.load_npz, data_in_ceph=self.data_in_ceph)
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang()
        elif self.except_lang:
            self.episode_lookup = self._build_file_indices_except_lang()
        else:
            self.episode_lookup = self._build_file_indices()

        if self.data_in_ceph:
            self.naming_pattern, self.n_digits = self.ceph_lookup_naming_pattern()
        else:
            self.naming_pattern, self.n_digits = lookup_naming_pattern(
                self.abs_datasets_dir, self.save_format
            )
    
    def ceph_lookup_naming_pattern(self):
        filenames = self.client.list(self.abs_datasets_dir)
        for filename in filenames:
            if self.save_format in filename:
                break
        filename = self.abs_datasets_dir + f"/{filename}"
        suffix = "." + self.save_format
        stem_suffix = filename.split('/')[-1]
        stem = stem_suffix.replace(suffix, "")
        aux_naming_pattern = re.split(r"\d+", stem)
        naming_pattern = (filename.replace(stem_suffix, aux_naming_pattern[0]), suffix)
        n_digits = len(re.findall(r"\d+", stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        if self.data_in_ceph:
            return f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        else:
            return Path(
                f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
            )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, # abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        try:
            if self.data_in_ceph:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir +f"/{self.lang_folder}/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", 
                                                  enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                    allow_pickle=True,
                ).item()
        except Exception:
            if self.data_in_ceph:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir + "/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
                ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if [start_idx, end_idx] not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
             
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def _build_file_indices_except_lang(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        lang_data = np.load(
            abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            allow_pickle=True,
        ).item()
        lang_ep_start_end_ids = lang_data["info"]["indx"]

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        ep_start_end_ids = np.load(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy").tolist()

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collator(self, sample):
        
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        seer_image_tensors = torch.stack([self.seer_image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        seer_gripper_tensors = torch.stack([self.seer_image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        vita_image_tensors = torch.stack([self.vita_image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        vita_gripper_tensors = torch.stack([self.vita_image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        seer_text_tensors = self.seer_text_fn(stacked_language)
        vita_text_tensors = None
         
        if self.rgb_pad != -1:
            bs, seq_len = seer_image_tensors.shape[:2]
            if self.traj_cons:
                seer_image_tensors = self.rgb_shift.forward_traj(seer_image_tensors)
            else:
                seer_image_tensors = seer_image_tensors.view(bs*seq_len, *seer_image_tensors.shape[2:])
                seer_image_tensors = self.rgb_shift(seer_image_tensors)
                seer_image_tensors = seer_image_tensors.view(bs, seq_len, *seer_image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = seer_gripper_tensors.shape[:2]
            if self.traj_cons:
                seer_gripper_tensors = self.gripper_shift.forward_traj(seer_gripper_tensors)
            else:
                seer_gripper_tensors = seer_gripper_tensors.view(bs * seq_len, *seer_gripper_tensors.shape[2:])
                seer_gripper_tensors = self.gripper_shift(seer_gripper_tensors)
                seer_gripper_tensors = seer_gripper_tensors.view(bs, seq_len, *seer_gripper_tensors.shape[1:])
         
        if self.rgb_pad != -1:
            bs, seq_len = vita_image_tensors.shape[:2]
            if self.traj_cons:
                vita_image_tensors = self.rgb_shift.forward_traj(vita_image_tensors)
            else:
                vita_image_tensors = vita_image_tensors.view(bs*seq_len, *vita_image_tensors.shape[2:])
                vita_image_tensors = self.rgb_shift(vita_image_tensors)
                vita_image_tensors = vita_image_tensors.view(bs, seq_len, *vita_image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = vita_gripper_tensors.shape[:2]
            if self.traj_cons:
                vita_gripper_tensors = self.gripper_shift.forward_traj(vita_gripper_tensors)
            else:
                vita_gripper_tensors = vita_gripper_tensors.view(bs * seq_len, *vita_gripper_tensors.shape[2:])
                vita_gripper_tensors = self.gripper_shift(vita_gripper_tensors)
                vita_gripper_tensors = vita_gripper_tensors.view(bs, seq_len, *vita_gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            seer_image_tensors = seer_image_tensors[:, :-(self.act_step-1)]
            seer_gripper_tensors = seer_gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return seer_image_tensors, vita_image_tensors, seer_text_tensors, vita_text_tensors, 
    action_tensors, seer_gripper_tensors, vita_gripper_tensors, state_tensors, robot_obs, stacked_language

    def load_pkl(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_npz(self, filename, data_in_ceph=False):
        if not data_in_ceph:
            return np.load(filename.as_posix())
        else:
            data_bytes = self.client.get(filename, enable_cache=True)
            data = io.BytesIO(data_bytes)
            try:
                data = np.load(data, allow_pickle=True)
            except:
                data = np.load(data)
            return data

def get_calvin_dataset(args, seer_image_processor, seer_tokenizer, vita_image_processor, 
                       vita_tokenizer, epoch=0, floor=False, except_lang=False):
    dataset_path = args.calvin_dataset
    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn_seer = functools.partial(
        preprocess_image, image_processor=seer_image_processor
    )
    preprocess_image_fn_vita = functools.partial(
        preprocess_image_vita, image_processor=vita_image_processor
    )
    preprocess_text_fn_seer = functools.partial(preprocess_text_calvin, tokenizer=seer_tokenizer)
    preprocess_text_fn_vita = functools.partial(preprocess_text_calvin, tokenizer=vita_tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/training"
    else:
        datasets_dir = Path(dataset_path) / "training"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        seer_image_fn=preprocess_image_fn_seer,
        seer_text_fn=preprocess_text_fn_seer,
        vita_image_fn=preprocess_image_fn_vita,
        vita_text_fn=preprocess_text_fn_vita,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        key='except_lang' if except_lang else 'lang',
    )
     
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  #
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
     

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=20,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def get_calvin_val_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/validation"
    else:
        datasets_dir = Path(dataset_path) / "validation"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=False,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

class BaseLiberoDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        image_primary_size=200,
        image_wrist_size=84,
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        language_mode: str = "language_instruction",
        primary_mode: str = "image_primary",
        dataset_info: str = "libero",
        small_size: int = 0,
        gripper_width: bool = False,
        load_libero_file: str = "h5", 
        **kwargs: Any,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.root_dir = root_dir 
        self.dataset_path = f'{root_dir}/{dataset_name}' 
        self.conf_path = '~/petreloss.conf'
        self.image_primary_size = image_primary_size
        self.image_wrist_size = image_wrist_size
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        self.pad = pad
        self.window_size = window_size
        self.language_mode = language_mode
        self.primary_mode = primary_mode
        self.small_size = small_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            raise NotImplementedError
        
        assert self.max_window_size == self.min_window_size
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        assert os.path.exists(f"{root_dir}/{self.dataset_info}.json")
        with open(f"{root_dir}/{self.dataset_info}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
        self.load_libero_file = load_libero_file

    # def process_rgb(
    #     self,
    #     episode: Dict[str, np.ndarray],
    #     observation_space: DictConfig,
    #     transforms: Dict,
    #     seq_idx: int = 0,
    #     window_size: int = 0,
    # ) -> Dict[str, Dict[str, torch.Tensor]]:
    #     rgb_obs_keys = observation_space["rgb_obs"]
    #     seq_rgb_obs_dict = {}
    #     for _, rgb_obs_key in enumerate(rgb_obs_keys):
    #         rgb_obs = episode[rgb_obs_key]
    #         # expand dims for single environment obs
    #         if len(rgb_obs.shape) != 4:
    #             rgb_obs = np.expand_dims(rgb_obs, axis=0)
    #         assert len(rgb_obs.shape) == 4
    #         if window_size == 0 and seq_idx == 0:  # single file loader
    #             # To Square image
    #             seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
    #         else:  # episode loader
    #             seq_rgb_obs_ = torch.from_numpy(
    #                 rgb_obs[seq_idx : seq_idx + window_size]
    #             ).byte()
            
    #         if rgb_obs_key in transforms:
    #             seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
    #         seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    #     # shape: N_rgb_obs x (BxHxWxC)

    #     return {"rgb_obs": seq_rgb_obs_dict}

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """

        return self.max_window_size - len(sequence["actions"])

    # @staticmethod
    # def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    #     """
    #     Pad a sequence Tensor by repeating last element pad_size times.

    #     Args:
    #         input_tensor: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded Tensor.
    #     """
    #     if head:
    #         last_repeated = torch.repeat_interleave(
    #             torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
    #         )
    #         padded = torch.vstack((last_repeated, input_tensor))
    #     else:
    #         last_repeated = torch.repeat_interleave(
    #             torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
    #         )
    #         padded = torch.vstack((input_tensor, last_repeated))

    #     return padded

    # @staticmethod
    # def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
    #     """
    #     Pad a Tensor with zeros.

    #     Args:
    #         input_tensor: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded Tensor.
    #     """
    #     zeros_repeated = torch.repeat_interleave(
    #         torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
    #         repeats=pad_size,
    #         dim=0,
    #     )
    #     if head:
    #         padded = torch.vstack((zeros_repeated, input_tensor))
    #     else:
    #         padded = torch.vstack((input_tensor, zeros_repeated))

    #     return padded

    # def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
    #     """
    #     Pad a sequence by repeating the last frame.

    #     Args:
    #         seq: Sequence to pad.
    #         pad_size: Number of frames to pad.

    #     Returns:
    #         Padded sequence.
    #     """
    #     seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
    #     seq.update(
    #         {
    #             "rgb_obs": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["rgb_obs"].items()
    #             }
    #         }
    #     )
    #     seq.update(
    #         {
    #             "depth_obs": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["depth_obs"].items()
    #             }
    #         }
    #     )
    #     #  todo: find better way of distinguishing rk and play action spaces
    #     if not self.relative_actions:
    #         if head:
    #             seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
    #         else:
    #             # repeat action for world coordinates action space
    #             seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
    #     else:
    #         # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
    #         if head:
    #             seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
    #         else:
    #             seq_acts = torch.cat(
    #                 [
    #                     self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
    #                     self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
    #                 ],
    #                 dim=-1,
    #             )
    #         seq.update({"actions": seq_acts})
    #     seq.update(
    #         {
    #             "state_info": {
    #                 k: self._pad_with_repetition(v, pad_size, head)
    #                 for k, v in seq["state_info"].items()
    #             }
    #         }
    #     )

    #     return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = pad_sequence(sequence, pad_size, head=head)

        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence

        def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        根据全局索引和窗口大小，加载并处理一个完整的数据序列。

        Args:
            idx: 数据集的全局步骤索引 (一个扁平化的索引，跨越所有 episode)。
            window_size: 需要加载的序列长度（步数）。
            head: (在此代码段中未使用) 控制采样位置的标志。

        Returns:
            一个字典，包含经过处理的、可直接用于模型输入的序列数据。
        """
        # --- 1. 定位 Episode 和起始/结束步骤 ---
        # self.accumulated_num_step 是一个预计算的列表，存储了每个 episode 结束时的累积步数。
        # 例如: [100, 250, 310] 表示 episode 0 有 100 步, episode 1 有 150 步, episode 2 有 60 步。
        # bisect.bisect_right 使用二分查找，高效地找到 `idx` 属于哪个 episode。
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)

        # 计算 `idx` 在其所属 episode 内的起始步骤 ID。
        if episode_id > 0:
            # 如果不是第一个 episode，用全局索引减去前一个 episode 的累积步数。
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            # 如果是第一个 episode，全局索引就是其内部的起始步骤 ID。
            start_id = idx

        # 确定序列的结束步骤 ID，同时确保不会超出当前 episode 的总步数。
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = min(start_id + window_size, num_step_per_episode)

        # 将数字索引映射到实际的 episode 标识符（例如，文件夹名称）。
        episode_id_str = self.episode_list[episode_id]

        # --- 2. 逐帧加载原始数据 ---
        episodes = []  # 用于存储从磁盘加载的每一帧的原始数据字典
        for step_id in range(start_id, end_id):
            data_dict = {}
            # 将整数 step_id 格式化为带前导零的字符串 (例如, 5 -> "0005") 以匹配文件名。
            str_step_id = str(step_id).zfill(4)

            # 如果数据格式是 HDF5，则打开对应的文件句柄。
            other_file = None
            if self.load_libero_file == "h5":
                other_path = f"{self.dataset_path}/episodes/{episode_id_str}/steps/{str_step_id}/other.h5"
                other_file = h5py.File(other_path)

            # 使用辅助函数从磁盘加载各种模态的数据。
            data_dict["rgb_static"] = self.load_primary_rgb(episode_id_str, str_step_id, self.primary_mode)
            data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id_str, str_step_id)
            data_dict["rel_actions"] = self.load_action(other_file)
            data_dict["robot_obs"] = self.load_robot_obs(other_file)
            data_dict["scene_obs"] = self.load_scene_obs(episode_id_str, str_step_id)
            
            episodes.append(data_dict)

        # --- 3. 聚合和堆叠数据 ---
        # 将数据从 "列表 of 字典" 转换为 "字典 of 列表/数组"。
        # 例如, [{'rgb': r1, 'act': a1}, {'rgb': r2, 'act': a2}] -> {'rgb': [r1, r2], 'act': [a1, a2]}
        # `np.stack` 会将列表转换为一个 NumPy 数组，增加一个时间维度。
        keys = list(chain(*self.observation_space.values())) # 获取所有需要处理的 observation keys
        keys.remove("language") # 语言是 episode 级别的，单独处理
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

        # 单独加载与整个 episode 相关的语言指令。
        episode["language"] = self.load_language_instruction(other_file, self.language_mode)

        # --- 4. 对堆叠好的数据进行后处理 ---
        # 调用外部的 `process_*` 函数对整个序列进行转换，例如：
        # - 归一化状态和动作
        # - 应用图像增强 (augmentation)
        # - Tokenize 语言指令
        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False # 添加辅助损失的标志
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)

        # --- 5. 组合成最终的返回字典 ---
        # 使用字典解包 `**` 将所有处理过的部分合并成一个大的字典。
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }
        
        # 添加元数据，可用于调试或特定的训练策略。
        seq_dict["idx"] = idx
        seq_dict["episode_id"] = episode_id_str

        return seq_dict

    def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
        image_primary_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
        #    - `.convert("RGB")`: 这是一个重要的健壮性步骤。它确保无论原始图像是 RGBA, P (palette),
        #      还是 L (grayscale) 格式，最终都会被统一转换为标准的 3 通道 RGB 格式。 
        image_primary = np.array(Image.open(image_primary_path).convert("RGB"))
        
        return image_primary

    def load_wrist_rgb(self, episode_id, step_id):
        image_wrist_path = f'{self.dataset_path}/episodes/{episode_id}/steps/{step_id}/image_wrist.jpg'
        image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))

        return image_wrist.astype(np.uint8)

    def load_language_instruction(self, other_file, language_mode="language_instruction"):
        if self.load_libero_file == "h5":
            language_instruction = other_file[language_mode][()].decode('utf-8')
        elif self.load_libero_file == "npz":
            language_instruction = other_file[language_mode].tobytes().decode('utf-8')
        else:
            raise NotImplementedError

        return language_instruction
        
    def load_action(self, other_file, max_rel_pos=0.02, max_rel_orn=0.05, 
                    magic_scaling_factor_pos=1.0, magic_scaling_factor_orn=1.0):
        if self.load_libero_file == "h5":
            action = other_file["action"][()]
        elif self.load_libero_file == "npz":
            action = other_file["action"]
        else:
            raise NotImplementedError
        
        return action

    def load_robot_obs(self, other_file):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        if self.load_libero_file == "h5":
            robot_obs[:6] = other_file['observation']['tcp_pose'][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file['observation']['gripper_state'][()]
            robot_obs[7:14] = other_file['observation']['proprio'][()]
            if self.gripper_width:
                robot_obs[-2:] = other_file['observation']['gripper_position'][()]
        elif self.load_libero_file == "npz":
            robot_obs[:6] = other_file["observation_tcp_pose"][:6]
            euler = R.from_euler("xyz", robot_obs[3:6], degrees=False)
            euler = euler.as_euler("xyz", degrees=False)
            robot_obs[3:6] = euler
            robot_obs[-1] = other_file["observation_gripper_state"]
            robot_obs[7:14] = other_file["observation_proprio"]
            if self.gripper_width:
                robot_obs[-2:] = other_file["observation_gripper_position"]
        else:
            raise NotImplementedError      

        return robot_obs

    def load_scene_obs(self, episode_id, step_id):
        scene_obs = np.zeros(self.proprio_state.n_scene_obs)

        return scene_obs

    def __len__(self):
        if self.small_size:
            return self.small_size
        else:
            return self.length

class DiskLiberoDataset(Dataset):
    def __init__(
        self, 
        # image_fn: Callable,
        # text_fn: Callable,
        seer_image_fn: Callable,
        seer_text_fn: Callable,
        vita_image_fn: Callable,
        vita_text_fn: Callable,
        dataset_names: List[str],
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        small_size: int = 0, 
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.datasets = [
            BaseLiberoDataset(
                *args, 
                dataset_name=dataset_name,
                act_step=act_step,
                small_size=small_size,
                gripper_width=gripper_width,
                **kwargs,
                
            ) for dataset_name in dataset_names
        ]
        # self.image_fn = image_fn
        # self.text_fn = text_fn
        self.seer_image_fn = seer_image_fn
        self.seer_text_fn = seer_text_fn
        self.vita_image_fn = vita_image_fn
        self.vita_text_fn = vita_text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]
    
    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    
    def collator(self, sample):
        
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        seer_image_tensors = torch.stack([self.seer_image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        seer_gripper_tensors = torch.stack([self.seer_image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        vita_image_tensors = torch.stack([self.vita_image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        vita_gripper_tensors = torch.stack([self.vita_image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        seer_text_tensors = self.seer_text_fn(stacked_language)
        vita_text_tensors = None
         
        if self.rgb_pad != -1:
            bs, seq_len = seer_image_tensors.shape[:2]
            if self.traj_cons:
                seer_image_tensors = self.rgb_shift.forward_traj(seer_image_tensors)
            else:
                seer_image_tensors = seer_image_tensors.view(bs*seq_len, *seer_image_tensors.shape[2:])
                seer_image_tensors = self.rgb_shift(seer_image_tensors)
                seer_image_tensors = seer_image_tensors.view(bs, seq_len, *seer_image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = seer_gripper_tensors.shape[:2]
            if self.traj_cons:
                seer_gripper_tensors = self.gripper_shift.forward_traj(seer_gripper_tensors)
            else:
                seer_gripper_tensors = seer_gripper_tensors.view(bs * seq_len, *seer_gripper_tensors.shape[2:])
                seer_gripper_tensors = self.gripper_shift(seer_gripper_tensors)
                seer_gripper_tensors = seer_gripper_tensors.view(bs, seq_len, *seer_gripper_tensors.shape[1:])
         
        if self.rgb_pad != -1:
            bs, seq_len = vita_image_tensors.shape[:2]
            if self.traj_cons:
                vita_image_tensors = self.rgb_shift.forward_traj(vita_image_tensors)
            else:
                vita_image_tensors = vita_image_tensors.view(bs*seq_len, *vita_image_tensors.shape[2:])
                vita_image_tensors = self.rgb_shift(vita_image_tensors)
                vita_image_tensors = vita_image_tensors.view(bs, seq_len, *vita_image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = vita_gripper_tensors.shape[:2]
            if self.traj_cons:
                vita_gripper_tensors = self.gripper_shift.forward_traj(vita_gripper_tensors)
            else:
                vita_gripper_tensors = vita_gripper_tensors.view(bs * seq_len, *vita_gripper_tensors.shape[2:])
                vita_gripper_tensors = self.gripper_shift(vita_gripper_tensors)
                vita_gripper_tensors = vita_gripper_tensors.view(bs, seq_len, *vita_gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            seer_image_tensors = seer_image_tensors[:, :-(self.act_step-1)]
            seer_gripper_tensors = seer_gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return seer_image_tensors, vita_image_tensors, seer_text_tensors, vita_text_tensors, 
    action_tensors, seer_gripper_tensors, vita_gripper_tensors, state_tensors, robot_obs, stacked_language

# def get_libero_pretrain_dataset(args, seer_image_processor, seer_tokenizer, 
#                                 vita_image_processor, vita_tokenizer, epoch=0, floor=False):
#     dataset_names = ["libero_90_converted"]
#     shared_epoch = SharedEpoch(epoch=epoch)
#     # preprocess_image_fn = functools.partial(
#     #     preprocess_image, image_processor=image_processor
#     # )
#     # preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
#     preprocess_image_fn_seer = functools.partial(
#         preprocess_image, image_processor=seer_image_processor
#     )
#     preprocess_image_fn_vita = functools.partial(
#         preprocess_image_vita, image_processor=vita_image_processor
#     )
#     preprocess_text_fn_seer = functools.partial(preprocess_text_calvin, tokenizer=seer_tokenizer)
#     preprocess_text_fn_vita = functools.partial(preprocess_text_calvin, tokenizer=vita_tokenizer)
    
    
#     libero_dataset = DiskLiberoDataset(
#         seer_image_fn=preprocess_image_fn_seer,
#         seer_text_fn=preprocess_text_fn_seer,
#         vita_image_fn=preprocess_image_fn_vita,
#         vita_text_fn=preprocess_text_fn_vita,
#         dataset_names=dataset_names,
#         rgb_pad=args.rgb_pad,
#         gripper_pad=args.gripper_pad,
#         traj_cons=args.traj_cons,
#         text_aug=args.text_aug,
#         act_step=args.multi_step_action,
#         root_dir=args.root_dir,
#         image_primary_size=args.image_primary_size,
#         image_wrist_size=args.image_wrist_size,
#         window_size=args.window_size,
#         dif_ws=args.dif_ws,
#         min_window_size=args.min_window_size,
#         max_window_size=args.max_window_size,
#         primary_mode=args.primary_mode,
#         small_size=args.small_size,
#         dataset_info='libero_90_converted',
#         gripper_width=args.gripper_width,
#         load_libero_file=args.load_libero_file,
#     )
#     round_fn = math.floor if floor else math.ceil
#     num_samples = len(libero_dataset)
#     global_batch_size = args.batch_size * args.world_size
#     num_batches = round_fn(num_samples / global_batch_size)
#     num_workers = max(1, args.workers)
#     num_worker_batches = round_fn(num_batches / num_workers)  
#     num_batches = num_worker_batches * num_workers
#     num_samples = num_batches * global_batch_size
#     sampler = DistributedSampler(
#         libero_dataset,
#         num_replicas=args.world_size,
#         rank=args.rank,
#         shuffle=True,
#         seed=args.seed,
#         drop_last=True,
#     )
#     dataloader = DataLoader(
#         libero_dataset,
#         batch_size=args.batch_size,
#         pin_memory=False,
#         num_workers=num_workers,
#         prefetch_factor=32,
#         sampler=sampler,
#         persistent_workers=True,
#         collate_fn=libero_dataset.collator,
#         drop_last=True
#     )
#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=libero_dataset)

def get_libero_finetune_dataset(args, seer_image_processor, seer_tokenizer, 
                                vita_image_processor, vita_tokenizer, epoch=0, floor=False):
    
    if 'finetune' in args.finetune_type:
        dataset_names = ["libero_10_converted"]
    elif 'pretrain' in args.finetune_type:
        dataset_names = ["libero_90_converted"]
    elif 'goal' in args.finetune_type:
        dataset_names = ["libero_goal_converted"]
    elif 'object' in args.finetune_type:
        dataset_names = ["libero_object_converted"]
    elif 'spatial' in args.finetune_type:
        dataset_names = ["libero_spatial_converted"]
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn_seer = functools.partial(
        preprocess_image, image_processor=seer_image_processor
    )
    preprocess_image_fn_vita = functools.partial(
        preprocess_image_vita, image_processor=vita_image_processor
    )
    preprocess_text_fn_seer = functools.partial(preprocess_text_calvin, tokenizer=seer_tokenizer)
    preprocess_text_fn_vita = functools.partial(preprocess_text_calvin, tokenizer=vita_tokenizer)
    
    libero_dataset = DiskLiberoDataset(
        seer_image_fn=preprocess_image_fn_seer,
        seer_text_fn=preprocess_text_fn_seer,
        vita_image_fn=preprocess_image_fn_vita,
        vita_text_fn=preprocess_text_fn_vita,
        dataset_names=dataset_names,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        image_primary_size=args.image_primary_size,
        image_wrist_size=args.image_wrist_size,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        primary_mode=args.primary_mode,
        small_size=args.small_size,
        dataset_info=dataset_names[0],
        gripper_width=args.gripper_width,
        load_libero_file=args.load_libero_file,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(libero_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        libero_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        libero_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=libero_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, 
                    sampler=sampler, dataset=libero_dataset)

# class BaseRealDataset(Dataset):
#     def __init__(
#         self,
#         dataset_name: str,
#         root_dir: str,
#         image_primary_size=200,
#         image_wrist_size=84,
#         obs_space: DictConfig = obs_config,
#         proprio_state: DictConfig = prop_state,
#         transforms: Dict = {},
#         window_size: int = 16,
#         min_window_size: int = 16,
#         max_window_size: int = 16,
#         pad: bool = True,
#         aux_lang_loss_window: int = 1,
#         text_aug=False,
#         dif_ws=False,
#         act_step: int = 1,
#         key: str = "lang",
#         language_mode: str = "language_instruction",
#         primary_mode: str = "image_primary",
#         dataset_info: str = "",
#         small_size: int = 0,
#         gripper_width: bool = False,
#         load_real_file: str = "npz", 
#         max_rel_pos: float = 0.02,
#         max_rel_orn: float = 0.05,
#         magic_scaling_factor_pos: float = 1.0,
#         magic_scaling_factor_orn: float = 1.0,
#         use_aug_data: bool = False,
#         **kwargs: Any,
#     ):
#         self.dataset_name = dataset_name
#         self.dataset_info = dataset_info
#         self.root_dir = root_dir 
#         self.dataset_path = f'{root_dir}/{dataset_name}'
#         self.conf_path = '~/petreloss.conf'
#         self.image_primary_size = image_primary_size
#         self.image_wrist_size = image_wrist_size
#         self.image_preprocess = None
#         self.observation_space = obs_space
#         self.proprio_state = proprio_state
#         self.transforms = transforms
#         self.with_lang = key == "lang"
#         self.relative_actions = "rel_actions" in self.observation_space["actions"]
#         self.pad = pad
#         self.window_size = window_size
#         self.language_mode = language_mode
#         self.primary_mode = primary_mode
#         self.small_size = small_size
#         self.use_aug_data = use_aug_data
#         if not dif_ws:
#             self.min_window_size = window_size + act_step - 1
#             self.max_window_size = window_size + act_step - 1
#         else:
#             raise NotImplementedError
#         assert self.max_window_size == self.min_window_size
#         self.aux_lang_loss_window = aux_lang_loss_window
#         self.text_aug = text_aug
#         self.act_step = act_step
#         self.max_rel_pos = max_rel_pos
#         self.max_rel_orn = max_rel_orn
#         self.magic_scaling_factor_pos = magic_scaling_factor_pos
#         self.magic_scaling_factor_orn = magic_scaling_factor_orn
#         logger.info(f"loading dataset at {root_dir}/{dataset_name}")
#         logger.info("finished loading dataset")
#         assert os.path.exists(f"./data_info/{self.dataset_info}.json")
#         with open(f"./data_info/{self.dataset_info}.json", 'r') as f:
#             self.episode_info_list = json.load(f)
#             self.episode_list = [f[0] for f in self.episode_info_list]
#             self.num_step_per_episode = [f[1] - self.max_window_size for f in self.episode_info_list]
#             self.num_episode = len(self.episode_list)
#         self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
#         self.length = self.accumulated_num_step[-1]
#         self.gripper_width = gripper_width
#         self.load_real_file = load_real_file

    # def process_rgb(
    #     self,
    #     episode: Dict[str, np.ndarray],
    #     observation_space: DictConfig,
    #     transforms: Dict,
    #     seq_idx: int = 0,
    #     window_size: int = 0,
    # ) -> Dict[str, Dict[str, torch.Tensor]]:
    #     rgb_obs_keys = observation_space["rgb_obs"]
    #     seq_rgb_obs_dict = {}
    #     for _, rgb_obs_key in enumerate(rgb_obs_keys):
    #         rgb_obs = episode[rgb_obs_key]
    #         # expand dims for single environment obs
    #         if len(rgb_obs.shape) != 4:
    #             rgb_obs = np.expand_dims(rgb_obs, axis=0)
    #         assert len(rgb_obs.shape) == 4
    #         if window_size == 0 and seq_idx == 0:  # single file loader
    #             # To Square image
    #             seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
    #         else:  # episode loader
    #             seq_rgb_obs_ = torch.from_numpy(
    #                 rgb_obs[seq_idx : seq_idx + window_size]
    #             ).byte()
            
    #         if rgb_obs_key in transforms:
    #             seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
    #         seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    #     # shape: N_rgb_obs x (BxHxWxC)
    #     return {"rgb_obs": seq_rgb_obs_dict}

#     def _get_pad_size(self, sequence: Dict) -> int:
#         """
#         Determine how many frames to append to end of the sequence

#         Args:
#             sequence: Loaded sequence.

#         Returns:
#             Number of frames to pad.
#         """
#         return self.max_window_size - len(sequence["actions"])

#     @staticmethod
#     def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
#         """
#         Pad a sequence Tensor by repeating last element pad_size times.

#         Args:
#             input_tensor: Sequence to pad.
#             pad_size: Number of frames to pad.

#         Returns:
#             Padded Tensor.
#         """
#         if head:
#             last_repeated = torch.repeat_interleave(
#                 torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
#             )
#             padded = torch.vstack((last_repeated, input_tensor))
#         else:
#             last_repeated = torch.repeat_interleave(
#                 torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
#             )
#             padded = torch.vstack((input_tensor, last_repeated))

#         return padded

#     @staticmethod
#     def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
#         """
#         Pad a Tensor with zeros.

#         Args:
#             input_tensor: Sequence to pad.
#             pad_size: Number of frames to pad.

#         Returns:
#             Padded Tensor.
#         """
#         zeros_repeated = torch.repeat_interleave(
#             torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
#             repeats=pad_size,
#             dim=0,
#         )
#         if head:
#             padded = torch.vstack((zeros_repeated, input_tensor))
#         else:
#             padded = torch.vstack((input_tensor, zeros_repeated))

#         return padded

#     def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
#         """
#         Pad a sequence by repeating the last frame.

#         Args:
#             seq: Sequence to pad.
#             pad_size: Number of frames to pad.

#         Returns:
#             Padded sequence.
#         """
#         seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
#         seq.update(
#             {
#                 "rgb_obs": {
#                     k: self._pad_with_repetition(v, pad_size, head)
#                     for k, v in seq["rgb_obs"].items()
#                 }
#             }
#         )
#         seq.update(
#             {
#                 "depth_obs": {
#                     k: self._pad_with_repetition(v, pad_size, head)
#                     for k, v in seq["depth_obs"].items()
#                 }
#             }
#         )
#         #  todo: find better way of distinguishing rk and play action spaces
#         if not self.relative_actions:
#             if head:
#                 seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
#             else:
#                 # repeat action for world coordinates action space
#                 seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
#         else:
#             # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
#             if head:
#                 seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
#             else:
#                 seq_acts = torch.cat(
#                     [
#                         self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
#                         self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
#                     ],
#                     dim=-1,
#                 )
#             seq.update({"actions": seq_acts})
#         seq.update(
#             {
#                 "state_info": {
#                     k: self._pad_with_repetition(v, pad_size, head)
#                     for k, v in seq["state_info"].items()
#                 }
#             }
#         )

#         return seq

#     def process_language(
#         self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
#     ):
#         return {"lang": episode["language"]}

#     def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
#         """
#         Get sequence of dataset.

#         Args:
#             idx: Index of the sequence.

#         Returns:
#             Loaded sequence.
#         """
#         if isinstance(idx, int):
#             if self.min_window_size == self.max_window_size:
#                 window_size = self.max_window_size
#             else:
#                 logger.error(
#                     f"min_window_size {self.min_window_size} != max_window_size {self.max_window_size}"
#                 )
#                 raise ValueError
#         else:
#             idx, window_size = idx
#         head = False
#         sequence = self._get_sequences(idx, window_size, head=head) # TODO

#         if self.pad:
#             pad_size = self._get_pad_size(sequence)
#             sequence = self._pad_sequence(sequence, pad_size, head=head)

#         new_list = []
#         np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
#         for i in range(np_rgb.shape[0]):
#             new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
#         sequence["rgb_obs"]["rgb_static"] = new_list
#         new_list = []
#         np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
#         for i in range(np_gripper.shape[0]):
#             new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
#         sequence["rgb_obs"]["rgb_gripper"] = new_list

#         return sequence

#     def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
#         episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
#         if episode_id - 1 >= 0:
#             start_id = idx - self.accumulated_num_step[episode_id - 1]
#         else:
#             start_id = idx
#         num_step_per_episode = self.num_step_per_episode[episode_id]
#         end_id = min(start_id + window_size, num_step_per_episode)

#         if self.use_aug_data:
#             demo_list = self.episode_info_list[episode_id][2:]
#             start_id, end_id = demo_list[start_id]
#         episode_id = self.episode_list[episode_id] 
#         exp_id = episode_id.split("/")[0]
#         episodes = []
#         for step_id in range(start_id, end_id):
#             data_dict = {}
#             try:
#                 str_step_id = str(step_id).zfill(4)
#                 if self.load_real_file == "npz":
#                     other_path = f"{self.dataset_path}/{episode_id}/steps/{str_step_id}/other.npz"
#                     other_file = np.load(other_path, allow_pickle=True)
#             except:
#                 print("episode_id :", episode_id)
#                 print("step_id :", str_step_id)
#                 print("num_step_per_episode :", num_step_per_episode)
#                 print("other_path", f"{self.dataset_path}/{episode_id}/steps/{str_step_id}/other.npz")
#             data_dict["rgb_static"] = self.load_primary_rgb(episode_id, str_step_id, self.primary_mode)
#             data_dict["rgb_gripper"] = self.load_wrist_rgb(episode_id, str_step_id)
#             data_dict["rel_actions"] = self.load_action(other_file, exp_id=exp_id)
#             data_dict["robot_obs"] = self.load_robot_obs(other_file)
#             data_dict["scene_obs"] = self.load_scene_obs(episode_id, str_step_id)
#             episodes.append(data_dict)
#         keys = list(chain(*self.observation_space.values()))
#         keys.remove("language")
#         keys.append("scene_obs")
#         episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
#         episode["language"] = self.load_language_instruction(other_file, self.language_mode)
#         seq_state_obs = process_state(
#             episode, self.observation_space, self.transforms, self.proprio_state
#         )
#         seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
#         seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
#         seq_acts = process_actions(episode, self.observation_space, self.transforms)
#         info = get_state_info_dict(episode)
#         info["use_for_aux_lang_loss"] = False
#         seq_lang = self.process_language(episode, self.transforms, self.with_lang)

#         seq_dict = {
#             **seq_state_obs,
#             **seq_rgb_obs,
#             **seq_depth_obs,
#             **seq_acts,
#             **info,
#             **seq_lang,
#         }  
#         seq_dict["idx"] = idx  
#         seq_dict["episode_id"] = episode_id

#         return seq_dict

#     def load_primary_rgb(self, episode_id, step_id, primary_mode="image_primary"):
#         image_primary_path = f'{self.dataset_path}/{episode_id}/steps/{step_id}/{primary_mode}.jpg'
#         image_primary = np.array(Image.open(image_primary_path).convert("RGB"))
        
#         return image_primary.astype(np.uint8)

#     def load_wrist_rgb(self, episode_id, step_id):
#         image_wrist_path = f'{self.dataset_path}/{episode_id}/steps/{step_id}/image_wrist.jpg'
#         image_wrist = np.array(Image.open(image_wrist_path).convert("RGB"))
        
#         return image_wrist.astype(np.uint8)

#     def load_language_instruction(self, other_file, language_mode="language_instruction"):
#         if self.load_real_file == "npz":
#             language_instruction = other_file[language_mode].tobytes().decode('utf-8')
#         else:
#             raise NotImplementedError
        
#         return language_instruction

#     def load_action(self, other_file, exp_id):
#         if self.load_real_file == "npz":
#             action = other_file["delta_cur_2_last_action"]
#             action[:3] /= (self.max_rel_pos * self.magic_scaling_factor_pos)
#             action[3:6] /= (self.max_rel_orn * self.magic_scaling_factor_orn)
#         else:
#             raise NotImplementedError
        
#         return action

#     def load_robot_obs(self, other_file):
#         robot_obs = np.zeros(self.proprio_state.n_state_obs)
#         if self.load_real_file == "npz":
#             robot_obs[:6] = other_file["gripper_pose"]
#             robot_obs[-1] = other_file["gripper_open_state"]
#             robot_obs[7:14] = other_file["joints"]
#         else:
#             raise NotImplementedError        
        
#         return robot_obs

#     def load_scene_obs(self, episode_id, step_id):
#         scene_obs = np.zeros(self.proprio_state.n_scene_obs)
        
#         return scene_obs

#     def __len__(self):
#         if self.small_size:
#             return self.small_size
#         else:
#             return self.length

# class DiskRealDataset(Dataset):
#     def __init__(
#         self, 
#         image_fn: Callable,
#         text_fn: Callable,
#         dataset_names: List[str],
#         *args: Any,
#         rgb_pad: int = -1,
#         gripper_pad: int = -1,
#         traj_cons: bool = False,
#         act_step : int = 1,
#         small_size: int = 0, 
#         gripper_width: bool = False,
#         **kwargs: Any,
#     ):
#         super().__init__()
#         self.dataset_names = dataset_names
#         self.datasets = [
#                 BaseRealDataset(
#                     *args, 
#                     dataset_name=dataset_name,
#                     act_step=act_step,
#                     small_size=small_size,
#                     gripper_width=gripper_width,
#                     **kwargs,
                    
#                 ) for dataset_name in dataset_names
#             ]
#         self.image_fn = image_fn
#         self.text_fn = text_fn
#         self.rgb_pad = rgb_pad
#         self.gripper_pad = gripper_pad
#         self.traj_cons = traj_cons
#         self.act_step = act_step
#         if self.rgb_pad != -1:
#             self.rgb_shift = RandomShiftsAug(rgb_pad)
#         self.gripper_pad = gripper_pad
#         if self.gripper_pad != -1:
#             self.gripper_shift = RandomShiftsAug(gripper_pad)
#         self.length_each_dataset = [len(dataset) for dataset in self.datasets]
#         self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

#     def register_image_preprocess_hook(self, func):
#         self.image_preprocess = func

#     def __len__(self):
#         return self.accumulated_length_each_dataset[-1]

#     def __getitem__(self, idx):
#         dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
#         if dataset_id - 1 >= 0:
#             local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
#         else:
#             local_idx = idx

#         return self.datasets[dataset_id].__getitem__(local_idx)

#     def collator(self, sample):
#         action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
#         state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
#         image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
#         gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
#         stacked_language = [s["lang"] for s in sample]
#         episode_id = [s["episode_id"] for s in sample]
#         text_tensors = self.text_fn(stacked_language)

#         if self.rgb_pad != -1:
#             bs, seq_len = image_tensors.shape[:2]
#             if self.traj_cons:
#                 image_tensors = self.rgb_shift.forward_traj(image_tensors)
#             else:
#                 image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
#                 image_tensors = self.rgb_shift(image_tensors)
#                 image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
#         if self.gripper_pad != -1:
#             bs, seq_len = gripper_tensors.shape[:2]
#             if self.traj_cons:
#                 gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
#             else:
#                 gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
#                 gripper_tensors = self.gripper_shift(gripper_tensors)
#                 gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
#         robot_obs = torch.zeros(1)

#         if self.act_step != 1:
        
#             actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
#             for b in range(action_tensors.shape[0]):
#                 for ix in range(self.window_size):
#                     actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

#             robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
#             for b in range(action_tensors.shape[0]):
#                 for ix in range(self.window_size):
#                     robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
#             robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
#             action_tensors = actions
#             image_tensors = image_tensors[:, :-(self.act_step-1)]
#             gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
#             state_tensors = state_tensors[:, :-(self.act_step-1)]

#         return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

# def get_real_finetune_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
#     dataset_names = [args.real_dataset_names]
#     shared_epoch = SharedEpoch(epoch=epoch)
#     preprocess_image_fn = functools.partial(
#         preprocess_image, image_processor=image_processor
#     )
#     preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
#     real_dataset = DiskRealDataset(
#         image_fn=preprocess_image_fn,
#         text_fn=preprocess_text_fn,
#         dataset_names=dataset_names,
#         rgb_pad=args.rgb_pad,
#         gripper_pad=args.gripper_pad,
#         traj_cons=args.traj_cons,
#         text_aug=args.text_aug,
#         act_step=args.multi_step_action,
#         root_dir=args.root_dir,
#         image_primary_size=args.image_primary_size,
#         image_wrist_size=args.image_wrist_size,
#         window_size=args.window_size,
#         dif_ws=args.dif_ws,
#         min_window_size=args.min_window_size,
#         max_window_size=args.max_window_size,
#         primary_mode=args.primary_mode,
#         small_size=args.small_size,
#         dataset_info=args.real_dataset_names,
#         gripper_width=args.gripper_width,
#         load_real_file="npz",
#         use_aug_data=args.use_aug_data,
#         max_rel_pos=args.max_rel_pos,
#         max_rel_orn=args.max_rel_orn,
#         magic_scaling_factor_pos=args.magic_scaling_factor_pos,
#         magic_scaling_factor_orn=args.magic_scaling_factor_orn,
#     )
#     round_fn = math.floor if floor else math.ceil
#     num_samples = len(real_dataset)
#     global_batch_size = args.batch_size * args.world_size
#     num_batches = round_fn(num_samples / global_batch_size)
#     num_workers = max(1, args.workers)
#     num_worker_batches = round_fn(num_batches / num_workers)  
#     num_batches = num_worker_batches * num_workers
#     num_samples = num_batches * global_batch_size
#     sampler = DistributedSampler(
#         real_dataset,
#         num_replicas=args.world_size,
#         rank=args.rank,
#         shuffle=True,
#         seed=args.seed,
#         drop_last=True,
#     )
#     dataloader = DataLoader(
#         real_dataset,
#         batch_size=args.batch_size,
#         pin_memory=False,
#         num_workers=num_workers,
#         prefetch_factor=3,
#         sampler=sampler,
#         persistent_workers=True,
#         collate_fn=real_dataset.collator,
#         drop_last=True
#     )
#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=real_dataset)


# if __name__ == "__main__":
#     from arguments_utils import get_args_and_cfg
#     from tqdm import tqdm
#     args, _ = get_args_and_cfg()
#     device='cuda'
#     model, image_processor = clip.load("ViT-B/32", device=device)
#     calvin_dataset = get_libero_pretrain_dataset(args, image_processor, clip, epoch=0)
#     calvin_dataset.set_epoch(epoch=0)
#     calvin_loader = calvin_dataset.dataloader
#     num_batches_per_epoch = calvin_loader.num_batches
#     t = tqdm(
#         enumerate(calvin_loader),
#         disable=args.rank != 0,
#         total=num_batches_per_epoch,
#     )
#     mv_avg_loss = []
#     for num_steps, batch in t:
#         if num_steps > 0:
#             torch.cuda.synchronize()
#             t2 = time.time()
#             print("t2 - t1", t2 - t1)
#         torch.cuda.synchronize()
#         t1 = time.time()
