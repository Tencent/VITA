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

from abc import ABC, abstractmethod

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import VitaGR00TTransform, ActionHeadInferenceTransform


def _get_common_transforms(
    video_keys, state_keys, action_keys, observation_indices, action_indices
):
    """
    获取通用的数据转换流程。
    Args:
        video_keys (list): 视频数据键名列表。
        state_keys (list): 状态数据键名列表。
        action_keys (list): 动作数据键名列表。
        observation_indices (list): 观测索引列表。
        action_indices (list): 动作索引列表。

    Returns:
        list: 包含一系列数据转换操作的列表。
    """
    # 初始化一个空列表，用于存放所有数据转换操作
    transformation_list = [
        # === 视频数据转换 ===
        # 将视频数据转换为张量（Tensor）格式
        VideoToTensor(apply_to=video_keys),
        # 对视频进行随机裁剪，保留95%的区域
        VideoCrop(apply_to=video_keys, scale=0.95),
        # 将视频尺寸调整为224x224，使用线性插值
        VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
        # 对视频进行颜色抖动，增强模型对颜色变化的鲁棒性
        VideoColorJitter(
            apply_to=video_keys,
            brightness=0.3,  # 亮度抖动范围
            contrast=0.4,    # 对比度抖动范围
            saturation=0.5,  # 饱和度抖动范围
            hue=0.08,        # 色调抖动范围
        ),
        # 将处理后的视频数据转回Numpy数组格式
        VideoToNumpy(apply_to=video_keys),
        # === 状态数据转换 ===
        # 将状态数据转换为张量格式
        StateActionToTensor(apply_to=state_keys),
        # 对状态数据进行处理，例如归一化
        StateActionTransform(
            apply_to=state_keys,
            # 对每个状态键都采用最大最小值归一化
            normalization_modes={key: "min_max" for key in state_keys},
        ),
        # === 动作数据转换 ===
        # 将动作数据转换为张量格式
        StateActionToTensor(apply_to=action_keys),
        # 对动作数据进行处理，例如归一化
        StateActionTransform(
            apply_to=action_keys,
            # 对每个动作键都采用最大最小值归一化
            normalization_modes={key: "min_max" for key in action_keys},
        ),
        # === 数据拼接转换 ===
        # 将不同模态的数据进行拼接
        ConcatTransform(
            video_concat_order=video_keys,    # 视频数据拼接顺序
            state_concat_order=state_keys,    # 状态数据拼接顺序
            action_concat_order=action_keys,  # 动作数据拼接顺序
        ),
        # === GR00T特定转换 ===
        # 应用VITA GR00T模型的特定转换
        VitaGR00TTransform(
            state_horizon=len(observation_indices),  # 状态观测的时间窗口长度
            action_horizon=len(action_indices),    # 动作序列的长度
            max_state_dim=64,                      # 状态向量的最大维度
            max_action_dim=32,                     # 动作向量的最大维度
        ),
    ]
    # 返回构建好的转换操作列表
    return transformation_list

class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


# -------


class LiberoObjectVitaDataConfig(BaseDataConfig):
    video_keys = [
        'video.image',
        'video.wrist_image']
    state_keys = [
        'state.state']
    action_keys = [
        'action.actions']

    language_keys = ['annotation.human.task_description']
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys)
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys)
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys)
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys)
        modality_configs = {
            'video': video_modality,
            'state': state_modality,
            'action': action_modality,
            'language': language_modality}
        return modality_configs

    def transform(self):
        common_transforms = _get_common_transforms(
            self.video_keys, self.state_keys, self.action_keys, self.observation_indices, self.action_indices)
        return ComposedModalityTransform(transforms=common_transforms)

# =====

class RealDataRobotVitaDataConfig(BaseDataConfig):
    """真实机器人数据的 VITA 数据配置类
    
    该配置用于处理真实机器人采集的数据，包括视频观察、机器人状态、动作序列和语言指令。
    配置包含完整的数据预处理和转换管道，适配 VITA-GR00T 模型的输入格式。
    
    主要特点：
    - 支持顶视角相机视频输入
    - 包含手部和机器人本体的状态信息
    - 处理手部和机器人的动作序列
    - 支持自然语言任务描述
    - 预设 16 步动作预测
    """
    
    video_keys_ = [  # 视频模态的数据键，指定从数据集中读取哪些视频流
        "video.top",  # 顶视角相机的视频数据
    ]
    
    state_keys_ = [  # 状态模态的数据键，定义机器人的状态信息来源
        "state.hand",  # 手部/夹爪的状态（如位置、姿态、开合度等）
        "state.robot",  # 机器人本体的状态（如关节角度、位置等）
    ]
    
    action_keys_ = [  # 动作模态的数据键，定义动作空间的组成部分
        "action.hand",  # 手部/夹爪的动作指令
        "action.robot",  # 机器人本体的动作指令
    ]

    language_keys_ = ["annotation.human.task_description"]  # 语言模态的数据键：人工标注的任务描述
    observation_indices_ = [0]  # 观察索引：[0] 表示只使用当前时间步的观察数据，如设置为 [0, -1] 则会使用当前和前一帧的观察
    action_indices_ = list(range(16))  # 动作索引：预测未来 16 步的动作序列，即模型一次推理输出 16 个连续动作

    def modality_config(self):
        """构建各模态的配置信息
        
        为视频、状态、动作和语言四种模态创建对应的 ModalityConfig 对象。
        每个配置指定了该模态的时间索引和数据键。
        
        Returns:
            dict[str, ModalityConfig]: 包含四种模态配置的字典
                - "video": 视频模态配置
                - "state": 状态模态配置  
                - "action": 动作模态配置
                - "language": 语言模态配置
        """
        # 视频模态配置：使用当前观察帧，包含所有视频流
        video_modality_ = ModalityConfig(
            delta_indices=self.observation_indices_,  # 时间索引：[0] 当前帧
            modality_keys=self.video_keys_,           # 数据键：顶视角相机
        )
        
        # 状态模态配置：使用当前状态，包含手部和机器人状态
        state_modality_ = ModalityConfig(
            delta_indices=self.observation_indices_,  # 时间索引：[0] 当前状态
            modality_keys=self.state_keys_,           # 数据键：手部和机器人状态
        )
        
        # 动作模态配置：预测未来 16 步动作，包含手部和机器人动作
        action_modality_ = ModalityConfig(
            delta_indices=self.action_indices_,       # 时间索引：[0-15] 未来 16 步
            modality_keys=self.action_keys_,          # 数据键：手部和机器人动作
        )
        
        # 语言模态配置：使用任务描述，与观察对齐
        language_modality_ = ModalityConfig(
            delta_indices=self.observation_indices_,  # 时间索引：[0] 与观察对齐
            modality_keys=self.language_keys_,        # 数据键：任务描述文本
        )
        
        # 将所有模态配置组织成字典返回
        modality_configs_ = {
            "video": video_modality_,
            "state": state_modality_,
            "action": action_modality_,
            "language": language_modality_,
        }
        return modality_configs_

    def transform(self):
        common_transforms_ = _get_common_transforms(
            self.video_keys_, self.state_keys_, self.action_keys_, self.observation_indices_, self.action_indices_
        )
        return ComposedModalityTransform(transforms=common_transforms_)

######

class RealDataRobotVitaActionHeadDataConfig(BaseDataConfig):
    """DataConfig for action head inference with robot state (no video/language processing)."""
    
    video_key = ["video.top"]
    state_key = ["state.hand", "state.robot"]
    action_key = ["action.hand", "action.robot"]
    observation_index = [0]
    action_index = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        state_modalities = ModalityConfig(
            delta_indices=self.observation_index,
            modality_keys=self.state_key,
        )
        action_modalities = ModalityConfig(
            delta_indices=self.action_index,
            modality_keys=self.action_key,
        )
        modality_config = {
            "state": state_modalities,
            "action": action_modalities,
        }
        return modality_config

    def transform(self) -> ModalityTransform:
        transform = [
            # state transforms
            StateActionToTensor(apply_to=self.state_key),
            StateActionTransform(
                apply_to=self.state_key,
                normalization_modes={key: "min_max" for key in self.state_key},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_key),
            StateActionTransform(
                apply_to=self.action_key,
                normalization_modes={key: "min_max" for key in self.action_key},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_key,
                state_concat_order=self.state_key,
                action_concat_order=self.action_key,
            ),
            # action head inference transform (handles state concatenation and processing)
            ActionHeadInferenceTransform(
                state_horizon=len(self.observation_index),
                max_state_dim=64,
                state_concat_order=self.state_key,
            ),
        ]
        return ComposedModalityTransform(transforms=transform)


#-=-=-

DATA_CONFIG_MAP = {
    "libero_vita": LiberoObjectVitaDataConfig(),
    "real_data_robot_vita": RealDataRobotVitaDataConfig(),
    "real_data_robot_vita_action_head": RealDataRobotVitaActionHeadDataConfig(),
}
