import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_vita import GR00T_N1
import os

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取给定状态的动作的抽象方法。

        Args:
            observations: 来自环境的观测。

        Returns:
            要在环境中采取的动作，以字典格式返回。
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        返回策略的模态配置。
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    Gr00t 模型检查点的包装器，处理模型加载、应用变换、
    进行预测和反向应用变换。这会加载一些与 Gr00t 模型检查点
    相关的自定义配置、统计数据和元数据。
    """

    def __init__(
        self,
        model_path_: str,
        embodiment_tag_: Union[str, EmbodimentTag],
        modality_config_: Dict[str, ModalityConfig],
        modality_transform_: ComposedModalityTransform,
        denoising_steps_: Optional[int] = None,
        device_: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化 Gr00tPolicy。

        Args:
            model_path (str): 模型检查点目录的路径或 huggingface hub id。
            modality_config (Dict[str, ModalityConfig]): 模型的模态配置。
            modality_transform (ComposedModalityTransform): 模型的模态变换。
            embodiment_tag (Union[str, EmbodimentTag]): 模型的具身标签。
            denoising_steps: 用于动作头的去噪步数。
            device (Union[int, str]): 运行模型的设备。
        """
        try:
            # 注意：这会返回模型的本地路径，通常保存在 ~/.cache/huggingface/hub/
            model_path_ = snapshot_download(model_path_, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path_}"
            )

        self._modality_config = modality_config_
        self._modality_transform = modality_transform_
        self._modality_transform.eval()  # 设置为评估模式
        self.model_path = Path(model_path_)
        self.device = device_

        # 如果需要，将字符串具身标签转换为 EmbodimentTag 枚举
        if isinstance(embodiment_tag_, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag_)
        else:
            self.embodiment_tag = embodiment_tag_

        # 加载模型
        self._load_model(model_path_)
        # 加载变换
        self._load_metadata(self.model_path / "experiment_cfg")
        # 加载时间范围
        self._load_horizons()

        if denoising_steps_ is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps_
                print(f"Set action denoising steps to {denoising_steps_}")

    def apply_transforms(self, obs_: Dict[str, Any]) -> Dict[str, Any]:
        """
        对观测应用变换。

        Args:
            obs (Dict[str, Any]): 要变换的观测。

        Returns:
            Dict[str, Any]: 变换后的观测。
        """
        # 在应用变换之前确保正确的维度
        return self._modality_transform(obs_)

    def unapply_transforms(self, action_: Dict[str, Any]) -> Dict[str, Any]:
        """
        对动作反向应用变换。

        Args:
            action (Dict[str, Any]): 要反向应用变换的动作。

        Returns:
            Dict[str, Any]: 反向变换后的动作。
        """
        return self._modality_transform.unapply(action_)

    def get_action(self, observations_: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用模型进行预测。
        Args:
            obs (Dict[str, Any]): 要进行预测的观测。

        例如 obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        或者使用批量输入:
        例如 obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: 预测的动作。
        """
        # 让 get_action 处理批量和单个输入
        is_batch_ = self._check_state_is_batched(observations_)
        if not is_batch_:
            observations_ = unsqueeze_dict_values(observations_)

        # 应用变换
        normalized_input_ = self.apply_transforms(observations_)

        normalized_action_ = self._get_action_from_normalized_input(normalized_input_)
        unnormalized_action_ = self._get_unnormalized_action(normalized_action_)

        if not is_batch_:
            unnormalized_action_ = squeeze_dict_values(unnormalized_action_)
            
        unnormalized_action_["prev_action_chunk"]=normalized_action_
        
        return unnormalized_action_

    def get_hidden_states(self, observations_: Dict[str, Any]) -> torch.Tensor:
        """
        从模型中获取隐藏状态而不生成动作。
        
        Args:
            observations (Dict[str, Any]): 要处理的观测。
            
        Returns:
            torch.Tensor: 来自主干模型的隐藏状态。
        """
        # 检查输入是否为批量
        is_batch_ = self._check_state_is_batched(observations_)
        if not is_batch_:
            observations_ = unsqueeze_dict_values(observations_)

        # 应用变换
        normalized_input_ = self.apply_transforms(observations_)

        print("observations!! ", observations_)

        # 从模型中获取隐藏状态
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            hidden_states_ = self.model.get_hidden_states(normalized_input_)
            
        return hidden_states_

    def _get_action_from_normalized_input(self, normalized_input_: Dict[str, Any]) -> torch.Tensor:
        # 如果需要，设置 autocast 上下文
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred_ = self.model.get_action(normalized_input_)
        # print(model_pred.keys())
        normalized_action_ = model_pred_["action_pred"].float()
        return normalized_action_

    def _get_unnormalized_action(self, normalized_action_: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action_.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        获取模型的模态配置，覆盖基类方法
        """
        return self._modality_config

    def get_realtime_action(self, 
                            observations_: Dict[str, Any]) -> Dict[str, Any]:
        inference_delay_ = observations_.get("inference_delay", 0)
        if "inference_delay" in observations_:
            del observations_["inference_delay"]
        execute_horizon_ = observations_.get("execute_horizon", 8)
        if "execute_horizon" in observations_:
            del observations_["execute_horizon"]
        prefix_attention_horizon_=self.model.config.action_horizon - execute_horizon_  # 16-8
        assert execute_horizon_>=inference_delay_, f"{execute_horizon_=} {inference_delay_=}"
        
        
        prev_action_chunk_ = observations_.get("prev_action_chunk")
        if prev_action_chunk_ is not None:
            prev_action_chunk_ = observations_["prev_action_chunk"]
            prev_action_chunk_ = torch.cat(
                    (
                        prev_action_chunk_[:, execute_horizon_:],
                        torch.zeros((prev_action_chunk_.shape[0], execute_horizon_, prev_action_chunk_.shape[2])).to(device=prev_action_chunk_.device)
                    ), dim=1)
        else:
            prev_action_chunk_ = self.get_action(observations_)["prev_action_chunk"]
            print("First Startup")
        del observations_["prev_action_chunk"]
        
        is_batch_ = self._check_state_is_batched(observations_)
        if not is_batch_:
            observations_ = unsqueeze_dict_values(observations_)

        # Apply transforms
        normalized_input_ = self.apply_transforms(observations_)
        with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred_ = self.model.get_realtime_action(normalized_input_,
                                                        prev_action_chunk_,
                                                        inference_delay_,
                                                        prefix_attention_horizon_)
        
        normalized_action_ = model_pred_["action_pred"].float()
        normalized_action_tmp_=normalized_action_
        normalized_action_ = torch.cat(
                (
                    prev_action_chunk_[:, :inference_delay_],
                    normalized_action_[:, inference_delay_:]
                ), dim=1)
        
        unnormalized_action_ = self._get_unnormalized_action(normalized_action_.detach())

        if not is_batch_:
            unnormalized_action_ = squeeze_dict_values(unnormalized_action_)
        # print("unnormalized_action!! ", unnormalized_action["action.actions"].shape) # (16, 7)
        
        unnormalized_action_["prev_action_chunk"]=normalized_action_
        unnormalized_action_["tmp_test"]=normalized_action_tmp_
        return unnormalized_action_


    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """获取视频增量索引。"""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """获取状态增量索引。"""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """获取去噪步数。"""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """设置去噪步数。"""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs_: Dict[str, Any]) -> bool:
        for k_, v_ in obs_.items():
            if "state" in k_ and len(v_.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path_):
        model_ = GR00T_N1.from_pretrained(model_path_, torch_dtype=COMPUTE_DTYPE)
        
        model_.eval()  # 设置模型为评估模式
        model_.to(device=self.device)  # type: ignore

        self.model = model_

    def _load_metadata(self, exp_cfg_dir_: Path):
        """加载模型的变换。"""
        # 加载归一化统计数据的元数据
        metadata_path_ = exp_cfg_dir_ / "metadata.json"
        with open(metadata_path_, "r") as f_:
            metadatas_ = json.load(f_)

        # 获取特定具身的元数据
        metadata_dict_ = metadatas_.get(self.embodiment_tag.value)
        if metadata_dict_ is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path_}",
            )

        metadata_ = DatasetMetadata.model_validate(metadata_dict_)

        self._modality_transform.set_metadata(metadata_)
        self.metadata = metadata_

    def _load_horizons(self):
        """加载模型所需的时间范围。"""
        # 获取模态配置
        # 视频时间范围
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # 状态时间范围（如果使用）
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices_: np.ndarray):
        """断言增量索引是否有效。"""
        # 所有增量索引应该是非正数，因为无法获取未来的观测
        assert np.all(delta_indices_ <= 0), f"{delta_indices_=}"
        # 最后一个增量索引应该是 0，因为不使用最新的观测没有意义
        assert delta_indices_[-1] == 0, f"{delta_indices_=}"
        if len(delta_indices_) > 1:
            # 步长是一致的
            assert np.all(
                np.diff(delta_indices_) == delta_indices_[1] - delta_indices_[0]
            ), f"{delta_indices_=}"
            # 并且步长是正数
            assert (delta_indices_[1] - delta_indices_[0]) > 0, f"{delta_indices_=}"


#######################################################################################################


# 辅助函数
def unsqueeze_dict_values(data_: Dict[str, Any]) -> Dict[str, Any]:
    """
    对字典的值进行 unsqueeze 操作。
    这会将数据转换为批量大小为 1 的批量数据。
    """
    unsqueezed_data_ = {}
    for k_, v_ in data_.items():
        if isinstance(v_, np.ndarray):
            unsqueezed_data_[k_] = np.expand_dims(v_, axis=0)
        elif isinstance(v_, torch.Tensor):
            unsqueezed_data_[k_] = v_.unsqueeze(0)
        else:
            unsqueezed_data_[k_] = v_
    return unsqueezed_data_


def squeeze_dict_values(data_: Dict[str, Any]) -> Dict[str, Any]:
    """
    对字典的值进行 squeeze 操作。这会移除批量维度。
    """
    squeezed_data_ = {}
    for k_, v_ in data_.items():
        if isinstance(v_, np.ndarray):
            squeezed_data_[k_] = np.squeeze(v_)
        elif isinstance(v_, torch.Tensor):
            squeezed_data_[k_] = v_.squeeze()
        else:
            squeezed_data_[k_] = v_
    return squeezed_data_
