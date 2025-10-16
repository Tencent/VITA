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

import gymnasium as gym
import numpy as np


class ObsIndexSelectionWrapper(gym.Wrapper):
    def __init__(self, env, video_delta_indices, state_delta_indices):
        super().__init__(env)
        self.video_delta_indices = video_delta_indices
        self.video_horizon = len(video_delta_indices)
        self.assert_delta_indices(self.video_delta_indices, self.video_horizon)

        if state_delta_indices is not None:
            self.state_delta_indices = state_delta_indices
            self.state_horizon = len(state_delta_indices)
            self.assert_delta_indices(self.state_delta_indices, self.state_horizon)
        else:
            self.state_horizon = None
            self.state_delta_indices = None

        self._observation_space = self.convert_observation_space(
            self.observation_space,
            self.video_horizon,
            self.state_horizon,
        )

    def assert_delta_indices(self, indices_delta: np.ndarray, horizon_len: int):
        # 检查偏移索引数组的长度是否等于指定的视窗长度。
        # 在这个包装器中，这似乎是多余的，因为我们是从偏移索引中获取的视窗长度。
        # 但在策略中，视窗长度不是从偏移索引中导出的，但我们需要使其保持一致。
        # 为了使函数保持一致，我们在这里保留这个检查。
        assert len(indices_delta) == horizon_len, f"{indices_delta=}, {horizon_len=}"
        # 所有的偏移索引都应该是非正数，因为无法获取未来的观测数据。
        assert np.all(indices_delta <= 0), f"{indices_delta=}"
        # 最后一个偏移索引应该是0，因为不使用最新的观测数据是没有意义的。
        assert indices_delta[-1] == 0, f"{indices_delta=}"
        # 如果偏移索引数组的长度大于1，则进行以下检查。
        if len(indices_delta) > 1:
            # 步长是一致的（因为在真实的机器人实验中，我们实际上使用dt来获取观测数据，这要求步长是一致的）。
            assert np.all(
                np.diff(indices_delta) == indices_delta[1] - indices_delta[0]
            ), f"{indices_delta=}"
            # 并且步长是正数。
            assert (indices_delta[1] - indices_delta[0]) > 0, f"{indices_delta=}"

    def select_steps_for_values(self, value_data, indices_delta):
        """
        data_value: [L, ...]
        delta_indices: np.ndarray[int], please check `assert_delta_indices` to see the requirements
        """
        L = value_data.shape[0]
        assert L >= len(indices_delta), f"{L=}, {len(indices_delta)=}"
        selected_indices = (L - 1) + indices_delta
        assert selected_indices[0] >= 0, f"{L=}, {selected_indices=}"
        return value_data[selected_indices]

    def select_steps_for_obs(self, obs):
        new_obs = {}
        for k in obs.keys():
            if k.startswith("video"):
                new_obs[k] = self.select_steps_for_values(obs[k], self.video_delta_indices)
            elif k.startswith("state"):
                if self.state_delta_indices is not None:
                    new_obs[k] = self.select_steps_for_values(obs[k], self.state_delta_indices)
                else:
                    # Don't include the state in the observation
                    continue
            else:
                raise ValueError(f"Unknown key: {k}")
        return new_obs

    def convert_observation_space(self, observation_space, video_horizon, state_horizon):
        new_observation_space = {}
        for obs_key in observation_space.keys():
            obs_box = observation_space[obs_key]
            if obs_key.startswith("video"):
                current_horizon = video_horizon
            elif obs_key.startswith("state"):
                if state_horizon is not None:
                    current_horizon = state_horizon
                else:
                    # Don't include the state in the observation space
                    continue
            else:
                raise ValueError(f"Unknown key: {obs_key}")

            new_observation_space[obs_key] = gym.spaces.Box(
                low=obs_box.low[:current_horizon],
                high=obs_box.high[:current_horizon],
                shape=(current_horizon, *obs_box.shape[1:]),
                dtype=obs_box.dtype,
            )
        return gym.spaces.Dict(new_observation_space)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self.select_steps_for_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.select_steps_for_obs(obs)
        return obs, reward, terminated, truncated, info
