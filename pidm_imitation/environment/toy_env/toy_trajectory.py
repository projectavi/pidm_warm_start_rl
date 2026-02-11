# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from pidm_imitation.constants import (
    ENV_CONFIG_FILE_SUFFIX,
    OBSERVATIONS_FILE_SUFFIX,
    STATES_FILE_SUFFIX,
    TRAJECTORY_DATA_FILE_SUFFIX,
)
from pidm_imitation.environment.toy_env.configs.toy_environment_config import (
    ToyEnvironmentConfig,
)
from pidm_imitation.environment.toy_env.toy_types import TerminationCondition
from pidm_imitation.utils import Trajectory, save_yaml
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.user_inputs import UserInputs, UserInputsLog, VideoTicks

log = Logger.get_logger(__name__)


def _get_max_possible_return_for_goal_env_config(
    env_config: ToyEnvironmentConfig,
) -> float:
    assert env_config.goal_config is not None
    num_goals = env_config.goal_config.num_goals
    term_cond = env_config.goal_config.termination_condition
    goal_reached_reward = env_config.goal_config.reward_per_reached_goal

    if term_cond == TerminationCondition.ALL_GOALS:
        return num_goals * goal_reached_reward
    elif term_cond == TerminationCondition.ANY_GOAL:
        return goal_reached_reward
    raise ValueError(f"Invalid termination condition {term_cond}")


def get_max_possible_return_for_env_config(env_config: ToyEnvironmentConfig) -> float:
    """Return maximum possible return (sum of rewards across episode) for given environment configuration."""
    return _get_max_possible_return_for_goal_env_config(env_config)


class ToyEnvironmentTrajectory(Trajectory):
    """This class represents information about recorded trajectories in the Toy Environment,
    including video frames, observations, states, actions, and environment configuration.
    """

    OBS_KEY: str = "obs"
    ACTION_KEY: str = "action"
    STATE_KEY: str = "state"
    REWARD_KEY: str = "reward"
    SUCCESS_KEY: str = "success"

    ADDITIONAL_LOAD_SAVE_KEYS = ["env_config", "observations", "states", "data"]
    LOAD_SAVE_KEYS = Trajectory.LOAD_SAVE_KEYS + ADDITIONAL_LOAD_SAVE_KEYS

    def __init__(
        self,
        video_frames: List[np.ndarray] | None = None,
        video_ticks: List[float] | None = None,
        observations: List[np.ndarray] | None = None,
        actions: List[np.ndarray] | None = None,
        states: List[np.ndarray] | None = None,
        rewards: List[float] | None = None,
        env_config: ToyEnvironmentConfig | None = None,
        video_fps: int = 10,
        color_format: str = "rgb",
    ):
        video_ticks_obj = VideoTicks()
        if video_ticks:
            video_ticks_obj.video_ticks = video_ticks

            # user inputs are recorded after a video tick and before the next video tick
            input_ticks = [
                ToyEnvironmentTrajectory.get_input_tick_from_video_tick(tick, video_fps)
                for tick in video_ticks[:-1]
            ]
        else:
            input_ticks = []
            video_ticks_obj = None

        user_inputs = ToyEnvironmentTrajectory.get_user_inputs_log(
            list(actions) if actions else [], input_ticks
        )
        super().__init__(
            video_frames=video_frames,
            video_ticks=video_ticks_obj,
            user_inputs=user_inputs,
            user_input_ticks=input_ticks,
            video_fps=video_fps,
            color_format=color_format,
        )
        self.observations = list(observations) if observations else []
        self.actions = list(actions) if actions else []
        self.states = list(states) if states else []
        self.rewards = list(rewards) if rewards else []

        self.env_config = env_config

        self.steps = max(0, len(self.video_frames) - 1)
        self.is_success: bool | None = None

    @staticmethod
    def get_input_tick_from_video_tick(video_tick: float, fps: int) -> float:
        return video_tick + 1 / (fps * 2)

    @staticmethod
    def init_from_dir(
        dirname: str,
        trajectory_name: str,
        video_width: int = -1,
        video_height: int = -1,
        color_format: str = "rgb",
    ) -> ToyEnvironmentTrajectory:
        """
        Load a trajectory from a directory.

        :param dirname: the directory containing the trajectory data.
        :param trajectory_name: the name of the trajectory.
        :param video_width: the width of the video frames.
        :param video_height: the height of the video frames.
        :param color_format: the color format of the video frames.
        :return: the loaded trajectory.
        """
        trajectory = ToyEnvironmentTrajectory()
        trajectory.load_from_dir(
            dirname,
            trajectory_name,
            video_width,
            video_height,
            color_format=color_format,
        )
        return trajectory

    @property
    def success(self) -> bool:
        if self.is_success is None:
            assert self.env_config is not None, "Environment configuration is not set."
            self.is_success = np.sum(
                self.rewards
            ) == get_max_possible_return_for_env_config(self.env_config)
        return self.is_success

    @success.setter
    def success(self, success_value: bool) -> None:
        self.is_success = success_value

    @staticmethod
    def get_data_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(
            dirname, trajectory_name + TRAJECTORY_DATA_FILE_SUFFIX + ".npz"
        )

    @staticmethod
    def get_config_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(dirname, trajectory_name + ENV_CONFIG_FILE_SUFFIX + ".yaml")

    @staticmethod
    def get_observations_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(
            dirname, trajectory_name + OBSERVATIONS_FILE_SUFFIX + ".npz"
        )

    @staticmethod
    def get_states_path(dirname: str, trajectory_name: str) -> str:
        return os.path.join(dirname, trajectory_name + STATES_FILE_SUFFIX + ".npz")

    @staticmethod
    def get_user_input(action: np.ndarray, tick: float):
        assert len(action) == 2, f"Invalid action shape: {action.shape}"
        return UserInputs(
            left_stick_x=float(action[0]),
            left_stick_y=float(action[1]),
            ticks=tick,
        )

    @staticmethod
    def get_user_inputs_log(
        actions: List[np.ndarray], action_ticks: List[float]
    ) -> UserInputsLog:
        """
        Get the user inputs log from the actions.

        :return: the user inputs log.
        """
        user_inputs_log = UserInputsLog()
        assert len(actions) == len(
            action_ticks
        ), "Number of actions and action ticks should match."
        for action, tick in zip(actions, action_ticks):
            user_input = ToyEnvironmentTrajectory.get_user_input(action, tick)
            user_inputs_log.record(user_input)
        return user_inputs_log

    def compute_episode_return(self, discount_factor: float = 1.0) -> float:
        """
        Computes the return of the episode.

        :param discount_factor: The discount factor to apply to future rewards. Default is 1.0 (no discounting).
        :return: The computed episode return.
        """
        ep_ret = 0.0
        for i, reward in enumerate(self.rewards):
            ep_ret += reward * (discount_factor**i)
        return ep_ret

    def compute_telemetry(self) -> np.ndarray:
        """
        Extracts telemetry information of the trajectory, requires state information.
        :return: 2D array of telemetry information of the agent throughout the trajectory
        """
        assert (
            len(self.states) > 0
        ), "States are required to extract telemetry information."
        return np.array(self.states)[:, :2]

    def save(self, filename_by_name: Dict[str, str], fourcc: str = "mp4v") -> List[str]:
        """
        Saves the trajectory data to files. Stores
            - video frames as .mp4 file
            - video ticks as .json file
            - user inputs as .json file
            - states as .npz file
            - observations as .npz file
            - trajectory data as .npz file (including all actions, observations, states,
                and rewards from gym env)
            - environment configuration as .yaml file

        :param video_filename: the filename to save the video frames.
        :param video_ticks_json_filename: the filename to save the video ticks.
        :param inputs_json_filename: the filename to save the user inputs.
        :param data_npz_filename: the filename to save the trajectory data.
        :param config_filename: the filename to save the environment configuration.
        :param observations_npz_filename: the filename to save the observations.
        :param states_npz_filename: the filename to save the states.
        :param fourcc: the fourcc codec to use for saving the video.
        :return: list of filenames where the trajectory data was saved.
        """
        super_filename_by_name = {}
        for k in Trajectory.LOAD_SAVE_KEYS:
            if (
                k in filename_by_name
                and k not in ToyEnvironmentTrajectory.ADDITIONAL_LOAD_SAVE_KEYS
            ):
                super_filename_by_name[k] = filename_by_name[k]

        saved_files = super().save(
            filename_by_name=super_filename_by_name, fourcc=fourcc
        )

        for name, filename in filename_by_name.items():
            if name in super_filename_by_name or not filename:
                continue

            if name == "env_config":
                save_yaml(self.env_config.get_config(), filename)
                saved_files.append(filename)
            elif name == "observations":
                np.savez(
                    filename, observations=np.array(self.observations, dtype=np.float32)
                )
                saved_files.append(filename)
            elif name == "states":
                np.savez(filename, states=np.array(self.states, dtype=np.float32))
                saved_files.append(filename)
            elif name == "data":
                data = {
                    ToyEnvironmentTrajectory.OBS_KEY: np.array(
                        self.observations, dtype=np.float32
                    ),
                    ToyEnvironmentTrajectory.ACTION_KEY: np.array(
                        self.actions, dtype=np.float32
                    ),
                    ToyEnvironmentTrajectory.STATE_KEY: np.array(
                        self.states, dtype=np.float32
                    ),
                    ToyEnvironmentTrajectory.REWARD_KEY: np.array(
                        self.rewards, dtype=np.float32
                    ),
                    ToyEnvironmentTrajectory.SUCCESS_KEY: np.array(self.success),
                }
                np.savez(filename, **data)  # type: ignore
                saved_files.append(filename)
            else:
                log.warning(
                    f"Unprocessed save name: {name}, only support {ToyEnvironmentTrajectory.LOAD_SAVE_KEYS}"
                )
        return saved_files

    def save_to_dir(
        self,
        dirname: str,
        trajectory_name: str,
        fourcc: str = "mp4v",
        save_video: bool = True,
    ) -> List[str]:
        """
        Saves the trajectory data to a directory. Stores
            - video frames as .mp4 file if save_video is True
            - video ticks as .json file
            - user inputs as .json file
            - states as .npz file
            - observations as .npz file
            - trajectory data as .npz file (including all actions, observations, states,
                and rewards from gym env)
            - environment configuration as .yaml file

        :param dirname: the directory to save the trajectory data.
        :param trajectory_name: the name of the trajectory.
        :param fourcc: the fourcc codec to use for saving the video.
        :param save_video: whether to save the video frames.
        :return: list of filenames where the trajectory data was saved.
        """
        os.makedirs(dirname, exist_ok=True)

        filename_by_name = {
            "video_ticks": ToyEnvironmentTrajectory.get_video_ticks_path(
                dirname, trajectory_name
            ),
            "inputs": ToyEnvironmentTrajectory.get_user_inputs_path(
                dirname, trajectory_name
            ),
            "env_config": ToyEnvironmentTrajectory.get_config_path(
                dirname, trajectory_name
            ),
            "observations": ToyEnvironmentTrajectory.get_observations_path(
                dirname, trajectory_name
            ),
            "states": ToyEnvironmentTrajectory.get_states_path(
                dirname, trajectory_name
            ),
            "data": ToyEnvironmentTrajectory.get_data_path(dirname, trajectory_name),
        }
        if save_video:
            filename_by_name["video"] = ToyEnvironmentTrajectory.get_video_path(
                dirname, trajectory_name
            )

        return self.save(
            filename_by_name=filename_by_name,
            fourcc=fourcc,
        )

    def load(
        self,
        filename_by_name: Dict[str, str],
        video_width: int = -1,
        video_height: int = -1,
    ):
        self.steps = None

        super_filename_by_name = {}
        for k in Trajectory.LOAD_SAVE_KEYS:
            if (
                k in filename_by_name
                and k not in ToyEnvironmentTrajectory.ADDITIONAL_LOAD_SAVE_KEYS
            ):
                super_filename_by_name[k] = filename_by_name[k]

        super().load(
            filename_by_name=super_filename_by_name,
            video_width=video_width,
            video_height=video_height,
        )

        # compute steps from loaded info
        if self.video_frames:
            self.steps = len(self.video_frames) - 1

        if self.video_ticks:
            if self.steps is None:
                self.steps = len(self.video_ticks.video_ticks) - 1
            else:
                assert len(self.video_ticks.video_ticks) == self.steps + 1

        for name, filename in filename_by_name.items():
            if name in super_filename_by_name or not filename:
                continue

            if name == "env_config":
                self.env_config = ToyEnvironmentConfig(filename)
            elif name == "data":
                data = np.load(filename, allow_pickle=True)
                self.observations = list(data[ToyEnvironmentTrajectory.OBS_KEY])
                self.actions = list(data[ToyEnvironmentTrajectory.ACTION_KEY])
                self.states = list(data[ToyEnvironmentTrajectory.STATE_KEY])
                self.rewards = list(data[ToyEnvironmentTrajectory.REWARD_KEY])
                self.success = data[ToyEnvironmentTrajectory.SUCCESS_KEY]
                assert (
                    len(self.observations)
                    == len(self.states)
                    == len(self.rewards) + 1
                    == len(self.actions) + 1
                )

                if self.steps is None:
                    self.steps = len(self.rewards)
                else:
                    assert len(self.rewards) == self.steps
            elif name == "observations":
                data = np.load(filename, allow_pickle=True)
                self.observations = list(data["observations"])
                assert len(self.observations) == self.steps + 1
            elif name == "states":
                data = np.load(filename, allow_pickle=True)
                self.states = list(data["states"])
                assert len(self.states) == self.steps + 1
            else:
                log.warning(
                    f"Unprocessed load name: {name}, only support {ToyEnvironmentTrajectory.LOAD_SAVE_KEYS}"
                )

    def load_from_dir(
        self,
        dirname: str,
        trajectory_name: str,
        video_width: int = -1,
        video_height: int = -1,
        color_format: str = "rgb",
    ):
        """
        Loads a trajectory and its data from a directory. The directory should contain
            - video frames as .mp4 file
            - video ticks as .json file
            - user inputs as .json file
            - trajectory data as .npz file (including all actions, observations, states,
                and rewards from gym env)
            - environment configuration as .yaml file

        :param dirname: the directory containing the trajectory data.
        :param video_width: the width of the video frames.
        :param video_height: the height of the video frames.
        :param color_format: the color format of the video frames.
        """
        self.color_format = color_format
        self.steps = None

        self.load(
            filename_by_name={
                "video": ToyEnvironmentTrajectory.get_video_path(
                    dirname, trajectory_name
                ),
                "video_ticks": ToyEnvironmentTrajectory.get_video_ticks_path(
                    dirname, trajectory_name
                ),
                "inputs": ToyEnvironmentTrajectory.get_user_inputs_path(
                    dirname, trajectory_name
                ),
                "data": ToyEnvironmentTrajectory.get_data_path(
                    dirname, trajectory_name
                ),
                "env_config": ToyEnvironmentTrajectory.get_config_path(
                    dirname, trajectory_name
                ),
                "observations": ToyEnvironmentTrajectory.get_observations_path(
                    dirname, trajectory_name
                ),
                "states": ToyEnvironmentTrajectory.get_states_path(
                    dirname, trajectory_name
                ),
            },
            video_width=video_width,
            video_height=video_height,
        )

    def add_step(
        self,
        frame: np.ndarray | None = None,
        video_tick: float | None = None,
        user_inputs: UserInputs | None = None,
        action_tick: float | None = None,
        other_data: dict = None,
    ):
        """
        Adds a step to the trajectory.

        :param frame: the video frame.
        :param observation: the observation.
        :param action: the action.
        :param state: the state.
        :param reward: the reward.
        """
        self.steps += 1

        if video_tick is None:
            video_tick = self.steps / self.video_fps
        if action_tick is None:
            action_tick = ToyEnvironmentTrajectory.get_input_tick_from_video_tick(
                video_tick, self.video_fps
            )
        if user_inputs is None:
            assert (
                "action" in other_data
            ), "User input is required if action is not provided."
            action = other_data["action"]
            user_inputs = ToyEnvironmentTrajectory.get_user_input(action, action_tick)

        super().add_step(
            frame=frame,
            video_tick=video_tick,
            user_inputs=user_inputs,
            action_tick=action_tick,
        )

        for key, storage in [
            (ToyEnvironmentTrajectory.OBS_KEY, self.observations),
            (ToyEnvironmentTrajectory.ACTION_KEY, self.actions),
            (ToyEnvironmentTrajectory.STATE_KEY, self.states),
            (ToyEnvironmentTrajectory.REWARD_KEY, self.rewards),
        ]:
            assert (
                key in other_data
            ), f"Key {key} is missing from other data for toy environment."
            assert isinstance(storage, list), f"Storage for key {key} is not a list."
            storage.append(other_data[key])
