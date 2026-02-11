# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Callable, List, Tuple

import torch
from tqdm import tqdm

from pidm_imitation.agents.supervised_learning.dataset.align_dataset.alignment_strategy import (
    ActionFrameAlignmentStrategy,
)
from pidm_imitation.constants import INPUTS_FILE_SUFFIX, VIDEO_METADATA_SUFFIX
from pidm_imitation.utils import (
    Logger,
    StateType,
    ValidControllerActions,
    load_state_file,
)
from pidm_imitation.utils.action_builder import ControllerActionBuilder
from pidm_imitation.utils.parsed_trajectory import ParsedTrajectory
from pidm_imitation.utils.user_inputs import UserInputsLog, VideoTicks

log = Logger.get_logger(__name__)


class StateActionTrajectory:
    """
    Class to create list[ParsedTrajectory] from state inputs and actions in a directory
    It loads states and the actions are represented arrays of floats.
    """

    def __init__(
        self,
        datapath: str,
        action_type: str,
        state_type: StateType,
        align_strategy: ActionFrameAlignmentStrategy | None = None,
        filter: Callable[[str], bool] | None = None,
    ):
        """
        :param datapath: load states and actions from this directory
        :param action_type: the type of action to use (ValidControllerActions.ALL)
        :param state_type: the type of state to use (OBSERVATIONS, STATES)
        :param align_strategy: Optional ActionFrameAlignmentStrategy (algorithm) to align/sync actions to frames in the
            dataset.
        :param filter: optional file name predicate to limit which files are loaded from the datapath.

        """
        self.state_type = (
            StateType.get_state_type_from_str(state_type)
            if isinstance(state_type, str)
            else state_type
        )
        self.action_type = action_type
        self.align_strategy = align_strategy
        assert os.path.exists(
            datapath
        ), f"Directory with data does not exist: {datapath}"
        self._parsed_traj, self._traj_lengths = self.load_all_trajectories(
            datapath, filter
        )

    @property
    def parsed_traj(self) -> List[ParsedTrajectory]:
        return self._parsed_traj

    @property
    def traj_lengths(self) -> List[int]:
        return self._traj_lengths

    @staticmethod
    def get_trajectory_names(
        datapath: str, filter: Callable[[str], bool] | None = None
    ) -> List[str]:
        # get all the trajectory names by reading *_inputs.json files
        return [
            os.path.splitext(f)[0][0 : -len(INPUTS_FILE_SUFFIX)]
            for f in os.listdir(datapath)
            if (filter is None or filter(f))
            and f.endswith(f"{INPUTS_FILE_SUFFIX}.json")
        ]

    @staticmethod
    def get_action_file(datapath: str, traj: str) -> str:
        return os.path.join(datapath, traj + f"{INPUTS_FILE_SUFFIX}.json")

    @staticmethod
    def get_state_file(datapath: str, traj: str, state_type: StateType) -> str:
        state_file_suffix = StateType.get_state_file_suffix(state_type)
        states_file = os.path.join(datapath, traj + state_file_suffix)
        return states_file

    @staticmethod
    def get_video_ticks_file(datapath: str, traj: str) -> str:
        return os.path.join(datapath, traj + f"{VIDEO_METADATA_SUFFIX}.json")

    def is_action_and_state_files_in_trajectory_path(
        self, datapath: str, trajectory_name: str
    ) -> bool:
        """
        Check if the action and state files exist in the trajectory path
        :param datapath: the path to the trajectory
        :param trajectory_name: the name of the trajectory
        :return: True if the action file exists in the trajectory path and the state file exists either in the
            trajectory path or in the cache
        """
        actions_file = StateActionTrajectory.get_action_file(datapath, trajectory_name)
        states_file = StateActionTrajectory.get_state_file(
            datapath, trajectory_name, self.state_type
        )
        return os.path.exists(actions_file) and os.path.exists(states_file)

    def load_trajectory(
        self,
        user_input_file: str,
        state_file: str,
        video_ticks_file: str = None,
        traj_name: str = None,
    ) -> ParsedTrajectory:
        """
        Give the user_input_file and the state_file, returns tuples of the states and actions array
        for a given trajectory
        """
        user_inputs = UserInputsLog()
        user_inputs.load(filename=user_input_file)
        action_builder = ControllerActionBuilder(self.action_type)
        actions_arr = action_builder.build_array_from_inputs(userInputsLog=user_inputs)
        action_ticks = [e.ticks for e in user_inputs.inputs]
        assert actions_arr.shape[-1] == ValidControllerActions.get_actions_dim(
            self.action_type
        )

        states_arr = load_state_file(file=state_file, state_key=self.state_type.value)

        state_ticks: List[float] | None = None
        if os.path.exists(video_ticks_file):
            state_ticks = VideoTicks.load(video_ticks_file).video_ticks

        actions = torch.tensor(actions_arr)
        states = torch.tensor(states_arr)

        return ParsedTrajectory(
            states=states,
            actions=actions,
            state_ticks=state_ticks,
            action_ticks=action_ticks,
            trajectory_name=traj_name,
        )

    def prepare_data_files(self, datapath, traj) -> None:
        """
        Prepare the data files for the trajectory. This method is called before loading the trajectory.
        It can be used to check if the data files are present and to create files if needed.

        :param datapath: the path to the trajectory
        :param traj: the name of the trajectory
        """
        pass

    def load_all_trajectories(
        self, datapath: str, filter: Callable[[str], bool] | None = None
    ) -> Tuple[List[ParsedTrajectory], List[int]]:
        """
        Loads all the trajectories (only trajectories with <traj_name>_<state_suffix>.npz are included) in the
        datapath and initializes the parsed_traj array
        """
        parsed_traj: List[ParsedTrajectory] = []
        traj_lengths: List[int] = []

        trajectory_names = StateActionTrajectory.get_trajectory_names(datapath, filter)
        for traj in (pbar := tqdm(trajectory_names)):
            actions_file = StateActionTrajectory.get_action_file(datapath, traj)
            states_file = StateActionTrajectory.get_state_file(
                datapath, traj, self.state_type
            )
            video_ticks_file = StateActionTrajectory.get_video_ticks_file(
                datapath, traj
            )

            self.prepare_data_files(datapath, traj)
            if not self.is_action_and_state_files_in_trajectory_path(datapath, traj):
                raise Exception(
                    f"Missing state file ({states_file}) or action file ({actions_file}) for trajectory!"
                )

            parsed_traj.append(
                self.load_trajectory(actions_file, states_file, video_ticks_file, traj)
            )
            traj_lengths.append(len(parsed_traj[-1].states))
            pbar.set_postfix(
                {
                    "Trajectory": {traj},
                    "Length": traj_lengths[-1],
                    "state_dim": parsed_traj[-1].states.shape[-1],
                    "action_dim": parsed_traj[-1].actions.shape[-1],
                }
            )

        if self.align_strategy is not None:
            self.align_strategy.align_dataset(parsed_traj)

        assert len(parsed_traj) > 0, "No trajectories have been parsed"
        assert all(length > 0 for length in traj_lengths), "Some trajectories are empty"
        return (parsed_traj, traj_lengths)


class StateActionTrajectoryFactory:
    """
    Factory class to create StateActionTrajectory objects based on state type
    """

    @staticmethod
    def get_trajectory_by_state_type(
        state_type: str | StateType,
        datapath: str,
        action_type: str,
        align_strategy: ActionFrameAlignmentStrategy | None = None,
        filter: Callable[[str], bool] | None = None,
    ) -> StateActionTrajectory:
        if isinstance(state_type, str):
            state_type = StateType.get_state_type_from_str(state_type)

        if state_type == StateType.OBSERVATIONS or state_type == StateType.STATES:
            return StateActionTrajectory(
                datapath=datapath,
                action_type=action_type,
                state_type=state_type,
                align_strategy=align_strategy,
                filter=filter,
            )
        raise ValueError(f"Invalid state type: {state_type}")
