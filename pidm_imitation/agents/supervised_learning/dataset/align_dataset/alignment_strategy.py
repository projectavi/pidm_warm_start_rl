# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC
from typing import List, Tuple

import numpy as np
import torch

from pidm_imitation.utils.parsed_trajectory import ParsedTrajectory

"""
This is an interface for different action/frame alignment strategies.
"""


class ActionFrameAlignmentStrategy(ABC):
    """
    Abstract class for aligning actions and frames.
    """

    def __init__(self, action_type: str) -> None:
        """
        Create a new ActionFrameAlignmentStrategy.

        :param action_type: str. The type of action we are using.
        This argument is automatically plumbed through by update_data_config_arg.
        """
        self.action_type = action_type

    def align_dataset(self, trajectories: List[ParsedTrajectory]) -> None:
        """
        Perform the alignment on all the trajectories.

        :param trajectories: List of ParsedTrajectory objects.

        Edits the ParsedTrajectory in place to achieve alignment and adds an aligned_steps array
        to each one containing the aligned tick information.

        This method is provided in case you want to do some alignment across all trajectories.
        The default implementation simply delegates to align_trajectory so you can easily
        implement an ActionFrameAlignmentStrategy that only aligns individual trajectories.
        """
        for trajectory in trajectories:
            self.align_trajectory(trajectory)

    def align_trajectory(self, trajectory: ParsedTrajectory) -> None:
        """
        Perform the alignment on the given actions and states given
        some timing information about those actions and states.

        :param trajectory: ParsedTrajectory object containing actions and frames
        and their timing information

        Edits the ParsedTrajectory in place to achieve alignment and adds an aligned_steps
        array containing the aligned tick information.
        """
        actions, frames, aligned_ticks = self.align_actions_and_frames(
            trajectory.actions.numpy(),
            trajectory.states.numpy(),
            action_ticks=trajectory.action_ticks,
            frame_ticks=trajectory.state_ticks,
        )
        trajectory.actions = torch.tensor(actions)
        trajectory.states = torch.tensor(frames)
        trajectory.aligned_ticks = aligned_ticks

    def align_actions_and_frames(
        self,
        actions: np.ndarray,
        frames: np.ndarray,
        action_ticks: List[float] | None = None,
        frame_ticks: List[float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Optional method that is here for compatibility with the old interface.
        You can choose to implement this method or the align_trajectory method.

        :param actions: numpy array of controller actions
        :param frames: numpy array of frames
        :param action_ticks: numpy array of action ticks
        :param frame_ticks: numpy array of frame timings
        :return: A tuple containing:
            - numpy array of aligned actions, one per step
            - numpy array of aligned frames
            - List of ticks for the aligned steps
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses or you can implement align_trajectory instead."
        )
