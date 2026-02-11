# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

import numpy as np

from pidm_imitation.agents.supervised_learning.dataset.align_dataset.alignment_strategy import (
    ActionFrameAlignmentStrategy,
)
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


class CausalActionFrameAlignmentStrategy(ActionFrameAlignmentStrategy):
    def __init__(self, action_type: str, max_n_frames: int = -1):
        """
        Create a new CausalActionFrameAlignmentStrategy. Causal alignment
         just ensures the ticks of actions and frames are
         consecutive in the recording.

        :action_type: str. The type of action we are using.
        :param max_n_frames: int. The maximum number of frames to consider.
        """
        super().__init__(action_type=action_type)
        self.max_n_frames = max_n_frames

    def align_actions_and_frames(
        self,
        actions: np.ndarray,
        frames: np.ndarray,
        action_ticks: list[float] | None = None,
        frame_ticks: list[float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        :param actions: numpy array of controller actions
        :param frames: numpy array of frames
        :param action_ticks: numpy array of action ticks
        :param frame_ticks: numpy array of frame timings
        :return: A tuple containing:
            - numpy array of aligned actions, one per step
            - numpy array of aligned frames
            - List of ticks for the aligned steps
        """
        assert len(action_ticks) == len(
            actions
        ), f"Action ticks ({len(action_ticks)}) and actions ({len(actions)}) must have the same length."
        assert frame_ticks is not None, "Frame ticks must be provided"
        assert len(frame_ticks) == len(
            frames
        ), f"Frame ticks ({len(frame_ticks)}) and frames ({len(frames)}) must have the same length."

        # Check that there are same number of actions and frames
        if len(actions) != len(frames):
            min_length = min(len(actions), len(frames))
            log.warning(
                f"Number of actions ({len(actions)}) and frames ({len(frames)}) do not match. "
                f"Truncating to the minimum length: {min_length}."
            )
            actions = actions[:min_length]
            frames = frames[:min_length]
            action_ticks = action_ticks[:min_length]
            frame_ticks = frame_ticks[:min_length]

        assert all(
            [a_t > f_t for a_t, f_t in zip(action_ticks, frame_ticks)]
        ), "Action ticks must be greater than frame ticks"

        if self.max_n_frames > 0:
            actions = actions[: self.max_n_frames]
            frames = frames[: self.max_n_frames]
            action_ticks = action_ticks[: self.max_n_frames]
        return actions, frames, action_ticks
