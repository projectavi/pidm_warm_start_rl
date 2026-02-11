# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import List

import numpy as np
import torch
from torch import Tensor

from pidm_imitation.agents.supervised_learning.inference_agents.utils.observation_handlers import (
    StateHandler,
)
from pidm_imitation.utils import ControllerActionBuilder, Logger, StateType, Trajectory

log = Logger.get_logger(__name__)


class ReferenceTrajectoryHandler:
    """
    Interface with method to get all raw states of the reference trajectory. A subclass of this interface is
    used by agents that require reference trajectory.
    """

    def __init__(self, reference_trajectory: Trajectory, action_type: str) -> None:
        self.reference_trajectory = reference_trajectory
        self.action_type = action_type
        self.action_builder = ControllerActionBuilder(self.action_type)

    @abc.abstractmethod
    def get_ref_trajectory_raw_states(self) -> Tensor:
        pass

    def get_ref_trajectory_actions(self) -> Tensor:
        reference_traj_actions = self.action_builder.build_array_from_inputs(
            userInputsLog=self.reference_trajectory.user_inputs
        )
        reference_traj_actions_tensor = torch.tensor(
            reference_traj_actions, dtype=torch.float32
        )
        return reference_traj_actions_tensor


class StateReferenceTrajectoryHandler(StateHandler, ReferenceTrajectoryHandler):

    def __init__(
        self,
        reference_trajectory: Trajectory,
        state_type: StateType,
        action_type: str,
    ) -> None:
        StateHandler.__init__(self, state_type=state_type)
        ReferenceTrajectoryHandler.__init__(
            self, reference_trajectory=reference_trajectory, action_type=action_type
        )

    def get_ref_trajectory_raw_states(self) -> Tensor:
        assert hasattr(
            self.reference_trajectory, self.state_type.value
        ), f"Reference trajectory does not have '{self.state_type.value}' attribute."
        states: List[np.ndarray] = getattr(
            self.reference_trajectory, self.state_type.value
        )
        return torch.tensor(np.array(states), dtype=torch.float32)
