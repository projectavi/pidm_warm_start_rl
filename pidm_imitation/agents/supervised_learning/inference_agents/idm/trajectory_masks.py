# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ValidTrajectoryMasking:
    NONE = "none"
    TOY_SAME_GOAL = "toy_same_goal"
    TIME_WINDOW = "time_window"
    ALL = [NONE, TOY_SAME_GOAL, TIME_WINDOW]

    @staticmethod
    def is_valid(masking: str) -> bool:
        return masking in ValidTrajectoryMasking.ALL


class TrajectoryMask(ABC):
    def __init__(self, device: str | torch.device):
        self.device = device

    @abstractmethod
    def get_mask(self, states: Tensor, current_state: Tensor, current_step: int) -> Tensor:
        """
        Given a tensor of states (N, state_dim), the current state (state_dim,) and the current step (int),
        return a boolean mask (N,) indicating which states are valid candidates for the planner.
        :param states: Tensor of shape (N, state_dim) corresponding to the states in a reference trajectory
        :param current_state: Tensor of shape (state_dim,)
        :param current_step: int
        """
        pass


class NoTrajectoryMask(TrajectoryMask):
    def __init__(self, device: str | torch.device):
        super().__init__(device)

    def get_mask(self, states: Tensor, current_state: Tensor, current_step: int) -> Tensor:
        return torch.ones(states.shape[0], dtype=torch.bool, device=self.device)


class ToySameGoalTrajectoryMask(TrajectoryMask):
    """
    Mask that only allows states that have the same goal (all but ) as the current state.
    This mask is only supported for toy environment states.
    """

    def __init__(self, device: str | torch.device):
        super().__init__(device)

    def get_mask(self, states: Tensor, current_state: Tensor, current_step: int) -> Tensor:
        # only allow states that have the same goal (last 2 dimensions) as the current state
        return torch.all(states[:, 2:] == current_state[2:], dim=1)


class WithinTimeWindowTrajectoryMask(TrajectoryMask):
    """
    Mask that only allows states that are within a certain distance to the current step.
    """

    def __init__(self, device: str | torch.device, max_distance: int):
        super().__init__(device)
        self.max_distance = max_distance

    def get_mask(self, states: Tensor, current_state: Tensor, current_step: int) -> Tensor:
        state_steps = torch.arange(states.shape[0], device=self.device)
        return (state_steps >= current_step - self.max_distance) & (state_steps <= current_step + self.max_distance)


class TrajectoryMaskFactory:
    @staticmethod
    def get_masking_function(masking: str, device: str | torch.device, **kwargs) -> TrajectoryMask:
        assert ValidTrajectoryMasking.is_valid(masking), f"Invalid masking function: {masking}"
        if masking == ValidTrajectoryMasking.NONE:
            return NoTrajectoryMask(device, **kwargs)
        elif masking == ValidTrajectoryMasking.TOY_SAME_GOAL:
            return ToySameGoalTrajectoryMask(device, **kwargs)
        elif masking == ValidTrajectoryMasking.TIME_WINDOW:
            return WithinTimeWindowTrajectoryMask(device, **kwargs)
        raise ValueError(f"Unknown masking function: {masking}")
