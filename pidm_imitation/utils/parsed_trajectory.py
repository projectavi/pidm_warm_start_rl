# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import List

import torch


class ParsedTrajectory:
    """
    A parsed Trajectory consists of (states, state_ticks, actions and action_ticks) for each trajectory.
    The state_ticks are the frame/state timestamps and the action_ticks are the action timestamps
    from the trajectory recording.
        states: array of states
        actions: array of actions taken between the observations
        state_ticks: List of ticks for every state in the trajectory
        action_ticks: List of ticks for every action in the trajectory
        aligned_ticks: List of ticks after aligning states and actions for every step
        trajectory_name: This is an optional name to identify trajectory
    """

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_ticks: List[float] | None = None,
        action_ticks: List[float] | None = None,
        aligned_ticks: List[float] | None = None,
        trajectory_name: str | None = None,
    ):
        self.states = states
        self.actions = actions
        self.state_ticks = state_ticks
        self.action_ticks = action_ticks
        self.aligned_ticks = aligned_ticks
        self.trajectory_name = trajectory_name
