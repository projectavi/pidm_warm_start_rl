# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_utils import (
    get_idm_lookahead,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.reference_trajectory_handlers import (
    ReferenceTrajectoryHandler,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.state_distances import (
    DistanceMetricFactory,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.trajectory_masks import (
    TrajectoryMask,
    TrajectoryMaskFactory,
)
from pidm_imitation.utils import Logger

log = Logger.get_logger(__name__)


class IdmPlanner(ABC):
    """
    Interface for a IDM agent planner model. IDM agents receive a lookahead state to plan towards.
    The planner model given a current state, identifies the lookahead horizon and returns a state.
    """

    def __init__(
        self,
        device: str | torch.device,
        eval_lookahead_type: str,
        eval_lookahead: int,
        train_lookahead_slice: List[int],
    ):
        self.device = device
        self.eval_lookahead_type = eval_lookahead_type
        self.eval_lookahead = eval_lookahead
        self.train_lookahead_slice = train_lookahead_slice
        self.eval_lookahead_k = self._get_lookahead_k(log_output=True)

    def _get_lookahead_k(self, log_output=False) -> int:
        return get_idm_lookahead(
            self.eval_lookahead_type,
            self.eval_lookahead,
            self.train_lookahead_slice,
            log_output=log_output,
        )

    @abstractmethod
    def get_lookahead_state_action_and_k(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Get the lookahead state and lookahead k for a given current state.

        :param current_state: The current state of the agent.
        :param current_action: The current action of the agent.
        :param current_step: The current step of the agent.
        :return: The lookahead state, action and lookahead k.
        """
        pass


class ReferenceTrajectoryPlanner(IdmPlanner):
    """
    IDM planner that given a single reference trajectory returns the lookahead state with index t + k
    in the given reference trajectory.
    """

    def __init__(
        self,
        device: str | torch.device,
        eval_lookahead_type: str,
        eval_lookahead: int,
        train_lookahead_slice: List[int],
        reference_trajectory_handler: ReferenceTrajectoryHandler,
    ):
        """
        Initialize the IDMReferenceTrajectoryPlanner.

        :param reference_trajectory: The reference trajectory to use for planning.
        """
        super().__init__(
            device=device,
            eval_lookahead_type=eval_lookahead_type,
            eval_lookahead=eval_lookahead,
            train_lookahead_slice=train_lookahead_slice,
        )
        self.ref_traj_raw_state = (
            reference_trajectory_handler.get_ref_trajectory_raw_states().to(self.device)
        )
        self.ref_traj_actions = (
            reference_trajectory_handler.get_ref_trajectory_actions().to(self.device)
        )

        self.ref_traj_length = len(self.ref_traj_raw_state)
        log.info(
            f"Reference trajectory IDM planner, steps in reference trajectory: {self.ref_traj_length}"
        )

    def get_lookahead_state_action_and_k(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Get the lookahead state and lookahead k for a given current state.

        :param current_state: The current state of the agent.
        :param current_action: The current action of the agent.
        :param current_step: The current step of the agent.
        :return: The lookahead state, lookahead action and lookahead k.
        """
        self.eval_lookahead_k = self._get_lookahead_k()
        lookahead_idx = current_step + self.eval_lookahead_k + 1
        assert lookahead_idx >= 1, "Lookahead index must be greater than or equal to 1"
        if lookahead_idx < self.ref_traj_length - 1:
            lookahead_state = self.ref_traj_raw_state[lookahead_idx]
            lookahead_action = self.ref_traj_actions[lookahead_idx - 1]
        else:
            lookahead_state = self.ref_traj_raw_state[-1]
            lookahead_action = self.ref_traj_actions[-2]

        lookahead_state = lookahead_state.to(device=self.device, dtype=torch.float32)
        lookahead_action = lookahead_action.to(device=self.device, dtype=torch.float32)
        return lookahead_state, lookahead_action, self.eval_lookahead_k


class InstanceBasedPlanner(IdmPlanner):
    """
    Abstract class for generic instanced-based IDM planners. An instance-based planner can use multiple reference
    trajectories to compute the lookahead state. This interface provides functionality to compute distance metrics
    and apply trajectory masks when searching for the closest state in all reference trajectories. Any instance-based
    planner must implement the `_get_lookahead_state` method that defines how the lookahead state is computed from the
    current state and step.
    """

    def __init__(
        self,
        device: str | torch.device,
        eval_lookahead_type: str,
        eval_lookahead: int,
        train_lookahead_slice: List[int],
        reference_trajectory_handlers: List[ReferenceTrajectoryHandler],
        distance_measure: str = "l2",
        masking_conditions: List[Tuple[str, dict]] | None = None,
    ):
        """
        Initialize a instance-based IDM planner.

        :param device: The device to use for planning.
        :param eval_lookahead_type: The type of lookahead to use for evaluation.
        :param eval_lookahead: The lookahead value to use for evaluation.
        :param train_lookahead_slice: The slice of the lookahead to use for training
        :param reference_trajectory_handlers: The reference trajectory handlers to extract information from all
            reference trajectories.
        :param distance_measure: The distance metric to use for finding the closest state. See `state_distances.py` for
            available distance metrics.
        :param masking_conditions: A list of tuples containing the name of the masking function and its kwargs to apply
            when searching for the closest state. See `trajectory_masks.py` for available masking functions.
        """
        super().__init__(
            device=device,
            eval_lookahead_type=eval_lookahead_type,
            eval_lookahead=eval_lookahead,
            train_lookahead_slice=train_lookahead_slice,
        )
        self.distance_measure = distance_measure
        self.ref_traj_raw_states = [
            reference_trajectory_handler.get_ref_trajectory_raw_states().to(self.device)
            for reference_trajectory_handler in reference_trajectory_handlers
        ]
        self.ref_traj_actions = [
            reference_trajectory_handler.get_ref_trajectory_actions().to(self.device)
            for reference_trajectory_handler in reference_trajectory_handlers
        ]

        if masking_conditions is None:
            masking_conditions = [("none", {})]
        self.trajectory_masks: List[TrajectoryMask] = []
        for masking_name, masking_kwargs in masking_conditions:
            trajectory_mask = TrajectoryMaskFactory.get_masking_function(
                masking_name, device, **masking_kwargs
            )
            self.trajectory_masks.append(trajectory_mask)
        self.distance_fn: Callable = DistanceMetricFactory.get_distance_function(
            distance_measure
        )

    @abstractmethod
    def _get_lookahead_state_and_action(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the lookahead state for a given current state, action, and step.
        :param current_state: A 1D tensor of shape (state_dim,) containing the current state.
        :param current_action: A 1D tensor of shape (action_dim,) containing the current action.
        :param current_step: The current step of the agent.
        :return: A tuple of tensors containing the lookahead state and action.
        """
        pass

    def _get_aggregate_mask(
        self, states: Tensor, current_state: Tensor, current_step: int
    ) -> Tensor:
        """
        Get the aggregate mask for all trajectory masks. All mask conditions must be fulfilled for a state to be
        considered valid.

        :param states: A 2D tensor of shape (num_states, state_dim) containing the states to mask.
        :param current_state: A 1D tensor of shape (state_dim,) containing the current state.
        :param current_step: The current step of the agent.
        :return: A 1D boolean tensor of shape (num_states,) containing the aggregate mask.
        """
        if len(self.trajectory_masks) == 1:
            return self.trajectory_masks[0].get_mask(
                states, current_state, current_step
            )

        aggregate_mask = torch.ones(
            states.shape[0], dtype=torch.bool, device=self.device
        )
        for trajectory_mask in self.trajectory_masks:
            mask = trajectory_mask.get_mask(states, current_state, current_step)
            aggregate_mask = aggregate_mask & mask
        return aggregate_mask

    def get_lookahead_state_action_and_k(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Get the lookahead state and lookahead k for a given current state.

        :param current_state: The current state of the agent.
        :param current_action: The current action of the agent.
        :param current_step: The current step of the agent.
        :return: The lookahead state, action and lookahead k.
        """
        self.eval_lookahead_k = self._get_lookahead_k()
        lookahead_state, lookahead_action = self._get_lookahead_state_and_action(
            current_state, current_action, current_step
        )
        lookahead_state = lookahead_state.to(self.device)
        lookahead_action = lookahead_action.to(self.device)
        return lookahead_state, lookahead_action, self.eval_lookahead_k


class ClosestReferencePlanner(InstanceBasedPlanner):
    """
    IDM planner that given all reference trajectories from the training data looks for the closest state and returns
    the lookahead state k steps ahead in that trajectory. If the lookahead index exceeds the length of the reference
    trajectory, the last state of that trajectory is returned. The closest state is determined by finding the state
    with the smallest distance to the current state.
    """

    def _get_lookahead_state_and_action(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the lookahead state for a given current state, action, and step.
        :param current_state: A 1D tensor of shape (state_dim,) containing the current state.
        :param current_action: A 1D tensor of shape (action_dim,) containing the current action.
        :param current_step: The current step of the agent.
        :return: A tuple of tensors containing the lookahead state and action.
        """
        closest_traj_idx = -1
        closest_state_idx = -1
        closest_distance = float("inf")
        for traj_idx, ref_traj_states in enumerate(self.ref_traj_raw_states):
            states_mask = self._get_aggregate_mask(
                ref_traj_states, current_state, current_step
            )
            masked_ref_traj_states = ref_traj_states[states_mask]
            if len(masked_ref_traj_states) == 0:
                # no trajectory fulfilling the mask condition in this trajectory
                continue
            distances = self.distance_fn(masked_ref_traj_states, current_state)
            min_distance, min_distance_idx = torch.min(distances, dim=0)
            if min_distance < closest_distance:
                closest_distance = min_distance.item()
                closest_traj_idx = traj_idx
                # get the index in the original trajectory (not the masked one)
                closest_state_idx = int(
                    torch.nonzero(states_mask)[min_distance_idx].item()
                )
        if closest_traj_idx == -1 or closest_state_idx == -1:
            raise ValueError("No closest state found")

        closest_traj_states = self.ref_traj_raw_states[closest_traj_idx]
        closest_traj_actions = self.ref_traj_actions[closest_traj_idx]
        lookahead_idx = closest_state_idx + self.eval_lookahead_k + 1
        assert lookahead_idx >= 1, "Lookahead index must be greater than or equal to 1"
        if lookahead_idx < len(closest_traj_states) - 1:
            return (
                closest_traj_states[lookahead_idx],
                closest_traj_actions[lookahead_idx - 1],
            )
        return closest_traj_states[-1], closest_traj_actions[-2]


class ZeroPlanner(IdmPlanner):
    """
    IDM planner that returns all-0 lookahead state. Can be used to ablate the planner.
    """

    def __init__(
        self,
        device: str | torch.device,
        eval_lookahead_type: str,
        eval_lookahead: int,
        train_lookahead_slice: List[int],
    ):
        """
        Initialize the IDMZeroPlanner.

        :param device: The device to use for planning.
        """
        super().__init__(
            device=device,
            eval_lookahead_type=eval_lookahead_type,
            eval_lookahead=eval_lookahead,
            train_lookahead_slice=train_lookahead_slice,
        )
        log.info("Disabled IDM planner --> all lookahead states are 0")

    def get_lookahead_state_action_and_k(
        self, current_state: Tensor, current_action: Tensor, current_step: int
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Get the lookahead state and lookahead k for a given current state.

        :param current_state: The current state of the agent.
        :param current_action: The current action of the agent.
        :param current_step: The current step of the agent.
        :return: The lookahead state, action and lookahead k.
        """
        self.eval_lookahead_k = self._get_lookahead_k()
        zero_state = torch.zeros_like(
            current_state, device=self.device, dtype=torch.float32
        )

        zero_action = (
            torch.zeros_like(
                current_action,
                device=self.device,
                dtype=torch.float32,
            )
            if current_action is not None
            else None
        )
        return zero_state, zero_action, self.eval_lookahead_k
