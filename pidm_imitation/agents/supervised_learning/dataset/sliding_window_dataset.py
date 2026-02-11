# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pidm_imitation.constants import (
    ACTION_HISTORY_KEY,
    ACTION_LOOKAHEAD_KEY,
    ACTION_TARGET_KEY,
    LOOKAHEAD_K_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    STATE_HISTORY_KEY,
    STATE_LOOKAHEAD_KEY,
)
from pidm_imitation.utils import Logger, StateType
from pidm_imitation.utils.padding_utils import ValidPadding, pad_tensor, pad_ticks
from pidm_imitation.utils.parsed_trajectory import ParsedTrajectory

log = Logger.get_logger(__name__)


class SlidingWindowDataset(Dataset):
    """
    Dataset that samples states and actions from a list of parsed trajectories using a sliding window approach
    and returns state history, action history, action targets, state lookahead, action lookahead, sampled lookahead_k,
    and one-hot encoded lookahead_k. The lookahead_k is an integer in the range [0, lookahead - 1] that determines
    how many steps into the future the lookahead is.

    The class can handle loading of states, and the actions are represented arrays of floats.
    """

    VALID_STATE_TYPES = StateType.get_valid_state_type_strings()

    def __init__(
        self,
        history: int,
        history_slice: List[int],
        lookahead: int,
        lookahead_slice: List[int],
        parsed_trajectories: List[ParsedTrajectory],
        padding_strategy: str = ValidPadding.NO_PADDING,
    ):
        """
        Pytorch Dataset that samples a sliding window of observations and actions from each trajectory. The sliding
        window has a length of history + lookahead + 1, where the final one corresponds to the current observation.
        Expects lookahead > 0, otherwise the SlidingWindowDataset should be used.

        :param history: discrete number of steps denoting the history length and determining the first part of the
            the sliding window of observations and actions.
        :param history_slice: indices of previous elements that will be included in the history
        :param lookahead: discrete number of steps denoting the observation into the future, and determining the third
            part of the sliding window (after the history and the current observation) of observations and actions.
        :param lookahead_slice: indices of candidates for the lookahead step. These are discrete indices that can be
            sampled from with each value in range [0, lookahead - 1].
        :param parsed_trajectories: List of trajectories, each one containing a tuple of (states, actions_array).
        :param padding_strategy: the strategy to use for padding the dataset trajectories to
            account for the sliding window. Default is ValidPadding.NO_PADDING which does no padding.
        """
        self.history = history
        self.history_slice = [
            h for h in history_slice
        ]  # make a copy of the history slice
        assert history >= 0, f"history must be >= 0 but got {history}"
        assert all(
            h >= 0 and h < history for h in history_slice
        ), f"history_slice must be in the range [0..{history}) for the sliding window dataset, got {history_slice}"
        if history:
            assert (
                len(history_slice) > 0
            ), f"history_slice must not be empty if history > 0 but got {history_slice}"
            assert (
                sorted(set(history_slice)) == history_slice
            ), f"history_slice must be sorted in ascending order with no repeated entries but got {history_slice}"
        self.history_slice.append(
            history
        )  # add the current state index to the history slice
        self.target_action_slice = [
            h + 1 for h in self.history_slice
        ]  # the target action is the next action after the history

        assert lookahead >= 0, f"lookahead must be >= 0 but got {lookahead}"
        if lookahead > 0:
            assert all(
                k >= 0 and k < lookahead for k in lookahead_slice
            ), f"lookahead_slice must be in [0..{lookahead-1}] for the lookahead dataset, got {lookahead_slice}"

            assert (
                len(lookahead_slice) > 0
            ), f"lookahead_slice must not be empty if lookahead > 0 but got {lookahead_slice}"
            assert (
                sorted(set(lookahead_slice)) == lookahead_slice
            ), f"lookahead_slice must be sorted in ascending order with no repeated entries but got {lookahead_slice}"
            self.lookahead = lookahead
            self.lookahead_slice = lookahead_slice
            self.num_lookahead_samples = len(lookahead_slice)
            self.one_hot_encoding = torch.eye(
                self.num_lookahead_samples, dtype=torch.long
            )
        else:
            self.lookahead = 0
            self.lookahead_slice = []
            self.num_lookahead_samples = 1
            self.one_hot_encoding = None

        self.sliding_wnd_len = history + 1 + lookahead  # +1 for the current observation
        self.padding_strategy = padding_strategy
        assert (
            self.padding_strategy in ValidPadding.ALL
        ), f"Invalid padding strategy: {self.padding_strategy}"
        self.parsed_trajs = self._pad_trajectories(parsed_trajectories)

        usable_lens = self._compute_usable_lengths()
        self._cumsum_usable_lens = np.cumsum(usable_lens)

    def _pad_trajectories(
        self, parsed_trajectories: List[ParsedTrajectory]
    ) -> List[ParsedTrajectory]:
        log.info(f"Padding trajectories with strategy: {self.padding_strategy}")
        for traj in parsed_trajectories:
            traj.states = pad_tensor(
                traj.states,
                pad_pre=self.history,
                pad_post=self.lookahead,
                mode=self.padding_strategy,
                dim=0,
            )
            traj.actions = pad_tensor(
                traj.actions,
                pad_pre=self.history,
                pad_post=self.lookahead,
                mode=self.padding_strategy,
                dim=0,
            )

            traj.aligned_ticks = pad_ticks(
                traj.aligned_ticks,
                pad_pre=self.history,
                pad_post=self.lookahead,
                mode=self.padding_strategy,
            )
        return parsed_trajectories

    def _compute_usable_lengths(self) -> List[int]:
        """
        Compute number of steps of every trajectory in the dataset, subtracting the sliding window
        size when computing the length, and asserting all trajectories are longer than the sliding window size.
        """
        traj_lens = []
        for traj in self.parsed_trajs:
            traj_len = len(traj.states)
            if traj_len > self.sliding_wnd_len:
                traj_lens.append(traj_len - self.sliding_wnd_len + 1)
            else:
                raise ValueError(
                    f"Trajectory {traj.trajectory_name} is shorter ({traj_len}) than the sliding window \
                      ({self.sliding_wnd_len})"
                )
        return traj_lens

    def __len__(self):
        return self._cumsum_usable_lens[-1] * self.num_lookahead_samples

    @property
    def state_dim(self) -> int:
        return self.parsed_trajs[0].states.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.parsed_trajs[0].actions.shape[-1]

    @property
    def sequence_length(self) -> int:
        return len(self.history_slice)

    def _find_trajectory(self, idx: int) -> int:
        """
        Find the trajectory index for a given sample index. It should be the index of the first value in
        `_cumsum_usable_lens` that is greater than the given index. `_cumsum_usable_lens` is sorted in
        ascending order, so we can use binary search to find the trajectory index efficiently.

        Examples:
        cumsum_usable_lens: [5, 10, 15, 28, 45]
        idx: 12
        --> traj_idx: 2 (the first value in cumsum_usable_lens that is greater than 12 is 15, which is at index 2)

        cumsum_usable_lens: [5, 10, 15, 28, 45]
        idx: 10
        --> traj_idx: 2 (the first value in cumsum_usable_lens that is greater than 10 is 15, which is at index 2)

        cumsum_usable_lens: [5, 10, 15, 28, 45]
        idx: 0
        --> traj_idx: 0 (the first value in cumsum_usable_lens that is greater than 0 is 5, which is at index 0)

        cumsum_usable_lens: [5, 10, 15, 28, 45]
        idx: 46
        --> ERROR (there is no value in cumsum_usable_lens that is greater than or equal to 46)
        """
        if idx < 0 or idx >= self._cumsum_usable_lens[-1]:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with length {len(self)}"
            )

        # Use binary search to find the trajectory index
        traj_idx = np.searchsorted(self._cumsum_usable_lens, idx, side="right")
        return int(traj_idx)  # type: ignore[return-value]

    def _find_state(self, idx: int, traj_idx: int) -> int:
        return idx if traj_idx == 0 else idx - self._cumsum_usable_lens[traj_idx - 1]

    def _find_trajectory_and_state_index(self, idx: int) -> Tuple[int, int]:
        traj_idx = self._find_trajectory(idx)
        state_idx = self._find_state(idx, traj_idx)
        return traj_idx, state_idx

    def _get_state_and_action_sequences(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the state and action sequences for the given index. This is a helper method to avoid code duplication.
        :param idx: index of the sample in the dataset, expected to be in the range [0, len(self)).
        :return: a tuple of (states, actions) where states is a tensor of shape (sliding_wnd_len, state_dim) and
            actions is a tensor of shape (sliding_wnd_len + 1, action_dim). The actions tensor includes the action
            taken before the first state in the sliding window (zero vector for first state in traj)
        """
        traj_idx, sample_idx = self._find_trajectory_and_state_index(idx)
        current_trajectory = self.parsed_trajs[traj_idx]
        states = current_trajectory.states[
            sample_idx : sample_idx + self.sliding_wnd_len
        ]
        if sample_idx > 0:
            # sample prior action until end of window
            actions = current_trajectory.actions[
                sample_idx - 1 : sample_idx + self.sliding_wnd_len
            ]
        else:
            # no prior action available --> add zero vector
            actions = current_trajectory.actions[
                sample_idx : sample_idx + self.sliding_wnd_len
            ]
            actions = pad_tensor(
                actions, pad_pre=1, pad_post=0, mode=ValidPadding.ZERO, dim=0
            )
        return states, actions

    def _get_lookahead_data(
        self, idx: int, states: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get the lookahead data for the given index.
        This method is always called when lookahead value > 0"""
        lookahead_idx = idx % self.num_lookahead_samples
        lookahead_k = self.lookahead_slice[lookahead_idx]
        lookahead_k_onehot = self.one_hot_encoding[lookahead_idx]

        lookahead_indices = [h + 1 + lookahead_k for h in self.history_slice]
        state_lookahead = states[lookahead_indices]
        action_lookahead = actions[lookahead_indices]

        return {
            LOOKAHEAD_K_KEY: torch.tensor(lookahead_k),
            LOOKAHEAD_K_ONEHOT_KEY: lookahead_k_onehot,
            STATE_LOOKAHEAD_KEY: state_lookahead,
            ACTION_LOOKAHEAD_KEY: action_lookahead,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Get a sample from the dataset at the given index. The index is expected to be in the range [0, len(self)).

        The function first samples the subsequence of a trajectory using the sliding window approach, and samples a
        lookahead idx. The returned sample is a dictionary containing:
        - 'state_history': a tensor of shape (history + 1, *state_dims) containing the trajectory states / observations
            from the history as a sequence.
        - 'action_history': a tensor of shape (history + 1, *action_dims) containing the corresponding actions history
            as a sequence. Note that the action at index i in the history corresponds to the action taken BEFORE state
            at index i in the history. If the history starts at the first state, then the action before the first state
            is included as a zero vector.
        - 'action_target': a tensor of shape (history + 1, *action_dims) containing the action targets for the given
            history. The action target at index i corresponds to the action taken AFTER the state at index i in the
            history.

        If lookahead > 0, the sample also contains:
        - `lookahead_k`: the sampled lookahead, which is an integer in the range [0, lookahead - 1], and determines how
            many steps into the future the lookahead is.
        - `lookahead_k_onehot`: a one-hot encoded tensor of shape (num_lookahead_samples,) representing the sampled
            lookahead index.
        - 'state_lookahead': a tensor of shape (history + 1, *state_dims) containing the trajectory states /
            observations `lookahead_k` steps into the future for each state in the history. The lookahead state at
            index i corresponds to the state `lookahead_k` steps after the state at index i in the history.
        - 'action_lookahead': a tensor of shape (history + 1, *action_dims) containing the corresponding actions
            `lookahead_k` steps into the future for each action in the history. The lookahead action at index i
            corresponds to the action taken `lookahead_k` steps after the action at index i in the history.
        """
        wnd_idx = idx // self.num_lookahead_samples

        states, actions = self._get_state_and_action_sequences(wnd_idx)

        # separate state and actions into history, target (only for actions)
        state_history = states[self.history_slice]
        action_history = actions[self.history_slice]

        action_target = actions[self.target_action_slice]

        sample = {
            STATE_HISTORY_KEY: state_history,
            ACTION_HISTORY_KEY: action_history,
            ACTION_TARGET_KEY: action_target,
        }

        if self.lookahead > 0:
            lookahead_data = self._get_lookahead_data(idx, states, actions)
            sample.update(lookahead_data)

        return sample

    def get_keys(self) -> List[str]:
        """
        Returns the keys of the dataset, which are the keys of the samples returned by __getitem__.
        """
        if self.lookahead == 0:
            return [
                STATE_HISTORY_KEY,
                ACTION_HISTORY_KEY,
                ACTION_TARGET_KEY,
            ]
        return [
            STATE_HISTORY_KEY,
            ACTION_HISTORY_KEY,
            ACTION_TARGET_KEY,
            LOOKAHEAD_K_KEY,
            LOOKAHEAD_K_ONEHOT_KEY,
            STATE_LOOKAHEAD_KEY,
            ACTION_LOOKAHEAD_KEY,
        ]
