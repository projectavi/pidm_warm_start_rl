# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List

import torch.nn as nn
from torch import Tensor

from pidm_imitation.agents.supervised_learning.base_models import ActionRegressor
from pidm_imitation.agents.supervised_learning.inference_agents.utils.action_handlers import (
    PytorchActionsHandler,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.sliding_window import (
    SlidingWindowModule,
)
from pidm_imitation.constants import (
    ACTION_HISTORY_KEY,
    ACTION_LOOKAHEAD_KEY,
    LOOKAHEAD_K_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    PREDICTED_ACTION_KEY,
    STATE_HISTORY_KEY,
    STATE_LOOKAHEAD_KEY,
)


def select_rollout_action(predicted_action: Tensor) -> Tensor:
    """
    Convert a model prediction into the single action to execute at rollout time.
    Sequence-valued policies use the last/current action in the predicted window.
    """
    if predicted_action.ndim == 3:
        return predicted_action[:, -1, :]
    if predicted_action.ndim == 2:
        return predicted_action
    if predicted_action.ndim == 1:
        return predicted_action.unsqueeze(0)
    raise ValueError(
        "Expected predicted action to have shape (batch, seq, action_dim), "
        "(batch, action_dim), or (action_dim,). "
        f"Got shape {tuple(predicted_action.shape)}."
    )


class SlidingWindowInferenceModel(nn.Module):
    """
    Pytorch model class that wraps a model trained on sliding windows of observations and actions
    into one that uses a SlidingWindowModule so that at inference time we can pass 1 input at a time
    instead of building up a sliding window outside of the model.
    """

    def __init__(
        self,
        action_regressor: ActionRegressor,
        action_handler: PytorchActionsHandler,
        window_size: int,
        slice: List[int],
        padding: str,
    ):
        super().__init__()
        self.state_history = SlidingWindowModule(window_size, slice, padding)
        self.action_history = SlidingWindowModule(window_size, slice, padding)
        self.action_regressor = action_regressor
        self.action_handler = action_handler

    def get_inputs(self, state: Tensor, action: Tensor) -> Dict[str, Tensor]:

        states = self.state_history(state).unsqueeze(0)
        actions = self.action_history(action).unsqueeze(0)
        inputs = {
            STATE_HISTORY_KEY: states,
            ACTION_HISTORY_KEY: actions,
        }
        return inputs

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        inputs = self.get_inputs(state, action)
        action = self.action_regressor.forward_policy_path(inputs)[PREDICTED_ACTION_KEY]
        action = select_rollout_action(action)
        action = self.action_handler.process_predicted_action(action)
        return action.flatten()

    def reset(self) -> None:
        self.state_history.reset()
        self.action_history.reset()
        self.action_regressor.reset()


class SlidingWindowInferenceIdmModel(nn.Module):
    """
    Pytorch model class that wraps an IDM ActionRegressor trained on sliding windows of observations, actions and
    lookahead, into one that uses a SlidingWindowModule so that at inference time we can pass 1 input at a time instead
    of building up a sliding window outside of the model.
    """

    def __init__(
        self,
        action_regressor: ActionRegressor,
        action_handler: PytorchActionsHandler,
        window_size: int,
        slice: List[int],
        padding: str,
        lookahead_k_tensor: Tensor,  # for inference we currently keep this static
        lookahead_k_onehot: Tensor,  # for inference we currently keep this static
    ):
        super().__init__()

        self.state_history = SlidingWindowModule(window_size, slice, padding)
        self.action_history = SlidingWindowModule(window_size, slice, padding)
        self.lookahead_state_history = SlidingWindowModule(window_size, slice, padding)
        self.lookahead_action_history = SlidingWindowModule(window_size, slice, padding)
        self.action_regressor = action_regressor
        self.action_handler = action_handler
        self.lookahead_k_tensor = lookahead_k_tensor
        self.lookahead_k_onehot = lookahead_k_onehot

    def get_inputs(
        self,
        state: Tensor,
        action: Tensor,
        lookahead: Tensor,
        lookahead_action: Tensor = None,
    ) -> Dict[str, Tensor]:
        states = self.state_history(state).unsqueeze(0)  # add batch dimension
        actions = self.action_history(action).unsqueeze(0)  # add batch dimension
        lookahead_states = self.lookahead_state_history(lookahead).unsqueeze(
            0
        )  # add batch dimension
        lookahead_actions: Tensor | None = None
        if lookahead_action is not None:
            lookahead_actions = self.lookahead_action_history(
                lookahead_action
            ).unsqueeze(
                0
            )  # add batch dimension
        inputs = {
            STATE_HISTORY_KEY: states,
            ACTION_HISTORY_KEY: actions,
            STATE_LOOKAHEAD_KEY: lookahead_states,
            LOOKAHEAD_K_KEY: self.lookahead_k_tensor,
            LOOKAHEAD_K_ONEHOT_KEY: self.lookahead_k_onehot,
            ACTION_LOOKAHEAD_KEY: lookahead_actions,
        }
        return inputs

    def forward(
        self, state: Tensor, action: Tensor, lookahead: Tensor, lookahead_action: Tensor
    ) -> Tensor:

        inputs = self.get_inputs(state, action, lookahead, lookahead_action)
        action = self.action_regressor.forward_policy_path(inputs)[PREDICTED_ACTION_KEY]
        action = select_rollout_action(action)
        action = self.action_handler.process_predicted_action(action)
        return action.flatten()

    def reset(self) -> None:
        self.state_history.reset()
        self.action_history.reset()
        self.lookahead_state_history.reset()
        self.lookahead_action_history.reset()
        self.action_regressor.reset()
