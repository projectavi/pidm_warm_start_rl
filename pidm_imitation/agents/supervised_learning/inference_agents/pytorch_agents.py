# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pidm_imitation.agents import Agent
from pidm_imitation.agents.supervised_learning.base_models import ActionRegressor
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_planners import (
    IdmPlanner,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_utils import (
    get_idm_lookahead_index,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.action_handlers import (
    PytorchActionsHandler,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.inference_models import (
    SlidingWindowInferenceIdmModel,
    SlidingWindowInferenceModel,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.observation_handlers import (
    ObservationHandler,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import (
    ACTION_HISTORY_KEY,
    ACTION_LOOKAHEAD_KEY,
    INFERENCE_ACTION_KEY,
    INFERENCE_LOOKAHEAD_ACTION_KEY,
    INFERENCE_LOOKAHEAD_KEY,
    INFERENCE_STATE_KEY,
    LOOKAHEAD_K_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    STATE_HISTORY_KEY,
    STATE_LOOKAHEAD_KEY,
)
from pidm_imitation.torch_utils.utils import get_device
from pidm_imitation.utils import Logger, StateType, ValidControllerActions
from pidm_imitation.utils.padding_utils import ValidPadding

log = Logger.get_logger(__name__)


# ###################################################
# Parameters readers
# ###################################################


class ParameterReader:
    @staticmethod
    def read_parameters_from_config(datamodule_hparams):
        # Parameters to be read from `config` dictionary loaded above.
        history: int = datamodule_hparams["history"]
        history_slice: List[int] = datamodule_hparams["history_slice"]
        padding_strategy: str = datamodule_hparams.get(
            "padding_strategy", ValidPadding.REPEAT
        )

        log.info(f"Model history size: {history}")
        log.info(f"History slices: {history_slice}")
        log.info(f"Padding strategy: {padding_strategy}")
        if padding_strategy == ValidPadding.NO_PADDING:
            log.warning(
                "NO_PADDING was used during training and will be replaced with ZERO for eval."
            )
        return history, history_slice, padding_strategy


class IdmParameterReader:
    @staticmethod
    def read_parameters_from_config(datamodule_hparams):
        # Parameters to be read from datamodule section of the `config` dictionary.
        lookahead: int = datamodule_hparams["lookahead"]
        assert (
            lookahead > 0
        ), f"Lookahead must be greater than zero, but model contains {lookahead}"
        include_k: bool = datamodule_hparams["include_k"]
        lookahead_slice: List[int] = datamodule_hparams["lookahead_slice"]
        num_lookahead_slices = len(lookahead_slice)

        log.info(f"Model lookahead size: {lookahead}, include K: {include_k}")
        log.info(f"Lookahead slices: {lookahead_slice}")
        return lookahead, include_k, lookahead_slice, num_lookahead_slices


# ###################################################
# Agents
# ###################################################


class PytorchAgent(Agent):
    """
    Base pytorch agent class that wraps a trained action regressor model and implements Agent interface.
    This class provides the current timestep state and action information to the model as inputs.
    """

    def __init__(
        self,
        config: OfflinePLConfigFile,
        model: nn.Module,
        model_hparams: Any,
        data_hparams: Any,
        model_path: str,
        observation_handler: ObservationHandler,
    ) -> None:
        assert isinstance(
            model, ActionRegressor
        ), "Model must be an instance of ActionRegressor"
        self.config = config
        self.model: nn.Module = model
        self.model_hparams = model_hparams
        self.datamodule_hparams = data_hparams
        self.model_path = model_path
        self.observation_handler = observation_handler
        self.action_handler = PytorchActionsHandler(config.action_config.type)
        self.history, self.history_slice, self.padding_strategy = (
            ParameterReader.read_parameters_from_config(self.datamodule_hparams)
        )
        self.history_slice.append(self.history)  # Add current timestep to the history
        self.device = get_device()
        self.prev_action: torch.Tensor | None = None

    @property
    def action_type(self) -> str:
        return self.config.action_config.type

    @property
    def state_type(self) -> StateType:
        return self.config.state_config.type

    def _get_raw_state(self, raw_obs, built_obs) -> Tensor:
        return self.observation_handler.get_raw_state(raw_obs, built_obs).to(
            self.device
        )

    def _get_prev_action(self) -> torch.Tensor:
        if self.prev_action is not None:
            prev_action = self.prev_action
        else:
            prev_action = torch.zeros(
                ValidControllerActions.get_actions_dim(self.action_type),
                device=self.device,
                dtype=torch.float32,
            )
        return self.action_handler.pre_process_action(prev_action)

    def _get_inputs(self, raw_obs: Any, built_obs: Any) -> Dict[str, Tensor]:
        # batch size and seq length are set to 1 by creating new dimensions with unsqueeze(*)
        curr_raw_state = (
            self._get_raw_state(raw_obs, built_obs).unsqueeze(0).unsqueeze(1).clone()
        )
        prev_action = self._get_prev_action().unsqueeze(0).unsqueeze(1).clone()
        return {
            STATE_HISTORY_KEY: curr_raw_state,
            ACTION_HISTORY_KEY: prev_action,
        }

    def _get_policy_path_predictions(self, raw_obs: Any, built_obs: Any) -> Tensor:
        inputs = self._get_inputs(raw_obs, built_obs)
        if isinstance(self.model, ActionRegressor):
            # takes a Dict[str, Tensor] as input.
            predicted_actions = self.model(inputs)
        else:
            predicted_actions = self.model(**inputs)
        return predicted_actions

    def get_action(self, raw_obs: Any, built_obs: Any) -> Any | None:
        """
        Returns the next action, if any, to be taken by the agent.
        """
        with torch.inference_mode():
            action = self._get_policy_path_predictions(raw_obs, built_obs)
            self.prev_action = self.action_handler.process_predicted_action(action)
            self.prev_action = self.prev_action.flatten()
            return self.prev_action.cpu().numpy()

    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """
        self.prev_action = None
        if hasattr(self.model, "reset"):
            self.model.reset()  # type: ignore

    def has_actions(self) -> bool:
        return True


class PytorchIdmAgent(PytorchAgent):
    """
    Base pytorch agent class for IDM models that wraps a trained action regressor model.
    This is the agent when the model is recurrent (RNN).
    This class provides the current timestep state and action information to the model as inputs.
    """

    def __init__(
        self,
        config: OfflinePLConfigFile,
        model: nn.Module,
        model_hparams: Any,
        data_hparams: Any,
        model_path: str,
        observation_handler: ObservationHandler,
        idm_planner: IdmPlanner,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            model_hparams=model_hparams,
            data_hparams=data_hparams,
            model_path=model_path,
            observation_handler=observation_handler,
        )

        (
            self.max_lookahead,
            self.model_input_k,
            self.train_lookahead_slice,
            self.num_lookahead_slices,
        ) = IdmParameterReader.read_parameters_from_config(self.datamodule_hparams)

        assert isinstance(
            model, ActionRegressor
        ), "Model must be an instance of ActionRegressor"

        self.idm_planner = idm_planner
        self.step_counter: int = (
            -1
        )  # Will be incremented to zero in get_action when used it for the first time.

    def _get_lookahead_k_and_onehot(self, lookahead_k: int) -> Tuple[Tensor, Tensor]:
        lookahead_k_index = get_idm_lookahead_index(
            lookahead_k, self.train_lookahead_slice
        )
        lookahead_k_tensor = torch.tensor(
            lookahead_k_index, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        one_hot_encoding = torch.eye(
            self.num_lookahead_slices, dtype=torch.long, device=self.device
        )
        lookahead_k_onehot = one_hot_encoding[lookahead_k_index].unsqueeze(0)
        return lookahead_k_tensor, lookahead_k_onehot

    def _get_inputs(self, raw_obs: Any, built_obs: Any) -> Dict[str, Tensor]:
        inputs = super()._get_inputs(raw_obs, built_obs)
        curr_raw_state = inputs[STATE_HISTORY_KEY]

        lookahead_state, lookahead_action, lookahead_k = (
            self.idm_planner.get_lookahead_state_action_and_k(
                current_state=curr_raw_state.squeeze(1),
                current_action=(
                    self.prev_action if self.prev_action is not None else None
                ),
                current_step=self.step_counter,
            )
        )
        lookahead_state = lookahead_state.unsqueeze(0).unsqueeze(1)
        lookahead_action = lookahead_action.unsqueeze(0).unsqueeze(1)
        lookahead_k_tensor, lookahead_k_onehot = self._get_lookahead_k_and_onehot(
            lookahead_k
        )
        return {
            **inputs,
            STATE_LOOKAHEAD_KEY: lookahead_state,
            ACTION_LOOKAHEAD_KEY: lookahead_action,
            LOOKAHEAD_K_KEY: lookahead_k_tensor,
            LOOKAHEAD_K_ONEHOT_KEY: lookahead_k_onehot,
        }

    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """
        super().reset()
        self.step_counter = (
            -1
        )  # It will be incremented to zero in get_action when used it for the first time.

    def get_action(self, raw_obs: Any, built_obs: Any) -> Any | None:
        """
        Returns the next action, if any, to be taken by the agent.
        """
        self.step_counter += 1
        return super().get_action(raw_obs, built_obs)


class PytorchSlidingWindowAgent(PytorchAgent):
    """
    Pytorch agent class that wraps a trained action regressor model and provides sequence information to the model as
    inputs where those sequences are created by a Sliding Window as implemented by SlidingWindowInferenceModel.
    """

    def __init__(
        self,
        config: OfflinePLConfigFile,
        model: ActionRegressor,
        model_hparams: Any,
        data_hparams: Any,
        model_path: str,
        observation_handler: ObservationHandler,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            model_hparams=model_hparams,
            data_hparams=data_hparams,
            model_path=model_path,
            observation_handler=observation_handler,
        )
        # wrap the ActionRegressor
        self.model = SlidingWindowInferenceModel(
            action_regressor=model,
            action_handler=self.action_handler,
            window_size=self.history + 1,  # +1 to add "current" slice
            slice=self.history_slice,
            padding=self.padding_strategy,
        )

    def _get_inputs(self, raw_obs: Any, built_obs: Any) -> Dict[str, Tensor]:
        curr_raw_state = self._get_raw_state(raw_obs, built_obs)
        prev_action = self._get_prev_action()
        action_tensor = prev_action.to(self.device)
        return {
            INFERENCE_STATE_KEY: curr_raw_state,
            INFERENCE_ACTION_KEY: action_tensor,
        }

    def get_action(self, raw_obs: Any, built_obs: Any) -> Any | None:
        # Simplified method because we moved action handler into SlidingWindowInferenceModel.
        with torch.inference_mode():
            self.prev_action = (
                self._get_policy_path_predictions(raw_obs, built_obs).detach().clone()
            )
            return self.prev_action.cpu().numpy()


class PytorchSlidingWindowIdmAgent(PytorchIdmAgent):
    """
    Pytorch agent class that wraps a trained IDM action regressor model and provides sequence information to the model
    as inputs where those sequences are created by a Sliding Window as implemented by SlidingWindowInferenceIdmModel.
    """

    def __init__(
        self,
        config: OfflinePLConfigFile,
        model: ActionRegressor,
        model_hparams: Any,
        data_hparams: Any,
        model_path: str,
        observation_handler: ObservationHandler,
        idm_planner: IdmPlanner,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            model_hparams=model_hparams,
            data_hparams=data_hparams,
            model_path=model_path,
            observation_handler=observation_handler,
            idm_planner=idm_planner,
        )
        assert (
            idm_planner.eval_lookahead_type == "fixed"
        ), "Lookahead random is no longer supported during eval"
        self.lookahead_k = idm_planner.eval_lookahead_k
        self.lookahead_k_tensor, self.lookahead_k_onehot = (
            self._get_lookahead_k_and_onehot(self.lookahead_k)
        )
        # wrap the ActionRegressor
        self.model = SlidingWindowInferenceIdmModel(
            action_regressor=model,
            action_handler=self.action_handler,
            window_size=self.history + 1,  # +1 to add "current" slice
            slice=self.history_slice,
            padding=self.padding_strategy,
            lookahead_k_tensor=self.lookahead_k_tensor,
            lookahead_k_onehot=self.lookahead_k_onehot,
        )

    def _get_inputs(self, raw_obs: Any, built_obs: Any) -> Dict[str, Tensor]:
        curr_raw_state = self._get_raw_state(raw_obs, built_obs)
        prev_action = self._get_prev_action()

        action_tensor = prev_action.to(self.device)
        lookahead_state, lookahead_action, lookahead_k = (
            self.idm_planner.get_lookahead_state_action_and_k(
                current_state=curr_raw_state,
                current_action=(
                    self.prev_action if self.prev_action is not None else None
                ),
                current_step=self.step_counter,
            )
        )

        assert lookahead_k == self.lookahead_k, "Lookahead k must be static"
        lookahead_state = lookahead_state.to(self.device)
        lookahead_action = (
            lookahead_action.to(self.device) if lookahead_action is not None else None
        )
        # The inputs keys have to match the args on SlidingWindowInferenceIdmModel.
        return {
            INFERENCE_STATE_KEY: curr_raw_state,
            INFERENCE_ACTION_KEY: action_tensor,
            INFERENCE_LOOKAHEAD_KEY: lookahead_state,
            INFERENCE_LOOKAHEAD_ACTION_KEY: lookahead_action,
        }

    def get_action(self, raw_obs: Any, built_obs: Any) -> Any | None:
        """Simplified method because we moved action handler into SlidingWindowInferenceIdmModel."""
        self.step_counter += 1
        with torch.inference_mode():
            self.prev_action = (
                self._get_policy_path_predictions(raw_obs, built_obs).detach().clone()
            )
            return self.prev_action.cpu().numpy()
