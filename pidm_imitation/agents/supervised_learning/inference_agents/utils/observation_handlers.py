# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc

import torch
from torch import Tensor

from pidm_imitation.utils import Logger, StateType

log = Logger.get_logger(__name__)


class ObservationHandler:
    """
    Interface with method to get the raw state that is used to build the input to the model. This is not directly
    used, but rather implemented by subclasses specific to different kinds of observations. A subclass of this
    interface is used by all agents.
    """

    @abc.abstractmethod
    def get_raw_state(self, raw_obs, built_obs) -> Tensor:
        pass


class StateHandler(ObservationHandler):

    VALID_STATE_TYPES = [StateType.STATES, StateType.OBSERVATIONS]

    def __init__(self, state_type: StateType) -> None:
        assert (
            state_type in self.VALID_STATE_TYPES
        ), f"State type must be in {self.VALID_STATE_TYPES} but got {state_type}"
        self.state_type = state_type

    def get_raw_state(self, raw_obs, built_obs) -> Tensor:
        assert isinstance(built_obs, dict), "built_obs must be a dictionary"
        assert (
            self.state_type.value in built_obs
        ), f"built_obs must contain value for '{self.state_type.value}' but only had {list(built_obs.keys())}"
        return torch.tensor(built_obs[self.state_type.value], dtype=torch.float32)
