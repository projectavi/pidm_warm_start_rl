# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pidm_imitation.utils.action_indices import LEFT_STICK_X_INDEX, LEFT_STICK_Y_INDEX
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.user_inputs import UserInputs, UserInputsLog
from pidm_imitation.utils.valid_controller_actions import ValidControllerActions

log = Logger.get_logger(__name__)


class BaseActionBuilder(ABC):
    """This class is responsible for building environment actions from a given policy output."""

    def __init__(self, action_type: str, flip_y_axis: bool = False):
        assert ValidControllerActions.is_valid_action_type(action_type), (
            f"Invalid action type {action_type}, must be one of "
            f"{', '.join(ValidControllerActions.get_valid_action_types())}"
        )
        self.action_type = action_type
        self.flip_y_axis = flip_y_axis
        self.action_dim = ValidControllerActions.get_actions_dim(self.action_type)

    @abstractmethod
    def build_action(self, action: Any) -> Any:
        """Build an action from a given policy output."""
        raise NotImplementedError("This method should be implemented by the subclass.")


class ControllerActionBuilder(BaseActionBuilder):
    """This class is building action vectors from controller inputs in the form of ActionRequest objects. It currently
    supports various Xbox controllers."""

    THROTTLE_INDEX = 0
    STEER_INDEX = 1

    def __init__(self, action_type: str):
        super().__init__(action_type)

    def _clamp_analog_value(self, value: float) -> float:
        """Clamp the analog value to be between -1 and 1."""
        return max(-1.0, min(1.0, value))

    def _build_gamepad_continuous_inputs(self, action: Any) -> UserInputs:
        """Map an action to a streaming sdk GamePad action"""
        continuous_action = action
        left_x = self._clamp_analog_value(continuous_action[LEFT_STICK_X_INDEX])
        left_y = self._clamp_analog_value(continuous_action[LEFT_STICK_Y_INDEX])

        if self.flip_y_axis:
            left_y = -left_y

        return UserInputs(left_stick_x=float(left_x), left_stick_y=float(left_y))

    def build_action(self, action: Any) -> UserInputs:
        assert (
            len(action) == self.action_dim
        ), f"Action dim mismatch: got {len(action)} but expected {self.action_dim}"
        user_input = self._build_gamepad_continuous_inputs(action)
        return user_input

    def build_array_from_inputs(self, userInputsLog: UserInputsLog) -> np.ndarray:
        action_arr_list = []

        for user_input in userInputsLog.inputs:
            action_arr = self.build_array_from_user_input(user_input)
            action_arr_list.append(action_arr)
        return np.array(action_arr_list)

    def build_array_from_user_input(self, user_input: UserInputs) -> np.ndarray:
        action_arr = np.zeros(self.action_dim, dtype=np.float32)
        action_arr[LEFT_STICK_X_INDEX] = user_input.get_left_stick_x()
        action_arr[LEFT_STICK_Y_INDEX] = user_input.get_left_stick_y()
        return action_arr
