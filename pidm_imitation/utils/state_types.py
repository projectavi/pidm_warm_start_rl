# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from typing import List

from pidm_imitation.constants import OBSERVATIONS_FILE_SUFFIX, STATES_FILE_SUFFIX

OBSERVATIONS_KEY = "observations"
STATES_KEY = "states"


class StateType(Enum):
    OBSERVATIONS = OBSERVATIONS_KEY
    STATES = STATES_KEY

    @staticmethod
    def get_state_type_from_str(state_type: str) -> "StateType":
        if state_type == OBSERVATIONS_KEY:
            return StateType.OBSERVATIONS
        elif state_type == STATES_KEY:
            return StateType.STATES
        raise ValueError(
            f"Unsupported state type. Supported types are are {StateType.get_valid_state_type_strings()}"
            + f"but got {state_type}"
        )

    @staticmethod
    def get_state_file_suffix(state_type: "StateType") -> str:
        if state_type == StateType.OBSERVATIONS:
            return f"{OBSERVATIONS_FILE_SUFFIX}.npz"
        elif state_type == StateType.STATES:
            return f"{STATES_FILE_SUFFIX}.npz"
        raise ValueError(
            f"Unsupported state type. Supported types are are {StateType.get_valid_state_types()} but got {state_type}"
        )

    @staticmethod
    def get_valid_state_types() -> List["StateType"]:
        return [StateType.OBSERVATIONS, StateType.STATES]

    @staticmethod
    def get_valid_state_type_strings() -> List[str]:
        return [StateType.OBSERVATIONS.value, StateType.STATES.value]
