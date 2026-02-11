# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict

from pidm_imitation.utils.valid_controller_actions import ValidControllerActions

LEFT_STICK_X_INDEX = 0
LEFT_STICK_Y_INDEX = 1

STICK_TRIGGER_INDEX_MAP = OrderedDict(
    [("left_stick_x", LEFT_STICK_X_INDEX), ("left_stick_y", LEFT_STICK_Y_INDEX)]
)


def get_stick_name(index: int) -> str:
    """
    Get the stick name from the index.
    """
    return list(STICK_TRIGGER_INDEX_MAP.keys())[index]


def get_continuous_signals_indices(action_type: str) -> list[int]:
    continuous_signal_indices = ValidControllerActions.get_actions_dim(
        action_type=action_type
    )
    return list(STICK_TRIGGER_INDEX_MAP.values())[:continuous_signal_indices]
