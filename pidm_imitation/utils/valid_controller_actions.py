# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List


class ValidControllerActions:
    """
    LEFT_STICK includes only the left stick.
    """

    LEFT_STICK = "left_stick"

    SUPPORTED_ACTIONS = [LEFT_STICK]

    ACTIONS = {
        LEFT_STICK: 2,
    }

    @staticmethod
    def get_valid_action_types() -> List[str]:
        return ValidControllerActions.SUPPORTED_ACTIONS

    @staticmethod
    def is_valid_action_type(action_type: str) -> bool:
        return action_type in ValidControllerActions.SUPPORTED_ACTIONS

    @staticmethod
    def get_actions_dim(action_type: str) -> int:
        assert action_type in ValidControllerActions.ACTIONS.keys(), f"Invalid action type: {action_type}"
        return ValidControllerActions.ACTIONS[action_type]
