# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class ValidInputFormats:
    # only states for history and lookahead
    STATE_ONLY = "state_only"
    # states and actions for history and lookahead
    STATE_AND_ACTION = "state_and_action"
    # states for history and lookahead and action only for history directly in policy
    STATE_AND_ACTION_HISTORY = "state_and_action_history"

    ALL = [STATE_ONLY, STATE_AND_ACTION, STATE_AND_ACTION_HISTORY]

    @staticmethod
    def is_valid_input_format(input_format: str) -> bool:
        """
        Check if the given input format is valid.
        :param input_format: The input format to check.
        :return: True if the input format is valid, False otherwise.
        """
        return input_format in ValidInputFormats.ALL
