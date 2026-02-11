# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch import Tensor

from pidm_imitation.utils import Logger

log = Logger.get_logger(__name__)


class PytorchActionsHandler:
    """
    Class to pre-process the latest actions and post-process the current action predicted by a pytorch model within one
    of the pytorch agents. Used by pytorch agents only.
    """

    def __init__(self, action_type: str):
        self.action_type = action_type

    def pre_process_action(self, action: Tensor) -> Tensor:
        """
        Process the latest action used in the environment in a form that can be added to the action history that will
        be fed to the pytorch model.
        """
        return action

    def process_predicted_action(self, pred_action: Tensor) -> Tensor:
        """
        Process model outputs to get environment actions.
        """
        return pred_action.detach().clone()
