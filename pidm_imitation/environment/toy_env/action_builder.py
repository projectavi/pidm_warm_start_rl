# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

import numpy as np

from pidm_imitation.utils.action_builder import BaseActionBuilder
from pidm_imitation.utils.action_indices import LEFT_STICK_X_INDEX, LEFT_STICK_Y_INDEX


class ToyEnvActionBuilder(BaseActionBuilder):
    """This class is responsible for building numpy actions for toy env from a given action vector."""

    def __init__(self, action_type: str):
        super().__init__(action_type)

    def build_action(self, action: Any) -> np.ndarray:
        assert (
            len(action) == self.action_dim
        ), f"Action dim mismatch: got {len(action)} but expected {self.action_dim}"
        action = np.array([action[LEFT_STICK_X_INDEX], action[LEFT_STICK_Y_INDEX]])

        return action
