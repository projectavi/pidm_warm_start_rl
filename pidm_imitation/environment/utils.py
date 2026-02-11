# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

from pidm_imitation.utils import Logger

log = Logger.get_logger(__name__)


class ValidIdmPlanners:
    DISABLED = "disabled"
    CLOSEST_REFERENCE = "closest_reference"
    REFERENCE_TRAJECTORY = "reference_trajectory"
    ALL = [DISABLED, CLOSEST_REFERENCE, REFERENCE_TRAJECTORY]

    @staticmethod
    def is_valid_planner(planner_name: str) -> bool:
        return planner_name in ValidIdmPlanners.ALL


def find_experiment_config(inputs_dir: Path, filename="config.yaml") -> Path | None:
    """Finds the config file in the given experiment inputs directory."""
    if os.path.isdir(inputs_dir):
        for name in os.listdir(inputs_dir):
            if name.endswith(filename):
                return inputs_dir / name
    return None
