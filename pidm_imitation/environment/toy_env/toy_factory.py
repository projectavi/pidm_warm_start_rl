# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.environment.toy_env.configs.toy_environment_config import (
    ToyEnvironmentConfig,
)
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.environment.toy_env.toy_environment_goal import (
    GoalReachingToyEnvironment,
)


class ToyEnvironmentFactory:
    @staticmethod
    def create_environment(config: ToyEnvironmentConfig) -> ToyEnvironment:
        return GoalReachingToyEnvironment(config)
