# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.environment.toy_env.configs import (
    ToyBCConfigFile,
    ToyIDMConfigFile,
    ToyPLConfigFile,
)
from pidm_imitation.evaluation.toy_valid_agents import ValidToyAgents
from pidm_imitation.utils.config_base import ConfigFile


def create_toy_config_parser(
    config_path: str,
    agent_name: str,
) -> ToyPLConfigFile:
    config: ConfigFile
    if agent_name in [ValidToyAgents.TOY_IDM]:
        config = ToyIDMConfigFile(config_path)
    elif agent_name == ValidToyAgents.TOY_BC:
        config = ToyBCConfigFile(config_path)
    else:
        raise NotImplementedError(
            f"Agent '{agent_name}' not implemented. Existing agents are: {ValidToyAgents.ALL}"
        )

    return config
