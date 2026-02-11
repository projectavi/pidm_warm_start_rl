# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Optional

import torch

from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_agents import (
    PytorchAgent,
)
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_agents_factory import (
    PytorchAgentFactory,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.environment.toy_env.action_builder import ToyEnvActionBuilder
from pidm_imitation.evaluation.utils import get_pytorch_agent_name
from pidm_imitation.utils import StateType


class ToyPytorchAgent(PytorchAgent):
    """
    An agent that generates actions from a trained policy for the toy environment.
    It also optionally synchronizes the rate of actions with a given
    user recording.
    """

    def __init__(self, pytorch_agent: PytorchAgent, flip_y_axis: bool = False):
        self._pytorch_agent = pytorch_agent
        self.action_builder: ToyEnvActionBuilder = ToyEnvActionBuilder(self.action_type)
        self.action_builder.flip_y_axis = flip_y_axis

    def reset(self):
        self._pytorch_agent.reset()
        self.ticks = []

    def get_action(self, raw_obs: Any, built_obs: Any) -> Optional[Any]:
        action = self._pytorch_agent.get_action(raw_obs=raw_obs, built_obs=built_obs)
        if torch.is_tensor(action):
            action = action.detach().cpu()
        return self.action_builder.build_action(action)

    def is_toy_agent(self) -> bool:
        return True

    @property
    def action_type(self) -> str:
        return self._pytorch_agent.action_type

    @property
    def state_type(self) -> StateType:
        return self._pytorch_agent.state_type


class ToyPytorchAgentWrapperFactory:

    @staticmethod
    def get_agent(
        agent_name: str,
        model_path: str,
        checkpoint_name: str | None,
        config: OfflinePLConfigFile,
        model_sub_directory: str | None = None,
        video_width: int | None = None,
        video_height: int | None = None,
        flip_y_axis: bool = False,
        **planner_kwargs: dict[str, Any],
    ) -> ToyPytorchAgent:
        assert isinstance(
            config, OfflinePLConfigFile
        ), f"Invalid config type: {type(config)}"
        pytorch_agent_name = get_pytorch_agent_name(agent_name)
        pytorch_agent = PytorchAgentFactory.get_agent(
            agent_name=pytorch_agent_name,
            model_path=model_path,
            checkpoint_name=checkpoint_name,
            config=config,
            model_sub_directory=model_sub_directory,
            video_width=video_width,
            video_height=video_height,
            **planner_kwargs,  # type: ignore
        )
        return ToyPytorchAgent(pytorch_agent, flip_y_axis)
