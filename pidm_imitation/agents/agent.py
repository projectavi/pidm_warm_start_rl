# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Any


class Agent(abc.ABC):
    """
    An agent is an entity that interacts with an environment by taking actions and
    receiving observations and rewards.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """
        pass

    @abc.abstractmethod
    def get_action(self, raw_obs: Any, built_obs: Any) -> Any | None:
        """
        Returns the next action, if any, to be taken by the agent.

        :param raw_obs: The raw unprocessed observation from the underlying environment.

        :param built_obs: The result of an associated ObservationBuilder.
        """
        pass

    @abc.abstractmethod
    def has_actions(self) -> bool:
        """
        Returns whether the agent has more actions to take.
        """
        pass

    def is_toy_agent(self) -> bool:
        return False
