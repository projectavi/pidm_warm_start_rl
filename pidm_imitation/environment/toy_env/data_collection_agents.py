# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import pygame

from pidm_imitation.constants import XBOX_CONTROLLERS
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.utils.joysticks import XBoxGamePadMap
from pidm_imitation.utils.user_inputs import UserInputs

HUMAN_AGENT_INITIAL_DEADZONE = (
    0.1  # initial deadzone for human agent to identify first active input
)
HUMAN_AGENT_ACTIVE_INPUT_WAIT_TIME = (
    0.01  # time to wait for active input from human agent
)
VALID_AGENT_TYPES = ["human", "random"]


class DataCollectionAgent(ABC):
    def __init__(self, env: ToyEnvironment):
        self.env: ToyEnvironment = env
        self.action_space: gym.Space = env.action_space

    @abstractmethod
    def act(self) -> np.ndarray:
        pass

    def wait_for_active_input(self) -> None:
        """Wait until the agent is actively providing input."""
        pass

    def reset(self):
        pass


class HumanDataCollectionAgent(DataCollectionAgent, XBoxGamePadMap):
    """Collect data as a human player using an Xbox controller."""

    def __init__(
        self,
        env: ToyEnvironment,
        initial_deadzone: float = HUMAN_AGENT_INITIAL_DEADZONE,
        initial_wait_time: float = HUMAN_AGENT_ACTIVE_INPUT_WAIT_TIME,
        **kwargs,
    ):
        """
        :param env: The toy environment in which the agent is moving.
        :param initial_deadzone: The initial deadzone to identify the first active input set.
        :param initial_wait_time: The wait time in between check for active input.
        """
        DataCollectionAgent.__init__(self, env)
        self.initial_deadzone = initial_deadzone
        self.initial_wait_time = initial_wait_time
        XBoxGamePadMap.__init__(self)

        pygame.init()
        pygame.joystick.init()

        # Initialize the Xbox controller
        self.joystick = None
        for i in range(pygame.joystick.get_count()):
            j = pygame.joystick.Joystick(i)
            j.init()
            if j.get_name() in XBOX_CONTROLLERS:
                self.joystick = j
                break

        if self.joystick is None:
            raise ValueError("No Xbox controller found.")

    def act(self) -> np.ndarray:
        user_inputs: UserInputs = self.get_user_inputs(self.joystick)
        return np.array(
            [
                user_inputs.left_stick_x,
                user_inputs.left_stick_y * -1,  # Invert y-axis
            ]
        )

    def _has_active_input(self) -> bool:
        """Check if the agent is actively providing input."""
        actions = self.act()
        return any(abs(action) > self.initial_deadzone for action in actions)

    def wait_for_active_input(self) -> None:
        """Wait until the agent is actively providing input."""
        # clear event queue to avoid processing old inputs
        pygame.event.clear()
        while not self._has_active_input():
            pygame.event.pump()  # Process events to keep the window responsive
            time.sleep(self.initial_wait_time)


class RandomDataCollectionAgent(DataCollectionAgent):
    """Collect data by taking random actions."""

    def __init__(self, env: ToyEnvironment, **kwargs):
        super().__init__(env)

    def act(self) -> np.ndarray:
        return self.action_space.sample().astype(np.float64)


def get_agent(env, agent_type: str, **kwargs) -> DataCollectionAgent:
    if agent_type == "human":
        return HumanDataCollectionAgent(env, **kwargs)
    if agent_type == "random":
        return RandomDataCollectionAgent(env, **kwargs)
    raise ValueError(
        f"Invalid agent type: {agent_type}, must be one of {', '.join(VALID_AGENT_TYPES)}"
    )
