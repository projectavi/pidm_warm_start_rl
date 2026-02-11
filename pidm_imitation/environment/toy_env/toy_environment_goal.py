# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Set, Tuple

import gymnasium as gym
import numpy as np
import pygame

from pidm_imitation.environment.toy_env.configs import (
    ToyEnvironmentConfig,
    ToyEnvironmentExogenousNoiseConfig,
    ToyEnvironmentRandomisationConfig,
)
from pidm_imitation.environment.toy_env.exogenous_noise_utils import (
    add_exogenous_noise_to_feature_vector,
)
from pidm_imitation.environment.toy_env.game_elements import (
    BlockSprite,
    Goal,
    Player,
    Wall,
    blocksprite_collide_with_all,
)
from pidm_imitation.environment.toy_env.toy_constants import NUM_TRIALS_RANDOM_SPAWN
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.environment.toy_env.toy_types import (
    ObservationType,
    TerminationCondition,
)
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


class GoalReachingToyEnvironment(ToyEnvironment):

    def __init__(
        self,
        config: ToyEnvironmentConfig | None,
    ):
        super().__init__(config)
        self.goal_config = config.goal_config
        self.num_goals: int = self.goal_config.num_goals
        self.goal_ordering: List[int] = self.goal_config.goal_ordering
        assert (
            len(self.goal_ordering) == self.num_goals
        ), f"Goal ordering length {len(self.goal_ordering)} does not match number of goals {self.num_goals}."
        self.randomise_config: ToyEnvironmentRandomisationConfig = (
            config.goal_config.randomise_config
        )
        self.observation_space: gym.Space = self.build_observation_space()

        self.goal_positions = self.positions["goals"]
        if len(self.goal_positions) < self.num_goals:
            log.warning(
                f"Found {len(self.goal_positions)} goal positions in layout, but {self.num_goals} are required."
            )

        self.goals: List[Goal] = []
        self._goals_group: pygame.sprite.Group | None = None
        self.current_goal_index: int | None = 0
        self.reached_current_goal_last_step: bool = False

    def _get_feature_state_observation_dim(self) -> int:
        """Feature state observations include
        - agent position (2)
        - for each goal (if observe_goals is True): goal position (2) and reached status (1)
        - exogenous noise values (if add_noise is True)
        """
        goal_values = 3 * self.num_goals if self.observation_config.observe_goals else 0
        noise_values = (
            self.observation_config.exogenous_noise_config.feature_vector_noise_dim
            if self.observation_config.exogenous_noise_config.add_noise
            else 0
        )
        return 2 + goal_values + noise_values

    @property
    def goals_set(self) -> Set[Goal]:
        return set(self.goals)

    @property
    def objects(self) -> Set[BlockSprite]:
        objects: Set[BlockSprite] = self.walls.union(self.goals_set)
        if self.player is not None:
            objects.add(self.player)
        return objects

    @property
    def goals_group(self) -> pygame.sprite.Group:
        return self._goals_group

    def parse_layout_string(self) -> Dict[str, np.ndarray]:
        """Parse a map layout of the environment from a string.

        Legend:
        - A: agent candidates (at least one is required)
        - G: goal candidate (at least one is required)
        - W: wall

        Example layout:
        WWWWWWWWWWW
        W         W
        W G     G W
        W         W
        W         W
        W    A    W
        W         W
        W         W
        W G     G W
        W         W
        WWWWWWWWWWW

        Spawn selection:
        - Agent: if randomise_config.randomise_agent_spawn = False, the agent will be spawned at a randomly
            selected position marked with 'A'
        - Goal: if randomise_config.randomise_goal_positions = False, `num_goals` goals will be spawned at
            the locations marked with 'G' in order from the top to the bottom, left to right, of the layout
            If randomise_config.randomise_goal_positions = True, num_goals goals will be spawned by randomly
            sampling from the goal candidates marked with 'G'

        :param layout: string with the layout
        :return: dictionary with keys 'agent', 'goals', and 'walls' containing the positions of the
            agent, goals, and walls
        """
        # get agent and walls positions
        positions = super().parse_layout_string()

        layout_rows: List[str] = [row for row in self.layout.split("\n") if row.strip()]
        layout_dims = (len(layout_rows), len(layout_rows[0]))
        goal_positions = []

        # get goal positions
        for y, row in enumerate(layout_rows):
            for x, char in enumerate(row):
                if char == "G":
                    goal_positions.append((float(x), float(y)))

        return {
            **positions,
            "goals": (
                np.stack(
                    [
                        ToyEnvironment.convert_position_to_room_size(
                            pos, layout_dims, self.room_size
                        )
                        for pos in goal_positions
                    ]
                )
                if goal_positions
                else np.array([])
            ),
        }

    def _update_next_goal_index(self):
        if self.goals[self.current_goal_index].reached:
            if self.current_goal_index < len(self.goals) - 1:
                self.current_goal_index += 1
                self.goals[self.current_goal_index].set_active()
            else:
                self.current_goal_index = None

    def _update_goals(self) -> None:
        if self.current_goal_index is not None:
            # active goal remaining
            current_goal: Goal = self.goals[self.current_goal_index]
            if not current_goal.reached and self.player.collide_with(current_goal):
                # reached current goal
                current_goal.set_reached()
                self._update_next_goal_index()
                self.reached_current_goal_last_step = True
            else:
                # no goal reached
                self.reached_current_goal_last_step = False

    def spawn_wall_in_position(
        self, name: str, position: np.ndarray, size: Tuple[int, int]
    ) -> Wall:
        wall = Wall(position, size, name=name)
        if (
            len(self.goals)
            and blocksprite_collide_with_all(wall, self.goals_group) is not None
        ):
            raise ValueError(f"Wall {wall.name} collides with goal")
        if self.player is not None and wall.collide_with(self.player):
            raise ValueError(f"Wall {wall.name} collides with agent")
        self.walls.add(wall)
        return wall

    def _spawn_agent(self):
        agent = Player(
            position=self.agent_positions[0], size=self.rendering_config.agent_size
        )
        if self.randomise_config.randomise_agent_spawn:
            self.spawn_agent_in_random_position(agent)
        else:
            # select random agent position from candidates
            agent_position = self._np_random.choice(self.agent_positions)
            self.spawn_agent_in_position(agent, position=agent_position)
        self.player = agent
        self._player_group = pygame.sprite.Group(self.player)

    def spawn_goal_in_position(self, position: np.ndarray) -> Goal:
        goal = Goal(position=position, size=self.rendering_config.goal_size)
        crash = blocksprite_collide_with_all(goal, self.objects_group)
        if crash is not None:
            names = ",".join([obj.name for obj in crash])
            raise ValueError(f"Goal {goal.name} collides with objects: {names}")
        self.goals.append(goal)
        return goal

    def spawn_goal_in_random_position(
        self, num_trials: int = NUM_TRIALS_RANDOM_SPAWN
    ) -> Goal:
        goal = Goal(
            position=self._get_random_position(), size=self.rendering_config.goal_size
        )
        self._spawn_sprite_in_random_position(goal, num_trials=num_trials)
        self.goals.append(goal)
        return goal

    def _spawn_goals(self):
        for goal_idx in self.goal_ordering:
            if self.randomise_config.randomise_goal_positions:
                if self.randomise_config.goal_randomisation_type == "candidates":
                    # spawn goal in random position among the goal candidates
                    # but ensure that the same position is not used twice
                    unused_goal_positions = [
                        goal_pos
                        for goal_pos in self.goal_positions
                        if not any(all(goal_pos == g.position) for g in self.goals)
                    ]
                    goal_position = self._np_random.choice(unused_goal_positions)
                    self.spawn_goal_in_position(goal_position)
                elif self.randomise_config.goal_randomisation_type == "anywhere":
                    self.spawn_goal_in_random_position()
            else:
                self.spawn_goal_in_position(self.goal_positions[goal_idx])
        self.goals[0].set_active()
        self._goals_group = pygame.sprite.Group(self.goals)

    def _clear_env(self):
        super()._clear_env()
        self.goals.clear()

        self.current_goal_index = 0
        self.reached_current_goal_last_step = False

    def _spawn_all(self):
        super()._spawn_all()
        self._spawn_goals()

    def _done(self) -> bool:
        if self.goal_config.termination_condition == TerminationCondition.ANY_GOAL:
            return any(goal.reached for goal in self.goals)
        elif self.goal_config.termination_condition == TerminationCondition.ALL_GOALS:
            return all(goal.reached for goal in self.goals)
        raise ValueError(
            f"Expected termination condition to be one of {TerminationCondition.get_valid_names()} but got "
            f"{self.goal_config.termination_condition}"
        )

    def _reward(self) -> float:
        if self.reached_current_goal_last_step:
            return self.goal_config.reward_per_reached_goal
        return 0.0

    def _get_feature_vector(
        self,
        include_goals: bool,
        positions_relative_to_agent: bool,
        exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig | None = None,
    ) -> np.ndarray:
        """Get the feature vector of the environment.

        vector contains: agent position, and for each goal its relative position and whether it has been reached
        (if goals are included)
        Optionally, exogenous noise values can be added to the state.

        :param include_goals: whether to include goal positions and reached status in the feature vector
        :param positions_relative_to_agent: whether to include goal positions relative to the agent's position
        :param exogenous_noise_config: configuration for exogenous noise
        :return: feature vector as a numpy array
        """
        feature_parts = [p for p in self.player.position]
        if include_goals:
            # include position and whether goal has been reached for each goal
            for goal in self.goals:
                goal_pos = (
                    goal.position - self.player.position
                    if positions_relative_to_agent
                    else goal.position
                )
                feature_parts.extend(list(goal_pos))
                feature_parts.append(int(goal.reached))
        feature_vector = np.array(feature_parts, dtype=np.float32)
        if exogenous_noise_config is not None and exogenous_noise_config.add_noise:
            feature_vector, self.last_exogenous_noise = (
                add_exogenous_noise_to_feature_vector(
                    feature_vector,
                    exogenous_noise_config,
                    self._np_random,
                    self.last_exogenous_noise,
                )
            )
        return np.array(feature_vector, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return self._get_feature_vector(
            include_goals=True,
            positions_relative_to_agent=self.state_config.positions_relative_to_agent,
            exogenous_noise_config=self.state_config.exogenous_noise_config,
        )

    def get_observation(self) -> np.ndarray:
        """Get a simplified image of the agent's view."""
        if self.observation_config.observation_type == ObservationType.FEATURE_STATE:
            return self._get_feature_vector(
                self.observation_config.observe_goals,
                self.observation_config.positions_relative_to_agent,
                self.observation_config.exogenous_noise_config,
            )
        elif self.observation_config.observation_type == ObservationType.IMAGE_STATE:
            return self._get_current_image(observation_processing=True)
        else:
            raise ValueError(
                f"Unsupported observation type {self.observation_config.observation_type}"
            )

    def set_state(self, state: np.ndarray) -> None:
        """
        Set the environment to a specific state by updating player position and goal states.
        :param state: State array where first 2 elements are player position and remaining are goal states
        """
        self.reset()
        # Set player position
        state_position = state[:2].astype(np.float32)
        self.player.position = state_position
        self.player.previous_position = state_position

        # Set goal states
        if len(state) > 2 and len(self.goals) > 0:
            goals_state = state[2:]
            current_goal_index = 0
            for i, goal in enumerate(self.goals):
                goal_info = goals_state[i * 3 : (i + 1) * 3]
                assert (
                    len(goal_info) == 3
                ), "Each goal state must have 3 elements: x, y, reached"
                assert (
                    goal.position[0] == goal_info[0]
                    and goal.position[1] == goal_info[1]
                ), (
                    f"Goal position mismatch at index {i}: expected ({goal.position[0]}, {goal.position[1]}), got "
                    f"({goal_info[0]}, {goal_info[1]})"
                )
                goal_reached = goal_info[2]
                assert goal_reached in (0, 1), "Goal reached status must be 0 or 1"
                if bool(goal_reached):
                    goal.set_reached()
                if goal.reached:
                    current_goal_index = i + 1
            if current_goal_index >= len(self.goals):
                # All goals reached
                self.current_goal_index = None

    def get_observation_from_state(
        self, state: np.ndarray, last_noise: np.ndarray | None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the corresponding observation for the current state. Only supported for feature vector observations

        :param state: the state to convert to an observation. Contains the agent position, and position + reached
            condition of each goal.
        :param last_noise: last exogenous noise added to the observation in case it is needed for the conversion.
        :return: observation as a numpy array, last exogenous noise as a numpy array
        """
        assert (
            self.observation_config.observation_type == ObservationType.FEATURE_STATE
        ), (
            f"Converting state to observation is only supported for feature vector observations, "
            f"but got {self.observation_config.observation_type}"
        )
        assert (
            len(state) == 2 + 3 * self.num_goals
        ), f"State must contain 2 + 3 * num_goals values, but got {len(state)} values."

        # extract information from the state
        player_position = state[:2]
        obs_parts = [p for p in player_position]
        if self.observation_config.observe_goals:
            # include position and whether goal has been reached for each goal
            for i in range(self.num_goals):
                goal_position = state[2 + 3 * i : 2 + 3 * i + 2]
                if self.state_config.positions_relative_to_agent:
                    # position in state is relative to agent so convert it to absolute position
                    goal_position += player_position

                obs_goal_position = goal_position
                if self.observation_config.positions_relative_to_agent:
                    # convert absolute goal position to relative position to agent for observation
                    obs_goal_position -= player_position
                obs_parts.extend(list(obs_goal_position))
                goal_reached = int(state[2 + 3 * i + 2])
                obs_parts.append(goal_reached)
        obs = np.array(obs_parts, dtype=np.float32)

        if (
            self.observation_config.exogenous_noise_config is not None
            and self.observation_config.exogenous_noise_config.add_noise
        ):
            obs, last_noise = add_exogenous_noise_to_feature_vector(
                obs,
                self.observation_config.exogenous_noise_config,
                self._np_random,
                last_noise,
            )
        return np.array(obs, dtype=np.float32), last_noise

    def _step_env(self, action: np.ndarray) -> bool:
        """
        Update env state from action
        :param action: player action
        :return: whether the player is stuck
        """
        stuck = super()._step_env(action)
        self._update_goals()
        return stuck
