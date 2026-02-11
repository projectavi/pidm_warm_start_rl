# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

from schema import And, Optional, Schema

from pidm_imitation.environment.toy_env.toy_types import (
    ExogenousNoiseType,
    ObservationType,
    TerminationCondition,
)
from pidm_imitation.utils.config_base import ConfigFile
from pidm_imitation.utils.subconfig import SubConfig


def parse_and_validate_layout(layout: str) -> str:
    """Validate that the layout is a valid rectangular grid and remove empty rows."""
    layout_rows = [row for row in layout.strip().split("\n") if row.strip()]
    layout_width = len(layout_rows[0])
    assert all(
        len(row) == layout_width for row in layout_rows
    ), "Only rectangular layouts are supported"
    return "\n".join(layout_rows)


class ToyEnvironmentExogenousNoiseConfig(SubConfig):
    KEY = "exogenous_noise"

    DEFAULT_NOISE_TYPE = "iid"
    DEFAULT_FEATURE_NOISE_DIM = 0
    DEFAULT_FEATURE_NOISE_MEAN = 0.0
    DEFAULT_FEATURE_NOISE_STD = 1.0
    DEFAULT_FEATURE_RANDOM_WALK_NOISE_MEAN = 0.0
    DEFAULT_FEATURE_RANDOM_WALK_NOISE_STD = 0.3
    DEFAULT_IMAGE_NOISE_MIN = 0
    DEFAULT_IMAGE_NOISE_MAX = 256
    DEFAULT_IMAGE_RANDOM_WALK_NOISE_MIN = -16
    DEFAULT_IMAGE_RANDOM_WALK_NOISE_MAX = 16

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.add_noise: bool
        # iid noise or correlated random walk noise
        self.noise_type: ExogenousNoiseType
        # Noise parameterisation for feature vector observations
        self.feature_vector_noise_dim: int
        self.feature_vector_noise_mean: float
        self.feature_vector_noise_std: float
        # Noise parameterisation for feature vector observations with random walk noise
        # Initial noise value is sampled as above but incremental noise added is sampled from Gaussian distribution
        # with following parameters
        self.feature_vector_random_walk_noise_mean: float
        self.feature_vector_random_walk_noise_std: float
        # Noise parameterisation for image observations
        self.image_noise_min: int
        self.image_noise_max: int
        # Noise parameterisation for image observations with random walk noise
        # Initial noise value is sampled as above but incremental noise added is sampled from uniform distribution
        # with following parameters
        self.image_random_walk_noise_min: int
        self.image_random_walk_noise_max: int
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("add_noise"): bool,
                Optional("noise_type"): str,
                Optional("feature_vector_noise_dim"): int,
                Optional("feature_vector_noise_mean"): float,
                Optional("feature_vector_noise_std"): float,
                Optional("feature_vector_random_walk_noise_mean"): float,
                Optional("feature_vector_random_walk_noise_std"): float,
                Optional("image_noise_min"): int,
                Optional("image_noise_max"): int,
                Optional("image_random_walk_noise_min"): int,
                Optional("image_random_walk_noise_max"): int,
            }
        )

    def _set_attributes(self) -> None:
        self.add_noise = self._config.get("add_noise", False)
        self.noise_type = ExogenousNoiseType.from_str(
            self._config.get("noise_type", self.DEFAULT_NOISE_TYPE)
        )
        self.feature_vector_noise_dim = self._config.get(
            "feature_vector_noise_dim", self.DEFAULT_FEATURE_NOISE_DIM
        )
        self.feature_vector_noise_mean = self._config.get(
            "feature_vector_noise_mean", self.DEFAULT_FEATURE_NOISE_MEAN
        )
        self.feature_vector_noise_std = self._config.get(
            "feature_vector_noise_std", self.DEFAULT_FEATURE_NOISE_STD
        )
        self.feature_vector_random_walk_noise_mean = self._config.get(
            "feature_vector_random_walk_noise_mean",
            self.DEFAULT_FEATURE_RANDOM_WALK_NOISE_MEAN,
        )
        self.feature_vector_random_walk_noise_std = self._config.get(
            "feature_vector_random_walk_noise_std",
            self.DEFAULT_FEATURE_RANDOM_WALK_NOISE_STD,
        )
        assert (
            self.feature_vector_noise_std >= 0.0
            and self.feature_vector_random_walk_noise_std >= 0.0
        ), "Feature vector noise std must be non-negative"
        assert (
            self.feature_vector_noise_dim >= 0
        ), "Feature vector noise dim must be non-negative"
        self.image_noise_min = self._config.get(
            "image_noise_min", self.DEFAULT_IMAGE_NOISE_MIN
        )
        self.image_noise_max = self._config.get(
            "image_noise_max", self.DEFAULT_IMAGE_NOISE_MAX
        )
        assert (
            0 <= self.image_noise_min <= self.image_noise_max <= 256
        ), "Image noise min and max must be between 0 and 256"
        self.image_random_walk_noise_min = self._config.get(
            "image_random_walk_noise_min", self.DEFAULT_IMAGE_RANDOM_WALK_NOISE_MIN
        )
        self.image_random_walk_noise_max = self._config.get(
            "image_random_walk_noise_max", self.DEFAULT_IMAGE_RANDOM_WALK_NOISE_MAX
        )

    def get_config(self) -> dict:
        return {
            "add_noise": self.add_noise,
            "noise_type": self.noise_type.value,
            "feature_vector_noise_dim": self.feature_vector_noise_dim,
            "feature_vector_noise_mean": self.feature_vector_noise_mean,
            "feature_vector_noise_std": self.feature_vector_noise_std,
            "feature_vector_random_walk_noise_mean": self.feature_vector_random_walk_noise_mean,
            "feature_vector_random_walk_noise_std": self.feature_vector_random_walk_noise_std,
            "image_noise_min": self.image_noise_min,
            "image_noise_max": self.image_noise_max,
            "image_random_walk_noise_min": self.image_random_walk_noise_min,
            "image_random_walk_noise_max": self.image_random_walk_noise_max,
        }


class ToyEnvironmentStateConfig(SubConfig):
    KEY = "state"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.positions_relative_to_agent: bool
        self.exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("positions_relative_to_agent"): bool,
                Optional(ToyEnvironmentExogenousNoiseConfig.KEY): object,
            }
        )

    def _set_attributes(self) -> None:
        self.positions_relative_to_agent = self._config.get(
            "positions_relative_to_agent", False
        )
        self.exogenous_noise_config = self._get_simple_config_obj(
            ToyEnvironmentExogenousNoiseConfig
        )
        if self.exogenous_noise_config is None:
            self.exogenous_noise_config = ToyEnvironmentExogenousNoiseConfig(
                self.config_path, {}
            )

    def get_config(self) -> dict:
        return {
            "positions_relative_to_agent": self.positions_relative_to_agent,
            ToyEnvironmentExogenousNoiseConfig.KEY: self.exogenous_noise_config.get_config(),
        }


class ToyEnvironmentObservationConfig(SubConfig):
    KEY = "observation"

    DEFAULT_OBS_GOALS = True
    DEFAULT_OBS_RELATIVE_POSITIONS = False
    DEFAULT_IMAGE_OBS_SIZE = (224, 224)

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.observation_type: ObservationType
        # only for feature vector observation: if True, the goals are included in the observation
        self.observe_goals: bool
        # only for feature vector observation and observing goals: if True, the positions are relative to the agent,
        # otherwise absolute
        self.positions_relative_to_agent: bool
        # only for image observation: size of the image observation
        self.img_obs_size: Tuple[int, int]
        # noise parameters
        self.exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig
        self._set_attributes()

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                "observation_type": object,
                Optional("observe_goals"): bool,
                Optional("positions_relative_to_agent"): bool,
                Optional("img_obs_size"): [int, int],
                Optional(ToyEnvironmentExogenousNoiseConfig.KEY): object,
            }
        )
        return schema

    def _set_attributes(self) -> None:
        self.observation_type = ObservationType.from_str(
            self._config["observation_type"]
        )
        self.observe_goals = self._config.get("observe_goals", self.DEFAULT_OBS_GOALS)
        self.positions_relative_to_agent = self._config.get(
            "positions_relative_to_agent", self.DEFAULT_OBS_RELATIVE_POSITIONS
        )
        self.img_obs_size = tuple(
            self._config.get("img_obs_size", self.DEFAULT_IMAGE_OBS_SIZE)
        )
        assert (
            self.img_obs_size[0] > 0 and self.img_obs_size[1] > 0
        ), "Image size must be positive"
        self.exogenous_noise_config = self._get_simple_config_obj(
            ToyEnvironmentExogenousNoiseConfig
        )
        if self.exogenous_noise_config is None:
            self.exogenous_noise_config = ToyEnvironmentExogenousNoiseConfig(
                self.config_path, {}
            )

    def get_config(self) -> dict:
        return {
            "observation_type": self.observation_type.value,
            "observe_goals": self.observe_goals,
            "positions_relative_to_agent": self.positions_relative_to_agent,
            "img_obs_size": list(self.img_obs_size),
            ToyEnvironmentExogenousNoiseConfig.KEY: self.exogenous_noise_config.get_config(),
        }


class ToyEnvironmentActionConfig(SubConfig):
    KEY = "action"

    DEFAULT_CONTROL_DEADZONE = 0.1
    DEFAULT_VELOCITY_SCALE = (20.0, 20.0)
    DEFAULT_ADD_TRANSITION_NOISE = False
    DEFAULT_TRANSITION_NOISE_MEAN = (0.0, 0.0)
    DEFAULT_TRANSITION_NOISE_STD = (0.0, 0.0)

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.control_deadzone: float
        self.velocity_scale: Tuple[float, float]
        self.add_transition_noise: bool
        self.transition_gaussian_noise_mean: Tuple[float, float]
        self.transition_gaussian_noise_std: Tuple[float, float]
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("control_deadzone"): float,
                Optional("velocity_scale"): tuple[float, float],
                Optional("add_transition_noise"): bool,
                Optional("transition_gaussian_noise_mean"): tuple[float, float],
                Optional("transition_gaussian_noise_std"): tuple[float, float],
            }
        )

    def _set_attributes(self) -> None:
        self.control_deadzone = self._config.get(
            "control_deadzone", self.DEFAULT_CONTROL_DEADZONE
        )
        assert self.control_deadzone >= 0.0, "Control deadzone must be non-negative"
        self.velocity_scale = self._config.get(
            "velocity_scale", self.DEFAULT_VELOCITY_SCALE
        )
        self.add_transition_noise = self._config.get(
            "add_transition_noise", self.DEFAULT_ADD_TRANSITION_NOISE
        )
        self.transition_gaussian_noise_mean = self._config.get(
            "transition_gaussian_noise_mean", self.DEFAULT_TRANSITION_NOISE_MEAN
        )
        self.transition_gaussian_noise_std = self._config.get(
            "transition_gaussian_noise_std", self.DEFAULT_TRANSITION_NOISE_STD
        )
        assert (
            self.transition_gaussian_noise_std[0] >= 0.0
            and self.transition_gaussian_noise_std[1] >= 0.0
        ), "Transition noise std must be non-negative"

    def get_config(self) -> dict:
        return {
            "control_deadzone": self.control_deadzone,
            "velocity_scale": list(self.velocity_scale),
            "add_transition_noise": self.add_transition_noise,
            "transition_gaussian_noise_mean": list(self.transition_gaussian_noise_mean),
            "transition_gaussian_noise_std": list(self.transition_gaussian_noise_std),
        }


class ToyEnvironmentRandomisationConfig(SubConfig):
    KEY = "randomisation"
    GOAL_RANDOMISATION_TYPES = ["candidates", "anywhere"]

    DEFAULT_RANDOMISE_AGENT_SPAWN = False
    DEFAULT_RANDOMISE_GOAL_POSITIONS = False
    DEFAULT_GOAL_RANDOMISATION_TYPE = "candidates"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.randomise_agent_spawn: bool
        self.randomise_goal_positions: bool
        self.goal_randomisation_type: str
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("randomise_agent_spawn"): bool,
                Optional("randomise_goal_positions"): bool,
                Optional("goal_randomisation_type"): And(
                    str,
                    lambda x: x in self.GOAL_RANDOMISATION_TYPES,
                    error="Invalid goal randomisation type",
                ),
            }
        )

    def _set_attributes(self) -> None:
        self.randomise_agent_spawn = self._config.get(
            "randomise_agent_spawn", self.DEFAULT_RANDOMISE_AGENT_SPAWN
        )
        self.randomise_goal_positions = self._config.get(
            "randomise_goal_positions", self.DEFAULT_RANDOMISE_GOAL_POSITIONS
        )
        self.goal_randomisation_type = self._config.get(
            "goal_randomisation_type", self.DEFAULT_GOAL_RANDOMISATION_TYPE
        )

    def get_config(self) -> dict:
        return {
            "randomise_agent_spawn": self.randomise_agent_spawn,
            "randomise_goal_positions": self.randomise_goal_positions,
            "goal_randomisation_type": self.goal_randomisation_type,
        }


class ToyEnvironmentRenderingConfig(SubConfig):
    KEY = "rendering"

    DEFAULT_GOAL_SIZE = 20
    DEFAULT_AGENT_SIZE = 20
    DEFAULT_TRACE_SIZE = 1
    DEFAULT_RENDER_PLAYER_TRACE = True
    DEFAULT_RENDER_PLANNER_TRACE = False
    DEFAULT_TRACE_LENGTH = -1

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.goal_size: int
        self.agent_size: int
        self.render_player_trace: bool
        self.render_planner_trace: bool
        self.trace_size: int
        self.max_trace_length: int
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("goal_size"): int,
                Optional("agent_size"): int,
                Optional("trace_size"): int,
                Optional("render_player_trace"): bool,
                Optional("render_planner_trace"): bool,
                Optional("max_trace_length"): int,
            }
        )

    def _set_attributes(self) -> None:
        self.goal_size = self._config.get("goal_size", self.DEFAULT_GOAL_SIZE)
        self.agent_size = self._config.get("agent_size", self.DEFAULT_AGENT_SIZE)
        self.trace_size = self._config.get("trace_size", self.DEFAULT_TRACE_SIZE)
        self.render_player_trace = self._config.get(
            "render_player_trace", self.DEFAULT_RENDER_PLAYER_TRACE
        )
        self.render_planner_trace = self._config.get(
            "render_planner_trace", self.DEFAULT_RENDER_PLANNER_TRACE
        )
        self.max_trace_length = self._config.get(
            "max_trace_length", self.DEFAULT_TRACE_LENGTH
        )
        assert self.goal_size > 0, "Goal size must be positive"
        assert self.agent_size > 0, "Agent size must be positive"
        assert self.trace_size > 0, "Trace size must be positive"
        assert (
            self.max_trace_length == -1 or self.max_trace_length > 0
        ), "Max trace length must be -1 (unlimited) or positive"

    def get_config(self) -> dict:
        return {
            "goal_size": self.goal_size,
            "agent_size": self.agent_size,
            "trace_size": self.trace_size,
            "render_player_trace": self.render_player_trace,
            "render_planner_trace": self.render_planner_trace,
            "max_trace_length": self.max_trace_length,
        }


class ToyEnvironmentGoalConfig(SubConfig):
    KEY = "goal"

    DEFAULT_REWARD_PER_GOAL = 1.0
    DEFAULT_TERMINATION_CONDITION = "all_goals"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.num_goals: int
        self.goal_ordering: List[int]
        self.reward_per_reached_goal: float
        self.termination_condition: TerminationCondition
        self.randomise_config: ToyEnvironmentRandomisationConfig
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                "num_goals": int,
                Optional("goal_ordering"): list,
                Optional("reward_per_reached_goal"): float,
                Optional("termination_condition"): object,
                Optional(ToyEnvironmentRandomisationConfig.KEY): object,
            }
        )

    def _set_attributes(self) -> None:
        self.num_goals = self._config["num_goals"]
        assert self.num_goals >= 1, "Need at least one goal!"
        self.goal_ordering = self._config.get(
            "goal_ordering", list(range(self.num_goals))
        )
        assert (
            len(self.goal_ordering) == self.num_goals
        ), "Goal ordering must match number of goals"
        assert set(self.goal_ordering) == set(
            range(self.num_goals)
        ), "Goal ordering must contain all goals in any order"

        self.reward_per_reached_goal = self._config.get(
            "reward_per_reached_goal", self.DEFAULT_REWARD_PER_GOAL
        )
        assert self.reward_per_reached_goal > 0.0, "Reward must be positive"

        self.termination_condition = TerminationCondition.from_str(
            self._config.get(
                "termination_condition", self.DEFAULT_TERMINATION_CONDITION
            )
        )

        self.randomise_config = self._get_simple_config_obj(
            ToyEnvironmentRandomisationConfig
        )

    def get_config(self) -> dict:
        return {
            "num_goals": self.num_goals,
            "goal_ordering": self.goal_ordering,
            "reward_per_reached_goal": self.reward_per_reached_goal,
            "termination_condition": self.termination_condition.value,
            ToyEnvironmentRandomisationConfig.KEY: (
                self.randomise_config.get_config() if self.randomise_config else None
            ),
        }


class ToyEnvironmentConfig(ConfigFile):

    DEFAULT_MAX_STEPS = 500

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.seed: int
        self.room_size: Tuple[int, int]
        self.layout: str
        self.max_steps: int

        self.observation_config: ToyEnvironmentObservationConfig
        self.state_config: ToyEnvironmentStateConfig
        self.action_config: ToyEnvironmentActionConfig
        self.rendering_config: ToyEnvironmentRenderingConfig
        self.goal_config: ToyEnvironmentGoalConfig

        self._set_attributes()

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                "seed": int,
                "room_size": [int, int],
                "layout": str,
                Optional("max_steps"): int,
                ToyEnvironmentObservationConfig.KEY: object,
                ToyEnvironmentStateConfig.KEY: object,
                ToyEnvironmentActionConfig.KEY: object,
                ToyEnvironmentRenderingConfig.KEY: object,
                ToyEnvironmentGoalConfig.KEY: object,
            }
        )
        return schema

    def _create_sub_configs(self) -> None:
        self.observation_config = self._get_simple_config_obj(
            ToyEnvironmentObservationConfig
        )
        self.state_config = self._get_simple_config_obj(ToyEnvironmentStateConfig)
        self.action_config = self._get_simple_config_obj(ToyEnvironmentActionConfig)
        self.rendering_config = self._get_simple_config_obj(
            ToyEnvironmentRenderingConfig
        )
        self.goal_config = self._get_simple_config_obj(ToyEnvironmentGoalConfig)

    def _set_attributes(self) -> None:
        self.seed = self._config["seed"]
        self.room_size = tuple(self._config["room_size"])
        self.layout = parse_and_validate_layout(self._config["layout"])
        self.max_steps = self._config.get("max_steps", self.DEFAULT_MAX_STEPS)
        assert self.max_steps > 0, "Max steps must be positive"

        self._create_sub_configs()

    def get_config(self) -> dict:
        return {
            "seed": self.seed,
            "room_size": list(self.room_size),
            "layout": self.layout,
            "max_steps": self.max_steps,
            ToyEnvironmentObservationConfig.KEY: self.observation_config.get_config(),
            ToyEnvironmentStateConfig.KEY: self.state_config.get_config(),
            ToyEnvironmentActionConfig.KEY: self.action_config.get_config(),
            ToyEnvironmentRenderingConfig.KEY: self.rendering_config.get_config(),
            ToyEnvironmentGoalConfig.KEY: self.goal_config.get_config(),
        }
