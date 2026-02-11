# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from pidm_imitation.constants import ENV_CONFIG_FILE_SUFFIX
from pidm_imitation.environment.toy_env.configs import (
    ToyEnvironmentConfig,
    ToyEnvironmentExogenousNoiseConfig,
)
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.environment.toy_env.toy_types import (
    ExogenousNoiseType,
    ObservationType,
    TerminationCondition,
)


def toy_round(
    value: float | Tuple[float, ...] | np.ndarray, cast_to_int: bool = False
) -> np.ndarray:
    value = np.round(value, decimals=0)
    if cast_to_int:
        return value.astype(int)
    return value


def add_toy_env_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment randomisation"
    )
    parser.add_argument(
        "--num_goals", type=int, default=None, help="Number of goals in the room"
    )
    parser.add_argument(
        "--goal_ordering",
        nargs="+",
        default=None,
        help="Order of goals to visit as sequence of numbers (e.g. '0 1 2 3')",
    )
    parser.add_argument(
        "--randomise_agent_spawn",
        action="store_true",
        help="Randomise the spawn location of the agent",
    )
    parser.add_argument(
        "--randomise_goal_positions",
        action="store_true",
        help="Randomise the position of the goals",
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        choices=ObservationType.get_valid_names(),
        help="Type of observation to use",
        default=None,
    )
    parser.add_argument(
        "--termination_condition",
        type=str,
        choices=TerminationCondition.get_valid_names(),
        help="Termination condition for the environment",
        default=None,
    )
    parser.add_argument(
        "--add_transition_noise",
        action="store_true",
        help="Whether to use stochastic transitions",
    )
    parser.add_argument(
        "--transition_noise_mean",
        type=float,
        default=None,
        help="Mean of the noise to add to the transitions",
    )
    parser.add_argument(
        "--transition_noise_std",
        type=float,
        default=None,
        help="Standard deviation of the noise to add to the transitions",
    )
    parser.add_argument(
        "--fps", type=int, help="Frames per second (default 30)", default=30
    )


def add_toy_exogenous_noise_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--add_exogenous_noise",
        action="store_true",
        help="Add exogenous noise to the observations",
    )
    parser.add_argument(
        "--exogenous_noise_type",
        type=str,
        default=None,
        choices=ExogenousNoiseType.get_valid_names(),
        help="Type of noise to add to the observations",
    )
    parser.add_argument(
        "--exogenous_feature_vector_noise_dim",
        type=int,
        default=None,
        help="Number of noise values to add to the observations (only used if"
        + "add_exogenous_noise = True and observation_type = feature_state)",
    )
    parser.add_argument(
        "--exogenous_feature_vector_noise_mean",
        type=float,
        default=None,
        help="Mean of the noise to add to feature vector observations",
    )
    parser.add_argument(
        "--exogenous_feature_vector_noise_std",
        type=float,
        default=None,
        help="Standard deviation of the noise to add to feature vector observations",
    )
    parser.add_argument(
        "--exogenous_feature_vector_random_walk_noise_mean",
        type=float,
        default=None,
        help="Mean of the noise to add to feature vector observations with random walk noise",
    )
    parser.add_argument(
        "--exogenous_feature_vector_random_walk_noise_std",
        type=float,
        default=None,
        help="Standard deviation of the noise to add to the observations with random walk noise",
    )
    parser.add_argument(
        "--exogenous_image_noise_min",
        type=int,
        default=None,
        help="Minimum noise value to add to image observations",
    )
    parser.add_argument(
        "--exogenous_image_noise_max",
        type=int,
        default=None,
        help="Maximum noise value to add to image observations",
    )
    parser.add_argument(
        "--exogenous_image_random_walk_noise_min",
        type=int,
        default=None,
        help="Minimum noise value to add to image observations with random walk noise",
    )
    parser.add_argument(
        "--exogenous_image_random_walk_noise_max",
        type=int,
        default=None,
        help="Maximum noise value to add to image observations with random walk noise",
    )


def overwrite_toy_config_from_exogenous_noise_args(
    exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig,
    input_type: ObservationType,
    args,
) -> None:
    if args.add_exogenous_noise:
        exogenous_noise_config.add_noise = True
    if args.exogenous_feature_vector_noise_dim:
        exogenous_noise_config.feature_vector_noise_dim = (
            args.exogenous_feature_vector_noise_dim
        )
        if (
            input_type != ObservationType.FEATURE_STATE
            or not exogenous_noise_config.add_noise
        ):
            warnings.warn(
                "exogenous_feature_vector_noise_dim is set but not used. To use noise values, set observation_type ="
                + " feature_state and add_exogenous_noise = True. Ignoring exogenous_feature_vector_noise_dim."
            )
    if (
        input_type == ObservationType.FEATURE_STATE
        and exogenous_noise_config.add_noise
        and exogenous_noise_config.feature_vector_noise_dim == 0
    ):
        warnings.warn(
            "Feature state observations are used and add_exogenous_noise = True but"
            + " exogenous_feature_vector_noise_dim=0. This will result in no noise being added to the observations."
            + " If you want to add noise to feature vectors, set feature_vector_noise_dim > 0 with the"
            + " --exogenous_feature_vector_noise_dim flag."
        )
    if args.exogenous_noise_type:
        exogenous_noise_config.noise_type = ExogenousNoiseType.from_str(
            args.exogenous_noise_type
        )
    if args.exogenous_feature_vector_noise_mean:
        exogenous_noise_config.feature_vector_noise_mean = (
            args.exogenous_feature_vector_noise_mean
        )
    if args.exogenous_feature_vector_noise_std:
        assert (
            args.exogenous_feature_vector_noise_std >= 0
        ), "Exogenous noise std must be greater than 0"
        exogenous_noise_config.feature_vector_noise_std = (
            args.exogenous_feature_vector_noise_std
        )
    if args.exogenous_feature_vector_random_walk_noise_mean:
        exogenous_noise_config.feature_vector_random_walk_noise_mean = (
            args.exogenous_feature_vector_random_walk_noise_mean
        )
    if args.exogenous_feature_vector_random_walk_noise_std:
        assert (
            args.exogenous_feature_vector_random_walk_noise_std >= 0
        ), "Exogenous noise std must be greater than 0"
        exogenous_noise_config.feature_vector_random_walk_noise_std = (
            args.exogenous_feature_vector_random_walk_noise_std
        )
    if args.exogenous_image_noise_min:
        exogenous_noise_config.image_noise_min = args.exogenous_image_noise_min
    if args.exogenous_image_noise_max:
        exogenous_noise_config.image_noise_max = args.exogenous_image_noise_max
    if args.exogenous_image_random_walk_noise_min:
        exogenous_noise_config.image_random_walk_noise_min = (
            args.exogenous_image_random_walk_noise_min
        )
    if args.exogenous_image_random_walk_noise_max:
        exogenous_noise_config.image_random_walk_noise_max = (
            args.exogenous_image_random_walk_noise_max
        )


def create_toy_config_from_args(args) -> ToyEnvironmentConfig:
    config = ToyEnvironmentConfig(args.config)
    config.seed = args.seed

    # env observation config
    if args.observation_type:
        config.observation_config.observation_type = ObservationType.from_str(
            args.observation_type
        )

    # env termination config
    if args.termination_condition:
        config.goal_config.termination_condition = TerminationCondition.from_str(
            args.termination_condition
        )

    # env goals
    if args.num_goals:
        config.goal_config.num_goals = args.num_goals
        if len(config.goal_config.goal_ordering) != config.goal_config.num_goals:
            raise ValueError(
                f"Goal ordering length ({len(config.goal_config.goal_ordering)}) does not match number of goals"
                f" ({config.goal_config.num_goals})."
            )

    if args.goal_ordering:
        config.goal_config.goal_ordering = [
            int(idx) for idx in list(args.goal_ordering)
        ]
        assert len(config.goal_config.goal_ordering) == config.goal_config.num_goals, (
            f"Goal ordering length ({len(config.goal_config.goal_ordering)}) does not match number of goals "
            f"({config.goal_config.num_goals})"
        )
        assert set(config.goal_config.goal_ordering) == set(
            range(config.goal_config.num_goals)
        ), (
            f"Goal ordering ({config.goal_config.goal_ordering}) does not match number of goals "
            f"({config.goal_config.num_goals})"
        )

    # env stochastic transitions
    if args.add_transition_noise:
        config.action_config.add_transition_noise = True
    if args.transition_noise_mean:
        config.action_config.transition_gaussian_noise_mean = (
            args.transition_noise_mean,
            args.transition_noise_mean,
        )
    if args.transition_noise_std:
        assert (
            args.transition_noise_std >= 0
        ), "Transition noise std must be greater than 0"
        config.action_config.transition_gaussian_noise_std = (
            args.transition_noise_std,
            args.transition_noise_std,
        )

    # env randomisation config
    if args.randomise_agent_spawn:
        config.goal_config.randomise_config.randomise_agent_spawn = True
    if args.randomise_goal_positions:
        config.goal_config.randomise_config.randomise_goal_positions = True

    # env exogenous noise config
    if config.observation_config.exogenous_noise_config is None:
        config.observation_config.exogenous_noise_config = (
            ToyEnvironmentExogenousNoiseConfig(config.config_path, {"add_noise": False})
        )
    overwrite_toy_config_from_exogenous_noise_args(
        config.observation_config.exogenous_noise_config,
        config.observation_config.observation_type,
        args,
    )

    return config


def get_trajectory_dir_and_name(env_config_file: Path) -> tuple[Path, str]:
    """Get the trajectory directory and name from the environment config file path.

    :param env_config_file: The path to the environment config file.
    :return: A tuple containing the trajectory directory and name.
    """
    trajectory_dir = env_config_file.parent
    trajectory_name = env_config_file.name[: -len(f"{ENV_CONFIG_FILE_SUFFIX}.yaml")]
    return trajectory_dir, trajectory_name


def get_trajectories(data_dir: Path):
    for env_config_file in tqdm(
        list(data_dir.glob(f"**/*{ENV_CONFIG_FILE_SUFFIX}.yaml")),
        desc="Processing trajectories",
    ):
        trajectory_dir, trajectory_name = get_trajectory_dir_and_name(env_config_file)
        trajectory = ToyEnvironmentTrajectory.init_from_dir(
            str(trajectory_dir), trajectory_name
        )
        yield trajectory
