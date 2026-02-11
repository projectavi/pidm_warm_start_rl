# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import numpy as np

from pidm_imitation.environment.toy_env.configs import (
    ToyEnvironmentExogenousNoiseConfig,
)
from pidm_imitation.environment.toy_env.toy_constants import BACKGROUND_COLOR
from pidm_imitation.environment.toy_env.toy_types import (
    ExogenousNoiseType,
    ObservationType,
)


def add_exogenous_noise_to_feature_vector(
    feature_vector: np.ndarray,
    exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig,
    np_random: np.random.Generator,
    last_noise: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add exogenous noise to the feature vector.

    :param feature_vector (np.ndarray): The feature vector to add noise to.
    :param exogenous_noise_config (ToyEnvironmentExogenousNoiseConfig): The configuration for the exogenous noise.
    :param np_random (np.random.Generator): The random number generator to use for adding noise.
    :param last_noise (np.ndarray | None): The last noise that has been added to the feature vector.
        (None if no noise has been added yet).
    :return (Tuple[np.ndarray, np.ndarray]): The feature vector with added noise and the noise that was added.
    """
    if exogenous_noise_config.noise_type == ExogenousNoiseType.IID or (
        exogenous_noise_config.noise_type == ExogenousNoiseType.RANDOM_WALK
        and last_noise is None
    ):
        noise = np_random.normal(
            loc=exogenous_noise_config.feature_vector_noise_mean,
            scale=exogenous_noise_config.feature_vector_noise_std,
            size=exogenous_noise_config.feature_vector_noise_dim,
        ).astype(np.float32)
    elif (
        exogenous_noise_config.noise_type == ExogenousNoiseType.RANDOM_WALK
        and last_noise is not None
    ):
        noise = np_random.normal(
            loc=exogenous_noise_config.feature_vector_random_walk_noise_mean,
            scale=exogenous_noise_config.feature_vector_random_walk_noise_std,
            size=exogenous_noise_config.feature_vector_noise_dim,
        ).astype(np.float32)
        assert (
            last_noise.shape == noise.shape
        ), f"last_noise shape {last_noise.shape} does not match noise shape {noise.shape}"
        noise += last_noise
    return np.concatenate([feature_vector, noise], axis=0), noise


def add_exogenous_noise_to_image(
    image_obs: np.ndarray,
    exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig,
    np_random: np.random.Generator,
    last_noise: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace background of image observation with exogenous noise.

    :param image_obs (np.ndarray): The image observation to add noise to.
    :param exogenous_noise_config (ToyEnvironmentExogenousNoiseConfig): The configuration for the exogenous noise.
    :param np_random (np.random.Generator): The random number generator to use for adding noise.
    :param last_noise (np.ndarray | None): The last noise that has been added to the feature vector.
        (None if no noise has been added yet).
    :return (Tuple[np.ndarray, np.ndarray]): The feature vector with added noise and the noise that was added.
    """
    # identify background pixels
    background_mask = np.all(image_obs == BACKGROUND_COLOR, axis=-1)[:, :, None].repeat(
        3, axis=-1
    )

    if exogenous_noise_config.noise_type == ExogenousNoiseType.IID or (
        exogenous_noise_config.noise_type == ExogenousNoiseType.RANDOM_WALK
        and last_noise is None
    ):
        noise = np_random.integers(
            exogenous_noise_config.image_noise_min,
            exogenous_noise_config.image_noise_max,
            image_obs.shape,
        )
    elif (
        exogenous_noise_config.noise_type == ExogenousNoiseType.RANDOM_WALK
        and last_noise is not None
    ):
        noise = np_random.integers(
            exogenous_noise_config.image_random_walk_noise_min,
            exogenous_noise_config.image_random_walk_noise_max,
            image_obs.shape,
        )
        noise += last_noise
    else:
        raise ValueError(
            f"Unknown noise type: {exogenous_noise_config.noise_type}. Supported types are: "
            f"{', '.join(ExogenousNoiseType.get_valid_names())}"
        )

    noise = np.clip(noise, 0, 255).astype(np.uint8)
    image_obs[background_mask] = noise[background_mask]
    return image_obs, noise


def add_exogenous_noise_to_observation(
    observation: np.ndarray,
    observation_type: ObservationType,
    exogenous_noise_config: ToyEnvironmentExogenousNoiseConfig,
    np_random: np.random.Generator,
    last_noise: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add exogenous noise to the observation.

    :param observation (np.ndarray): The observation to add noise to.
    :param observation_type (ObservationType): The type of observation (feature vector or image).
    :param exogenous_noise_config (ToyEnvironmentExogenousNoiseConfig): The configuration for the exogenous noise.
    :param np_random (np.random.Generator): The random number generator to use for adding noise.
    :param last_noise (np.ndarray | None): The last noise that has been added to the feature vector.
        (None if no noise has been added yet).
    :return (Tuple[np.ndarray, np.ndarray]): The feature vector with added noise and the noise that was added.
    """
    if observation_type == ObservationType.FEATURE_STATE:
        return add_exogenous_noise_to_feature_vector(
            observation,
            exogenous_noise_config,
            np_random,
            last_noise,
        )
    elif observation_type == ObservationType.IMAGE_STATE:
        return add_exogenous_noise_to_image(
            observation,
            exogenous_noise_config,
            np_random,
            last_noise,
        )
    raise ValueError(
        f"Unknown observation type: {observation_type}. "
        f"Supported types are: {', '.join(ObservationType.get_valid_names())}"
    )
