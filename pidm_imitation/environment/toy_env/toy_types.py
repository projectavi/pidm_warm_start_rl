# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from enum import Enum
from typing import List


class TerminationCondition(Enum):
    NONE = "none"
    ANY_GOAL = "any_goal"
    ALL_GOALS = "all_goals"

    @staticmethod
    def from_str(term_cond: str) -> TerminationCondition:
        if term_cond.lower() == TerminationCondition.NONE.value:
            return TerminationCondition.NONE
        if term_cond.lower() == TerminationCondition.ANY_GOAL.value:
            return TerminationCondition.ANY_GOAL
        if term_cond.lower() == TerminationCondition.ALL_GOALS.value:
            return TerminationCondition.ALL_GOALS
        raise ValueError(
            f"Invalid termination condition {term_cond}, must be one of {TerminationCondition.get_valid_names()}"
        )

    @staticmethod
    def get_valid_names() -> List[str]:
        return [term_cond.value.lower() for term_cond in TerminationCondition]

    @staticmethod
    def get_valid_values() -> List[TerminationCondition]:
        return [term_cond for term_cond in TerminationCondition]


class ObservationType(Enum):
    FEATURE_STATE = "feature_state"  # feature vector of environment
    IMAGE_STATE = "image_state"  # image of full environment

    @staticmethod
    def from_str(obs_type: str) -> ObservationType:
        if obs_type.lower() == ObservationType.FEATURE_STATE.value:
            return ObservationType.FEATURE_STATE
        if obs_type.lower() == ObservationType.IMAGE_STATE.value:
            return ObservationType.IMAGE_STATE
        raise ValueError(f"Invalid observation type: {obs_type}, must be one of {ObservationType.get_valid_names()}")

    @staticmethod
    def get_valid_names() -> List[str]:
        return [obs_type.value.lower() for obs_type in ObservationType]

    @staticmethod
    def get_valid_values() -> List[ObservationType]:
        return [obs_type for obs_type in ObservationType]


class ExogenousNoiseType(Enum):
    IID = "iid"  # independently sample random noise values
    RANDOM_WALK = (
        "random_walk"  # sample initial random noise values and subsequently add sampled noise to the previous value
    )

    @staticmethod
    def from_str(noise_type: str) -> ExogenousNoiseType:
        if noise_type.lower() == ExogenousNoiseType.IID.value:
            return ExogenousNoiseType.IID
        if noise_type.lower() == ExogenousNoiseType.RANDOM_WALK.value:
            return ExogenousNoiseType.RANDOM_WALK
        raise ValueError(f"Invalid noise type: {noise_type}, must be one of {ExogenousNoiseType.get_valid_names()}")

    @staticmethod
    def get_valid_names() -> List[str]:
        return [noise_type.value.lower() for noise_type in ExogenousNoiseType]

    @staticmethod
    def get_valid_values() -> List[ExogenousNoiseType]:
        return [noise_type for noise_type in ExogenousNoiseType]
