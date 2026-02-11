# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import Tensor


class ValidDistances:
    L1 = "l1"
    L2 = "l2"
    COSINE = "cosine"
    ALL = [L1, L2, COSINE]

    @staticmethod
    def is_valid(distance_measure: str) -> bool:
        return distance_measure in ValidDistances.ALL


def l1_distance(states: Tensor, current_state: Tensor) -> Tensor:
    return torch.nn.functional.l1_loss(states, current_state.unsqueeze(0).expand_as(states), reduction="none").sum(
        dim=-1
    )


def l2_distance(states: Tensor, current_state: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(states, current_state.unsqueeze(0).expand_as(states), reduction="none").sum(
        dim=-1
    )


def cosine_distance(states: Tensor, current_state: Tensor) -> Tensor:
    return -torch.nn.functional.cosine_similarity(states, current_state, dim=-1)


class DistanceMetricFactory:
    @staticmethod
    def get_distance_function(distance_measure: str):
        assert ValidDistances.is_valid(distance_measure), f"Invalid distance metric: {distance_measure}"
        if distance_measure == ValidDistances.L1:
            return l1_distance
        elif distance_measure == ValidDistances.L2:
            return l2_distance
        elif distance_measure == ValidDistances.COSINE:
            return cosine_distance
        raise ValueError(f"Unknown distance metric: {distance_measure}")
