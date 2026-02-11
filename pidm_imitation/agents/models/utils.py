# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from torch import nn

from pidm_imitation.agents.models.norms import BNLayer
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


LAYERS_IN_EVAL_MODE_WHEN_FROZEN = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    BNLayer,
    nn.LayerNorm,
]


def get_output_dim(model: nn.Module) -> int:
    """
    Get the output dimension of a model.
    :param model: The model to get the output dimension from.
    :return: The output dimension as an integer.
    """
    if hasattr(model, "out_dim"):
        return model.out_dim  # type: ignore
    elif hasattr(model, "output_dim"):
        return model.output_dim  # type: ignore
    raise ValueError("Model does not have an output dimension defined.")


def get_collapse_sequence(model: nn.Module) -> bool:
    """
    Get whether the model collapses the sequence dimension.
    :param model: The model to get info from.
    :return: True if the model collapses the sequence dimension, False otherwise.
    """
    if hasattr(model, "collapse_sequence"):
        return model.collapse_sequence  # type: ignore
    log.warning(
        "Model does not have a collapse_sequence attribute defined. Defaulting to False."
    )
    return False


def is_recurrent(model: nn.Module) -> bool:
    """
    Check if the model is recurrent.
    :param model: The model to check.
    :return: True if the model is recurrent, False otherwise.
    """
    if hasattr(model, "is_recurrent"):
        return model.is_recurrent  # type: ignore
    log.warning(
        "Model does not have an is_recurrent attribute defined. Defaulting to False."
    )
    return False


def reset_model(model: nn.Module) -> Any:
    """
    Reset the model if it has a reset method.
    :param model: The model to reset.
    """
    if hasattr(model, "reset"):
        return model.reset()  # type: ignore
    log.warning("Model does not have a reset method defined. No action taken.")
