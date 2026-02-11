# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

import torch

SUPPORTED_OPTIMIZERS = ["adam", "adamw"]


def get_optimizer(name: str, parameters: Any, **kwargs) -> torch.optim.Optimizer:
    """
    Returns an optimizer instance based on the name and keyword arguments.
    :param name: Name of the optimizer (e.g., 'adam', 'adamw').
    :param parameters: Parameters to optimize, typically model parameters.
    :param kwargs: Additional keyword arguments for the optimizer.
    :return: An instance of the specified optimizer.
    """
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    elif name.lower() == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)

    # try to import the optimizer from torch.optim
    fn = getattr(torch.optim, name, None)
    if fn is not None:
        return fn(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}. This supports {SUPPORTED_OPTIMIZERS} or any from torch.optim.")
