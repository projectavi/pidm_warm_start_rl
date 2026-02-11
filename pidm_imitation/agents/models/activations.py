# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Any

import torch.nn as nn

SUPPORTED_ACTIVATIONS = [
    "linear",
    "relu",
    "tanh",
    "elu",
    "swish",
    "silu",
    "leaky_relu",
    "selu",
    "gelu",
]


def get_activation_fn(name: str | None, **kwargs) -> Any:
    """Given the name of the activation,
    returns the corresponding from torch actionation function.
    kwargs can be used to pass additional parameters
    to the activation function.
    """

    # Check if the name is in torch.nn
    if hasattr(nn, name):
        fn = getattr(nn, name, None)
    elif name in ["linear", None]:
        fn = nn.Identity
    elif name in ["swish", "silu"]:
        fn = nn.SiLU
    elif name == "relu":
        fn = nn.ReLU
    elif name == "tanh":
        fn = nn.Tanh
    elif name == "elu":
        fn = nn.ELU
    elif name == "leaky_relu":
        fn = nn.LeakyReLU
    elif name == "selu":
        fn = nn.SELU
    elif name == "gelu":
        fn = nn.GELU
    else:
        raise ValueError(f"Unknown activation {name}. Supports one of {SUPPORTED_ACTIVATIONS} or import from torch.nn.")

    if fn is not None:
        return partial(fn, **kwargs)
