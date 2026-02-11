# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from torch import nn

SUPPORTED_NORM_LAYERS = [None, "batch_norm", "layer_norm", "batch_norm2d"]


class BNLayer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self._h_dim = h_dim
        self.model = nn.BatchNorm1d(h_dim)

    def forward(self, input):
        in_shape = input.shape
        assert (
            in_shape[-1] == self._h_dim
        ), f"ERROR: Expected the last input dim to be {self._h_dim}, but got {in_shape}."
        return self.model(input.reshape(-1, in_shape[-1])).reshape(in_shape)


def get_norm_layer(name: str | None, h_dim: int) -> Any:
    """Given the name of the normalization layer,
    returns the corresponding torch normalization layer.
    """
    assert name in SUPPORTED_NORM_LAYERS, f"Unsupported norm layer: {name}. Supported layers: {SUPPORTED_NORM_LAYERS}."

    if name is None:
        return nn.Identity()
    elif name == "batch_norm":
        return BNLayer(h_dim)
    elif name == "layer_norm":
        return nn.LayerNorm(h_dim)
    elif name == "batch_norm2d":
        return nn.BatchNorm2d(h_dim)
    raise ValueError(f"Unknown normalization layer {name}. Supported layers: {SUPPORTED_NORM_LAYERS}.")
