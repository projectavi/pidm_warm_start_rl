# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

import torch


class ValidPadding:
    ZERO = "zero"
    REPEAT = "repeat"
    NO_PADDING = "no_padding"

    ALL = [ZERO, REPEAT, NO_PADDING]


def pad_tensor(tensor: torch.Tensor, pad_pre: int, pad_post: int, dim: int, mode: str) -> torch.Tensor:
    """
    Pad a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        pad_pre (int): The amount of padding to add before the tensor along the specified dimension.
        pad_post (int): The amount of padding to add after the tensor along the specified dimension.
        dim (int): The dimension along which to pad the tensor.
        mode (str): The padding mode to use. One of ValidPadding.ALL.
            ValidPadding.ZERO: Pad with zeros.
            ValidPadding.REPEAT: Pad by repeating the edge values of the tensor.
            ValidPadding.NO_PADDING: Do not pad the tensor.

    Returns:
        torch.Tensor: The padded tensor.

    Raises:
        ValueError: If an invalid padding mode is provided.

    """
    if mode not in ValidPadding.ALL:
        raise ValueError(f"Invalid padding mode. Expected one of {ValidPadding.ALL}, got {mode}")

    if mode == ValidPadding.ZERO or mode == ValidPadding.REPEAT:
        n_steps = tensor.size(dim)
        pre_padding = tensor.index_select(dim, torch.tensor([0])) * padding_factor(mode)
        post_padding = tensor.index_select(dim, torch.tensor([n_steps - 1])) * padding_factor(mode)
        if pad_pre > 0:
            tensor = torch.cat([pre_padding] * pad_pre + [tensor], dim=dim)
        if pad_post > 0:
            tensor = torch.cat([tensor] + [post_padding] * pad_post, dim=dim)
        return tensor
    elif mode == ValidPadding.NO_PADDING:
        return tensor
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def pad_ticks(ticks: List[float], pad_pre: int, pad_post: int, mode: str) -> List[float]:
    """
    Pad ticks. Different from padding tensors,
     padding with zeros does not make sense for ticks.
     We always pad with the edge values.

    Args:
        lst (List[float]): The input list to be padded.
        pad_pre (int): The amount of padding to add before the list.
        pad_post (int): The amount of padding to add after the list.
        mode (str): The padding mode to use. One of ValidPadding.ALL.
            ValidPadding.ZERO: Pad with zeros.
            ValidPadding.REPEAT: Pad by repeating the edge values of the list.
            ValidPadding.NO_PADDING: Do not pad the list.

    Returns:
        List[float]: The padded list.

    Raises:
        ValueError: If an invalid padding mode is provided.

    """
    if mode not in ValidPadding.ALL:
        raise ValueError(f"Invalid padding mode. Expected one of {ValidPadding.ALL}, got {mode}")

    if mode == ValidPadding.ZERO or mode == ValidPadding.REPEAT:
        return [ticks[0]] * pad_pre + ticks + [ticks[-1]] * pad_post
    elif mode == ValidPadding.NO_PADDING:
        return ticks
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def padding_factor(padding_strategy: str) -> float:
    """
    Determine the padding factor based on the padding strategy used for filing the
     observation queue for the agent during eval.
    """

    if padding_strategy == ValidPadding.ZERO:
        return 0.0
    elif padding_strategy == ValidPadding.REPEAT:
        return 1.0
    elif padding_strategy == ValidPadding.NO_PADDING:
        # To keep backward compatibility, the factor is set to 1.0
        # when no padding is applied during traning.
        return 1.0
    else:
        raise ValueError(f"Unsupported padding mode: {padding_strategy}")
