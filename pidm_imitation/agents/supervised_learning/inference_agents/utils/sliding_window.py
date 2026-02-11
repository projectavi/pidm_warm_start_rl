# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

import torch
from torch import Tensor, nn

from pidm_imitation.utils.padding_utils import ValidPadding


class SlidingWindowModule(nn.Module):
    """
    Maintains a fixed length (window_size) temporal buffer (oldest -> newest) of incoming tensors.

    On every forward call a single tensor `x` (no batch dimension) is appended. When the window is
    full the oldest element is dropped (sliding window). After inserting the new element the model
    returns the rows at indices given by `slices` (in ascending order) from the internal buffer.

    Notes / Conventions:
    - `slices` are absolute indices into the window (0 == oldest, window_size-1 == newest).
    - Before the window is fully populated, un-filled slots are zero (they're pre‑zeroed on first
      allocation). This keeps logic simple; downstream code may mask/ignore if desired.
    - The first time `forward` is called the buffer is allocated to match the shape & dtype of `x`.
    - A `reset()` method clears the buffer.
    """

    def __init__(self, window_size: int, slices: List[int], padding: str):
        super().__init__()
        assert window_size > 0, f"window_size must be > 0, got {window_size}"
        assert all(
            0 <= s < window_size for s in slices
        ), f"All slice indices must be in [0, {window_size-1}], got {slices}"
        self.window_size = window_size
        self.slices = slices
        self.padding = padding
        self._buffer: Tensor

        # Buffer is lazily created on first forward when we know the feature shape & device & dtype.
        self.register_buffer("_buffer", torch.empty(0), persistent=False)

    def reset(self) -> None:
        """Reset the sliding window (next forward re-initializes with duplicated first input)."""
        # Make buffer empty so next forward call re-initializes it.
        self._buffer = torch.empty(0)

    def _init_buffer(self, x: Tensor) -> None:
        """Initialize the buffer on first forward call by duplicating first input across window.

        This ensures the very first output has a fully-populated window (no zeros) which can be
        important for models expecting a complete history on the first step.
        """
        if self.padding == ValidPadding.REPEAT:
            self._buffer = x.unsqueeze(0).expand(self.window_size, -1).contiguous()
        else:  # ZERO and NO_PADDING both use zero padding during eval.
            buffer_shape = (self.window_size, *x.shape)
            self._buffer = torch.zeros(buffer_shape, dtype=x.dtype, device=x.device)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        Add one item to the buffer and return the desired slices from that buffer.

        :param x: Incoming tensor representing a single timestep (no batch dimension). If a batch
                  dimension is desired the caller should manage batching externally.
        :return: Tensor stacked from selected indices: shape (len(slices), *x.shape)
        """
        if torch.numel(self._buffer) == 0:
            self._init_buffer(x)

        # Window already initialized -> slide and insert new item
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=0)
        self._buffer[-1] = x

        return self._buffer[self.slices]
