# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
import math
from typing import Any, List

VALID_SLICE_CLASSES = ["LinearSlicer", "GeometricSlicer", "FixedSlicer"]


class Slicer:
    """
    Interface for slicers. Child classes have to define the self._slice property.
    """

    def __init__(self, *args, **kwargs):
        self._slice = None

    @property
    def slice(self) -> list[int] | None:
        if self._slice is None:
            raise NotImplementedError("Slicer classes must define property self._slice property.")
        return self._slice


class LinearSlicer(Slicer):
    def __init__(self, num_data_samples: int, num_samples: int):
        """
        Class that returns slicing indices that are linearly spaced. For example, if num_data_samples=32,
        num_samples=4, the indices will be [7, 15, 23, 31]. The numbers will be spread such that it gets
        as close to `num_data_samples - 1` as possible so it utilizes the full range of the data.

        Args:
            num_data_samples (int): The total number of data samples.
            num_samples (int): The desired number of samples to include in the slice indices.

        Raises:
            AssertionError: If `num_data_samples` is less than `num_samples`.
        """
        if num_data_samples == 0:
            self._slice = []
        else:
            assert num_data_samples >= num_samples, f"Num samples ({num_data_samples}) < target ({num_samples})."
            spacing = num_data_samples / num_samples
            self._slice = [math.floor(i * spacing) - 1 for i in range(1, num_samples + 1)]


class GeometricSlicer(Slicer):
    def __init__(self, num_data_samples: int, num_samples: int, ratio: int):
        """
        Class that returns slicing indices that are geometrically spaced. For example, if num_data_samples=32,
        num_samples=4, and ratio=2, the indices will be [1, 2, 4, 8].
        It is possible to specify a large ratio such that it is not possible to get the desired
        num_samples from the num_data_samples, in which case an AssertionError is raised.

        Args:
            num_data_samples (int): The total number of data samples.
            num_samples (int): The desired maximum number of samples to include in the slices.
            ratio (int): The geometric ratio (exponent) used to compute the indices of the slice.
            direction (int): The positive or negative sign of the slicing indices.

        Raises:
            AssertionError: If `num_data_samples` is less than `num_samples`.
            AssertionError: If `ratio` is less than or equal to 1.
        """
        if num_data_samples == 0:
            self._slice = []
        else:
            assert num_data_samples >= num_samples, f"Num samples ({num_data_samples}) < target ({num_samples})."
            assert ratio > 1, f"Ratio ({ratio}) must be greater than 1."
            self._slice = self._geometric_range(start=1, end=num_data_samples, target=num_samples, ratio=ratio)

    @staticmethod
    def _geometric_range(start, end, target, ratio) -> List[int]:
        slice: List[int] = []
        current = start
        while current < end and len(slice) < target:
            slice.append(current)
            current *= ratio
        assert len(slice) == target, (
            f"Length of slice {slice} ({len(slice)}) smaller than expected ({target}). Try changing the ratio "
            + "or the number of samples so more indices can be included to meet your target num_samples."
        )
        return slice


class FixedSlicer(Slicer):
    def __init__(self, slices: List[int]):
        """
        Class that is given the exact slices to return. It is up to the caller to ensure the given indices
        fit within the specified number of samples.

        Args:
            slices (int): The indices of the frames to return.

        """
        self._slice = slices


def compute_slices(num_samples: int, slice_spec_dict: dict[str, Any]) -> List[int]:
    slice_specs = deepcopy(slice_spec_dict)  # Avoid modifying the original dict in case used multiple times
    class_name = slice_specs.pop("class_name")
    slices: List[int] = []
    if class_name == "LinearSlicer":
        slices = LinearSlicer(num_samples, **slice_specs).slice
    elif class_name == "GeometricSlicer":
        slices = GeometricSlicer(num_samples, **slice_specs).slice
    elif class_name == "FixedSlicer":
        slices = FixedSlicer(**slice_specs).slice
    else:
        raise NotImplementedError(f"Invalid slice type {class_name}, expecting one of {VALID_SLICE_CLASSES}.")

    return slices


def check_slices(name: str, num_samples: int, slice: list[int]) -> None:
    if num_samples > 0:
        assert len(slice) > 0, f"{name} slices must be non-empty"
        positive_indices = all([s >= 0 for s in slice])
        assert positive_indices, f"{name} slices must be positive"
        assert num_samples > slice[-1], f"{num_samples} is less than last slice {slice[-1]}"
        assert all(
            [slice[i] > slice[i - 1] for i in range(1, len(slice))]
        ), f"{name} slices must be in increasing order"
