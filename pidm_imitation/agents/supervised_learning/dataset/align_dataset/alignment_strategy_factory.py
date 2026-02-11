# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from pidm_imitation.agents.supervised_learning.dataset.align_dataset import (
    ActionFrameAlignmentStrategy,
)
from pidm_imitation.agents.supervised_learning.dataset.align_dataset.valid_alignment_strategies import (
    ValidAlignmentStrategies,
)


class AlignmentStrategyFactory:
    """Factory class to create alignment strategy instances"""

    @staticmethod
    def _get_alignment_strategy_kwargs(
        alignment_strategy: str,
        action_type: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Returns the keyword arguments for the dataset."""
        kwargs = {
            "action_type": action_type,
        }
        if alignment_strategy == ValidAlignmentStrategies.CAUSAL_ACTION_FRAME_ALIGNMENT:
            kwargs.update(
                {
                    "max_n_frames": kwargs.get("max_n_frames", -1),
                }
            )
            return kwargs
        raise ValueError(
            f"Unknown alignment strategy: {alignment_strategy}, valid strategies are {ValidAlignmentStrategies.ALL}"
        )

    @staticmethod
    def get_alignment_strategy(
        alignment_strategy: str,
        action_type: str,
        **kwargs: Any,
    ) -> ActionFrameAlignmentStrategy:
        """Creates and returns an alignment strategy based on the provided parameters."""
        alignment_class = ValidAlignmentStrategies.get_alignment_strategy_class(
            alignment_strategy
        )
        dataset_kwargs = AlignmentStrategyFactory._get_alignment_strategy_kwargs(
            alignment_strategy, action_type, **kwargs
        )
        return alignment_class(**dataset_kwargs)
