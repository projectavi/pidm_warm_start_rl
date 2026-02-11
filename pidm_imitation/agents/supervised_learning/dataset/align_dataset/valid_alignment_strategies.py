# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.agents.supervised_learning.dataset.align_dataset.causal_alignment_strategy import (
    CausalActionFrameAlignmentStrategy,
)


class ValidAlignmentStrategies:
    """
    This class defines the valid alignment strategies for aligning datasets.
    """

    CAUSAL_ACTION_FRAME_ALIGNMENT = "causal_action_frame_alignment"
    ALL = [CAUSAL_ACTION_FRAME_ALIGNMENT]

    @staticmethod
    def is_valid_alignment_strategy(strategy: str) -> bool:
        """
        Check if the provided strategy is a valid alignment strategy.
        """
        return strategy in ValidAlignmentStrategies.ALL

    @staticmethod
    def get_alignment_strategy_class(strategy: str) -> type:
        """
        Get the class corresponding to the provided alignment strategy.
        """
        if strategy == ValidAlignmentStrategies.CAUSAL_ACTION_FRAME_ALIGNMENT:
            return CausalActionFrameAlignmentStrategy
        raise ValueError(
            f"Invalid alignment strategy: {strategy}, must be one of {ValidAlignmentStrategies.ALL}"
        )
