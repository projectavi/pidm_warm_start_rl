# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from pidm_imitation.constants import (
    ACTION_HISTORY_KEY,
    ACTION_LOOKAHEAD_KEY,
    HISTORY_EMB_KEY,
    LOOKAHEAD_EMB_KEY,
    LOOKAHEAD_K_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    STATE_HISTORY_KEY,
    STATE_LOOKAHEAD_KEY,
)
from pidm_imitation.utils import Logger

log = Logger().get_root_logger()


# input keys that represent sequences
SEQUENCE_KEYS = set(
    [
        STATE_HISTORY_KEY,
        ACTION_HISTORY_KEY,
        STATE_LOOKAHEAD_KEY,
        ACTION_LOOKAHEAD_KEY,
    ]
)

SEQUENCE_EMB_KEYS = set([HISTORY_EMB_KEY, LOOKAHEAD_EMB_KEY])


def is_sequence_key(key: str, state_encoder_collapse_sequence: bool = False) -> bool:
    return key in SEQUENCE_KEYS or (
        not state_encoder_collapse_sequence and key in SEQUENCE_EMB_KEYS
    )


def get_total_dim(
    keys: List[str],
    state_dim: int,
    action_dim: int,
    sequence_length: int,
    num_lookaheads: int,
    state_encoder_dim: int | None = None,
    state_encoder_collapse_sequence: bool = False,
    collapse_sequence: bool = False,
):
    """
    Get the total dimension based on a list of input keys and other parameters.
    :param keys: List of input keys to determine the dimensions.
    :param state_dim: Dimension of the state input.
    :param action_dim: Dimension of the action input.
    :param sequence_length: Length of the sequence for the input.
    :param num_lookaheads: Number of lookaheads for one-hot encoding.
    :param state_encoder_dim: dimension of the state encoder output or None if not used.
    :param state_encoder_collapse_sequence: If True, the state encoder returns only one embedding for the sequence.
    :param collapse_sequence: If True, the inputs are stacked along the sequence dimension.
    :return: The total input dimension as an integer.
    """
    total_dim = 0
    seq_multiplier = sequence_length if collapse_sequence else 1
    for k in keys:
        if k in (STATE_HISTORY_KEY, STATE_LOOKAHEAD_KEY):
            k_dim = state_dim
        elif k in (ACTION_HISTORY_KEY, ACTION_LOOKAHEAD_KEY):
            k_dim = action_dim
        elif k in (LOOKAHEAD_K_ONEHOT_KEY):
            k_dim = num_lookaheads
        elif k == LOOKAHEAD_K_KEY:
            k_dim = 1
        elif k in (
            HISTORY_EMB_KEY,
            LOOKAHEAD_EMB_KEY,
        ):
            assert (
                state_encoder_dim is not None
            ), f"Input key '{k}' requires state_encoder_dim to be set, but it is None."
            k_dim = state_encoder_dim
        else:
            raise ValueError(
                f"Need to implement dimension calculation for input key '{k}'."
            )

        if is_sequence_key(k, state_encoder_collapse_sequence):
            total_dim += k_dim * seq_multiplier
        else:
            total_dim += k_dim

    return total_dim
