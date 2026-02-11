# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import List

from pidm_imitation.agents.subconfig import IdmPlannerConfig
from pidm_imitation.utils import Logger

log = Logger.get_logger(__name__)


def get_idm_lookahead(
    lookahead_type: str,
    lookahead: int,
    train_lookahead_slice: List[int],
    log_output: bool = False,
) -> int:
    """
    Determine the lookahead k to use for evaluation. Ensure that any lookahead is included in the lookaheads
    used during training.

    :param lookahead_type: The type of lookahead to use. Must be in IdmPlannerConfig.VALID_LOOKAHEAD_TYPES.
    :param lookahead: The lookahead to use for evaluation if lookahead_type is "fixed".
    :param datamodule_lookahead_slice: The lookahead slice used during training.
    :param log_output: Whether to log the output.
    :return: lookahead k.
    """
    if lookahead_type == "fixed":
        # take fixed specified lookahead --> ensure it is in the training lookahead slice
        assert (
            lookahead in train_lookahead_slice
        ), f"Invalid fixed lookahead value, not found in training lookahead slice ({train_lookahead_slice})"
        if log_output:
            log.info(f"Using fixed lookahead {lookahead} for evaluation")
        return lookahead
    else:
        raise ValueError(
            f"Invalid lookahead type {lookahead_type}, must be in {IdmPlannerConfig.VALID_LOOKAHEAD_TYPES}"
        )


def get_idm_lookahead_index(
    lookahead_k: int,
    train_lookahead_slice: List[int],
):
    """Get the index of the lookahead k in the training lookahead slice.

    :param lookahead_k: The lookahead k to use.
    :param train_lookahead_slice: The training lookahead slice.
    :return: The index of the lookahead k among the training slices
    """
    assert (
        lookahead_k in train_lookahead_slice
    ), f"Invalid fixed lookahead value, not found in training lookahead slice ({train_lookahead_slice})"
    return train_lookahead_slice.index(lookahead_k)


def get_local_train_data_path(train_dir: str) -> str:
    """
    Get the local path to the training data, and verify that the directory exists locally.

    :param train_dir: The training data directory
    :return: Local path to the training data directory
    """
    # training data should already be present locally
    assert os.path.isdir(
        train_dir
    ), f"Training data directory {train_dir} does not exist."
    return train_dir
