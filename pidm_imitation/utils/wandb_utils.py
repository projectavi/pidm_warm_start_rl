# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from typing import List

import wandb

from pidm_imitation.utils.git_utils import (
    get_branch_name,
    get_head_commitid,
    is_git_repo,
)

this_dir = os.path.abspath(os.path.dirname(__file__))


def initialize_wandb(
    group: str | None,
    project: str | None,
    tags: List[str] | None,
    notes: str | None,
    config_dict: dict,
    experiment_name: str,
) -> None:

    wandb.init(
        project=project,
        name=experiment_name,
        group=group,
        tags=tags,
        notes=notes,
        config=add_git_info(config_dict),
    )


def add_git_info(config_dict: dict) -> dict:
    config_dict = config_dict.copy()
    config_dict["command"] = sys.argv[1:]
    location = os.path.dirname(sys.argv[0])
    if is_git_repo(cwd=location):
        config_dict["git"] = {
            "branch": get_branch_name(cwd=location),
            "commit_id": get_head_commitid(cwd=location),
        }
    return config_dict
