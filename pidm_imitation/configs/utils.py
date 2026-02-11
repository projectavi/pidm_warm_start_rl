# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import MODEL_DIRECTORY
from pidm_imitation.utils.valid_controller_actions import ValidControllerActions


def sanity_check_config(config: OfflinePLConfigFile):
    # check action type is supported
    assert ValidControllerActions.is_valid_action_type(config.action_config.type), (
        f"ERROR: Unsupported action type {config.action_config.type} for the offline PyTorch Lightning models. "
        + f"The supported actions are: {ValidControllerActions.get_valid_action_types()}."
    )


def get_checkpoint_dir(config: OfflinePLConfigFile) -> str:
    """
    Returns the directory where checkpoints will be stored.
    """
    checkpoints_dir = (
        config.callbacks_config.get_model_checkpoint_dir()
        if config.callbacks_config
        else None
    )
    if checkpoints_dir is None:
        checkpoints_dir = MODEL_DIRECTORY
    return checkpoints_dir
