# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, Tuple

from pidm_imitation.agents.supervised_learning.extract_args_utils import (
    ExtractArgsFromConfig,
)
from pidm_imitation.agents.supervised_learning.utils.valid_input_formats import (
    ValidInputFormats,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import (
    ACTION_HISTORY_KEY,
    ACTION_LOOKAHEAD_KEY,
    HISTORY_EMB_KEY,
    HISTORY_KEY,
    LOOKAHEAD_EMB_KEY,
    LOOKAHEAD_K_ONEHOT_KEY,
    LOOKAHEAD_KEY,
    POLICY_HEAD_KEY,
    STATE_ENCODER_MODEL_KEY,
    STATE_HISTORY_KEY,
    STATE_LOOKAHEAD_KEY,
)
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.valid_models import ValidModels

logger = Logger()
log = logger.get_root_logger()


class InputsFactory:
    """
    Utility class to extract inputs for submodels from the config.
    """

    @staticmethod
    def _get_bc_inputs(config: OfflinePLConfigFile) -> Dict[str, Tuple[str, Any]]:
        input_format = ExtractArgsFromConfig.get_input_format(config)
        model_config = ExtractArgsFromConfig.get_model_config(config)
        submodel_names = ExtractArgsFromConfig.get_submodel_names_from_model_config(
            model_config
        )

        if STATE_ENCODER_MODEL_KEY in submodel_names:
            if input_format in [
                ValidInputFormats.STATE_ONLY,
                ValidInputFormats.STATE_AND_ACTION_HISTORY,
            ]:
                state_encoder_input_groups = {HISTORY_KEY: [STATE_HISTORY_KEY]}
            elif input_format == ValidInputFormats.STATE_AND_ACTION:
                state_encoder_input_groups = {
                    HISTORY_KEY: [STATE_HISTORY_KEY, ACTION_HISTORY_KEY]
                }
            else:
                raise ValueError(
                    f"Invalid input format '{input_format}' for BC model. Expected one of {ValidInputFormats.ALL}."
                )
            policy_head_inputs = [HISTORY_EMB_KEY]
            if input_format == ValidInputFormats.STATE_AND_ACTION_HISTORY:
                policy_head_inputs.append(ACTION_HISTORY_KEY)
            return {
                STATE_ENCODER_MODEL_KEY: ("input_groups", state_encoder_input_groups),
                POLICY_HEAD_KEY: ("input_keys", policy_head_inputs),
            }
        else:
            if input_format == ValidInputFormats.STATE_ONLY:
                return {POLICY_HEAD_KEY: ("input_keys", [STATE_HISTORY_KEY])}
            elif input_format in [
                ValidInputFormats.STATE_AND_ACTION,
                ValidInputFormats.STATE_AND_ACTION_HISTORY,
            ]:
                return {
                    POLICY_HEAD_KEY: (
                        "input_keys",
                        [STATE_HISTORY_KEY, ACTION_HISTORY_KEY],
                    )
                }
            raise ValueError(
                f"Invalid input format '{input_format}' for BC model. Expected one of {ValidInputFormats.ALL}."
            )

    @staticmethod
    def _get_idm_inputs(config: OfflinePLConfigFile) -> Dict[str, Tuple[str, Any]]:
        input_format = ExtractArgsFromConfig.get_input_format(config)
        model_config = ExtractArgsFromConfig.get_model_config(config)
        submodel_names = ExtractArgsFromConfig.get_submodel_names_from_model_config(
            model_config
        )

        if STATE_ENCODER_MODEL_KEY in submodel_names:
            if input_format in [
                ValidInputFormats.STATE_ONLY,
                ValidInputFormats.STATE_AND_ACTION_HISTORY,
            ]:
                state_encoder_input_groups = {
                    HISTORY_KEY: [STATE_HISTORY_KEY],
                    LOOKAHEAD_KEY: [STATE_LOOKAHEAD_KEY],
                }
            elif input_format == ValidInputFormats.STATE_AND_ACTION:
                state_encoder_input_groups = {
                    HISTORY_KEY: [STATE_HISTORY_KEY, ACTION_HISTORY_KEY],
                    LOOKAHEAD_KEY: [STATE_LOOKAHEAD_KEY, ACTION_LOOKAHEAD_KEY],
                }
            else:
                raise ValueError(
                    f"Invalid input format '{input_format}' for IDM model. Expected one of {ValidInputFormats.ALL}."
                )
            policy_head_inputs = [
                HISTORY_EMB_KEY,
                LOOKAHEAD_EMB_KEY,
                LOOKAHEAD_K_ONEHOT_KEY,
            ]
            if input_format == ValidInputFormats.STATE_AND_ACTION_HISTORY:
                policy_head_inputs.append(ACTION_HISTORY_KEY)
            return {
                STATE_ENCODER_MODEL_KEY: ("input_groups", state_encoder_input_groups),
                POLICY_HEAD_KEY: ("input_keys", policy_head_inputs),
            }
        else:
            if input_format == ValidInputFormats.STATE_ONLY:
                return {
                    POLICY_HEAD_KEY: (
                        "input_keys",
                        [
                            STATE_HISTORY_KEY,
                            STATE_LOOKAHEAD_KEY,
                            LOOKAHEAD_K_ONEHOT_KEY,
                        ],
                    ),
                }
            elif input_format == ValidInputFormats.STATE_AND_ACTION_HISTORY:
                return {
                    POLICY_HEAD_KEY: (
                        "input_keys",
                        [
                            STATE_HISTORY_KEY,
                            ACTION_HISTORY_KEY,
                            STATE_LOOKAHEAD_KEY,
                            LOOKAHEAD_K_ONEHOT_KEY,
                        ],
                    ),
                }
            elif input_format == ValidInputFormats.STATE_AND_ACTION:
                return {
                    POLICY_HEAD_KEY: (
                        "input_keys",
                        [
                            STATE_HISTORY_KEY,
                            ACTION_HISTORY_KEY,
                            STATE_LOOKAHEAD_KEY,
                            ACTION_LOOKAHEAD_KEY,
                            LOOKAHEAD_K_ONEHOT_KEY,
                        ],
                    ),
                }
            raise ValueError(
                f"Invalid input format '{input_format}' for IDM model. Expected one of {ValidInputFormats.ALL}."
            )

    @staticmethod
    def get_inputs(config: OfflinePLConfigFile) -> Dict[str, Tuple[str, Any]]:
        """
        Get the input groups for the state encoder from the config.
        :param config: The OfflinePLConfigFile containing the state encoder configuration.
        :return: A dictionary mapping input group names to lists of input keys.
        """
        algorithm_name = ExtractArgsFromConfig.get_algorithm(config)
        input_format = ExtractArgsFromConfig.get_input_format(config)
        assert ValidInputFormats.is_valid_input_format(
            input_format
        ), f"Invalid input format '{input_format}', expected one of {ValidInputFormats.ALL}."

        if algorithm_name == ValidModels.BC:
            inputs = InputsFactory._get_bc_inputs(config)
        elif algorithm_name == ValidModels.IDM:
            inputs = InputsFactory._get_idm_inputs(config)
        else:
            raise ValueError(
                f"Invalid algorithm '{algorithm_name}' in config. Expected one of {ValidModels.ALL}."
            )

        log.info(
            f"Extracted inputs for algorithm '{algorithm_name}' with input format '{input_format}':"
        )
        for submodel, (key_type, keys) in inputs.items():
            log.info(f"\t{submodel}: {key_type} = {keys}")

        return inputs
