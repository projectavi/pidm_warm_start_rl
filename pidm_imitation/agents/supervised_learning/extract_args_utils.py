# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, Tuple

from pidm_imitation.agents.supervised_learning.config.subconfig import ModelConfig
from pidm_imitation.agents.supervised_learning.dataset.datamodule import DataModule
from pidm_imitation.agents.supervised_learning.dataset.slicer import compute_slices
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import POLICY_HEAD_KEY, STATE_ENCODER_MODEL_KEY
from pidm_imitation.utils.state_types import StateType
from pidm_imitation.utils.valid_controller_actions import ValidControllerActions


class ExtractArgsFromConfig:
    @staticmethod
    def get_action_dim(config: OfflinePLConfigFile) -> int:
        return ValidControllerActions.get_actions_dim(config.action_config.type)

    @staticmethod
    def get_sequence_length(config: OfflinePLConfigFile) -> int:
        history = config.data_config.history
        history_slice_specs = config.data_config.history_slice_specs
        history_slice = compute_slices(history, history_slice_specs)
        return len(history_slice) + 1  # +1 for the current state

    @staticmethod
    def get_num_lookahead_samples(config: OfflinePLConfigFile) -> int:
        lookahead = config.data_config.lookahead
        if lookahead:
            lookahead_slice_specs = config.data_config.lookahead_slice_specs
            lookahead_slice = compute_slices(lookahead, lookahead_slice_specs)
            return len(lookahead_slice)
        return 0

    @staticmethod
    def get_model_config(config: OfflinePLConfigFile) -> ModelConfig:
        return config.model_config

    @staticmethod
    def get_model_init_args(config: OfflinePLConfigFile) -> Dict[str, Any]:
        return config.model_config.init_args

    @staticmethod
    def get_algorithm(config: OfflinePLConfigFile) -> str:
        return config.model_config.algorithm

    @staticmethod
    def get_input_format(config: OfflinePLConfigFile) -> str:
        return config.model_config.input_format

    @staticmethod
    def get_action_type(config: OfflinePLConfigFile) -> str:
        return config.action_config.type

    @staticmethod
    def get_state_type(config: OfflinePLConfigFile) -> StateType:
        return config.state_config.type

    @staticmethod
    def get_optimizer_kwargs(config: OfflinePLConfigFile) -> Dict[str, Any]:
        optimizer = config.pl_config.optimizer
        optimizer_kwargs = config.pl_config.optimizer_kwargs
        return {"optimizer_name": optimizer, "optimizer_kwargs": optimizer_kwargs}

    @staticmethod
    def get_scheduler_kwargs(config: OfflinePLConfigFile) -> Dict[str, Any]:
        scheduler = config.pl_config.scheduler
        scheduler_kwargs = config.pl_config.scheduler_kwargs
        if (
            "max_steps" in config.pl_config.trainer
            and "num_training_steps" not in scheduler_kwargs
        ):
            # use maximum training steps for learning rate scheduling
            scheduler_kwargs["num_training_steps"] = config.pl_config.trainer[
                "max_steps"
            ]
        return {"scheduler": scheduler, "scheduler_kwargs": scheduler_kwargs}

    @staticmethod
    def _get_policy_head_config(config: OfflinePLConfigFile) -> Dict[str, Any]:
        model_config = ExtractArgsFromConfig.get_model_config(config)
        return model_config.submodel_configs[POLICY_HEAD_KEY]

    @staticmethod
    def get_policy_head_name(config: OfflinePLConfigFile) -> str:
        policy_submodel_config = ExtractArgsFromConfig._get_policy_head_config(config)
        return policy_submodel_config["class_name"]

    @staticmethod
    def _get_policy_head_init_args(config: OfflinePLConfigFile) -> Dict[str, Any]:
        policy_submodel_config = ExtractArgsFromConfig._get_policy_head_config(config)
        return policy_submodel_config["init_args"]

    @staticmethod
    def _get_policy_head_policy_model_kwargs(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        policy_submodel_kwargs = ExtractArgsFromConfig._get_policy_head_init_args(
            config
        )
        return policy_submodel_kwargs["policy_model"]

    @staticmethod
    def get_policy_head_policy_model_name(config: OfflinePLConfigFile) -> str:
        policy_model_args = ExtractArgsFromConfig._get_policy_head_policy_model_kwargs(
            config
        )
        return policy_model_args["class_name"]

    @staticmethod
    def get_policy_head_policy_model_init_args(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        policy_model_args = ExtractArgsFromConfig._get_policy_head_policy_model_kwargs(
            config
        )
        return policy_model_args["init_args"]

    @staticmethod
    def get_policy_head_init_args_without_policy_model(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        # copy the init args before popping the policy_model to avoid modifying the original config
        policy_submodel_kwargs = dict(
            ExtractArgsFromConfig._get_policy_head_init_args(config)
        )
        policy_submodel_kwargs.pop("policy_model")
        return policy_submodel_kwargs

    @staticmethod
    def has_state_encoder_submodel(config: OfflinePLConfigFile) -> bool:
        model_config = ExtractArgsFromConfig.get_model_config(config)
        return STATE_ENCODER_MODEL_KEY in model_config.submodel_configs

    @staticmethod
    def _get_state_encoder_config(config: OfflinePLConfigFile) -> Dict[str, Any]:
        model_config = ExtractArgsFromConfig.get_model_config(config)
        if STATE_ENCODER_MODEL_KEY in model_config.submodel_configs:
            return model_config.submodel_configs[STATE_ENCODER_MODEL_KEY]
        return {}

    @staticmethod
    def get_state_encoder_name(config: OfflinePLConfigFile) -> str:
        state_encoder_submodel_config = ExtractArgsFromConfig._get_state_encoder_config(
            config
        )
        if state_encoder_submodel_config:
            return state_encoder_submodel_config["class_name"]
        return ""

    @staticmethod
    def _get_state_encoder_init_args(config: OfflinePLConfigFile) -> Dict[str, Any]:
        state_encoder_submodel_config = ExtractArgsFromConfig._get_state_encoder_config(
            config
        )
        if state_encoder_submodel_config:
            return state_encoder_submodel_config["init_args"]
        return {}

    @staticmethod
    def _get_state_encoder_encoder_model_kwargs(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        state_submodel_kwargs = ExtractArgsFromConfig._get_state_encoder_init_args(
            config
        )
        return state_submodel_kwargs["encoder_model"]

    @staticmethod
    def get_state_encoder_encoder_model_name(config: OfflinePLConfigFile) -> str:
        state_encoder_model_args = (
            ExtractArgsFromConfig._get_state_encoder_encoder_model_kwargs(config)
        )
        return state_encoder_model_args["class_name"]

    @staticmethod
    def get_state_encoder_encoder_model_init_args(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        state_encoder_model_args = (
            ExtractArgsFromConfig._get_state_encoder_encoder_model_kwargs(config)
        )
        return state_encoder_model_args["init_args"]

    @staticmethod
    def get_state_encoder_init_args_without_encoder_model(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        # copy the init args before popping the encoder_model to avoid modifying the original config
        state_encoder_submodel_kwargs = dict(
            ExtractArgsFromConfig._get_state_encoder_init_args(config)
        )
        state_encoder_submodel_kwargs.pop("encoder_model")
        return state_encoder_submodel_kwargs

    @staticmethod
    def get_info_from_submodel_config(
        submodel_config: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract information from the submodel configuration. Expects the config to contain:
        - 'class_name': The class name of the submodel.
        - 'init_args': A dictionary of keyword arguments to initialise the submodel.
        :return: A tuple containing the submodel name, and kwargs.
        """
        submodel_name = submodel_config["class_name"]
        submodel_kwargs = submodel_config["init_args"]
        return submodel_name, submodel_kwargs

    @staticmethod
    def get_submodel_names_from_model_config(config: ModelConfig) -> List[str]:
        """
        Get the name of all submodels defined in the model configuration.
        :param config: The model configuration containing submodel definitions.
        :return: A list of submodel names.
        """
        return list(config.submodel_configs.keys())


class ExtractArgsFromDataModule:
    @staticmethod
    def get_state_dim(datamodule: DataModule):
        return datamodule.state_dim

    @staticmethod
    def get_action_dim(datamodule: DataModule):
        return datamodule.action_dim

    @staticmethod
    def get_sequence_length(datamodule: DataModule):
        return datamodule.sequence_length

    @staticmethod
    def get_num_history_samples(datamodule: DataModule):
        return datamodule.num_history_samples

    @staticmethod
    def get_num_lookahead_samples(datamodule: DataModule):
        return datamodule.num_lookahead_samples

    @staticmethod
    def get_include_k(datamodule: DataModule):
        return datamodule.include_k
