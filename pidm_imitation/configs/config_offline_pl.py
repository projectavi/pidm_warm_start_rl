# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from schema import Optional, Schema

from pidm_imitation.agents.supervised_learning.config.subconfig import ModelConfig
from pidm_imitation.agents.supervised_learning.dataset.config.subconfig import (
    DataConfig,
)
from pidm_imitation.configs.subconfig import (
    CallbacksConfig,
    ControllerActionConfig,
    PytorchLightningConfig,
    StateConfig,
    WandbConfig,
)
from pidm_imitation.utils.config_base import ConfigFile


class OfflinePLConfigFile(ConfigFile):
    """
    Config parser class used for training offline PyTorch Lightning agents. This parser
    is agnostic to the environment being used, as during training
    we only consider a set of videos, regardless of where those were
    recorded. For evaluation, use the config file classes for offline PyTorch Lightning
    specific to the environment being used.
    """

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.experiment_name: str
        self.data_config: DataConfig
        self.model_config: ModelConfig
        self.state_config: StateConfig
        self.action_config: ControllerActionConfig
        self.pl_config: PytorchLightningConfig
        self.callbacks_config: CallbacksConfig
        self.wandb_config: WandbConfig | None
        self._set_attributes()

    @property
    def pl_parameters_dict(self) -> dict:
        return self.pl_config._config

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                self.EXPERIMENT_NAME: str,
                DataConfig.KEY: object,
                ModelConfig.KEY: object,
                StateConfig.KEY: object,
                ControllerActionConfig.KEY: object,
                PytorchLightningConfig.KEY: object,
                Optional(CallbacksConfig.KEY): object,
                Optional(WandbConfig.KEY): object,
            },
            ignore_extra_keys=True,
        )
        return schema

    def _create_sub_configs(self) -> None:
        self.experiment_name = self._config[self.EXPERIMENT_NAME]
        self.data_config = self._get_simple_config_obj(DataConfig)
        self.model_config = self._get_simple_config_obj(ModelConfig)
        self.state_config = self._get_simple_config_obj(StateConfig)
        self.action_config = self._get_simple_config_obj(ControllerActionConfig)
        self.pl_config = self._get_simple_config_obj(PytorchLightningConfig)
        self.callbacks_config = self._get_simple_config_obj(CallbacksConfig)
        self.wandb_config = self._get_simple_config_obj(WandbConfig)

    def _set_attributes(self) -> None:
        self.experiment_name = self._config[self.EXPERIMENT_NAME]
        self._create_sub_configs()
