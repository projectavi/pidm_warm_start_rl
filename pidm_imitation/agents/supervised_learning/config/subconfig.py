# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict

from schema import Optional, Schema

from pidm_imitation.utils.subconfig import SubConfig

SUBMODEL_KEYS = {"class_name", "init_args"}


class ModelConfig(SubConfig):
    KEY = "model"

    def __init__(self, config_path, config):
        super().__init__(config_path, config)
        self.algorithm: str
        self.input_format: str
        self.submodel_configs: Dict[str, Dict[str, Any]]
        self.init_args: Dict[str, Any]
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                "algorithm": str,
                "input_format": str,
                "submodels": dict,
                Optional("init_args"): dict,
            }
        )

    def _set_attributes(self):
        self.algorithm = self._config["algorithm"]
        self.input_format = self._config["input_format"]
        self.submodel_configs = self._config["submodels"]
        for submodel_config in self.submodel_configs.values():
            assert (
                "class_name" in submodel_config
            ), f"Submodel configuration must contain 'class_name' key but only got {submodel_config.keys()}."
            assert (
                "init_args" in submodel_config
            ), f"Submodel configuration must contain 'init_args' key but only got {submodel_config.keys()}."
            submodel_config_keys = set(submodel_config.keys())
            assert submodel_config_keys.issubset(SUBMODEL_KEYS), (
                f"Submodel configuration contains unsupported keys "
                f"{submodel_config_keys - SUBMODEL_KEYS}."
            )
        self.init_args = self._config.get("init_args", {})
