# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from shutil import copyfile
from typing import Dict, List

from pidm_imitation.constants import CONFIG_FILE_KEY
from pidm_imitation.utils.ioutils import FileMetadata, read_yaml
from pidm_imitation.utils.subconfig import SubConfig


class ConfigFile(SubConfig):
    """
    Abstract class used to create the logic for a specific scenario. Each scenario
    is defined by the pair of (algorithm, environment), since we have different requirements
    depending on which algorithm or environment we're using. The child classes of ``ConfigFile``
    receive a config file path, read that file and extract all parameter blocks in it. It
    then creates multiple ``SubConfig`` classes internally, and pass the correct block of
    parameters to the corresponding ``SubConfig`` class. The ``ConfigFile`` classes are the
    ones that creates and uses the ``SubConfig`` classes, and are the ones we should
    instantiate in our training/evaluation scripts.
    """

    EXPERIMENT_NAME = "experiment_name"

    def __init__(self, config_path: str) -> None:
        """
        :param config_path: the config file' path.
        """
        self._config_path: str
        self.experiment_name: str
        self._set_config(config_path)

    @property
    def config_dict(self) -> dict:
        assert self._config is not None
        return self._config

    def create_config_copy(self, outfile: str):
        copyfile(self._config_path, outfile)

    def _set_config(self, config_path: str) -> None:
        self._config_path = os.path.realpath(str(config_path))
        assert os.path.exists(
            self._config_path
        ), f"ERROR: the config file was not found: {self._config_path}."
        config = read_yaml(self._config_path)
        SubConfig.__init__(self, self._config_path, config)

    def get_config_files_dict(self) -> Dict[str, str | List[FileMetadata]]:
        """Returns a dictionary containing the list of all external files
        associated with this config file including a special key named
        CONFIG_FILE_KEY for the path of config file itself."""
        return {CONFIG_FILE_KEY: self._config_path}
