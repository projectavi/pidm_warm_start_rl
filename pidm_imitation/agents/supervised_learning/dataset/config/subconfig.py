# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import warnings

from schema import And, Optional, Or, Schema

from pidm_imitation.utils.ioutils import (
    resolve_default_dataset_folder,
    resolve_path,
    resolve_repo_directory_folder,
)
from pidm_imitation.utils.padding_utils import ValidPadding
from pidm_imitation.utils.subconfig import SubConfig


class DataConfig(SubConfig):
    KEY = "data"

    def __init__(self, config_path, config):
        super().__init__(config_path, config)
        self.train_data_dir: str
        self.validation_data_dir: str | None
        self.num_train_samples: int | None
        self.num_validation_samples: int | None
        self.shuffle_trajectories: bool
        self.shuffle_seed: int | None
        self.batch_size: int
        self.num_workers: int

        self.history: int
        self.history_slice_specs: dict
        self.lookahead: int
        self.lookahead_slice_specs: dict
        self.include_k: bool
        self.padding_strategy: str

        self.alignment_strategy: str
        self.alignment_strategy_kwargs: dict

        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                "train_data_dir": str,
                "batch_size": And(
                    int,
                    lambda x: x > 0,
                    error="ERROR: Batch size must be a positive integer.",
                ),
                Optional("validation_data_dir"): Or(None, str),
                Optional("num_train_samples"): Or(None, int),
                Optional("num_validation_samples"): Or(None, int),
                Optional("shuffle_trajectories"): Or(None, bool),
                Optional("shuffle_seed"): Or(None, int),
                Optional("num_workers"): And(
                    int,
                    lambda x: x >= 0,
                    error="ERROR: Number of workers must be a non-negative integer.",
                ),
                Optional("history"): And(
                    int,
                    lambda x: x >= 0,
                    error="ERROR: History must be a positive integer.",
                ),
                Optional("history_slice_specs"): dict,
                Optional("lookahead"): And(
                    int,
                    lambda x: x >= 0,
                    error="ERROR: Lookahead must be a non-negative integer.",
                ),
                Optional("lookahead_slice_specs"): dict,
                Optional("include_k"): bool,
                Optional("padding_strategy"): str,
                Optional("alignment_strategy"): str,
                Optional("alignment_strategy_kwargs"): dict,
            }
        )

    def _set_attributes(self):
        self.train_data_dir = self._config["train_data_dir"]
        self.batch_size = self._config["batch_size"]

        self.num_train_samples = self._config.get("num_train_samples", None)
        self.validation_data_dir = self._config.get("validation_data_dir", None)
        self.num_validation_samples = self._config.get("num_validation_samples", None)
        self.shuffle_trajectories = self._config.get("shuffle_trajectories", True)
        self.shuffle_seed = self._config.get("shuffle_seed", None)
        if self.shuffle_trajectories and self.shuffle_seed is None:
            warnings.warn(
                "data.shuffle_seed is not set while data.shuffle_trajectories is True. Will fallback to pytorch "
                "seed_everything seed for trajectory shuffling."
            )
        self.num_workers = self._config.get("num_workers", 0)

        self.history = self._config.get("history", 0)
        self.history_slice_specs = self._config.get("history_slice_specs", {})
        self.lookahead = self._config.get("lookahead", 0)
        self.lookahead_slice_specs = self._config.get("lookahead_slice_specs", {})
        self.include_k = self._config.get("include_k", False)
        self.padding_strategy = self._config.get(
            "padding_strategy", ValidPadding.NO_PADDING
        )

        self.alignment_strategy = self._config.get("alignment_strategy", None)
        self.alignment_strategy_kwargs = self._config.get(
            "alignment_strategy_kwargs", {}
        )

    @property
    def training_dir(self) -> str:
        self.train_data_dir = resolve_default_dataset_folder(self.train_data_dir)
        self.train_data_dir = resolve_repo_directory_folder(self.train_data_dir)
        local_path = os.path.expandvars(self.train_data_dir)
        return resolve_path(self._config_path, local_path)

    @property
    def validation_dir(self) -> str | None:
        self.validation_data_dir = resolve_default_dataset_folder(
            self.validation_data_dir
        )
        self.validation_data_dir = resolve_repo_directory_folder(
            self.validation_data_dir
        )
        if self.validation_data_dir is None:
            return self.validation_data_dir

        local_path = os.path.expandvars(self.validation_data_dir)
        return resolve_path(self._config_path, local_path)
