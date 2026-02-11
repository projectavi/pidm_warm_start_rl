# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pidm_imitation.agents.supervised_learning.dataset.align_dataset.alignment_strategy_factory import (
    AlignmentStrategyFactory,
)
from pidm_imitation.agents.supervised_learning.dataset.datamodule import DataModule
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile


class DataModuleFactory:

    @staticmethod
    def get_datamodule(
        config: OfflinePLConfigFile,
        output_dir: str,
    ) -> DataModule:
        state_type = config.state_config.type.value
        action_type = config.action_config.type
        data_config = config.data_config

        align_strategy = AlignmentStrategyFactory.get_alignment_strategy(
            alignment_strategy=data_config.alignment_strategy,
            action_type=action_type,
            **data_config.alignment_strategy_kwargs,
        )

        shuffle_seed = (
            data_config.shuffle_seed
            if data_config.shuffle_seed is not None
            else config.pl_config.seed_everything
        )

        datamodule_kwargs = {
            "state_type": state_type,
            "action_type": action_type,
            "train_data_dir": data_config.training_dir,
            "history": data_config.history,
            "history_slice_specs": data_config.history_slice_specs,
            "lookahead": data_config.lookahead,
            "lookahead_slice_specs": data_config.lookahead_slice_specs,
            "include_k": data_config.include_k,
            "padding_strategy": data_config.padding_strategy,
            "align_strategy": align_strategy,
            "validation_data_dir": data_config.validation_dir,
            "batch_size": data_config.batch_size,
            "num_workers": data_config.num_workers,
            "num_train_samples": data_config.num_train_samples,
            "num_validation_samples": data_config.num_validation_samples,
            "shuffle_trajectories": data_config.shuffle_trajectories,
            "shuffle_seed": shuffle_seed,
            "output_dir": output_dir,
        }

        return DataModule(**datamodule_kwargs)  # type: ignore
