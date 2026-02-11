# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from typing import Any, List

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger

from pidm_imitation.agents.supervised_learning.dataset.datamodule_factory import (
    DataModuleFactory,
)
from pidm_imitation.agents.supervised_learning.model_factory import ModelFactory
from pidm_imitation.agents.supervised_learning.utils.train_utils import (
    CustomCheckpointFunctionFactory,
    CustomModelCheckpoint,
    add_common_args,
    get_number_of_parameters,
    prepare_checkpoint_dir,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.configs.utils import get_checkpoint_dir, sanity_check_config
from pidm_imitation.utils import Logger
from pidm_imitation.utils.wandb_utils import add_git_info

log = Logger().get_root_logger()


def parse_args():
    parser = argparse.ArgumentParser("Train model using offline learning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Overwrite any existing checkpoints and logs.",
    )
    add_common_args(parser)

    args = parser.parse_args()
    return args


def get_wandb_logger(config: OfflinePLConfigFile) -> WandbLogger | None:
    """
    Returns a WandbLogger instance if wandb is configured in the OfflinePLConfigFile.
    :param config: OfflinePLConfigFile instance containing the configuration.
    :return: WandbLogger instance if wandb is configured, otherwise None.
    """
    if config.wandb_config:
        wandb_logger = WandbLogger(
            project=config.wandb_config.project,
            group=config.wandb_config.train_group,
            name=(
                config.wandb_config.train_name
                if config.wandb_config.train_name
                else config.experiment_name
            ),
            save_dir=config.wandb_config.save_dir,
            log_model=config.wandb_config.log_model,
            offline=config.wandb_config.offline,
            tags=config.wandb_config.tags,
            notes=config.wandb_config.notes,
            config=add_git_info(config._config),
        )
        return wandb_logger
    return None


def get_callbacks(config: OfflinePLConfigFile) -> List[Callback]:
    """
    Returns a list of callbacks based on the configuration.
    :param config: OfflinePLConfigFile instance containing the configuration.
    :return: List of lightning callbacks to be used in the training process.
    """
    if config.callbacks_config is None:
        return []

    callbacks: List[Callback] = []
    # Adding the learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    # adjusts depth from default 1 to display summary of heads by name
    callbacks.append(ModelSummary(max_depth=2))
    if config.callbacks_config.checkpoint_callback:
        args = config.callbacks_config.checkpoint_callback_kwargs
        if "every_n_steps_custom" in args:
            custom_fn_config = args.pop("every_n_steps_custom")
            name = custom_fn_config["classname"]
            init_args = custom_fn_config.get("init_args", {})
            fn = CustomCheckpointFunctionFactory.create(name, init_args)
            args["checkpoint_function"] = fn
            callbacks.append(CustomModelCheckpoint(**args))
        else:
            callbacks.append(ModelCheckpoint(**args))
    return callbacks


if __name__ == "__main__":
    args = parse_args()
    config = OfflinePLConfigFile(args.config)
    sanity_check_config(config)

    if config.pl_config.seed_everything is not None:
        pl.seed_everything(config.pl_config.seed_everything, workers=True)
    else:
        log.warning(
            "No seed specified! Please set a seed in pytorch_lightning.seed_everything for reproducibility."
        )

    logger: Any = get_wandb_logger(config)
    if not logger:
        # If no wandb logger is configured, use TensorBoardLogger
        log.info("Using TensorBoardLogger as no WandbLogger is configured.")
        log.info("You can view these logs using: tensorboard --logdir logs")
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(
            save_dir="./logs",
            name=config.experiment_name,
            version=1,
            log_graph=False,
        )

    checkpoint_dir = get_checkpoint_dir(config)
    prepare_checkpoint_dir(checkpoint_dir, args.new)
    datamodule = DataModuleFactory.get_datamodule(
        config=config, output_dir=checkpoint_dir
    )
    model = ModelFactory.get_model(config, datamodule)

    callbacks = get_callbacks(config)

    num_params, num_train_params = get_number_of_parameters(model)
    log.info(
        f"Model architecture with {num_params} total parameters ({num_train_params} trainable parameters):"
    )
    log.info(model)

    # initiate trainer and start training
    trainer_kwargs = config.pl_config.trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )
    trainer.fit(model=model, datamodule=datamodule, **config.pl_config.fit_kwargs)
