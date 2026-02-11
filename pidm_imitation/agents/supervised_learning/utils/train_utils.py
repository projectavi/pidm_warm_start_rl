# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from abc import abstractmethod
from datetime import timedelta
from shutil import rmtree
from typing import Any, Dict, Literal, Optional, Tuple

import lightning.pytorch as pl
from git import Union
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT

from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


def add_common_args(parser) -> None:
    parser.add_argument(
        "--username",
        type=str,
        help="Send output to the WANDB account for this user name if WANDB is enabled in the config. \
            Defaults to current user name.",
    )


def prepare_checkpoint_dir(checkpoints_dir: str, wipe_existing: bool = False):
    """
    Prepare the checkpoint directory for training.
    If `wipe_existing` is True, it will remove any existing checkpoints in the directory.
    If the directory does not exist, it will create it.
    If the directory already exists and `wipe_existing` is False, it will raise an error.
    :param checkpoints_dir: The directory where checkpoints will be stored.
    :param wipe_existing: If True, existing checkpoints will be removed.
    """
    if os.path.isdir(checkpoints_dir):
        if wipe_existing:
            log.warning("Wiping existing checkpoints ...")
            rmtree(checkpoints_dir, ignore_errors=True)
        else:
            raise ValueError(
                f"ERROR: The directory '{checkpoints_dir}' already exists. Please start the run from a new directory "
                " to ensure you don't end up with a confusing combination checkpoints from different training runs in "
                "the same directory. To delete the previous checkpoints you can provide the '--new' argument."
            )
    # create empty checkpoints directory so that other runs fail with error above
    os.makedirs(checkpoints_dir, exist_ok=True)


def get_number_of_parameters(model: pl.LightningModule) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters in the model.
    :param model: The model instance to check.
    :return: The number of total and trainable parameters in the model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_train_params


class CustomCheckpointFunction:
    @abstractmethod
    def compute_next_n_steps(self, global_step: int) -> int:
        pass


class CheckpointExponentialFunction(CustomCheckpointFunction):
    def __init__(
        self,
        every_n_train_steps: int = 100,
        increase_power: int = 10,
        increase_on_n_steps: int = None,
    ):
        self.every_n_train_steps = every_n_train_steps
        self.increase_on_n_steps = increase_on_n_steps or every_n_train_steps
        self.increase_power = increase_power

    def compute_next_n_steps(self, global_step: int) -> int:
        # make the next n steps increase by the given power, for example, if the start is 100 and increase
        # by a factor of ten you will get 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000
        # 7000,8000,9000,10000,20000,30000,40000... etc
        if global_step == self.increase_on_n_steps * self.increase_power:
            self.increase_on_n_steps *= self.increase_power
            self.every_n_train_steps *= self.increase_power
        return self.every_n_train_steps


class CustomCheckpointFunctionFactory:
    @staticmethod
    def create(
        function_name: str, init_args: Dict[str, Any]
    ) -> CustomCheckpointFunction:
        if function_name == "exponential":
            return CheckpointExponentialFunction(**init_args)
        else:
            raise ValueError(f"Unknown checkpoint function: {function_name}")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        checkpoint_function: CustomCheckpointFunction | None = None,
    ):
        self.checkpoint_function = checkpoint_function
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.checkpoint_function is not None:
            self._every_n_train_steps = self.checkpoint_function.compute_next_n_steps(
                trainer.global_step
            )
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
