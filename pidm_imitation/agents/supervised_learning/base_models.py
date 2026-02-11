# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Tuple, no_type_check

import lightning.pytorch as pl
import torch
from torch import Tensor, nn

from pidm_imitation.agents.models.optimizers import get_optimizer
from pidm_imitation.agents.models.schedulers import get_scheduler
from pidm_imitation.agents.supervised_learning.submodels.base_head import Head
from pidm_imitation.agents.supervised_learning.submodels.policy_heads import (
    PolicyHeadBase,
)
from pidm_imitation.agents.supervised_learning.submodels.state_encoder_model import (
    StateEncoderModel,
)
from pidm_imitation.constants import (
    EMB_KEY_SUFFIX,
    POLICY_HEAD_KEY,
    PREDICTED_ACTION_KEY,
    TRAIN_LOSS,
    VALIDATION_LOSS,
)
from pidm_imitation.torch_utils import TrainingProgressMixin
from pidm_imitation.utils import Logger

warnings.filterwarnings(
    "ignore", category=UserWarning, message="No positive samples in targets*"
)
log = Logger.get_logger(__name__)


class ActionRegressor(pl.LightningModule, TrainingProgressMixin):
    """
    Lightning Module base class for multi-head action regressors.
    """

    def __init__(
        self,
        state_encoder_model: StateEncoderModel | None,
        heads: OrderedDict[str, Head],
        optimizer_name: str,
        optimizer_kwargs: Dict[str, Any],
        scheduler: str,
        scheduler_kwargs: Dict[str, Any],
    ):
        """
        :param state_encoder_model: state encoder model to encoder state(-action) sequences
        :param heads: ordered dictionary from head names to Head instances (nn.Module).
        :param optimizer_name: Name of the optimizer to use for training the model.
        :param optimizer_kwargs: Keyword arguments for the optimizer.
        :param scheduler: Name of the learning rate scheduler to use for training the model.
        :param scheduler_kwargs: Keyword arguments for the scheduler.
        """
        super().__init__()
        self.initialize_watchers()

        self.state_encoder_model = state_encoder_model
        self.heads: Mapping[str, Head] = nn.ModuleDict(heads)  # type: ignore
        assert POLICY_HEAD_KEY in heads, "Policy head must be included in heads"
        assert isinstance(
            self.policy_head, PolicyHeadBase
        ), f"Expected policy head to be PolicyHead instance, but got {type(self.policy_head)}"

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.save_hyperparameters()

    @property
    def policy_head(self) -> PolicyHeadBase:
        return self.heads[POLICY_HEAD_KEY]  # type: ignore[return-value]

    @property
    def is_recurrent(self) -> bool:
        return self.state_encoder_model.is_recurrent or any(
            head.is_recurrent for head in self.heads.values()
        )

    def get_input_keys(self) -> List[str]:
        """
        Get the names of the inputs required by the model.
        :return: List of input keys.
        """
        input_names = set()
        if self.state_encoder_model:
            input_names.update(self.state_encoder_model.get_input_keys())
        for head in self.heads.values():
            input_names.update(head.get_input_keys())
        return list(input_names)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.optimizer_name, self.parameters(), **self.optimizer_kwargs
        )
        scheduler = get_scheduler(self.scheduler, optimizer, **self.scheduler_kwargs)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
        ]

    def reset(self) -> None:
        if self.state_encoder_model:
            self.state_encoder_model.reset()
        for head in self.heads.values():
            head.reset()

    def _compute_encoder_embeddings(
        self,
        inputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Computes the embeddings from the state encoder model if available. The embeddings are added  under the key
        <group_name> + EMB_KEY_SUFFIX for each group.
        :param inputs: dictionary of inputs mapping input name to tensors.
        :return: dictionary of embeddings mapping group names to tensors.
        """
        embeddings = {}
        if self.state_encoder_model:
            encoder_embedding_groups = self.state_encoder_model.get_embeddings(inputs)
            for group_name, group_embeddings in encoder_embedding_groups.items():
                embeddings[group_name + EMB_KEY_SUFFIX] = group_embeddings
        return embeddings

    def forward_policy_path(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass from the input to the output of the policy head, which includes the predicted action.
        This is the method used by the pytorch agent to interact with the environment, as it is the one that
        predicts the next action.
        :param inputs: dictionary of inputs mapping input name to tensors.
        :return: dictionary mapping output keys to tensors, including the predicted action.
        """
        encoder_embeddings = self._compute_encoder_embeddings(inputs)

        policy_inputs = {**inputs, **encoder_embeddings}
        policy_predictions = self.policy_head(policy_inputs)
        return {
            **encoder_embeddings,
            **policy_predictions,
        }

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass through the state encoder and policy head to get the predicted action.
        :param inputs: dictionary of inputs mapping input name to tensors.
        :return: predicted action tensor.
        """
        return self.forward_policy_path(inputs)[PREDICTED_ACTION_KEY]

    def _compute_losses(
        self,
        predicted: Dict[str, Dict[str, Tensor | Tuple[Tensor, Tensor]]],
        target: Dict[str, Tensor],
        training: bool = True,
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        """
        Compute losses and loss logs for all heads.
        :param predicted: dictionary mapping head names to their outputs.
        :param target: dictionary mapping head names to their targets.
        :param training: boolean flag indicating if the step is for training or validation.
        :return: total loss for the step and a dictionary of loss logs.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        losses_dict = {}
        for name, head in self.heads.items():
            if head.use_at_train_time:
                head_target = (
                    target[name][:, -1].unsqueeze(1)
                    if head.collapse_sequence
                    else target[name]
                )
                head_total_loss, head_losses_dict = head.compute_loss(
                    predicted[name], head_target, training
                )
                losses_dict.update(head_losses_dict)
                total_loss += head_total_loss * head.loss_weight
        total_loss_name = TRAIN_LOSS if training else VALIDATION_LOSS
        losses_dict[total_loss_name] = total_loss
        return total_loss, losses_dict

    def _log_losses(self, losses_dict: dict[str, Tensor], training: bool) -> None:
        self.log_dict(losses_dict, on_step=True, sync_dist=False if training else True)

    def _compute_total_loss_and_log_losses(
        self,
        predicted: Dict[str, Dict[str, Tensor | Tuple[Tensor, Tensor]]],
        target: Dict[str, Tensor],
        training: bool = True,
    ) -> Tensor:
        """
        Compute losses and log all losses
        :param predicted: dictionary mapping head names to their outputs.
        :param target: dictionary mapping head names to their targets.
        :param training: boolean flag indicating if the step is for training or validation.
        :return: total loss
        """
        total_loss, losses_dict = self._compute_losses(predicted, target, training)
        self._log_losses(losses_dict, training)
        return total_loss

    @no_type_check
    def _log_on_training_step(self) -> None:
        if self.global_rank == 0 and self.global_step == 0:
            log.info(
                f"Training Data: {self.trainer.datamodule.train_datasize} Trajectories, \
                      {self.trainer.datamodule.train_samples} Dataset size"
            )
            log.info(
                f"Validation Data: {self.trainer.datamodule.validation_datasize} Trajectories, \
                      {self.trainer.datamodule.validation_samples} Dataset size"
            )

    @no_type_check
    def _training_first_step(self):
        log.info(
            f"Training Data: {self.trainer.datamodule.train_datasize} Trajectories, \
                {self.trainer.datamodule.train_samples} Dataset size"
        )
        log.info(
            f"Validation Data: {self.trainer.datamodule.validation_datasize} Trajectories, \
                {self.trainer.datamodule.validation_samples} Dataset size"
        )

    @abstractmethod
    def _common_training_validation_step(
        self, batch: Dict[str, Tensor], training: bool = True
    ) -> Tensor:
        """
        Common training and validation step logic.
        :param batch: batch of data as a dictionary with input keys and tensors.
        :param training: boolean flag indicating if the step is for training or validation.
        :return: total loss for the step.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses. It should contain the common logic for training and "
            "validation steps."
        )

    def training_step(self, batch, batch_idx):
        self._log_on_training_step()
        loss = self._common_training_validation_step(batch, training=True)
        if self.global_rank == 0 and self.global_step == 0:
            self._training_first_step()

        self.log("train_loss", loss, prog_bar=True)  # type: ignore[union-attr]
        self.log("global_step", self.global_step, prog_bar=True)  # type: ignore[union-attr]

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_training_validation_step(batch, training=False)
        return loss
