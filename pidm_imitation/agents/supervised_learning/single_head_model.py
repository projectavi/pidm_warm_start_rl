# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import Any, Dict

from torch import Tensor

from pidm_imitation.agents.supervised_learning.base_models import ActionRegressor
from pidm_imitation.agents.supervised_learning.submodels.policy_heads import PolicyHead
from pidm_imitation.agents.supervised_learning.submodels.state_encoder_model import (
    StateEncoderModel,
)
from pidm_imitation.constants import POLICY_HEAD_KEY


class SingleHeadActionRegressor(ActionRegressor):
    """
    Action regressor model based on multi-head architecture but with only single policy head. Depending on the inputs
    given to the state encoder model (defined by `state_encoder_inputs` argument) and policy head (as defined in the
    `policy_inputs`argument when creating the head) this class can be used for a IDM or BC model.
    """

    def __init__(
        self,
        state_encoder_model: StateEncoderModel | None,
        policy_head: PolicyHead,
        optimizer_name: str,
        optimizer_kwargs: Dict[str, Any],
        scheduler: str,
        scheduler_kwargs: Dict[str, Any],
    ):
        """
        :param state_encoder_model: nn.Module with the state encoder of our action regressor architecture.
        :param policy_head: head for policy network
        :param optimizer_name: Name of the optimizer to use for training the model.
        :param optimizer_kwargs: Keyword arguments for the optimizer.
        :param scheduler: Name of the learning rate scheduler to use for training the model.
        :param scheduler_kwargs: Keyword arguments for the scheduler.
        """
        super().__init__(
            state_encoder_model=state_encoder_model,
            heads=OrderedDict({POLICY_HEAD_KEY: policy_head}),
            optimizer_name=optimizer_name,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )

    def _common_training_validation_step(
        self, batch: Dict[str, Tensor], training: bool = True
    ) -> Tensor:
        """
        Common training and validation step logic.
        :param batch: batch of data as a dictionary with input keys and tensors.
        :param training: boolean flag indicating if the step is for training or validation.
        :return: total loss for the step.
        """
        self.reset()
        state_embedddings = self._compute_encoder_embeddings(batch)

        assert (
            self.policy_head.use_at_train_time
        ), "Policy head is the only head and must be active"
        policy_inputs = {**batch, **state_embedddings}
        policy_outputs = self.policy_head(policy_inputs)
        policy_target_key = self.policy_head.get_target_key()

        head_predictions = {POLICY_HEAD_KEY: policy_outputs}
        head_targets = {POLICY_HEAD_KEY: batch[policy_target_key]}

        return self._compute_total_loss_and_log_losses(
            predicted=head_predictions,
            target=head_targets,
            training=training,
        )
