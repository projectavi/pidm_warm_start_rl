# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch import Tensor, nn

from pidm_imitation.utils import ValidControllerActions


class ActionLoss(nn.Module):
    """
    Factory class to compute the loss based on the action type.
    """

    L1 = "l1"
    MSE = "mse"
    VALID_CONT_LOSS = [L1, MSE]

    def __init__(
        self,
        action_type: str,
        continuous_loss: str | nn.Module | None,
        sequence_training: bool = True,
    ):
        """
        Class used to create the loss functions used by the IDM model. It receives the action type being used by the
        model, which is used to determine if the agent uses sticks and/or buttons, and it also receives which
        continuous loss and binary classification loss are used for the continuous and classification outputs of the
        model, respectively.

        :param action_type: a string defining which action type is used by the agent. Must be an action from
            ``ValidControllerActions.ALL``;
        :param continuous_loss: Continuous loss function used for sticks or any continuous button. Valid values
            are defined in ``ActionLoss.VALID_CONT_LOSS``.
        :param binary_loss: Binary classification loss function used for discrete buttons. If the value
            provided is a string, it must be one of the strings defined in ``ActionLoss.VALID_BINARY_LOSS``.
            This way, a loss object is created based on the string provided, but using only default values
            for the loss function. If a Pytorch loss object is provided, then use that as the classification
            loss. If non-default parameters are needed, then directly provide a valid loss object.
        :param cont_binary_loss_weight: a list with two values: the weight assigned to the continuous loss
            function and the weight of the binary classification loss (in that order). By default, both loss
            functions have a weight of 1.0. This can also be a list of lists defining the precise weights for
            each individual action continuous and discrete action.
        :param sequence_training: if True, the model is trained with sequences of target actions and is expected to
            predict a sequence of actions. If False, the model is trained with a single target action and is
            expected to predict a single action.
        """
        super().__init__()
        self._action_type = action_type
        self._sequence_training = sequence_training

        self._cont_dim = ValidControllerActions.get_actions_dim(self._action_type)
        self._setup_loss_functions(action_type, continuous_loss)

    def _check_inputs(self, inputs: Tensor):
        """Check model predictions based on sequence training flag
        :param inputs: The inputs to the model, which can be a Tensor or a tuple of Tensors.
        """
        if not self._sequence_training:
            # check that predicted action logits have sequence length of 1
            assert (
                inputs.ndim == 3 and inputs.shape[1] == 1
            ), "When sequence_training is False, the inputs should be a Tensor of shape (batch_size, 1, in_dim)."

    def _check_and_get_targets(self, target: Tensor) -> Tensor:
        """Check the target actions based on sequence training flag
        :param target: The target actions of shape (batch_size, seq_len, in_dim).
        :return: The target actions as a Tensor of shape (batch_size, seq_len, in_dim) or (batch_size, 1, in_dim)
            if no sequence training.
        """
        assert (
            target.ndim == 3
        ), "Expected target actions to be a Tensor of shape (batch_size, seq_len, in_dim). "
        if not self._sequence_training:
            target = target[:, -1, :].unsqueeze(
                1
            )  # Get the last action in the sequence
        return target

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """Call this function to compute the action loss."""
        self._check_inputs(inputs)
        target = self._check_and_get_targets(target)

        if self._action_type == ValidControllerActions.LEFT_STICK:
            return self._compute_cont_loss(inputs, target)  # type: ignore
        raise ValueError(f"Unsupported action specified: {self._action_type}")

    def _setup_loss_functions(
        self,
        action_type: str,
        continuous_loss_name: str | nn.Module | None,
    ) -> None:
        if action_type == ValidControllerActions.LEFT_STICK:
            self._cont_loss_fn = self.get_cont_loss_fn(continuous_loss_name)
        else:
            raise ValueError(f"Unsupported action specified: {action_type}")

    def _compute_cont_loss(
        self, pred_actions: Tensor, target_actions: Tensor
    ) -> Tensor:
        return self._cont_loss_fn(pred_actions, target_actions)

    def get_cont_loss_fn(self, loss: str | nn.Module) -> nn.Module:
        # Check if loss is already a valid loss object
        if isinstance(loss, nn.Module):
            return loss

        # If it's a string, then create the loss object
        if loss == ActionLoss.L1:
            return nn.L1Loss()
        if loss == ActionLoss.MSE:
            return nn.MSELoss()
        raise ValueError(
            f"Invalid loss function specified: {loss}, expected one of {ActionLoss.VALID_CONT_LOSS}."
        )
