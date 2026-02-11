# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch import nn

from pidm_imitation.agents.models.utils import (
    get_collapse_sequence,
    get_output_dim,
    is_recurrent,
    reset_model,
)
from pidm_imitation.utils.valid_controller_actions import ValidControllerActions


class FinalLayerFactory:
    """
    Final Layer of the NN for left_right_sticks and sticks_and_buttons controller action
    """

    @staticmethod
    def get_final_layer_for_continous_actions(
        action_type: str, in_dim: int
    ) -> nn.Module:
        action_dim = ValidControllerActions.get_actions_dim(action_type=action_type)
        return FinalLayer(in_dim, action_dim, FinalLayer.TANH_ACTIVATION)

    @staticmethod
    def get_final_layer(action_type: str, in_dim: int) -> nn.Module:
        if action_type == ValidControllerActions.LEFT_STICK:
            return FinalLayerFactory.get_final_layer_for_continous_actions(
                action_type, in_dim
            )
        raise ValueError(f"Unsupported action specified: {action_type}")


class FinalLayer(nn.Module):
    TANH_ACTIVATION = "tanh"
    SIGMOID_ACTIVATION = "sigmoid"
    NO_ACTIVATION = "none"
    ALL_ACTIVATION = [TANH_ACTIVATION, SIGMOID_ACTIVATION, NO_ACTIVATION]

    def __init__(self, in_dim: int, out_dim: int, activation: str):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert (
            activation in self.ALL_ACTIVATION
        ), f"ERROR: Invalid activation function, got {activation} but expected one of {self.ALL_ACTIVATION}."
        activation_fn = self._get_activation_fn(activation)
        self.model: nn.Module
        if activation_fn is None:
            self.model = nn.Linear(in_dim, out_dim)
        else:
            self.model = nn.Sequential(nn.Linear(in_dim, out_dim), activation_fn)

    def _get_activation_fn(self, activation: str) -> nn.Module | None:
        if activation == self.TANH_ACTIVATION:
            return nn.Tanh()
        elif activation == self.SIGMOID_ACTIVATION:
            return nn.Sigmoid()
        elif activation == self.NO_ACTIVATION:
            return None
        raise ValueError(
            f"ERROR: Invalid activation function, got {activation} but expected one of {self.ALL_ACTIVATION}."
        )

    def forward(self, x):
        return self.model(x)


class PolicyNetwork(nn.Module):
    """
    Wrapper around a network to create a policy network. The wrapper adds the ability to define
    a final output layer based on the action type.
    """

    def __init__(self, base_model: nn.Module, action_type: str):
        super().__init__()
        self.base_model = base_model
        self.final_layer = FinalLayerFactory.get_final_layer(
            action_type, get_output_dim(self.base_model)
        )

    def forward(self, x):
        """Forward pass through the policy network to get action logits
        :param x: Input tensor of shape (batch_size, seq_len, in_dim).
        :return: Action logits as tensor of shape (batch_size, seq_len, out_dim).
        """
        x = self.base_model(x)
        return self.final_layer(x)

    def reset(self):
        reset_model(self.base_model)

    @property
    def collapse_sequence(self) -> bool:
        """Check if inputs sequences should be collapsed"""
        return get_collapse_sequence(self.base_model)

    @property
    def is_recurrent(self) -> bool:
        return is_recurrent(self.base_model)
