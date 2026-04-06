# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import torch
from torch import Tensor, nn

from pidm_imitation.utils.valid_controller_actions import ValidControllerActions


class StructuredSSMCore(nn.Module):
    """
    Shared scaffold for the future SSIDM core.

    The real implementation will replace these placeholder feed-forward paths with the
    structured convolutional training operator and recurrent rollout operator. The method
    surface is already aligned with that end state so the surrounding scaffold can settle now.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self._cached_state: Tensor | None = None

    def _project(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        return self.output_projection(self.activation(projected))

    def forward_convolution(self, inputs: Tensor) -> Tensor:
        return self._project(inputs)

    def forward_recurrent(self, inputs: Tensor) -> Tensor:
        outputs = self._project(inputs)
        self._cached_state = outputs[:, -1, :].detach()
        return outputs

    def step(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        outputs = self._project(inputs)
        self._cached_state = outputs[:, -1, :].detach()
        return outputs[:, -1:, :]

    def reset(self) -> None:
        self._cached_state = None


class SSIDMPolicyNetwork(nn.Module):
    """
    Shared scaffold policy for both PSSIDM and LSSIDM.

    Version 1 of this class deliberately keeps the surrounding repo contracts stable first:
    - sequence-valued forward pass for training
    - step/recurrent signatures for rollout integration
    - optional internal shared latent encoder for LSSIDM
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        action_type: str,
        hidden_dim: int = 128,
        latent_encoder_dim: int = 128,
        use_latent_encoder: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.action_type = action_type
        self.hidden_dim = hidden_dim
        self.latent_encoder_dim = latent_encoder_dim
        self.use_latent_encoder = use_latent_encoder

        assert (
            action_type == ValidControllerActions.LEFT_STICK
        ), f"Unsupported action type for SSIDM scaffold: {action_type}"
        assert (
            input_dim >= 2 * state_dim
        ), f"SSIDM scaffold expects at least two state streams, got input_dim={input_dim}, state_dim={state_dim}."

        self.executed_dim = state_dim
        self.reference_dim = input_dim - state_dim

        core_stream_dim = self.executed_dim
        if self.use_latent_encoder:
            assert self.reference_dim == self.executed_dim, (
                "Shared latent encoder scaffold expects executed and reference streams "
                f"to have matching dimensions, got {self.executed_dim} and {self.reference_dim}."
            )
            self.shared_latent_encoder = nn.Sequential(
                nn.Linear(self.executed_dim, latent_encoder_dim),
                nn.ReLU(),
            )
            core_stream_dim = latent_encoder_dim
        else:
            self.shared_latent_encoder = None

        self.core = StructuredSSMCore(
            input_dim=2 * core_stream_dim,
            hidden_dim=hidden_dim,
            action_dim=ValidControllerActions.get_actions_dim(action_type),
        )

    def _split_streams(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        executed = inputs[..., : self.executed_dim]
        reference = inputs[..., self.executed_dim :]
        return executed, reference

    def _combine_streams(self, inputs: Tensor) -> Tensor:
        executed, reference = self._split_streams(inputs)
        if self.shared_latent_encoder is not None:
            executed = self.shared_latent_encoder(executed)
            reference = self.shared_latent_encoder(reference)
        return torch.cat([executed, reference], dim=-1)

    def forward_convolution(self, inputs: Tensor) -> Tensor:
        return self.core.forward_convolution(self._combine_streams(inputs))

    def forward_recurrent(self, inputs: Tensor) -> Tensor:
        return self.core.forward_recurrent(self._combine_streams(inputs))

    def forward_step(self, inputs: Tensor) -> Tensor:
        return self.core.step(self._combine_streams(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_convolution(inputs)

    def reset(self) -> None:
        self.core.reset()

    @property
    def collapse_sequence(self) -> bool:
        return False

    @property
    def is_recurrent(self) -> bool:
        # The scaffold uses the sliding-window rollout path first. The real SSIDM recurrent
        # agent path can be enabled once the recurrent core is implemented and validated.
        return False
