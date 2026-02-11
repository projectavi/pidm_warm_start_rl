# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import nullcontext
from typing import List, Tuple

import torch
from torch import Tensor, nn

from pidm_imitation.agents.models.layer_types import (
    RECURRENT_NETWORK_CLASSES,
    NetworkConfig,
)


class NetworkBlock(nn.Module):
    """This class supports custom "layer" definition in yaml so it can do a combination different types of layers and
    operations. It processes sequences of data of shape (batch_size, seq_len, input_dim) and outputs sequences of shape
    (batch_size, seq_len, output_dim). For supported types of network layers, normalization layers, and activations,
    see the `SUPPORTED_LAYER_TYPES`, `SUPPORTED_NORM_LAYERS`, and `SUPPORTED_ACTIVATIONS` constants.

    This model caches the hidden states of all RNN layers. The hidden states can be reset using the `reset()` method.

    Example config:

          layers:
            - type: gru
              size: 256
              layers: 2
            - type: relu
            - type: batch_norm
            - type: linear
              size: 256
            - type: relu

    """

    def __init__(
        self,
        network_config: List[dict] | NetworkConfig,
        input_dim: int,
        collapse_sequence: bool = False,
    ):
        """Define a network block with a sequence of layers.
        :param network_config: List of dictionaries defining the layers in the block. Each dictionary should contain
            the type of layer and its parameters, see above for example.
        :param input_dim: The dimension of the input to the first layer.
        :param collapse_sequence: If True, the input sequences will be collapsed into a single sequence
            by reshaping the input tensor to (batch_size, 1, seq_len * input_dim). This is useful for models that
            stack inputs together.
        """
        super().__init__()
        if isinstance(network_config, list):
            self.network_config = NetworkConfig(network_config)
        else:
            assert isinstance(
                network_config, NetworkConfig
            ), "network_config must be a list of dictionaries or a LayerConfig object."
            self.network_config = network_config
        self.input_dim = input_dim
        self.collapse_sequence = collapse_sequence

        self.layers, self.output_dim = self.network_config.build_network(self.input_dim)
        num_recurrent_layers = self.network_config.get_num_recurrent_layers()
        self.hidden_states: List[Tensor | Tuple[Tensor, ...] | None] = [
            None
        ] * num_recurrent_layers
        self.frozen = False

    def _forward_recurrent_layer(
        self, layer: nn.Module, x: Tensor, rnn_count: int
    ) -> Tensor:
        # Get current hidden state for the recurrent layer
        hidden_state = self.hidden_states[rnn_count]
        x, hidden_state = layer(x, hidden_state)
        # Store detached hidden state
        if isinstance(hidden_state, Tensor):
            self.hidden_states[rnn_count] = hidden_state.detach()
        elif isinstance(hidden_state, tuple) and all(
            isinstance(h, Tensor) for h in hidden_state
        ):
            # For LSTM, hidden_state is a tuple (h_n, c_n)
            self.hidden_states[rnn_count] = tuple(h.detach() for h in hidden_state)
        else:
            raise ValueError(
                f"Unexpected hidden state type: {type(hidden_state)}, expected tuple or Tensor"
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers of the network block. For each non-recurrent layer, the input is simply
        passed through the layer. For recurrent layers (GRU, LSTM), the input is processed with the current hidden state
        and the hidden state is updated. The hidden states are stored in `self.hidden_states` and detached to avoid
        backpropagation through time.
        """
        with torch.inference_mode() if self.frozen else nullcontext():
            assert (
                x.dim() == 3
            ), "Input tensor must be of shape (batch_size, seq_len, input_dim)"

            if self.collapse_sequence:
                batch_size, seq_len, in_dim = x.shape
                x = x.reshape(batch_size, 1, seq_len * in_dim)

            assert (
                x.size(-1) == self.input_dim
            ), f"Input dimension mismatch: expected {self.input_dim}, got {x.size(-1)}"

            rnn_count = 0
            for layer in self.layers:
                if isinstance(layer, tuple(RECURRENT_NETWORK_CLASSES)):
                    x = self._forward_recurrent_layer(layer, x, rnn_count)
                    rnn_count += 1
                else:
                    x = layer(x)
            return x.detach() if self.frozen else x

    def reset(self):
        self.hidden_states = [None] * len(self.hidden_states)

    @property
    def out_dim(self) -> int:
        return self.output_dim

    @property
    def in_dim(self) -> int:
        return self.input_dim

    @property
    def is_recurrent(self) -> bool:
        return any(
            module_type.is_recurrent()
            for module_type in self.network_config.module_types
        )

    @property
    def is_frozen(self) -> bool:
        return self.frozen
