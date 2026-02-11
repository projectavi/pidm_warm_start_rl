# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import nn

from pidm_imitation.agents.models.activations import (
    SUPPORTED_ACTIVATIONS,
    get_activation_fn,
)
from pidm_imitation.agents.models.norms import SUPPORTED_NORM_LAYERS, get_norm_layer
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()

SUPPORTED_NETWORK_TYPES = ["linear", "gru", "lstm"]
RECURRENT_NETWORK_TYPES = ["gru", "lstm"]
RECURRENT_NETWORK_CLASSES = [nn.GRU, nn.LSTM]


class NetworkModuleType(ABC):
    @abstractmethod
    def build_layer(self, input_size: int) -> Tuple[nn.Module, int]:
        """
        Builds the configured layer for the given input size
        :param input_size: The size of the input to the layer.
        :return: A PyTorch nn.Module representing the layer. and the output size of the layer.
        """
        raise NotImplementedError(
            "Method `build_layer` must be implemented in child class."
        )

    def is_recurrent(self) -> bool:
        """
        Returns True if the layer is recurrent which requires a hidden state.
        """
        return False


class NetworkLayerType(NetworkModuleType):
    """Class to define and build network layers such as GRU, LSTM, or Linear layers."""

    def __init__(self, layer_type: str, size: int, num_layers: int = 1, **kwargs):
        assert (
            layer_type in SUPPORTED_NETWORK_TYPES
        ), f"Unsupported layer type: {layer_type}. Supported types are: {SUPPORTED_NETWORK_TYPES}"
        self.layer_type = layer_type
        self.size = size
        self.num_layers = num_layers
        self.extra_kwargs = kwargs

    def build_layer(self, input_size: int) -> Tuple[nn.Module, int]:
        network: nn.Module
        if self.layer_type == "gru":
            network = nn.GRU(
                input_size,
                self.size,
                num_layers=self.num_layers,
                batch_first=True,
                **self.extra_kwargs,
            )
        elif self.layer_type == "lstm":
            network = nn.LSTM(
                input_size,
                self.size,
                num_layers=self.num_layers,
                batch_first=True,
                **self.extra_kwargs,
            )
        elif self.layer_type == "linear":
            assert self.num_layers == 1, (
                "Specified a linear layer with multiple layers, which is not supported to ensure non-linear activations"
                " are applied between layers. Use a sequence of linear layers with activations instead."
            )
            network = nn.Linear(input_size, self.size, **self.extra_kwargs)
        else:
            raise ValueError(
                f"Unsupported layer type: {self.layer_type}, expected one of: {SUPPORTED_NETWORK_TYPES}"
            )
        return network, self.size

    def is_recurrent(self) -> bool:
        """
        Returns True if the layer is recurrent which requires a hidden state.
        """
        return self.layer_type in RECURRENT_NETWORK_TYPES


class NetworkActivationType(NetworkModuleType):
    """Class to define and build activation layers such as ReLU, Sigmoid, etc."""

    def __init__(self, activation_type: str, **extra_kwargs):
        self.activation_type = activation_type
        self.extra_kwargs = extra_kwargs

    def build_layer(self, input_size: int) -> Tuple[nn.Module, int]:
        activation = get_activation_fn(self.activation_type, **self.extra_kwargs)()
        assert isinstance(
            activation, nn.Module
        ), f"Activation function {self.activation_type} must return a PyTorch nn.Module, got {type(activation)}"
        return activation, input_size


class NormalizationType(NetworkModuleType):
    """Class to define and build normalization layers such as layer normalization, batch normalization, etc."""

    def __init__(self, norm_type: str):
        if norm_type not in SUPPORTED_NORM_LAYERS:
            raise ValueError(
                f"Unsupported normalization type: {norm_type}. Supported types are: {SUPPORTED_NORM_LAYERS}"
            )
        self.norm_type = norm_type

    def build_layer(self, input_size: int) -> Tuple[nn.Module, int]:
        return get_norm_layer(self.norm_type, input_size), input_size


class NetworkConfig:
    """Config class to define a sequence of layers based on a configuration list."""

    def __init__(self, module_configs: List[dict]):
        self.module_configs = module_configs
        self.module_types: List[NetworkModuleType] = []
        for module_config in module_configs:
            if "type" in module_config:
                module_type = module_config["type"].lower()  # type: ignore
                module_kwargs = {k: v for k, v in module_config.items() if k != "type"}
                if module_type in SUPPORTED_NORM_LAYERS:
                    self.module_types.append(NormalizationType(module_type))
                elif module_type in SUPPORTED_NETWORK_TYPES:
                    assert (
                        "size" in module_config
                    ), f"Module of type {module_type} must have a 'size' key in the configuration."
                    self.module_types.append(
                        NetworkLayerType(module_type, **module_kwargs)
                    )
                elif module_type in SUPPORTED_ACTIVATIONS:
                    self.module_types.append(
                        NetworkActivationType(module_type, **module_kwargs)
                    )
                else:
                    raise ValueError(
                        f"Unsupported module type: {module_type}. Supported types are {SUPPORTED_NETWORK_TYPES} for "
                        f"network layers and {SUPPORTED_NORM_LAYERS} for normalization layers."
                    )
            else:
                raise ValueError(
                    "Each layer configuration must have a 'type' key specifying the layer type."
                )

    def build_network(self, input_dim: int) -> Tuple[nn.ModuleList, int]:
        """
        Builds the network based on the layer types defined in the configuration.
        :param input_dim: The dimension of the input to the first layer.
        :return: A pytorch module list containing the layers of the network and the output dimension of the last layer.
        """
        layers = []
        dim = input_dim
        activation_since_last_layer = True
        for layer_type in self.module_types:
            if isinstance(layer_type, NetworkLayerType):
                if not activation_since_last_layer:
                    log.warning(
                        "Network configuration specifies network layers following each other without an activation "
                        "layer in between. Please review your configuration and consider adding intermediate "
                        "activations."
                    )
                activation_since_last_layer = False
            elif isinstance(layer_type, NetworkActivationType):
                activation_since_last_layer = True
            layer, dim = layer_type.build_layer(dim)
            layers.append(layer)
        return nn.ModuleList(layers), dim

    def get_num_recurrent_layers(self) -> int:
        """
        Returns the number of recurrent layers in the configuration.
        """
        return sum(1 for layer_type in self.module_types if layer_type.is_recurrent())

    @property
    def has_recurrent_layer(self) -> bool:
        """
        Returns True if the layer configuration has at least one recurrent layer.
        """
        return any(layer_type.is_recurrent() for layer_type in self.module_types)
