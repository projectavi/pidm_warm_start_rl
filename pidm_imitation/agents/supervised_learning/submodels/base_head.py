# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

import torch
from torch import Tensor

from pidm_imitation.agents.supervised_learning.submodels.submodel import SubModel
from pidm_imitation.agents.supervised_learning.utils.utils import is_sequence_key
from pidm_imitation.constants import TRAIN_PREFIX, VALIDATION_PREFIX


class Head(SubModel, ABC):
    def __init__(self, *args, **kwds):
        """
        Must be called by the child class to initialize the Module.
        """
        super().__init__(*args, **kwds)
        self._checkpoint_folder: str | None = None

    @property
    def use_at_test_time(self) -> bool:
        """
        Flag to indicate if the head should be used not only for training and validation, but also during rollouts
        (test time = evaluation = rollouts).
        """
        raise NotImplementedError(
            "Property `use_at_test_time` must be implemented in child class"
        )

    @property
    def use_at_train_time(self) -> bool:
        """
        Flag to indicate if the head should be used during training and validation
        By default it is set to true
        """
        return True

    @property
    @abstractmethod
    def collapse_sequence(self) -> bool:
        """
        Flag to indicate if the head model expects the inputs to be collapsed along the sequence dimension.
        If True, the inputs are expected to be of shape (batch_size, in_dim) and not (batch_size, seq_len, in_dim).
        """
        raise NotImplementedError(
            "Property `collapse_sequence` must be implemented in child class"
        )

    @property
    @abstractmethod
    def is_recurrent(self) -> bool:
        """
        Flag to indicate if the head model is recurrent. If True, the head model expects the inputs to be of shape
        (batch_size, seq_len, in_dim) and not (batch_size, in_dim).
        """
        raise NotImplementedError(
            "Property `is_recurrent` must be implemented in child class"
        )

    @abstractmethod
    def get_input_keys(self) -> List[str]:
        """
        Returns the names of expected inputs for the model
        :return: list of expected input keys to be passed to the model.
        """
        raise NotImplementedError(
            "This method must be implemented in the child classes to return the expected inputs for the model."
        )

    @abstractmethod
    def get_output_keys(self) -> List[str]:
        """
        Returns the names of outputs for the model.
        :return: list of output keys to be returned by the model.
        """
        raise NotImplementedError(
            "This method must be implemented in the child classes to return the output keys for the model."
        )

    @abstractmethod
    def get_target_key(self) -> str:
        """
        Returns the name of the target for the model.
        :return: expected target key to be passed to the model.
        """
        raise NotImplementedError(
            "This method must be implemented in the child classes to return the expected targets for the model."
        )

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the head model, typically used to reset the internal state of the model.
        """
        raise NotImplementedError("Method `reset` must be implemented in child class")

    def get_input_tensor(
        self, inputs: Dict[str, Tensor], detach_inputs: Set[str] | None = None
    ) -> Tensor:
        """
        Returns the input tensor for the head.
        :param inputs: dictionary of input tensors.
        :param detach_inputs: list of input keys that should be detached before passing to the head.
        :return: Single tensor concatenated from the expected inputs of shape (batch_size, seq_len, in_dim).
        """
        input_keys = self.get_input_keys()
        assert all(
            key in inputs for key in input_keys
        ), f"Expected inputs {input_keys} but received {list(inputs.keys())}"

        if detach_inputs is None:
            detach_inputs = set()

        inputs = {
            k: inputs[k].detach() if k in detach_inputs else inputs[k]
            for k in input_keys
        }
        sequence_inputs = [t for k, t in inputs.items() if is_sequence_key(k)]
        non_sequence_inputs = [
            t.unsqueeze(1) if t.ndim < 3 else t
            for k, t in inputs.items()
            if not is_sequence_key(k)
        ]
        if sequence_inputs:
            batch_size, seq_len, _ = sequence_inputs[0].shape
        else:
            batch_size = non_sequence_inputs[0].shape[0]
            seq_len = 1

        if non_sequence_inputs:
            if self.collapse_sequence:
                # If the model collapses sequences, we expect only one input for each non-sequence input,
                # so we need to combine the sequence dimension for sequence inputs and concatenate after
                sequence_inputs = [
                    t.reshape(batch_size, 1, -1) for t in sequence_inputs
                ]
            else:
                # If the model expects sequences, we need a sequence for the non-sequence inputs as well
                # so we repeat them along the sequence dimension
                non_sequence_inputs = [
                    t.repeat(1, seq_len, 1) for t in non_sequence_inputs
                ]
        input_tensors = sequence_inputs + non_sequence_inputs
        assert inputs, "No inputs provided to the head"
        if len(input_tensors) > 1:
            return torch.cat(input_tensors, dim=-1)
        return input_tensors[0]

    @abstractmethod
    def forward(
        self, inputs: Dict[str, Tensor], **kwargs
    ) -> Dict[str, Tensor | Tuple[Tensor, Tensor]]:
        """
        Forward pass through the head models. Gets any available input tensors, constructs its inputs
        and returns all outputs as a dictionary mapping output keys to tensors (or a pair of tensors)
        """
        raise NotImplementedError("Method `forward` must be implemented in child class")

    @abstractmethod
    def compute_loss(
        self,
        predicted: Dict[str, Tensor | Tuple[Tensor, Tensor]],
        target: Tensor,
        training: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Computes the loss for the head and return it together with a dict of intermediate losses we want to log.
        """
        raise NotImplementedError(
            "Method `compute_loss` must be implemented in child class"
        )

    @property
    def loss_weight(self) -> float:
        """
        Weight to scale the head loss before adding it to the total loss of the model.
        """
        return 1.0

    def _get_prefix(self, training: bool) -> str:
        return TRAIN_PREFIX if training else VALIDATION_PREFIX
