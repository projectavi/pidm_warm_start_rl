# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import Dict, List

import torch
from torch import Tensor, nn

from pidm_imitation.agents.models.utils import (
    get_collapse_sequence,
    get_output_dim,
    is_recurrent,
    reset_model,
)
from pidm_imitation.agents.supervised_learning.submodels.submodel import SubModel


class StateEncoderModel(SubModel):
    """Network to encode history or lookahead of states and/ or actions to embeddings."""

    def __init__(
        self,
        model: nn.Module,
        input_groups: Dict[str, List[str]],
    ):
        """
        Wrapper for state encoder model to encode (sequences of) states and/ or actions
        :param model: The state encoder model to be wrapped.
        :param input_groups: Dictionary mapping group names to lists of input keys. Each group corresponds
            to a list of input keys that will be concatenated together before being passed to the model
            Can be used with sequence model state encoder to encode a sequence of states but only return the
            last embedding that encodes the entire sequence for following models.

            Example for BC with state and action inputs: {"history": ["state_history", "action_history"]}
            Example for BC with state only inputs: {"history": ["state_history"]}
            Example for IDM with state and action inputs:
            {
                "history": ["state_history", "action_history"],
                "lookahead": ["state_lookahead", "action_lookahead"]
            }
        """
        super().__init__()
        self.model = model
        self.input_groups = OrderedDict(input_groups)

    def get_input_keys(self) -> List[str]:
        """
        Get the names of the inputs required by the model.
        :return: List of input keys.
        """
        return [
            input_key
            for input_group_keys in self.input_groups.values()
            for input_key in input_group_keys
        ]

    def _build_batch(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Build batch for processing. The tensors within each group are concatenated along the last dimension.
        Concatenated tensors across groups are then concatenated along the batch size dimension for parallel processing.

        Each tensor in each input list should be either a 2D tensor (batch_size, dim) or a 3D tensor
        (batch_size, seq_len, dim). Reshape all tensors to 3D before concatenation.

        :param group_inputs: Dictionary mapping group names to lists of tensors to be concatenated. Each tensor can be
            (batch_size, dim), or (batch_size, seq_len, dim).
        :return: Tensor with shape (batch_size * (num_groups), seq_len, total_dim), where total_dim
            is the sum of the dimensions of all input tensors within each group
        """
        group_inputs = {
            group_name: [inputs[key] for key in group_inputs]
            for group_name, group_inputs in self.input_groups.items()
        }

        seq_len = None
        in_dim = None

        all_tensors = []

        for group_name, tensors in group_inputs.items():
            group_in_dim = 0
            group_tensors = []
            for tensor in tensors:
                assert tensor.ndim in (2, 3), (
                    f"Input '{group_name}' has unexpected shape {tensor.shape}. "
                    "Expected 2D (batch_size, dim) or 3D (batch_size, seq_len, dim) tensor"
                )
                if tensor.ndim == 2:
                    # add sequence dimension
                    tensor = tensor.unsqueeze(1)

                seq_len = tensor.size(1) if seq_len is None else seq_len
                group_in_dim += tensor.size(-1)

                group_tensors.append(tensor)

            group_tensor = torch.cat(group_tensors, dim=-1)
            assert seq_len == group_tensor.size(
                1
            ), f"Input '{group_name}' has sequence length {group_tensor.size(1)} but expected {seq_len}."
            if in_dim is None:
                in_dim = group_in_dim
            else:
                assert (
                    in_dim == group_in_dim
                ), f"Input '{group_name}' has input dimension {group_in_dim} but expected {in_dim}."

            all_tensors.append(group_tensor)

        batch_tensor = torch.cat(all_tensors, dim=0)
        assert batch_tensor.ndim == 3, (
            f"Concatenated tensor has unexpected shape {batch_tensor.shape}. "
            "Expected 3D tensor (batch_size * num_groups, seq_len, total_dim)."
        )
        return batch_tensor

    def _split_batch(self, batch: Tensor) -> Dict[str, Tensor]:
        """
        Split a batch tensor into a dictionary of tensors based on group names.
        Each group name corresponds to a tensor in the batch, which is split along the first dimension.
        The batch tensor is expected to have shape (batch_size * num_groups, seq_len, num_patches, total_dim).
        :param batch: A tensor with shape (batch_size * num_groups, seq_len, num_patches, total_dim).
        :param group_names: A list of group names corresponding to the tensors in the batch.
        :return: A dictionary mapping group names to tensors with shape (batch_size, seq_len, num_patches, dim).
        """
        assert batch.ndim == 3, (
            f"Batch tensor has unexpected shape {batch.shape}. "
            "Expected 3D tensor (batch_size * num_groups, seq_len, total_dim)."
        )

        group_names = list(self.input_groups.keys())
        batch_size = batch.size(0) // len(group_names)
        assert (
            batch.size(0) % len(group_names) == 0
        ), f"Batch size {batch.size(0)} is not divisible by the number of groups {len(group_names)}."

        group_tensors = torch.split(batch, batch_size, dim=0)
        return {
            group_name: group_tensor
            for group_name, group_tensor in zip(group_names, group_tensors)
        }

    def get_embeddings(
        self,
        inputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute the embeddings for the state encoder model. First, the given inputs are grouped by the list of input
        keys defined in input_groups. Then, the grouped inputs are concatenated along the last dimension and passed
        through the state encoder model. The outputs are then split by the input groups and returned as a dictionary
        mapping group names to the corresponding embeddings.
        :param inputs: dictionary of inputs mapping input name to tensors.
        :return: A dictionary mapping input group names to tensor with the embeddings from the state encoder model of
            shape (batch_size, seq_len, out_dim).
        """
        batch_inputs = self._build_batch(inputs)
        encoder_embeddings = self.model(batch_inputs)
        return self._split_batch(encoder_embeddings)

    def get_embeddings_from_tensor(self, tensors: List[Tensor]) -> Tensor:
        """
        Compute embeddings by concatenating a list of input tensors and forwarding through the model.

        This relaxed helper skips key/group checks. It's the caller's responsibility to pass
        tensors with compatible batch and sequence dimensions expected by the encoder model.

        Args:
            tensors: List of tensors, each shaped either [batch, dim] or [batch, seq, dim].

        Returns:
            Tensor of shape [batch, seq, out_dim] with the embeddings.
        """
        assert len(tensors) > 0, "Expected at least one tensor to encode."

        seq_len = None
        prepared: List[Tensor] = []
        for t in tensors:
            assert t.ndim in (2, 3), (
                f"Input has unexpected shape {t.shape}. "
                "Expected 2D (batch, dim) or 3D (batch, seq, dim)."
            )
            if t.ndim == 2:
                t = t.unsqueeze(1)
            seq_len = t.size(1) if seq_len is None else seq_len
            prepared.append(t)

        group_tensor = torch.cat(prepared, dim=-1)  # [batch, seq, total_dim]

        # Forward through the underlying model
        embeddings = self.model(group_tensor)
        return embeddings

    def reset(self):
        reset_model(self.model)

    @property
    def out_dim(self) -> int:
        return get_output_dim(self.model)

    @property
    def collapse_sequence(self) -> bool:
        return get_collapse_sequence(self.model)

    @property
    def is_recurrent(self) -> bool:
        return is_recurrent(self.model)
