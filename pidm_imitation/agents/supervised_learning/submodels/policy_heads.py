# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Set, Tuple

from torch import Tensor

from pidm_imitation.agents.models.policy_models import PolicyNetwork
from pidm_imitation.agents.models.utils import get_collapse_sequence, is_recurrent
from pidm_imitation.agents.supervised_learning.submodels.base_head import Head
from pidm_imitation.agents.supervised_learning.utils.action_loss import ActionLoss
from pidm_imitation.constants import ACTION_TARGET_KEY, PREDICTED_ACTION_KEY

PREDICTED_ACTION_TYPE = Tensor | tuple[Tensor, Tensor]
POLICY_MODEL_OUTPUT_TYPE = PREDICTED_ACTION_TYPE | dict[str, PREDICTED_ACTION_TYPE]


class PolicyHeadBase(Head):
    def __init__(
        self,
        policy_model: PolicyNetwork,
        action_loss: ActionLoss,
        use_at_train_time: bool = True,
        use_at_test_time: bool = True,
    ):
        """
        Base class for policy head models. Children classes must implement the forward method.
        :param policy_model: model that takes the output of the encoder and predicts the action logits.
        :param action_loss: Action loss function used to compute the loss for the model. This is an instance of
            the ActionLoss class that defines how to compute the loss for the actions predicted by the model.
        :param use_at_test_time: flag to indicate if the head should be used not only for training and validation,
            but also at test time.
        """
        super().__init__()
        self.policy_model = policy_model
        self._loss_fn = action_loss

        self._use_at_train_time = use_at_train_time
        self._use_at_test_time = use_at_test_time

        self.expected_inputs = self.get_input_keys()

    @property
    def use_at_train_time(self) -> bool:
        if not hasattr(self, "_use_at_train_time"):
            self._use_at_train_time = True
        return self._use_at_train_time

    @property
    def use_at_test_time(self) -> bool:
        if not hasattr(self, "_use_at_test_time"):
            self._use_at_test_time = True
        return self._use_at_test_time

    @property
    def collapse_sequence(self) -> bool:
        return get_collapse_sequence(self.policy_model)

    @property
    def is_recurrent(self) -> bool:
        return is_recurrent(self.policy_model)

    def get_output_keys(self) -> List[str]:
        """
        Returns the names of outputs for the model.
        :return: list of output keys to be returned by the model.
        """
        return [PREDICTED_ACTION_KEY]

    def get_target_key(self) -> str:
        return ACTION_TARGET_KEY

    def reset(self) -> None:
        self.policy_model.reset()

    def compute_loss(
        self,
        predicted: Dict[str, Tensor | Tuple[Tensor, Tensor]],
        target: Tensor,
        training: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Computes the loss for the policy head and return intermediate continuous and binary losses, as well as their
        weighted versions.
        """
        assert (
            self.use_at_train_time
        ), "Training and loss computation is not supported for this policy head."
        prefix = self._get_prefix(training)
        loss_dict = {}

        action_prediction = predicted[PREDICTED_ACTION_KEY]
        loss = self._loss_fn(action_prediction, target)
        loss_dict[f"{prefix}_policy_loss"] = loss
        return loss, loss_dict

    def _training_first_step(self, checkpoint_folder: str | None = None) -> None:
        """
        Method called at the first training step to initialize the head. It can be used to set up any necessary
        resources or configurations.
        :param checkpoint_folder: folder where the checkpoints are saved, if available.
        """
        self._checkpoint_folder = checkpoint_folder

    def _forward_policy_model(
        self,
        inputs: Dict[str, Tensor],
        detach_inputs: Set[str] = set(),
    ) -> Dict[str, Tensor | Tuple[Tensor, Tensor]]:
        """
        Forward pass through the policy model.
        :param inputs: dictionary of input tensors to the policy model.
        :param detach_inputs: list of input keys that should be detached before passing to the policy model.
        :return: dictionary with the predicted action or actions at key PREDICTED_ACTION_KEY.
            Can also return additional outputs at other keys, depending on the policy model implementation.
        """
        policy_input = self.get_input_tensor(inputs, detach_inputs)
        policy_output = self.policy_model(policy_input)
        return {PREDICTED_ACTION_KEY: policy_output}

    def forward(
        self,
        inputs: Dict[str, Tensor],
        detach_inputs: Set[str] = set(),
        **kwargs,
    ) -> Dict[str, Tensor | tuple[Tensor, Tensor]]:
        return self._forward_policy_model(inputs, detach_inputs=detach_inputs)


class PolicyHead(PolicyHeadBase):
    """
    Policy head that expects the expected inputs to be specified.
    Example inputs are:
      - ["history_emb"] --> only history embedding of encoder model goes into policy (e.g. BC with encoder)
      - ["history_emb", "action_history"] --> history embedding of encoder model and raw action history
      - ["history_emb", "lookahead_emb"] --> history and lookahead embedding of encoder model (e.g. IDM with encoder)
      - ["history_emb", "lookahead_emb", "lookahead_k_onehot"] --> history, lookahead embeddings of encoder model and
        one-hot encoding of sampled IDM lookahead k
    """

    def __init__(
        self,
        policy_model: PolicyNetwork,
        action_loss: ActionLoss,
        input_keys: List[str],
        use_at_train_time: bool = True,
        use_at_test_time: bool = True,
    ):
        """
        Create policy head for BC.
        :param policy_model: policy network definition
        :param action_loss: policy loss definition
        :param policy_inputs: list of inputs that can be keys of data samples (see constants.py) or outputs
            of the encoder model or other predictions (see constants.py)
        :param use_at_test_time: whether to use this head at test time
        """
        self.input_keys = input_keys
        super().__init__(policy_model, action_loss, use_at_train_time, use_at_test_time)

    def get_input_keys(self) -> List[str]:
        return self.input_keys
