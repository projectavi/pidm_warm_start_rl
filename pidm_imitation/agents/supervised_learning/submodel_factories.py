# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List

from torch import nn

from pidm_imitation.agents.models.network_block import NetworkBlock
from pidm_imitation.agents.models.policy_models import PolicyNetwork
from pidm_imitation.agents.models.ssidm import SSIDMPolicyNetwork
from pidm_imitation.agents.supervised_learning.base_models import Head
from pidm_imitation.agents.supervised_learning.extract_args_utils import (
    ExtractArgsFromConfig,
)
from pidm_imitation.agents.supervised_learning.submodels.policy_heads import PolicyHead
from pidm_imitation.agents.supervised_learning.submodels.state_encoder_model import (
    StateEncoderModel,
)
from pidm_imitation.agents.supervised_learning.utils.action_loss import ActionLoss
from pidm_imitation.agents.supervised_learning.utils.utils import get_total_dim
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.utils.valid_models import ValidModels


class BaseModelFactory:
    NETWORK_BLOCK = "NetworkBlock"
    VALID_BASE_MODELS = [NETWORK_BLOCK]

    @staticmethod
    def _get_base_model_class(class_name: str) -> type:
        if class_name == BaseModelFactory.NETWORK_BLOCK:
            return NetworkBlock
        raise ValueError(
            f"Invalid base model {class_name}, must be one of {BaseModelFactory.VALID_BASE_MODELS}"
        )

    @staticmethod
    def _get_input_dim(
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        collapse_sequence: bool = False,
        state_encoder_dim: int = None,
        state_encoder_collapse_sequence: bool = False,
    ) -> int:
        action_dim = ExtractArgsFromConfig.get_action_dim(config)
        sequence_length = ExtractArgsFromConfig.get_sequence_length(config)
        num_lookaheads = ExtractArgsFromConfig.get_num_lookahead_samples(config)

        return get_total_dim(
            input_keys,
            state_dim,
            action_dim,
            sequence_length,
            num_lookaheads,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
            collapse_sequence=collapse_sequence,
        )

    @staticmethod
    def _get_network_block_kwargs(
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        state_encoder_dim: int = None,
        state_encoder_collapse_sequence: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        input_dim = BaseModelFactory._get_input_dim(
            config=config,
            input_keys=input_keys,
            state_dim=state_dim,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
            collapse_sequence=kwargs.get("collapse_sequence", False),
        )
        return {
            "input_dim": input_dim,
            **kwargs,  # type: ignore
        }

    @staticmethod
    def _get_base_model_kwargs(
        class_name: str,
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        state_encoder_dim: int = None,
        state_encoder_collapse_sequence: bool = False,
        **kwargs: Any,
    ):
        if class_name == BaseModelFactory.NETWORK_BLOCK:
            return BaseModelFactory._get_network_block_kwargs(
                config=config,
                input_keys=input_keys,
                state_dim=state_dim,
                state_encoder_dim=state_encoder_dim,
                state_encoder_collapse_sequence=state_encoder_collapse_sequence,
                **kwargs,
            )
        raise ValueError(
            f"Unknown base model: {class_name}, must be one of {BaseModelFactory.VALID_BASE_MODELS}"
        )

    @staticmethod
    def get_base_model(
        class_name: str,
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        state_encoder_dim: int = None,
        state_encoder_collapse_sequence: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        model_class = BaseModelFactory._get_base_model_class(class_name)
        model_kwargs = BaseModelFactory._get_base_model_kwargs(
            class_name=class_name,
            config=config,
            input_keys=input_keys,
            state_dim=state_dim,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
            **kwargs,
        )
        return model_class(**model_kwargs)


class StateEncoderFactory:
    """Factory class to create state encoder models."""

    STATE_ENCODER = "StateEncoder"
    VALID_STATE_ENCODER_MODELS = [STATE_ENCODER]

    @staticmethod
    def _get_state_encoder_class(class_name: str) -> type:
        if class_name == StateEncoderFactory.STATE_ENCODER:
            return StateEncoderModel
        raise ValueError(
            f"Invalid state encoder model {class_name}, must be one of {StateEncoderFactory.VALID_STATE_ENCODER_MODELS}"
        )

    @staticmethod
    def _get_state_encoder_kwargs(
        class_name: str,
        **kwargs: dict,
    ) -> Dict[str, Any]:
        if class_name == StateEncoderFactory.STATE_ENCODER:
            return {**kwargs}
        raise ValueError(
            f"Unknown state encoder model: {class_name}, needs to be one of "
            f"{StateEncoderFactory.VALID_STATE_ENCODER_MODELS}"
        )

    @staticmethod
    def get_state_encoder(
        config: OfflinePLConfigFile,
        state_dim: int,
    ) -> StateEncoderModel:
        assert ExtractArgsFromConfig.has_state_encoder_submodel(
            config
        ), "State encoder submodel is not defined in the config."
        state_encoder_name = ExtractArgsFromConfig.get_state_encoder_name(config)
        state_encoder_class = StateEncoderFactory._get_state_encoder_class(
            state_encoder_name
        )
        state_encoder_kwargs = (
            ExtractArgsFromConfig.get_state_encoder_init_args_without_encoder_model(
                config
            )
        )
        model_name = ExtractArgsFromConfig.get_state_encoder_encoder_model_name(config)
        model_kwargs = ExtractArgsFromConfig.get_state_encoder_encoder_model_init_args(
            config
        )
        input_groups = state_encoder_kwargs["input_groups"]
        input_keys = list(input_groups.values())[0]  # type: ignore
        encoder_model = BaseModelFactory.get_base_model(
            class_name=model_name,
            config=config,
            input_keys=input_keys,
            state_dim=state_dim,
            **model_kwargs,
        )
        return state_encoder_class(model=encoder_model, **state_encoder_kwargs)


class PolicyModelFactory:
    """
    This factory creates the policy model that is needed when creating the policy head (below).
    """

    POLICY_NETWORK = "PolicyNetwork"
    SSIDM_POLICY_NETWORK = "SSIDMPolicyNetwork"
    VALID_POLICY_MODELS = [POLICY_NETWORK, SSIDM_POLICY_NETWORK]

    @staticmethod
    def _get_policy_model_class(class_name: str) -> type:
        if class_name == PolicyModelFactory.POLICY_NETWORK:
            return PolicyNetwork
        if class_name == PolicyModelFactory.SSIDM_POLICY_NETWORK:
            return SSIDMPolicyNetwork
        raise ValueError(
            f"Invalid policy model {class_name}, must be one of {PolicyModelFactory.VALID_POLICY_MODELS}"
        )

    @staticmethod
    def _get_policy_model_args(
        class_name: str,
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        state_encoder_dim: int | None,
        state_encoder_collapse_sequence: bool = False,
    ) -> Dict[str, Any]:
        if class_name == PolicyModelFactory.POLICY_NETWORK:
            action_type = ExtractArgsFromConfig.get_action_type(config)

            # need to build the base model for the policy network wrapper
            policy_model_kwargs = (
                ExtractArgsFromConfig.get_policy_head_policy_model_init_args(config)
            )
            policy_base_model_config = policy_model_kwargs["base_model"]
            policy_base_model_name = policy_base_model_config["class_name"]
            policy_base_model_kwargs = policy_base_model_config["init_args"]
            base_model = BaseModelFactory.get_base_model(
                class_name=policy_base_model_name,
                config=config,
                input_keys=input_keys,
                state_dim=state_dim,
                state_encoder_dim=state_encoder_dim,
                state_encoder_collapse_sequence=state_encoder_collapse_sequence,
                **policy_base_model_kwargs,
            )
            return {
                "base_model": base_model,
                "action_type": action_type,
            }
        if class_name == PolicyModelFactory.SSIDM_POLICY_NETWORK:
            input_dim = BaseModelFactory._get_input_dim(
                config=config,
                input_keys=input_keys,
                state_dim=state_dim,
                state_encoder_dim=state_encoder_dim,
                state_encoder_collapse_sequence=state_encoder_collapse_sequence,
                collapse_sequence=False,
            )
            policy_model_kwargs = (
                ExtractArgsFromConfig.get_policy_head_policy_model_init_args(config)
            )
            return {
                **policy_model_kwargs,
                "input_dim": input_dim,
                "state_dim": state_dim,
                "action_type": ExtractArgsFromConfig.get_action_type(config),
                "use_latent_encoder": (
                    ExtractArgsFromConfig.get_algorithm(config) == ValidModels.LSSIDM
                ),
            }
        raise ValueError(
            f"Unknown policy model: {class_name}, must be one of {PolicyModelFactory.VALID_POLICY_MODELS}"
        )

    @staticmethod
    def get_policy_model(
        config: OfflinePLConfigFile,
        input_keys: List[str],
        state_dim: int,
        state_encoder_dim: int | None,
        state_encoder_collapse_sequence: bool = False,
    ) -> nn.Module:
        policy_model_name = ExtractArgsFromConfig.get_policy_head_policy_model_name(
            config
        )
        policy_model_class = PolicyModelFactory._get_policy_model_class(
            policy_model_name
        )
        policy_model_kwargs = PolicyModelFactory._get_policy_model_args(
            class_name=policy_model_name,
            config=config,
            input_keys=input_keys,
            state_dim=state_dim,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
        )
        return policy_model_class(**policy_model_kwargs)


class PolicyHeadFactory:
    """
    This factory creates the policy head. It uses MultiheadPolicyModelFactory to create the policy model.
    """

    POLICY_HEAD = "PolicyHead"
    VALID_POLICY_HEADS = [POLICY_HEAD]

    @staticmethod
    def _get_policy_head_class(class_name: str) -> type:
        if class_name == PolicyHeadFactory.POLICY_HEAD:
            return PolicyHead
        raise ValueError(
            f"Unknown policy head: {class_name}, must be one of {PolicyHeadFactory.VALID_POLICY_HEADS}"
        )

    @staticmethod
    def _get_policy_head_base_args(
        config: OfflinePLConfigFile,
    ) -> Dict[str, Any]:
        head_args = (
            ExtractArgsFromConfig.get_policy_head_init_args_without_policy_model(config)
        )
        loss_args = head_args.get("action_loss", {})
        policy_head_kwargs = {
            k: v for k, v in head_args.items() if k not in ["action_loss"]
        }
        return {
            "action_loss": ActionLoss(
                action_type=ExtractArgsFromConfig.get_action_type(config),
                **loss_args,  # type: ignore
            ),
            **policy_head_kwargs,  # type: ignore
        }

    @staticmethod
    def _get_policy_head_args(
        class_name: str,
        config: OfflinePLConfigFile,
        state_dim: int,
        state_encoder_dim: int | None,
        state_encoder_collapse_sequence: bool = False,
    ) -> Dict[str, Any]:
        head_args = PolicyHeadFactory._get_policy_head_base_args(config)
        if class_name == PolicyHeadFactory.POLICY_HEAD:
            policy_model = PolicyModelFactory.get_policy_model(
                config=config,
                input_keys=head_args["input_keys"],
                state_dim=state_dim,
                state_encoder_dim=state_encoder_dim,
                state_encoder_collapse_sequence=state_encoder_collapse_sequence,
            )
            return {
                "policy_model": policy_model,
                **head_args,  # type: ignore
            }
        return head_args

    @staticmethod
    def get_policy_head(
        config: OfflinePLConfigFile,
        state_dim: int,
        state_encoder_dim: int | None,
        state_encoder_collapse_sequence: bool = False,
    ) -> Head:
        head_name = ExtractArgsFromConfig.get_policy_head_name(config)
        head_class = PolicyHeadFactory._get_policy_head_class(head_name)
        head_args = PolicyHeadFactory._get_policy_head_args(
            class_name=head_name,
            config=config,
            state_dim=state_dim,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
        )
        return head_class(**head_args)
