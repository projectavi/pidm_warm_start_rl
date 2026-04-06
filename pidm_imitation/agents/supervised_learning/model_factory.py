# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, Set, Tuple

from pidm_imitation.agents.supervised_learning.base_models import ActionRegressor
from pidm_imitation.agents.supervised_learning.dataset.datamodule import DataModule
from pidm_imitation.agents.supervised_learning.extract_args_utils import (
    ExtractArgsFromConfig,
    ExtractArgsFromDataModule,
)
from pidm_imitation.agents.supervised_learning.inputs_factory import InputsFactory
from pidm_imitation.agents.models.ssidm import SSIDMPolicyNetwork
from pidm_imitation.agents.supervised_learning.single_head_model import (
    SingleHeadActionRegressor,
)
from pidm_imitation.agents.supervised_learning.submodel_factories import (
    PolicyHeadFactory,
    StateEncoderFactory,
)
from pidm_imitation.agents.supervised_learning.submodels.state_encoder_model import (
    StateEncoderModel,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import (
    ACTION_LOOKAHEAD_KEY,
    POLICY_HEAD_KEY,
    STATE_ENCODER_MODEL_KEY,
)
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.valid_models import ValidModels

log = Logger().get_root_logger()


class ModelFactory:
    @staticmethod
    def _is_ssidm_algorithm(alg: str) -> bool:
        return alg in [ValidModels.PSSIDM, ValidModels.LSSIDM]

    @staticmethod
    def _get_base_kwargs_and_dims(
        config: OfflinePLConfigFile, datamodule: DataModule
    ) -> Tuple[Dict[str, Any], int, int | None, bool]:
        model_config = ExtractArgsFromConfig.get_model_config(config)
        submodel_names = ExtractArgsFromConfig.get_submodel_names_from_model_config(
            model_config
        )

        state_dim = ExtractArgsFromDataModule.get_state_dim(datamodule)

        state_encoder_model: StateEncoderModel | None = None
        state_encoder_dim: int | None = None
        state_encoder_collapse_sequence: bool = False
        if STATE_ENCODER_MODEL_KEY in submodel_names:
            state_encoder_model = StateEncoderFactory.get_state_encoder(
                config=config,
                state_dim=state_dim,
            )
            state_encoder_dim = state_encoder_model.out_dim
            state_encoder_collapse_sequence = state_encoder_model.collapse_sequence

        assert (
            POLICY_HEAD_KEY in submodel_names
        ), f"Policy head must be specified in the config under '{POLICY_HEAD_KEY}' for {model_config.algorithm}."
        policy_head = PolicyHeadFactory.get_policy_head(
            config=config,
            state_dim=state_dim,
            state_encoder_dim=state_encoder_dim,
            state_encoder_collapse_sequence=state_encoder_collapse_sequence,
        )

        return (
            {
                "state_encoder_model": state_encoder_model,
                "policy_head": policy_head,
                **ExtractArgsFromConfig.get_optimizer_kwargs(config),
                **ExtractArgsFromConfig.get_scheduler_kwargs(config),
                **ExtractArgsFromConfig.get_model_init_args(config),
            },
            state_dim,
            state_encoder_dim,
            state_encoder_collapse_sequence,
        )

    @staticmethod
    def _get_singlehead_kwargs(
        config: OfflinePLConfigFile, datamodule: DataModule
    ) -> Dict[str, Any]:
        algorithm_kwargs, _, _, _ = ModelFactory._get_base_kwargs_and_dims(
            config=config,
            datamodule=datamodule,
        )
        return algorithm_kwargs

    @staticmethod
    def _get_singlehead_bc_model_kwargs(
        config: OfflinePLConfigFile, datamodule: DataModule
    ) -> Dict[str, Any]:
        return ModelFactory._get_singlehead_kwargs(config, datamodule)

    @staticmethod
    def _get_singlehead_idm_model_kwargs(
        config: OfflinePLConfigFile, datamodule: DataModule
    ) -> Dict[str, Any]:
        return ModelFactory._get_singlehead_kwargs(config, datamodule)

    @staticmethod
    def get_model_class(alg: str) -> type[ActionRegressor]:
        """
        Returns the class of the action regressor based on the algorithm specified in the config.
        """
        if alg == ValidModels.BC:
            return SingleHeadActionRegressor
        if alg == ValidModels.IDM:
            return SingleHeadActionRegressor
        if ModelFactory._is_ssidm_algorithm(alg):
            return SingleHeadActionRegressor
        raise ValueError(
            f"Unsupported algorithm '{alg}'. Supported algorithms are {ValidModels.ALL}."
        )

    @staticmethod
    def _add_input_keys(config: OfflinePLConfigFile):
        inputs_dict = InputsFactory.get_inputs(config)
        global_inputs = inputs_dict.pop("global", None)
        if global_inputs:
            key, value = global_inputs
            config.model_config.init_args[key] = value
        for submodel, (key, value) in inputs_dict.items():
            config.model_config.submodel_configs[submodel]["init_args"][key] = value

    @staticmethod
    def get_model_kwargs(
        alg: str, config: OfflinePLConfigFile, datamodule: DataModule
    ) -> Dict[str, Any]:
        """
        Returns the base kwargs for the action regressor based on the algorithm specified in the config.
        """
        ModelFactory._add_input_keys(config)

        # Singlehead architecture
        if alg in [ValidModels.BC, ValidModels.IDM] or ModelFactory._is_ssidm_algorithm(
            alg
        ):
            return ModelFactory._get_singlehead_kwargs(config, datamodule)

        raise ValueError(
            f"Unsupported algorithm '{alg}'. Supported algorithms are {ValidModels.ALL}."
        )

    @staticmethod
    def _get_bc_required_and_valid_submodels() -> Tuple[Set[str], Set[str]]:
        return {POLICY_HEAD_KEY}, {POLICY_HEAD_KEY, STATE_ENCODER_MODEL_KEY}

    @staticmethod
    def _get_idm_required_and_valid_submodels() -> Tuple[Set[str], Set[str]]:
        return {POLICY_HEAD_KEY}, {POLICY_HEAD_KEY, STATE_ENCODER_MODEL_KEY}

    @staticmethod
    def _get_ssidm_required_and_valid_submodels() -> Tuple[Set[str], Set[str]]:
        return {POLICY_HEAD_KEY}, {POLICY_HEAD_KEY}

    @staticmethod
    def assert_required_and_valid_submodels(config: OfflinePLConfigFile):
        algorithm_name = ExtractArgsFromConfig.get_algorithm(config)
        model_config = ExtractArgsFromConfig.get_model_config(config)
        submodel_names = set(
            ExtractArgsFromConfig.get_submodel_names_from_model_config(model_config)
        )
        if algorithm_name == ValidModels.BC:
            required, valid = ModelFactory._get_bc_required_and_valid_submodels()
        elif algorithm_name == ValidModels.IDM:
            required, valid = ModelFactory._get_idm_required_and_valid_submodels()
        elif ModelFactory._is_ssidm_algorithm(algorithm_name):
            required, valid = ModelFactory._get_ssidm_required_and_valid_submodels()
        else:
            raise ValueError(
                f"Unsupported algorithm '{algorithm_name}'. Supported algorithms are {ValidModels.ALL}."
            )
        assert required.issubset(submodel_names), (
            f"Model configuration for {algorithm_name} must contain the following submodels: {required}. "
            f"Got {submodel_names}."
        )
        assert submodel_names.issubset(valid), (
            f"Model configuration for {algorithm_name} contains unsupported submodels: {submodel_names - valid}. "
            f"Supported submodels are {valid}."
        )

    @staticmethod
    def assert_valid_model_datamodule(
        model: ActionRegressor, datamodule: DataModule
    ) -> None:
        """
        Asserts that the model and datamodule are compatible.
        Checks:
        1. No leakage of action targets through action lookahead if given to model.
        """
        if ACTION_LOOKAHEAD_KEY in model.get_input_keys():
            # to avoid leaking action targets through action lookahead, the first action lookahead input
            # must be after the last action target
            history = datamodule.history
            min_lookahead = min(datamodule.lookahead_slice)
            # index `history` in sequence is current timestep --> last part of history input and index
            # of last target action
            # `min_lookahead` + 1 is the first lookahead input index
            # --> min_lookahead >= history means that the first lookahead input is after the last action target
            assert min_lookahead >= history, (
                f"Action lookahead input {ACTION_LOOKAHEAD_KEY} is given to model but the first lookahead input "
                f"is at or before the last action target for given history {history} and lookahead slice "
                f"{datamodule.lookahead_slice}. This can leak action targets into the model input! Please ensure that "
                "the first lookahead input is after the last action target."
            )

        if isinstance(model.policy_head.policy_model, SSIDMPolicyNetwork):
            assert (
                datamodule.lookahead
            ), "PSSIDM/LSSIDM require lookahead data, but datamodule lookahead is disabled."
            assert len(datamodule.lookahead_slice) == 1, (
                "PSSIDM/LSSIDM currently require a single fixed lookahead slice. "
                f"Got {datamodule.lookahead_slice}."
            )
            assert datamodule.lookahead_slice[0] == 0, (
                "PSSIDM/LSSIDM scaffold expects next-state lookahead semantics "
                "(repo lookahead_k = 0). "
                f"Got {datamodule.lookahead_slice}."
            )
            assert not datamodule.include_k, (
                "PSSIDM/LSSIDM scaffold expects include_k = False because the "
                "strict fixed-horizon core should not consume lookahead_k inputs."
            )

    @staticmethod
    def get_model(
        config: OfflinePLConfigFile, datamodule: DataModule
    ) -> ActionRegressor:
        ModelFactory.assert_required_and_valid_submodels(config)

        alg = ExtractArgsFromConfig.get_algorithm(config)
        model_class = ModelFactory.get_model_class(alg)
        model_kwargs = ModelFactory.get_model_kwargs(alg, config, datamodule)
        model = model_class(**model_kwargs)

        ModelFactory.assert_valid_model_datamodule(model, datamodule)

        return model
