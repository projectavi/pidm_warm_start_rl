# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List

import torch

from pidm_imitation.agents.subconfig import IdmAgentConfig
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_planners import (
    ClosestReferencePlanner,
    IdmPlanner,
    ReferenceTrajectoryPlanner,
    ZeroPlanner,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_utils import (
    get_local_train_data_path,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.reference_trajectory_handlers import (
    ReferenceTrajectoryHandler,
    StateReferenceTrajectoryHandler,
)
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.configs.subconfig import ReferenceTrajectoryConfig
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.environment.utils import ValidIdmPlanners
from pidm_imitation.utils import StateType, Trajectory


class IdmPlannerFactory:

    @staticmethod
    def _get_planner_class(planner_type: str) -> type[IdmPlanner]:
        if planner_type == ValidIdmPlanners.DISABLED:
            return ZeroPlanner
        if planner_type == ValidIdmPlanners.REFERENCE_TRAJECTORY:
            return ReferenceTrajectoryPlanner
        if planner_type == ValidIdmPlanners.CLOSEST_REFERENCE:
            return ClosestReferencePlanner
        raise ValueError(
            f"Unknown planner type: {planner_type}, must be in {ValidIdmPlanners.ALL}"
        )

    @staticmethod
    def _get_idm_planner_base_kwargs(
        config: OfflinePLConfigFile,
        device: str | torch.device,
        train_lookahead_slice: List[int],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        idm_agent_config = config.get_subconfig_att(IdmAgentConfig)
        assert idm_agent_config, "IDM agent config not found in the config file."
        kwargs = idm_agent_config.planner_config.get_kwargs()
        kwargs.update(
            {
                "device": device,  # type: ignore
                "train_lookahead_slice": train_lookahead_slice,  # type: ignore
            }
        )
        return kwargs

    @staticmethod
    def _get_reference_trajectory_handler(
        state_type: StateType,
        action_type: str,
        reference_trajectory: Trajectory,
        **kwargs: Dict[str, Any],
    ) -> ReferenceTrajectoryHandler:
        assert (
            action_type is not None
        ), "action_type is required for reference trajectory handler."
        if state_type in [StateType.STATES, StateType.OBSERVATIONS]:
            return StateReferenceTrajectoryHandler(
                reference_trajectory=reference_trajectory,
                state_type=state_type,
                action_type=action_type,  # type: ignore
            )
        raise ValueError(
            f"Unknown state type {state_type} for reference trajectory handler, expected one of"
            f" {StateType.get_valid_state_types()}"
        )

    @staticmethod
    def _get_reference_trajectory_planner_kwargs(
        config: OfflinePLConfigFile,
        device: str | torch.device,
        train_lookahead_slice: List[int],
        **kwargs: Dict[str, Any],
    ):
        # extracting the reference trajectory from the config
        reference_trajectory_config = config.get_subconfig_att(
            ReferenceTrajectoryConfig
        )
        assert (
            reference_trajectory_config
        ), "Reference trajectory config not found in the config file."
        reference_trajectory = reference_trajectory_config.get_trajectory_obj(
            data_dir=config.data_config.training_dir
        )
        # creating the reference trajectory handler
        reference_trajectory_handler = (
            IdmPlannerFactory._get_reference_trajectory_handler(
                state_type=config.state_config.type,
                action_type=config.action_config.type,
                reference_trajectory=reference_trajectory,
                **kwargs,
            )
        )
        kwargs = IdmPlannerFactory._get_idm_planner_base_kwargs(
            config, device, train_lookahead_slice
        )
        kwargs.update(
            {
                "reference_trajectory_handler": reference_trajectory_handler,  # type: ignore
            }
        )
        return kwargs

    @staticmethod
    def _get_closest_reference_planner_kwargs(
        config: OfflinePLConfigFile,
        device: str | torch.device,
        train_lookahead_slice: List[int],
        input_trajectories: dict,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        train_dir = config.data_config.training_dir
        train_dir_path = get_local_train_data_path(train_dir)

        # get names for trajectories used for training in dataset
        assert (
            "train" in input_trajectories
        ), "Input trajectories with 'train' key must be provided."
        train_traj_names = input_trajectories["train"]
        # load trajectories and create reference trajectory handlers to allow for easy access to states
        train_trajectories = [
            ToyEnvironmentTrajectory.init_from_dir(train_dir_path, traj_name)
            for traj_name in train_traj_names
        ]
        reference_trajectory_handlers = [
            IdmPlannerFactory._get_reference_trajectory_handler(
                state_type=config.state_config.type,
                action_type=config.action_config.type,
                reference_trajectory=trajectory,
                **kwargs,
            )
            for trajectory in train_trajectories
        ]
        kwargs = IdmPlannerFactory._get_idm_planner_base_kwargs(
            config, device, train_lookahead_slice
        )
        kwargs.update(
            {
                "reference_trajectory_handlers": reference_trajectory_handlers,  # type: ignore
            }
        )
        return kwargs

    @staticmethod
    def _get_planner_kwargs(
        planner_type: str,
        config: OfflinePLConfigFile,
        device: str | torch.device,
        train_lookahead_slice: List[int],
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if planner_type == ValidIdmPlanners.REFERENCE_TRAJECTORY:
            return IdmPlannerFactory._get_reference_trajectory_planner_kwargs(
                config,
                device,
                train_lookahead_slice,
                **kwargs,  # type: ignore
            )
        if planner_type == ValidIdmPlanners.DISABLED:
            return IdmPlannerFactory._get_idm_planner_base_kwargs(
                config,
                device,
                train_lookahead_slice,
                **kwargs,  # type: ignore
            )
        if planner_type == ValidIdmPlanners.CLOSEST_REFERENCE:
            return IdmPlannerFactory._get_closest_reference_planner_kwargs(
                config,
                device,
                train_lookahead_slice,
                **kwargs,  # type: ignore
            )
        raise ValueError(
            f"Unknown planner type: {planner_type}, must be in {ValidIdmPlanners.ALL}"
        )

    @staticmethod
    def get_idm_planner(
        config: OfflinePLConfigFile,
        device: str | torch.device,
        train_lookahead_slice: List[int] | None,
        **kwargs: Dict[str, Any],
    ) -> IdmPlanner:
        idm_agent_config = config.get_subconfig_att(IdmAgentConfig)
        assert idm_agent_config, "IDM agent config not found in the config file."
        planner_type = idm_agent_config.planner_type
        assert ValidIdmPlanners.is_valid_planner(
            planner_type
        ), f"Unknown IDM planner type: {planner_type}"
        assert (
            train_lookahead_slice is not None and len(train_lookahead_slice) > 0
        ), "Lookahead slice is required for IDM agents."

        planner_class = IdmPlannerFactory._get_planner_class(planner_type)
        planner_kwargs = IdmPlannerFactory._get_planner_kwargs(
            planner_type,
            config,
            device,
            train_lookahead_slice,
            **kwargs,
        )
        return planner_class(**planner_kwargs)
