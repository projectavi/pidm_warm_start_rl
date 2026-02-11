# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from typing import Dict, List

from schema import Optional, Schema

from pidm_imitation.agents.subconfig import AgentConfig, IdmAgentConfig
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.configs.subconfig import ReferenceTrajectoryConfig
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.utils import FileMetadata


class ToyEnvRefTrajectoryConfig(ReferenceTrajectoryConfig):
    def __init__(
        self,
        config_path: str,
        config: dict,
    ) -> None:
        super().__init__(config_path, config)
        self.loaded_trajectory: ToyEnvironmentTrajectory | None = None

    def get_trajectory_obj(self, **kwargs) -> ToyEnvironmentTrajectory:
        if self.loaded_trajectory is None:
            data_dir = kwargs.get("data_dir", self.trajectory_dir)
            self.loaded_trajectory = ToyEnvironmentTrajectory.init_from_dir(
                dirname=data_dir,
                trajectory_name=self.trajectory_name,
            )
        return self.loaded_trajectory


class ToyPLConfigFile(OfflinePLConfigFile):
    """
    Config parser class for all experiments using the offline Pytorch Lightning agent in
    the toy environment. This class is only used during evaluation, as config files are not
    used while training offline Pytorch Lightning agents.
    """

    def __init__(self, config_path: str) -> None:
        self.ref_traj_config: ToyEnvRefTrajectoryConfig
        self.agent_config: AgentConfig
        super().__init__(config_path)

    def _get_schema(self) -> Schema:
        base_schema = super()._get_schema()
        schema = Schema(
            {
                **base_schema.schema,
                ToyEnvRefTrajectoryConfig.KEY: object,
                Optional(AgentConfig.KEY): object,
            }
        )
        return schema

    def _create_sub_configs(self) -> None:
        super()._create_sub_configs()
        self.ref_traj_config = self._get_simple_config_obj(ToyEnvRefTrajectoryConfig)
        agent_config_class = self.get_agent_config_class()
        self.agent_config = self._get_simple_config_obj(agent_config_class)

    def _set_attributes(self) -> None:
        self.experiment_name = self._config[self.EXPERIMENT_NAME]
        self._create_sub_configs()
        self.ref_traj_config.assert_video_file_is_valid()

    def get_config_files_dict(self) -> Dict[str, str | List[FileMetadata]]:
        result = OfflinePLConfigFile.get_config_files_dict(self)
        result.update(self.ref_traj_config.get_files_dict())
        return result

    @abstractmethod
    def get_agent_config_class(self) -> type[AgentConfig]:
        pass


class ToyIDMConfigFile(ToyPLConfigFile):
    def get_agent_config_class(self) -> type[AgentConfig]:
        return IdmAgentConfig


class ToyBCConfigFile(ToyPLConfigFile):
    def get_agent_config_class(self) -> type[AgentConfig]:
        return AgentConfig
