# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from schema import And, Optional, Or, Schema

from pidm_imitation.environment.utils import ValidIdmPlanners
from pidm_imitation.utils.subconfig import SubConfig


class AgentConfig(SubConfig):
    KEY = "agent"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.agent_type: str
        self.seed: int | None
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                "type": str,
                Optional("seed"): Or(int, None),  # Add 'seed' as an optional key
            }
        )

    def _set_attributes(self) -> None:
        self.agent_type = self._config["type"]
        self.seed = self._config.get("seed", None)


class IdmAgentConfig(AgentConfig):

    def __init__(self, config_path: str, config: dict) -> None:
        self.planner_type: str
        self.planner_config: IdmPlannerConfig
        super().__init__(config_path, config)

    def _get_schema(self) -> Schema:
        return Schema(
            {
                **super()._get_schema()._schema,
                Optional("planner_type"): And(
                    str,
                    lambda x: x in ValidIdmPlanners.ALL,
                    error=f"ERROR: Invalid planner_type. Valid values are: {ValidIdmPlanners.ALL}.",
                ),
                Optional(IdmPlannerConfig.KEY): object,
            }
        )

    def _set_attributes(self) -> None:
        super()._set_attributes()
        self.planner_type = self._config.get(
            "planner_type", ValidIdmPlanners.REFERENCE_TRAJECTORY
        )
        self.planner_config = self._get_simple_config_obj(IdmPlannerConfig)
        if self.planner_config is None:
            # if the planner config is not in the config, create a default one
            self.planner_config = IdmPlannerConfig(self.config_path, {})


class IdmPlannerConfig(SubConfig):
    KEY = "idm_planner"
    VALID_LOOKAHEAD_TYPES = ["fixed"]

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.eval_lookahead_type: str
        self.eval_lookahead: int
        self.planner_kwargs: dict
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("eval_lookahead_type"): And(
                    str,
                    lambda x: x in self.VALID_LOOKAHEAD_TYPES,
                    error=f"ERROR: Invalid eval_lookahead_type. Valid values are: {self.VALID_LOOKAHEAD_TYPES}.",
                ),
                Optional("eval_lookahead"): And(
                    int, lambda x: x >= 0, error=self.POSITIVE_NUMBER_ERR
                ),
                Optional("planner_kwargs"): dict,
            }
        )

    def _set_attributes(self) -> None:
        self.eval_lookahead_type = self._config.get("eval_lookahead_type", "fixed")
        self.eval_lookahead = self._config.get("eval_lookahead", 0)
        self.planner_kwargs = self._config.get("planner_kwargs", {})

    def get_kwargs(self) -> dict:
        return {
            "eval_lookahead_type": self.eval_lookahead_type,
            "eval_lookahead": self.eval_lookahead,
            **self.planner_kwargs,
        }
