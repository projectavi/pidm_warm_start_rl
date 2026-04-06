# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import List
from typing import Optional as OptionalType

from schema import And, Optional, Or, Schema

from pidm_imitation.constants import INPUTS_FILE_SUFFIX, VIDEO_FILE_SUFFIX
from pidm_imitation.utils import StateType, Trajectory, resolve_path
from pidm_imitation.utils.logger import Logger
from pidm_imitation.utils.subconfig import SubConfig
from pidm_imitation.utils.valid_controller_actions import ValidControllerActions

logger = Logger()
log = logger.get_root_logger()


class WandbConfig(SubConfig):
    KEY = "wandb"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.project: str
        self.username: str | None
        self.train_group: str | None
        self.eval_group: str | None
        self.train_name: str | None
        self.eval_name: str | None
        self.save_dir: str
        self.log_model: bool
        self.offline: bool
        self.tags: List[str] | None
        self.notes: str | None
        self._set_attributes()

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                "project": str,
                Optional("username"): str,
                Optional("train_group"): str,
                Optional("train_name"): str,
                Optional("eval_group"): str,
                Optional("eval_name"): str,
                Optional("save_dir"): str,
                Optional("log_model"): bool,
                Optional("offline"): bool,
                Optional("tags"): list,
                Optional("notes"): str,
            }
        )
        return schema

    def _set_attributes(self) -> None:
        self.project = self._config["project"]
        self.username = self._config.get("username", None)
        self.train_group = self._config.get("train_group", None)
        self.train_name = self._config.get("train_name", None)
        self.eval_group = self._config.get("eval_group", None)
        self.eval_name = self._config.get("eval_name", None)
        self.save_dir = self._config.get("save_dir", "./logs")
        self.log_model = self._config.get("log_model", False)
        self.offline = self._config.get("offline", False)
        self.tags = self._config.get("tags", None)
        self.notes = self._config.get("notes", None)
        if self.tags:
            for i in range(len(self.tags)):
                try:
                    self.tags[i] = str(self.tags[i])
                except:
                    raise ValueError(
                        "The wandb.tags parameter only accepts strings or numbers."
                    )


class ReferenceTrajectoryConfig(SubConfig):
    KEY = "reference_trajectory"

    VIDEO_ERR = "Invalid reference trajectory video file. It must be a .mp4 file."
    INPUTS_ERR = "Invalid reference trajectory inputs file. It must be a .json file."

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.video_file: str | None
        self.inputs_file: str | None
        self._set_attributes()

    @property
    def trajectory_name(self) -> str:
        self.assert_video_file_is_valid()
        self.assert_input_file_is_valid()
        # get general trajectory name by removing suffix
        trajectory_name = os.path.basename(self.video_file)[
            : -len(f"{VIDEO_FILE_SUFFIX}.mp4")
        ]
        input_file_name = os.path.basename(self.inputs_file)[
            : -len(f"{INPUTS_FILE_SUFFIX}.json")
        ]
        assert (
            trajectory_name == input_file_name
        ), "ERROR: The video and inputs files must have the same name."
        return trajectory_name

    @property
    def trajectory_dir(self) -> str | None:
        if self.has_video_file:
            return os.path.dirname(self.video_file)
        elif self.has_inputs_file:
            return os.path.dirname(self.inputs_file)
        return None

    @property
    def has_inputs_file(self) -> bool:
        return self.inputs_file is not None

    @property
    def has_video_file(self) -> bool:
        return self.video_file is not None

    def assert_input_file_is_valid(self) -> None:
        assert (
            self.has_inputs_file
        ), "ERROR: The current experiment requires an inputs file, but none was provided."

    def assert_video_file_is_valid(self) -> None:
        assert (
            self.has_video_file
        ), "ERROR: The current experiment requires a video file, but none was provided."

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                Optional("video"): And(
                    str, lambda x: x.endswith(".mp4"), error=self.VIDEO_ERR
                ),
                Optional("inputs"): And(
                    str, lambda x: x.endswith(".json"), error=self.INPUTS_ERR
                ),
            }
        )
        return schema

    def _get_file_path(self, current_path: str | None) -> str | None:
        if current_path is not None:
            current_path = os.path.expandvars(current_path)
            current_path = resolve_path(self._config_path, current_path)
        return current_path

    def _set_attributes(self) -> None:
        self.video_file = self._config.get("video", None)
        self.inputs_file = self._config.get("inputs", None)
        if self.video_file:
            self.video_file = self._get_file_path(self.video_file)
        if self.inputs_file:
            self.inputs_file = self._get_file_path(self.inputs_file)

    def get_files_dict(self) -> dict:
        """Returns a dictionary with the reference trajectory files."""
        result = {}
        # do not attempt to upload and relativize 'input_trajectories/{train[0]}_video.mp4' paths.
        if self.has_video_file:
            result["reference_trajectory.video"] = self.video_file
        if self.has_inputs_file:
            result["reference_trajectory.inputs"] = self.inputs_file
        return result

    def get_trajectory_obj(self, **kwargs) -> Trajectory:
        """
        This method must return the ``Trajectory`` object of the reference
        trajectory used in any of the environment subclasses.
        """
        raise NotImplementedError("Not implemented for the base class.")


class StateConfig(SubConfig):
    KEY = "state"

    STATE_ERR = f"Invalid state type. The allowed types are: {StateType.get_valid_state_type_strings()}"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.type: StateType
        self._set_attributes()

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                "type": And(
                    str,
                    lambda x: x in StateType.get_valid_state_type_strings(),
                    error=self.STATE_ERR,
                ),
            },
        )
        return schema

    def _set_attributes(self) -> None:
        self.type = StateType.get_state_type_from_str(self._config["type"])


class ControllerActionConfig(SubConfig):
    KEY = "action"

    ACTION_ERR = f"Invalid action value. The allowed values are: {ValidControllerActions.get_valid_action_types()}"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.type: str
        self._set_attributes()

    def _get_schema(self) -> Schema:
        schema = Schema(
            {
                "type": And(
                    str,
                    lambda x: x in ValidControllerActions.get_valid_action_types(),
                    error=self.ACTION_ERR,
                ),
            },
        )
        return schema

    def _set_attributes(self) -> None:
        self.type = self._config["type"]


class PytorchLightningConfig(SubConfig):
    KEY = "pytorch_lightning"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.optimizer: str
        self.optimizer_kwargs: dict
        self.seed_everything: int | None
        self.trainer: dict
        self.fit_kwargs: dict
        self.scheduler: str
        self.scheduler_kwargs: dict
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                "optimizer": str,
                "optimizer_kwargs": dict,
                Optional("seed_everything"): Or(int, None),
                Optional("trainer"): dict,
                Optional("fit_kwargs"): dict,
                Optional("scheduler"): str,
                Optional("scheduler_kwargs"): dict,
            }
        )

    def _set_attributes(self) -> None:
        self.optimizer = self._config["optimizer"]
        self.optimizer_kwargs = self._config["optimizer_kwargs"]
        self.seed_everything = self._config.get("seed_everything", None)
        if self.seed_everything is None:
            log.warning("No seed provided for PyTorch Lightning.")
        self.trainer = self._config.get("trainer", {})
        if "devices" not in self.trainer:
            self.trainer["devices"] = 1
        self.fit_kwargs = self._config.get("fit_kwargs", {})
        self.scheduler = self._config.get("scheduler", "constant")
        self.scheduler_kwargs = self._config.get("scheduler_kwargs", {})


class CallbacksConfig(SubConfig):
    KEY = "callbacks"

    def __init__(self, config_path: str, config: dict) -> None:
        super().__init__(config_path, config)
        self.checkpoint_callback: bool
        self.checkpoint_callback_kwargs: dict
        self._set_attributes()

    def _get_schema(self) -> Schema:
        return Schema(
            {
                Optional("checkpoint_callback"): bool,
                Optional("checkpoint_callback_kwargs"): dict,
            }
        )

    def _set_attributes(self) -> None:
        self.checkpoint_callback = self._config.get("checkpoint_callback", False)
        self.checkpoint_callback_kwargs = self._config.get(
            "checkpoint_callback_kwargs", {}
        )

    def get_model_checkpoint_dir(self) -> OptionalType[str]:
        if self.checkpoint_callback:
            return self.checkpoint_callback_kwargs.get("dirpath", None)
        return None
