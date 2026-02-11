# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from argparse import Namespace

from pidm_imitation.configs.subconfig import WandbConfig
from pidm_imitation.utils.config_base import ConfigFile


class EvaluationContext:
    """This class represents all the common context needed to perform an evaluation in an environment.
    It is essentially a strongly typed bag of command line arguments, plus a few extra computed args.
    Subclass this to add more environment specific arguments"""

    def __init__(self, args: Namespace):
        # add_run_model_args
        self.config_file: str = args.config
        self.agent_name: str = args.agent
        self.checkpoint: str = args.checkpoint
        self.log_level: str = args.log_level
        self.log_file: str = args.log_file

        # add_eval_base_args
        self.experiment: str = args.experiment
        self.episodes: int = args.episodes
        self.output_dir: str = args.output_dir
        self.recording: str | None = args.recording
        self.recording_video: str | None = args.recording_video

        # add_eval_wandb_args
        self.wandb_log: bool = args.enable_wandb
        self.wandb_project: str = args.wandb_project
        self.log_video: bool = args.log_video

        # computed
        self.checkpoint_dir: str = ""
        self.config: ConfigFile | None = None
        self.wandb_config: WandbConfig | None = None
        self.rollout_folder: str = ""
        self.input_trajectories: dict = {}
