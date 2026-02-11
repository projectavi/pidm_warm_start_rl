# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
from typing import Callable, Dict, List, Tuple

import yaml
from pytorch_lightning import seed_everything

from pidm_imitation.agents.subconfig import AgentConfig
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.configs.subconfig import ReferenceTrajectoryConfig, WandbConfig
from pidm_imitation.constants import INPUT_TRAJECTORIES_FILE
from pidm_imitation.environment.context import EvaluationContext
from pidm_imitation.utils import Logger
from pidm_imitation.utils.config_base import ConfigFile
from pidm_imitation.utils.ioutils import read_json
from pidm_imitation.utils.wandb_utils import initialize_wandb

logger = Logger()
log = logger.get_root_logger()


def get_pytorch_agent_name(environment_agent_type: str) -> str:
    """Extract the training agent name from the evaluation agent type
    by removing the environment prefix, e.g. 'toy_'."""
    i = environment_agent_type.find("_")
    if i > 0:
        return environment_agent_type[i + 1 :]
    return environment_agent_type


def evaluate_common_setup(
    args: EvaluationContext,
    logger: Logger,
    config_parser_function: Callable[[str, str], ConfigFile],
) -> EvaluationContext:
    """Common setup logic for evaluation args.

    It returns EvaluationContext"""
    logger.set_log_level(args.log_level)
    if args.log_file:
        logger.set_log_file(os.path.join(args.output_dir, args.log_file))

    config_file = args.config_file
    checkpoint_dir, checkpoint, config_file, agent = setup_model(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        config_file=args.config_file,
    )

    config = config_parser_function(config_file, agent)
    assert isinstance(config, OfflinePLConfigFile)

    # Setting the evaluation seed
    agent_config = config.get_subconfig_att(AgentConfig)
    assert agent_config is not None, "config is missing agent subconfig"
    seed = agent_config.seed
    if seed is not None:
        seed_everything(seed, workers=True)

    input_trajectories: Dict[str, List[str]] = {}
    if agent is not None:
        input_trajectories = read_input_trajectories(checkpoint_dir)

    wandb_config = setup_eval_wandb_config(config, args.wandb_log, args.wandb_project)
    folder_name = f"rollout_{checkpoint}" if checkpoint else f"rollout_{agent}"
    rollout_folder = get_evaluate_output_folder(args.output_dir, folder_name)

    get_and_overwrite_ref_traj_config(config, args.recording, args.recording_video)

    # return computed eval context.
    args.checkpoint_dir = checkpoint_dir
    args.checkpoint = checkpoint
    args.config = config
    args.wandb_config = wandb_config
    args.rollout_folder = rollout_folder
    if input_trajectories:
        args.input_trajectories = input_trajectories
    return args


def setup_eval_wandb_config(
    config: ConfigFile, enable_wandb: bool, wandb_project: str | None
) -> WandbConfig | None:
    wandb_config = config.get_subconfig_att(WandbConfig)
    if not enable_wandb:
        if wandb_config:
            return None
    elif wandb_config is None:
        log.warning(
            "### --enable_wandb is ignored because config file has no wandb information"
        )
    elif wandb_project:
        wandb_config.project = wandb_project

    if wandb_config:
        experiment_name = (
            wandb_config.eval_name if wandb_config.eval_name else config.experiment_name
        )
        wandb_group = (
            wandb_config.eval_group
            if wandb_config.eval_group
            else wandb_config.train_group
        )
        prefix = wandb_group + "/"
        if experiment_name.startswith(prefix):
            # trim the group name in this case so it is not redundant with the WANDB group name.
            experiment_name = experiment_name[len(prefix) :]
        initialize_wandb(
            group=wandb_group,
            project=wandb_config.project,
            tags=wandb_config.tags,
            notes=wandb_config.notes,
            config_dict=config.config_dict,
            experiment_name=experiment_name,
        )
    return wandb_config


def get_and_overwrite_ref_traj_config(
    config: OfflinePLConfigFile,
    input_file: str | None,
    video_file: str | None,
) -> ReferenceTrajectoryConfig:
    """Get the reference trajectory config from the config file and replace any fields that
    is provided with CLI arguments.
    """
    ref_traj_config = config.get_subconfig_att(ReferenceTrajectoryConfig)
    assert ref_traj_config

    # Overwrites with the arguments passed in the command line if they are not None
    if input_file:
        ref_traj_config.inputs_file = input_file
    if video_file:
        ref_traj_config.video_file = video_file

    return ref_traj_config


def get_evaluate_output_folder(output_dir: str, folder_name: str) -> str:
    """Get a unique name for the next rollout to be performed given any existing local rollout
    folders in folder_name"""
    if output_dir == ".":
        output_dir = os.getcwd()
    rollout_path = os.path.join(output_dir, folder_name)
    rollout_folders = []
    if os.path.isdir(rollout_path):
        rollout_folders = os.listdir(os.path.dirname(rollout_path))

    if len(rollout_folders):
        # Hmmm, this rollout has already been done, so let's uniquify with some auto-generated _N suffixes.
        reg = re.compile(f"{folder_name}_[\\d]+")
        existing_rollouts = [
            int(x.split("_")[-1]) for x in rollout_folders if reg.fullmatch(x)
        ]
        if len(existing_rollouts):
            next_rollout = max(existing_rollouts) + 1
        else:
            next_rollout = 1
        folder_name = f"{folder_name}_{next_rollout}"
    else:
        folder_name = f"{folder_name}_0"
    rollout_path = os.path.join(output_dir, folder_name)
    log.info(f"Rollout output folder: {folder_name}")
    return rollout_path


def find_agent_name(config_file: str) -> str | None:
    # Find the agent name in the config file
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
        agent_type = data.get("agent", {}).get("type", None)
        if agent_type is not None:
            return agent_type
        else:
            raise ValueError(
                f"Could not find agent type in config file {config_file}. "
                + "Please ensure the config file has an 'agent' section with a 'type' field."
            )


def setup_model(
    checkpoint: str,
    output_dir: str,
    config_file: str | None = None,
) -> Tuple[str, str, str, str]:
    """Setup the local files to run evaluation and return the
    - checkpoint directory
    - name of checkpoint
    - path to the config_file
    - agent name
    """
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.isfile(config_file), f"Could not find config file {config_file}"
    agent_name = find_agent_name(config_file)

    if checkpoint and os.path.isfile(checkpoint):
        # if the checkpoint is a full path, use it directly.
        checkpoint = os.path.realpath(checkpoint)
    assert os.path.isfile(
        checkpoint
    ), f"Could not find local checkpoint file at {checkpoint}."

    checkpoint_dir = os.path.dirname(os.path.realpath(checkpoint))
    checkpoint_file = os.path.splitext(os.path.basename(checkpoint))[0]
    return checkpoint_dir, checkpoint_file, config_file, agent_name


def read_input_trajectories(
    output_dir: str,
) -> Dict[str, List[str]]:
    """Read the input trajectories from the experiment folder."""
    os.makedirs(output_dir, exist_ok=True)
    local_input_trajectories_file = os.path.join(output_dir, INPUT_TRAJECTORIES_FILE)
    input_trajectories = {}
    # try also the parent folder
    local_parent_input_trajectories_file = os.path.realpath(
        os.path.join(output_dir, "..", INPUT_TRAJECTORIES_FILE)
    )
    if os.path.isfile(local_input_trajectories_file):
        input_trajectories = read_json(local_input_trajectories_file)
    elif os.path.isfile(local_parent_input_trajectories_file):
        input_trajectories = read_json(local_parent_input_trajectories_file)
    else:
        raise FileNotFoundError(
            f"Could not find {INPUT_TRAJECTORIES_FILE} in {output_dir} or its parent folder."
            + " Please ensure the input_trajectories.json file is present in the model folder."
        )

    train_trajectories = input_trajectories.get("train", [])
    validation_trajectories = input_trajectories.get("validation", [])
    log.info(
        "Found input trajectories:\n"
        f"{len(train_trajectories)} training trajectories: {train_trajectories}\n"
        f"{len(validation_trajectories)} validation trajectories: {validation_trajectories}"
    )

    return input_trajectories
