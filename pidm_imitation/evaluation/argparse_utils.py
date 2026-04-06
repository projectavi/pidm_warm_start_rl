# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def add_run_model_args(parser, valid_agents) -> None:
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help=f"Type of agent to be evaluated, must be one of {valid_agents}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the checkpoint to load.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        help="Logging level (default INFO)",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Name of the log file, will be stored in --output_dir",
        default=None,
    )


def add_eval_wandb_args(parser) -> None:
    parser.add_argument(
        "--enable_wandb",
        help="Enable logging in Weight & Biases (default false).",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_project",
        help="Specifies an override for the wandb_config project name in the "
        + "config yaml (default comes from your config file).",
    )
    parser.add_argument(
        "--log_video",
        help="Whether to log video to wandb (default false).",
        action="store_true",
    )


def add_eval_base_args(parser) -> None:
    parser.add_argument(
        "--experiment",
        type=str,
        help="The name of the experiment to evaluate. Used to find the folder with checkpoints",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes/ runs to evaluate for.",
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Location to save the output (default .). Rollout log files will be stored in a folder"
        + "named `rollout_N` where N is an incrementally assigned number.",
        default=".",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Full path and file name for the inputs json file (default: comes from --config file under "
        + "'reference_trajectory.inputs'.)",
    )
    parser.add_argument(
        "--recording_video",
        type=str,
        default=None,
        help="Full path and file name for the recording video mp4 file (default: comes from --config file under "
        + "reference_trajectory.video.)",
    )


def add_toy_eval_args(parser, valid_agents) -> None:
    parser.add_argument("--toy_config", type=str, required=True, help="Path to config file of the toy environment.")
    add_run_model_args(parser, valid_agents)
    add_eval_base_args(parser)
    add_eval_wandb_args(parser)
    parser.add_argument("--save_trajectories", action="store_true", help="Save trajectory data of each episode")
    parser.add_argument(
        "--save_video", action="store_true", help="Save video recording of each episode (only with --save_trajectories)"
    )
    parser.add_argument("--save_results", action="store_true", help="Save final metrics to a results.json file in the output directory")
