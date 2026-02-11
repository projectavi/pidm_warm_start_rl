# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pidm_imitation.constants import ENV_CONFIG_FILE_SUFFIX
from pidm_imitation.environment.toy_env.configs.toy_environment_config import (
    ToyEnvironmentConfig,
)
from pidm_imitation.environment.toy_env.game_elements import Ribbon
from pidm_imitation.environment.toy_env.toy_constants import BACKGROUND_COLOR
from pidm_imitation.environment.toy_env.toy_environment_goal import (
    GoalReachingToyEnvironment,
)
from pidm_imitation.environment.toy_env.toy_factory import ToyEnvironmentFactory
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.utils import Logger

COLORS = plt.colormaps["tab10"]

logger = Logger()
log = logger.get_root_logger()


def get_args():
    parser = ArgumentParser("Draw toy environment with traces")
    parser.add_argument(
        "--data_dirs",
        type=Path,
        nargs="+",
        help="Directories containing the trajectories",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Names of the algorithms/ trajectory groups (in order of data_dirs)",
    )
    parser.add_argument(
        "--save_path", type=Path, default=None, help="Path to save the image to"
    )
    parser.add_argument("--no_legend", action="store_true", help="Do not draw a legend")
    parser.add_argument(
        "--plot_agent", action="store_true", help="Plot the agent in the environment"
    )
    return parser.parse_args()


def float_to_rgb(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


def _get_telemetry_trace(
    telemetry: np.ndarray, trace_color: Any, trace_size: int = 1
) -> Ribbon:
    """Get a ribbon trace of the agent telemetry for a given trajectory.

    :param telemtry: The telemtry information to be traced.
    :return: A Ribbon object with all agent telemetry positions as points.
    """
    ribbon = Ribbon(size=trace_size, color=trace_color)
    assert (
        len(telemetry.shape) == 2 and telemetry.shape[1] == 2
    ), "Telemetry must be of shape (n, 2)"
    for x, y in telemetry:
        ribbon.add_point((int(x), int(y)))
    return ribbon


def get_env_frame(
    env_config: ToyEnvironmentConfig,
    telemetry_traces: Dict[str, List[np.ndarray]] | None = None,
    trajectory_colors: Dict[str, Any] | None = None,
    plot_agent: bool = True,
    black_goals: bool = False,
) -> np.ndarray:
    env = ToyEnvironmentFactory.create_environment(env_config)
    env.reset()
    if plot_agent:
        # plot the agent in green
        env.player.color = (0, 255, 0)
    else:
        # plot the agent in background color to hide it
        env.player.color = BACKGROUND_COLOR

    # make all goals active to show them in color
    if isinstance(env, GoalReachingToyEnvironment):
        if black_goals:
            # set all goals to black
            for goal in env.goals:
                goal.color = (0, 0, 0)
        else:
            for goal in env.goals:
                goal.set_active()

    if telemetry_traces is not None:
        for i, (name, telemetries) in enumerate(telemetry_traces.items()):
            if trajectory_colors is not None and name in trajectory_colors:
                color = trajectory_colors[name]
            else:
                color = COLORS(i)[:3]
            if isinstance(color, tuple) and all(
                isinstance(c, float) and 0 <= c <= 1 for c in color
            ):
                color = float_to_rgb(color)
            for j, telemetry in enumerate(telemetries):
                ribbon = _get_telemetry_trace(telemetry, color)
                env.ribbons[f"{name}_{j + 1}"] = ribbon

    env._update_screen()
    frame = env.render()
    return frame


def draw_toy_env_with_traces(
    env_config: ToyEnvironmentConfig,
    trajectories_by_name: Dict[str, List[ToyEnvironmentTrajectory]],
    plot_agent: bool = True,
    black_goals: bool = False,
    trajectory_colors: Dict[str, Any] | None = None,
    save_path: Path | None = None,
    no_legend: bool = False,
) -> np.ndarray:
    """
    Draw the toy environment with traces for each trajectory.

    :param env_config: The environment configuration.
    :param trajectories_by_name: A dictionary mapping names to lists of trajectories.
    :param plot_agent: Whether to plot the agent in the environment.
    :param black_goals: Whether to draw goals in black.
    :param trajectory_colors: A dictionary mapping names to colors.
    :param save_path: The path to save the image to.
    :param no_legend: Whether to draw a legend.
    :return: The drawn frame as a numpy array.
    """
    telemetry_by_names = {
        name: [traj.compute_telemetry() for traj in trajectories]
        for name, trajectories in trajectories_by_name.items()
    }
    return draw_telemetry_traces(
        env_config,
        telemetry_by_names,
        plot_agent,
        black_goals,
        trajectory_colors,
        save_path,
        no_legend,
    )


def draw_telemetry_traces(
    env_config: ToyEnvironmentConfig,
    telemetry_by_names: Dict[str, List[np.ndarray]],
    plot_agent: bool,
    black_goals: bool = False,
    trajectory_colors: Dict[str, Any] | None = None,
    save_path: Path | None = None,
    no_legend: bool = False,
) -> np.ndarray:
    """
    Draw the toy environment with traces for each telemetry array.

    :param env_config: The environment configuration.
    :param telemetry_by_names: A dictionary mapping names to lists of telemetry arrays.
    :param plot_agent: Whether to plot the agent in the environment.
    :param black_goals: Whether to draw goals in black.
    :param trajectory_colors: A dictionary mapping names to colors.
    :param save_path: The path to save the image to.
    :param no_legend: Whether to draw a legend.
    :return: The drawn frame as a numpy array.
    """
    fake_legend = []
    for i, (name, _) in enumerate(telemetry_by_names.items()):
        color = COLORS(i)[:3]
        fake_legend.append(plt.Line2D([0], [0], color=color, lw=4, label=name))
    frame = get_env_frame(
        env_config,
        telemetry_by_names,
        trajectory_colors,
        plot_agent=plot_agent,
        black_goals=black_goals,
    )

    # Draw the frame with a legend outside the plot above
    plt.imshow(frame)
    plt.axis("off")
    if not no_legend:
        plt.legend(handles=fake_legend, loc="upper left", bbox_to_anchor=(1, 1))
    if save_path:
        log.info(f"Saving plot under {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return frame


def load_trajectories_from_dir(directory: Path) -> List[ToyEnvironmentTrajectory]:
    """
    Load all trajectories from a directory.

    :param directory: The directory containing the trajectory files.
    :return: A list of trajectories.
    """
    log.info(f"Loading trajectories from {directory}")
    env_config_paths = [
        f for f in directory.iterdir() if ENV_CONFIG_FILE_SUFFIX in f.name
    ]
    trajectory_names = [
        f.name[: -len(ENV_CONFIG_FILE_SUFFIX + ".yaml")] for f in env_config_paths
    ]
    trajectories = [
        ToyEnvironmentTrajectory.init_from_dir(str(directory), name)
        for name in trajectory_names
    ]
    return trajectories


def main():
    args = get_args()
    assert len(args.data_dirs) > 0, "Must provide at least one data directory"
    names = args.names if args.names else args.data_dirs
    data_directories = [Path(data_dir) for data_dir in args.data_dirs]
    assert all(d.is_dir() for d in data_directories), "All data directories must exist"

    trajectories_by_name = {
        name: load_trajectories_from_dir(data_dir)
        for name, data_dir in zip(names, data_directories)
    }

    env_config = trajectories_by_name[names[0]][0].env_config
    draw_toy_env_with_traces(
        env_config,
        trajectories_by_name,
        args.plot_agent,
        args.save_path,
        args.no_legend,
    )


if __name__ == "__main__":
    main()
