# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import math
import os
import time

import pygame
from tqdm import tqdm

from pidm_imitation.environment.toy_env.configs import ToyEnvironmentConfig
from pidm_imitation.environment.toy_env.data_collection_agents import (
    VALID_AGENT_TYPES,
    DataCollectionAgent,
    get_agent,
)
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.environment.toy_env.toy_factory import ToyEnvironmentFactory
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.environment.toy_env.utils import (
    add_toy_env_args,
    add_toy_exogenous_noise_args,
    create_toy_config_from_args,
)
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


def parse_color(s):
    try:
        r, g, b = map(int, s.split(","))
        return r, g, b
    except:
        raise argparse.ArgumentTypeError("Color must be r,g,b")


def get_arg_parser(test_run=False):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        "Record human trajectories in toy environment",
        add_help=not test_run,
        exit_on_error=not test_run,
    )

    add_toy_env_args(parser)
    add_toy_exogenous_noise_args(parser)

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=VALID_AGENT_TYPES,
        help=f"Choose one of {', '.join(VALID_AGENT_TYPES)} for the agent to play the game",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the user's input and save trajectories",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes to play (default 30)",
        default=30,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment (required if recording)",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Location to save the experiment files in",
        default=".",
    )
    return parser


def parse_args():
    parser = get_arg_parser(True)
    args, _ = parser.parse_known_args()
    parser = get_arg_parser(False)
    return parser.parse_args()


def play_episode(
    env: ToyEnvironment,
    agent: DataCollectionAgent,
    fps: int,
    wait_for_fps: bool = False,
) -> ToyEnvironmentTrajectory:
    """Record agent gameplay in the game environment.

    :param env: The game environment
    :param agent: The agent to control the environment
    :param fps: Frames per second
    :param wait_for_fps: Whether to wait for the specified FPS
    :return: The recorded trajectory
    """
    clock = pygame.time.Clock()
    obs, _ = env.reset()
    frame = env.render()
    state = env.get_state()
    agent.reset()

    recording = ToyEnvironmentTrajectory(
        video_frames=[frame],
        video_ticks=[0],
        observations=[obs],
        states=[state],
        env_config=env.config,
        video_fps=fps,
        color_format="rgb",
    )

    done = False
    agent.wait_for_active_input()
    start_time = time.time()
    while not done:
        # Step the environment
        action = agent.act()
        obs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        state = env.get_state()
        frame = env.render()
        recording.add_step(
            frame=frame,
            other_data={
                "obs": obs,
                "action": action,
                "state": state,
                "reward": reward,
            },
        )

        if wait_for_fps:
            # Control the frame rate for human agents
            clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Exception("User closed the window")

    record_time = time.time() - start_time
    log.debug(f"Episode finished in {record_time:.2f} seconds.")

    return recording


def play_episodes(
    env_config: ToyEnvironmentConfig,
    agent_type: str,
    episodes: int,
    record: bool,
    experiment_name: str,
    output_dir: str,
    fps: int = 30,
):
    env = ToyEnvironmentFactory.create_environment(env_config)
    agent = get_agent(env, agent_type)

    if record:
        # prepare recording
        assert experiment_name is not None, "Experiment name is required when recording"
        experiment_dir = os.path.join(output_dir, experiment_name)
        if os.path.exists(experiment_dir):
            log.warning(
                f"Directory {experiment_dir} already exists. Will overwrite stored data."
            )
        digits = math.ceil(math.log10(episodes))

    pbar = tqdm(total=episodes, desc="Episodes", unit="episode")
    episode = 0
    while episode < episodes:
        try:
            recorded_trajectory = play_episode(
                env, agent, fps, wait_for_fps=agent_type == "human"
            )
            if record:
                traj_name = f"{experiment_name}_{episode:0{digits}d}"
                recorded_trajectory.save_to_dir(experiment_dir, traj_name)
            episode += 1
            pbar.update(1)
        except KeyboardInterrupt:
            log.info("Session interrupted by user.")
            break


if __name__ == "__main__":
    args = parse_args()
    env_config = create_toy_config_from_args(args)

    play_episodes(
        env_config,
        args.agent,
        args.episodes,
        args.record,
        args.experiment,
        args.output_dir,
        args.fps,
    )
