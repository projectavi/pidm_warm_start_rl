# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import wandb

from pidm_imitation.agents import Agent
from pidm_imitation.environment.context import EvaluationContext
from pidm_imitation.environment.toy_env.draw_toy_trajectories import (
    draw_telemetry_traces,
)
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.environment.toy_env.toy_trajectory import ToyEnvironmentTrajectory
from pidm_imitation.utils import GameTimer, Logger, StateType

log = Logger.get_logger(__name__)


class ToyEvaluationContext(EvaluationContext):
    def __init__(self, args: Namespace):
        super().__init__(args)

        # optional add_toy_eval_args
        self.toy_config: str = args.toy_config
        self.save_trajectories: bool = args.save_trajectories
        self.save_video: bool = args.save_video


class ToyEnvExperimentResult:

    def __init__(self, reference_trajectory: ToyEnvironmentTrajectory):
        self.reference_trajectory: ToyEnvironmentTrajectory = reference_trajectory

        self.number_of_episodes = 0
        self.ep_steps: List[int] = []
        self.ep_durations: List[float] = []
        self.ep_successes: List[int] = []
        self.ep_metrics: Dict[str, List[float]] = defaultdict(list)
        self.extra_metrics: Dict[str, List[Any]] = defaultdict(list)

    def _compute_telemetry_distance(self, rollout: ToyEnvironmentTrajectory) -> float:
        rollout_telemetry = rollout.compute_telemetry()
        reference_telemetry = self.reference_trajectory.compute_telemetry()
        assert (
            len(rollout_telemetry.shape) == 2 and len(reference_telemetry.shape) == 2
        ), "Telemetry must be 2D"
        assert (
            rollout_telemetry.shape[1] == 2 and reference_telemetry.shape[1] == 2
        ), "Telemetry must be (x, y) coordinates"

        # compute minimum x and y to normalise the telemetry in [0, 1]
        min_x = min(np.min(rollout_telemetry[:, 0]), np.min(reference_telemetry[:, 0]))
        max_x = max(np.max(rollout_telemetry[:, 0]), np.max(reference_telemetry[:, 0]))
        min_y = min(np.min(rollout_telemetry[:, 1]), np.min(reference_telemetry[:, 1]))
        max_y = max(np.max(rollout_telemetry[:, 1]), np.max(reference_telemetry[:, 1]))

        # normalise telemetry in [0, 1]
        rollout_telemetry = (rollout_telemetry - [min_x, min_y]) / [
            max_x - min_x,
            max_y - min_y,
        ]
        reference_telemetry = (reference_telemetry - [min_x, min_y]) / [
            max_x - min_x,
            max_y - min_y,
        ]

        # pad telemetry with last entrance to have same length
        if len(rollout_telemetry) < len(reference_telemetry):
            num_pad_values = len(reference_telemetry) - len(rollout_telemetry)
            rollout_telemetry = np.pad(
                rollout_telemetry, ((0, num_pad_values), (0, 0)), mode="edge"
            )
        elif len(rollout_telemetry) > len(reference_telemetry):
            num_pad_values = len(rollout_telemetry) - len(reference_telemetry)
            reference_telemetry = np.pad(
                reference_telemetry, ((0, num_pad_values), (0, 0)), mode="edge"
            )

        # compute L2 distance between the two normalised telemetry
        telemetry_distance = float(
            np.linalg.norm(rollout_telemetry - reference_telemetry)
        )
        log.info(f"Telemetry distance from recorded path = {telemetry_distance}")
        return telemetry_distance

    def _compute_goal_completion_count(
        self, rollout: ToyEnvironmentTrajectory
    ) -> float | None:
        """Calculate the number of goals reached for a rollout if the trajectory is for a goal reaching task."""
        rollout_telemetry = np.array(rollout.states)

        num_goals = rollout.env_config.goal_config.num_goals
        state_dim = 2 + num_goals * 3
        # we assume the state is in the form of [x, y, goal1_x, goal1_y, goal1_reached, ...] + (optional noise)
        assert rollout_telemetry.shape[1] >= state_dim

        # Get goal completions
        goal_completions = rollout_telemetry[-1][4:state_dim:3]
        assert (
            self.reference_trajectory.success
        ), "Reference trajectory has not reached all goals"

        # Calculate the number of goals reached
        goal_completion_count = goal_completions.sum()
        log.info(f"Goals completed in rollout: {goal_completion_count}")

        return goal_completion_count

    def _compute_final_goal_distances(
        self, rollout: ToyEnvironmentTrajectory
    ) -> float | None:
        """Calculate the distance to the goal for a rollout."""
        if rollout.env_config.goal_config is None:
            # not a goal-reaching task --> no goal completion count
            return None

        rollout_telemetry = np.array(rollout.states)
        reference_telemetry = np.array(self.reference_trajectory.states)

        num_goals = rollout.env_config.goal_config.num_goals
        state_dim = 2 + num_goals * 3
        # we assume the state is in the form of [x, y, goal1_x, goal1_y, goal1_reached, ...] + (optional noise)
        assert rollout_telemetry.shape[1] >= state_dim

        last_position = rollout_telemetry[-1][:2]
        last_goal_position = reference_telemetry[-1][:2]

        # Calculate the distance to the goal
        goal_distance = float(np.linalg.norm(last_position - last_goal_position))

        return goal_distance

    def compute_metrics(
        self,
        ep_time: float,
        ep_step_count: int,
        done: bool,
        rollout: ToyEnvironmentTrajectory,
        info: dict,
    ):
        rollout.compute_telemetry()
        self.reference_trajectory.compute_telemetry()

        self.number_of_episodes += 1
        self.ep_steps.append(ep_step_count)
        self.ep_durations.append(ep_time)
        self.ep_successes.append(int(rollout.success))

        telemetry_distance = self._compute_telemetry_distance(rollout)
        episode_return = rollout.compute_episode_return()
        self.ep_metrics["telemetry_distance"].append(telemetry_distance)
        self.ep_metrics["episode_return"].append(episode_return)

        # Add goal metrics if rollout is for a goal-reaching task
        if rollout.env_config.goal_config is not None:
            goal_completion_count = self._compute_goal_completion_count(rollout)
            goal_distance = self._compute_final_goal_distances(rollout)
            self.ep_metrics["goal_completion_count"].append(goal_completion_count)
            self.ep_metrics["final_goal_distance"].append(goal_distance)

        for k, v in info.items():
            self.extra_metrics[k].append(v)

    def get_log_dict(self) -> Dict[str, float | int]:
        log_dict = {
            "step_count": self.ep_steps[-1],
            "episode_time": self.ep_durations[-1],
            "success": self.ep_successes[-1],
            "avg_step_count": float(np.mean(self.ep_steps)),
            "avg_episode_time": float(np.mean(self.ep_durations)),
            "avg_success": float(np.mean(self.ep_successes)),
        }

        # add episode metrics
        for metric, values in self.ep_metrics.items():
            latest_value = values[-1]
            avg_value = float(np.mean(values))
            log_dict.update({f"avg_{metric}": avg_value, metric: latest_value})

        # Add extra metrics
        for metric, values in self.extra_metrics.items():
            latest_value = values[-1]
            avg_value = float(np.mean(values))
            log_dict.update({f"avg_{metric}": avg_value, metric: latest_value})

        return log_dict


class ToyEnvExperiment:
    """
    This class runs an experiment using a Agent and a toy environment. The agent will interact with the environment
    by taking actions and receiving environment information (toy env observations or states; states will always be
    telemetry-based feature vectors, observations can be similar telemetry-based feature vectors or image-based
    arrays based on the environment configuration).
    """

    def __init__(
        self,
        context: ToyEvaluationContext,
        name: str,
        env: ToyEnvironment,
        agent: Agent,
        reference_trajectory: ToyEnvironmentTrajectory,
    ):
        """
        Args:
            context (ToyEvaluationContext): The context for the evaluation.
            name (str): The name of the experiment.
            env (ToyEnvironment): The toy environment to interact with
            agent (Agent): The agent used in the experiment.
            reference_trajectory (ToyEnvironmentTrajectory): The reference trajectory.
        """
        self.name = name
        self.env = env
        self.agent = agent
        self.num_episodes = context.episodes

        assert isinstance(self.env, ToyEnvironment)
        self.wandb_log = context.wandb_log
        self.save_trajectories = context.save_trajectories
        self.save_video = context.save_video

        self.output_dir = context.rollout_folder
        os.makedirs(self.output_dir, exist_ok=True)
        self.reference_trajectory = reference_trajectory
        self.rollout_telemetries: List[np.ndarray] = []
        self.start_time = datetime.now(timezone.utc)
        self._exp_result: ToyEnvExperimentResult | None = None

    def run(self) -> ToyEnvExperimentResult:
        self._exp_result = ToyEnvExperimentResult(
            reference_trajectory=self.reference_trajectory
        )

        log.info(
            f"Running experiment {self.name} for {self.num_episodes} episodes in process {os.getpid()}"
        )

        self.start_time = datetime.now(timezone.utc)

        for episode in range(self.num_episodes):
            log.info(f"Episode: {episode + 1} of {self.num_episodes}")
            ep_timer = GameTimer()
            step_timer = GameTimer()
            step_times = []

            # initialise environment and trajectory for new episode
            obs, info = self.env.reset()
            frame = self.env.render()
            state = self.env.get_state()
            recording = ToyEnvironmentTrajectory(
                video_frames=[frame],
                video_ticks=[0],
                observations=[obs],
                states=[state],
                env_config=self.env.config,
                color_format="rgb",
            )

            self.agent.reset()
            done = False
            ep_finished = False
            ep_steps = 0
            ep_timer.start()

            while not ep_finished:
                step_timer.start()
                action = self.agent.get_action(
                    raw_obs=frame,
                    built_obs={
                        StateType.OBSERVATIONS.value: obs,
                        StateType.STATES.value: state,
                        "frame": frame,
                    },
                )
                obs, reward, done, truncated, info = self.env.step(action)
                frame = self.env.render()
                state = self.env.get_state()
                recording.add_step(
                    frame=frame,
                    other_data={
                        ToyEnvironmentTrajectory.OBS_KEY: obs,
                        ToyEnvironmentTrajectory.ACTION_KEY: action,
                        ToyEnvironmentTrajectory.STATE_KEY: state,
                        ToyEnvironmentTrajectory.REWARD_KEY: reward,
                    },
                )
                step_times.append(int(step_timer.ticks() * 1000))
                ep_steps += 1

                ep_finished = done or truncated

                if done:
                    log.info(
                        f"Episode {episode} reached 'done' state after {recording.steps} steps"
                    )
                    break
                elif truncated:
                    log.info(
                        f"Episode {episode} reached 'truncated' state after {recording.steps} steps"
                    )
                    break

            ep_time = ep_timer.ticks()

            if len(step_times) > 0:
                log.info(
                    f"Step times avg: {int(np.mean(step_times))} ms,\
                            min={min(step_times)} ms, max={max(step_times)} ms"
                )

            self.rollout_telemetries.append(recording.compute_telemetry())
            self._log_episode_metrics(episode, ep_time, ep_steps, done, recording, info)
            recording = None

        # draw telemetry traces of all trajectories and reference trajectory in the environment
        telemetries_by_name = {
            "reference": [self.reference_trajectory.compute_telemetry()],
            "rollouts": self.rollout_telemetries,
        }
        trajectories_path = Path(self.output_dir) / "rollout_trajectories.jpg"
        assert isinstance(self.env, ToyEnvironment)
        draw_telemetry_traces(
            self.env.config,
            telemetries_by_name,
            plot_agent=True,
            save_path=trajectories_path,
        )

        if self.wandb_log:
            wandb.log({"rollout_trajectories": wandb.Image(str(trajectories_path))})

        return self._exp_result

    def _log_episode_metrics(
        self,
        episode: int,
        ep_time: float,
        ep_steps: int,
        done: bool,
        recording: ToyEnvironmentTrajectory,
        final_info: dict,
    ) -> List[str]:
        trajectory_name = f"ep_{str(episode)}"
        files_saved: List[str] = []

        self._exp_result.compute_metrics(
            ep_time, ep_steps, done, recording, info=final_info
        )
        log_dict = self._exp_result.get_log_dict()
        if self.save_trajectories:
            files_saved = recording.save_to_dir(
                self.output_dir, trajectory_name, save_video=self.save_video
            )

        if self.wandb_log:
            wandb.log(log_dict)
        return files_saved
