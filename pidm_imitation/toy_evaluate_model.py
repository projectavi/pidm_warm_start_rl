# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import sys

import wandb

from pidm_imitation.agents import Agent
from pidm_imitation.environment.context import EvaluationContext
from pidm_imitation.environment.toy_env.configs import (
    ToyEnvironmentConfig,
    ToyEnvRefTrajectoryConfig,
    ToyPLConfigFile,
)
from pidm_imitation.environment.toy_env.toy_environment_base import ToyEnvironment
from pidm_imitation.environment.toy_env.toy_factory import ToyEnvironmentFactory
from pidm_imitation.evaluation.argparse_utils import add_toy_eval_args
from pidm_imitation.evaluation.toy_agents import ToyPytorchAgentWrapperFactory
from pidm_imitation.evaluation.toy_env_experiments import (
    ToyEnvExperiment,
    ToyEnvExperimentResult,
    ToyEvaluationContext,
)
from pidm_imitation.evaluation.toy_utils import create_toy_config_parser
from pidm_imitation.evaluation.toy_valid_agents import ValidToyAgents
from pidm_imitation.evaluation.utils import evaluate_common_setup
from pidm_imitation.utils import GameTimer, Logger

logger = Logger()
log = logger.get_root_logger()


def parse_args():
    parser = argparse.ArgumentParser("Evaluate models in toy environment")
    add_toy_eval_args(parser, ValidToyAgents.ALL)
    return parser.parse_args()


def toy_env_evaluate(
    context: ToyEvaluationContext,
    config: ToyPLConfigFile,
    env_config: ToyEnvironmentConfig,
) -> ToyEnvExperimentResult | None:
    experiment = config.experiment_name

    log.info("Running an Agent experiment in toy environment")
    ref_traj_config = config.get_subconfig_att(ToyEnvRefTrajectoryConfig)
    assert ref_traj_config is not None

    env = ToyEnvironmentFactory.create_environment(env_config)

    result: ToyEnvExperimentResult | None = None
    agent = create_agent(
        context=context,
        config=config,
        env_config=env_config,
        env=env,
    )

    exp = ToyEnvExperiment(
        context=context,
        name=experiment,
        env=env,
        agent=agent,
        reference_trajectory=ref_traj_config.get_trajectory_obj(
            data_dir=config.data_config.training_dir
        ),
    )

    try:
        while result is None:
            timer = GameTimer()
            timer.start()
            result = exp.run()
            log.info(
                f"Evaluation of {result.number_of_episodes} rollouts completed in {timer.ticks():.3f} seconds."
            )
    finally:
        env.close()

    return result


def create_agent(
    context: EvaluationContext,
    config: ToyPLConfigFile,
    env_config: ToyEnvironmentConfig,
    env: ToyEnvironment,
) -> Agent:
    video_width, video_height = env_config.room_size

    assert ValidToyAgents.is_toy_agent(
        context.agent_name
    ), f"Invalid agent name: {context.agent_name}"
    return ToyPytorchAgentWrapperFactory.get_agent(
        agent_name=context.agent_name,
        model_path=context.checkpoint_dir,
        checkpoint_name=context.checkpoint,
        config=config,  # type: ignore
        video_height=video_height,
        video_width=video_width,
        env=env,  # type: ignore
        input_trajectories=context.input_trajectories,  # type: ignore
    )


def main() -> int:

    args = ToyEvaluationContext(parse_args())

    evaluate_common_setup(args, logger, create_toy_config_parser)

    env_config = ToyEnvironmentConfig(args.toy_config)
    assert isinstance(args.config, ToyPLConfigFile)

    config: ToyPLConfigFile = args.config

    try:
        toy_env_evaluate(context=args, config=config, env_config=env_config)
        return 0
    except Exception as e:
        log.error(f"Error during evaluation: {e}", exc_info=True, stack_info=True)
        return 1
    finally:
        if args.wandb_config is not None:
            wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
