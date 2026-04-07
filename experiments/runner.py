from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

from experiments.common import (
    filter_experiments,
    find_checkpoint,
    get_results_file,
    launch_command,
    load_manifest,
    promote_latest_results,
    resolve_manifest_path,
    run_command,
)
from oss_configs.generate_configs_human import (
    DEFAULT_SUITE_PATH,
    generate_suite,
    load_yaml,
    resolve_output_dir,
)

SMOKE_TEST_ENV = "multiroom"
SMOKE_TEST_NUM_SAMPLES = 50
SMOKE_TEST_NUM_SEEDS = 1
SMOKE_TEST_EPISODES = 3
SMOKE_TEST_MAX_STEPS = 100


@dataclass
class ExperimentRunState:
    index: int
    experiment: dict[str, Any]
    checkpoint_path: Path | None = None
    training_required: bool = False
    training_failed: bool = False


@dataclass
class ActiveTrainingJob:
    state: ExperimentRunState
    process: subprocess.Popen[Any]
    log_handle: TextIO
    log_path: Path


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configs, run experiments, and create plots for the toy PIDM suite."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE_PATH,
        help="Path to the suite registry YAML.",
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=None,
        help="Override the suite config output directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the generated suite batch size. Defaults to the OSS paper value.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override trainer.max_steps in the generated configs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override the generated dataloader worker count.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=1,
        help="Number of training jobs to run side by side. Evaluation stays serial.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a small proof-of-life configuration intended to finish in a few minutes.",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="Reuse the existing manifest instead of regenerating configs.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and only evaluate existing checkpoints.",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Retrain even if a checkpoint already exists. Implies fresh evaluation unless --skip_eval is set.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation.",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip visualization generation.",
    )
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Re-run evaluation even if a results.json already exists.",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Forward --enable_wandb to evaluation runs.",
    )
    parser.add_argument(
        "--only_env",
        action="append",
        default=[],
        help="Restrict runs to one environment. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--only_method",
        action="append",
        default=[],
        help="Restrict runs to one method. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        action="append",
        default=[],
        help="Restrict runs to one training sample count. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Restrict runs to specific seeds.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Restrict runs to seeds less than this value.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List the experiments that match the filters without running them.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where run artifacts such as CSVs and plots will be written.",
    )
    return parser.parse_args()


def apply_smoke_test_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke_test:
        return args

    log_progress("Applying smoke-test preset.")
    if not args.only_env:
        args.only_env = [SMOKE_TEST_ENV]
    if not args.num_samples:
        args.num_samples = [SMOKE_TEST_NUM_SAMPLES]
    if args.num_seeds is None:
        args.num_seeds = SMOKE_TEST_NUM_SEEDS
    if args.episodes == 50:
        args.episodes = SMOKE_TEST_EPISODES
    if args.max_steps is None:
        args.max_steps = SMOKE_TEST_MAX_STEPS

    log_progress(
        "Smoke-test settings: "
        f"envs={args.only_env}, "
        f"num_samples={args.num_samples}, "
        f"num_seeds={args.num_seeds}, "
        f"episodes={args.episodes}, "
        f"max_steps={args.max_steps}."
    )
    return args


def ensure_manifest(args: argparse.Namespace) -> dict:
    if args.skip_generate and (
        args.batch_size is not None
        or args.max_steps is not None
        or args.num_workers is not None
    ):
        raise ValueError(
            "--batch_size/--max_steps/--num_workers require config regeneration; remove --skip_generate."
        )
    if not args.skip_generate:
        log_progress("Generating experiment configs and manifest.")
        return generate_suite(
            suite_path=args.suite,
            config_dir_override=args.config_dir,
            batch_size_override=args.batch_size,
            max_steps_override=args.max_steps,
            num_workers_override=args.num_workers,
        )

    suite_data = load_yaml(args.suite.resolve())
    config_dir = resolve_output_dir(args.suite, suite_data, args.config_dir)
    log_progress(f"Loading existing manifest from {config_dir}.")
    return load_manifest(config_dir)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "all"


def build_run_output_dir(args: argparse.Namespace, experiments: list[dict]) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    env_part = slugify("-".join(sorted(set(args.only_env)))) if args.only_env else "all-envs"
    method_part = (
        slugify("-".join(sorted(set(args.only_method)))) if args.only_method else "all-methods"
    )
    sample_part = (
        "samples-" + "-".join(str(sample) for sample in sorted(set(args.num_samples)))
        if args.num_samples
        else "all-samples"
    )
    seed_part = (
        "seeds-" + "-".join(str(seed) for seed in args.seeds)
        if args.seeds is not None
        else (
            f"seed-limit-{args.num_seeds}"
            if args.num_seeds is not None
            else "all-seeds"
        )
    )
    mode_part = "smoke" if args.smoke_test else "full"
    step_part = f"steps-{args.max_steps}" if args.max_steps is not None else "paper-steps"
    batch_part = f"batch-{args.batch_size}" if args.batch_size is not None else "paper-batch"
    worker_part = (
        f"workers-{args.num_workers}" if args.num_workers is not None else "paper-workers"
    )
    parallel_part = f"parallel-{args.parallel_jobs}"
    experiment_count = f"{len(experiments)}runs"

    run_name = "__".join(
        [
            timestamp,
            env_part,
            method_part,
            sample_part,
            seed_part,
            mode_part,
            step_part,
            batch_part,
            worker_part,
            parallel_part,
            experiment_count,
        ]
    )
    return Path("outputs") / run_name


def print_selected_experiments(experiments: list[dict]) -> None:
    print(f"Selected {len(experiments)} experiment configurations.", flush=True)
    for experiment in experiments:
        print(
            f" - {experiment['config_stem']}: "
            f"env={experiment['environment']} method={experiment['method']} "
            f"samples={experiment['num_train_samples']} seed={experiment['seed']}",
            flush=True,
        )


def build_wandb_env(enable_wandb: bool) -> dict[str, str]:
    if enable_wandb:
        return {}
    return {
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled",
    }


def copy_experiment_results(
    manifest: dict, experiment: dict, run_output_dir: Path
) -> None:
    results_dir = resolve_manifest_path(manifest, experiment["results_dir"])
    if not results_dir.exists():
        return
    dest = run_output_dir / "results" / experiment["config_stem"]
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(results_dir, dest)
    log_progress(f"Copied {experiment['config_stem']} results into {dest}.")


def build_training_command(manifest: dict, experiment: dict) -> list[str]:
    config_path = resolve_manifest_path(manifest, experiment["config_path"])
    return [
        sys.executable,
        "-u",
        "-m",
        "pidm_imitation.agents.supervised_learning.train",
        "--config",
        str(config_path),
        "--new",
    ]


def train_experiment(manifest: dict, experiment: dict, enable_wandb: bool) -> bool:
    return (
        run_command(
            build_training_command(manifest, experiment),
            env_overrides=build_wandb_env(enable_wandb),
        )
        == 0
    )


def resolve_experiment_states(
    manifest: dict,
    experiments: list[dict[str, Any]],
    force_train: bool,
    skip_train: bool,
    failed_training: list[str],
) -> list[ExperimentRunState]:
    states: list[ExperimentRunState] = []
    total = len(experiments)
    for index, experiment in enumerate(experiments, start=1):
        checkpoint_path = None if force_train else find_checkpoint(manifest, experiment)
        state = ExperimentRunState(
            index=index,
            experiment=experiment,
            checkpoint_path=checkpoint_path,
            training_required=checkpoint_path is None,
            training_failed=False,
        )
        if checkpoint_path is None and skip_train:
            print("Checkpoint missing and training is disabled, skipping.", flush=True)
            failed_training.append(experiment["config_stem"])
            state.training_failed = True
        elif checkpoint_path is None:
            if force_train:
                log_progress(
                    f"Experiment {index}/{total} will retrain from scratch: {experiment['config_stem']}."
                )
            else:
                log_progress(
                    f"Experiment {index}/{total} is missing a checkpoint and will be trained: {experiment['config_stem']}."
                )
        else:
            log_progress(
                f"Found checkpoint for {experiment['config_stem']} at {checkpoint_path}."
            )
        states.append(state)
    return states


def launch_training_job(
    manifest: dict,
    state: ExperimentRunState,
    enable_wandb: bool,
    logs_dir: Path,
) -> ActiveTrainingJob:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{state.experiment['config_stem']}.train.log"
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = launch_command(
            build_training_command(manifest, state.experiment),
            env_overrides=build_wandb_env(enable_wandb),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    return ActiveTrainingJob(
        state=state,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
    )


def run_training_jobs(
    manifest: dict,
    states: list[ExperimentRunState],
    enable_wandb: bool,
    parallel_jobs: int,
    logs_dir: Path,
    failed_training: list[str],
) -> None:
    pending_states = [
        state
        for state in states
        if state.training_required and not state.training_failed
    ]
    if not pending_states:
        return

    total = len(pending_states)
    active_jobs: list[ActiveTrainingJob] = []
    started = 0
    finished = 0
    log_progress(
        f"Launching {total} training jobs with parallel_jobs={parallel_jobs}. Training logs will be written to {logs_dir}."
    )

    while pending_states or active_jobs:
        while pending_states and len(active_jobs) < parallel_jobs:
            state = pending_states.pop(0)
            started += 1
            try:
                job = launch_training_job(
                    manifest=manifest,
                    state=state,
                    enable_wandb=enable_wandb,
                    logs_dir=logs_dir,
                )
            except OSError as exc:
                state.training_failed = True
                failed_training.append(state.experiment["config_stem"])
                log_progress(
                    f"Failed to launch training for {state.experiment['config_stem']}: {exc}."
                )
                continue
            active_jobs.append(job)
            log_progress(
                f"Started training {started}/{total}: {state.experiment['config_stem']} -> {job.log_path}"
            )

        if not active_jobs:
            continue

        finished_any = False
        for job in list(active_jobs):
            returncode = job.process.poll()
            if returncode is None:
                continue
            finished_any = True
            active_jobs.remove(job)
            job.log_handle.close()
            finished += 1

            state = job.state
            config_stem = state.experiment["config_stem"]
            if returncode != 0:
                state.training_failed = True
                failed_training.append(config_stem)
                log_progress(
                    f"Training failed for {config_stem} with return code {returncode}. See {job.log_path}."
                )
                continue

            checkpoint_path = find_checkpoint(manifest, state.experiment)
            if checkpoint_path is None:
                state.training_failed = True
                failed_training.append(config_stem)
                log_progress(
                    f"Training for {config_stem} completed without a checkpoint. See {job.log_path}."
                )
                continue

            state.checkpoint_path = checkpoint_path
            log_progress(
                f"Completed training {finished}/{total}: {config_stem} -> {checkpoint_path}"
            )

        if active_jobs and not finished_any:
            time.sleep(1)


def resolve_reference_recording(
    manifest: dict, experiment: dict, checkpoint_path: Path
) -> tuple[Path, Path] | None:
    input_trajectories_path = checkpoint_path.parent / "input_trajectories.json"
    if not input_trajectories_path.exists():
        return None

    with input_trajectories_path.open("r", encoding="utf-8") as handle:
        input_trajectories = json.load(handle)

    train_trajectories = input_trajectories.get("train", [])
    if not train_trajectories:
        return None

    trajectory_name = train_trajectories[0]
    dataset_dir = resolve_manifest_path(manifest, experiment["dataset_dir"])
    recording_path = dataset_dir / f"{trajectory_name}_inputs.json"
    recording_video_path = dataset_dir / f"{trajectory_name}_video.mp4"
    if not recording_path.exists() or not recording_video_path.exists():
        return None
    return recording_path, recording_video_path


def evaluate_experiment(
    manifest: dict,
    experiment: dict,
    checkpoint_path: Path,
    episodes: int,
    enable_wandb: bool,
    run_output_dir: Path,
) -> bool:
    config_path = resolve_manifest_path(manifest, experiment["config_path"])
    toy_config = resolve_manifest_path(manifest, experiment["toy_config"])
    results_dir = resolve_manifest_path(manifest, experiment["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        "-m",
        "pidm_imitation.toy_evaluate_model",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--toy_config",
        str(toy_config),
        "--agent",
        experiment["eval_agent"],
        "--episodes",
        str(episodes),
        "--output_dir",
        str(results_dir),
        "--save_results",
    ]
    resolved_recording = resolve_reference_recording(
        manifest=manifest,
        experiment=experiment,
        checkpoint_path=checkpoint_path,
    )
    if resolved_recording is not None:
        recording_path, recording_video_path = resolved_recording
        log_progress(
            f"Resolved reference trajectory to {recording_path.name} for evaluation."
        )
        command.extend(
            [
                "--recording",
                str(recording_path),
                "--recording_video",
                str(recording_video_path),
            ]
        )
    else:
        log_progress(
            "Could not resolve a concrete reference trajectory from input_trajectories.json; "
            "falling back to config defaults."
        )
    if enable_wandb:
        command.append("--enable_wandb")

    if run_command(command, env_overrides=build_wandb_env(enable_wandb)) != 0:
        return False

    promoted_results = promote_latest_results(manifest, experiment)
    if promoted_results is None:
        print(f"No evaluation results were produced for {experiment['config_stem']}.")
        return False
    copy_experiment_results(manifest, experiment, run_output_dir)
    print(f"Saved stable results to {promoted_results}.")
    return True


def main() -> int:
    start_time = datetime.now(timezone.utc)
    args = parse_args()
    args = apply_smoke_test_defaults(args)
    if args.force_train and args.skip_train:
        raise ValueError("--force_train cannot be combined with --skip_train.")
    if args.parallel_jobs <= 0:
        raise ValueError("--parallel_jobs must be a positive integer.")
    manifest = ensure_manifest(args)
    log_progress(
        f"Loaded suite with {len(manifest.get('experiments', []))} generated experiment configs."
    )
    experiments = filter_experiments(
        manifest=manifest,
        environments=args.only_env,
        methods=args.only_method,
        num_samples=args.num_samples,
        seeds=args.seeds,
        num_seeds=args.num_seeds,
    )

    if not experiments:
        print("No experiments matched the provided filters.", flush=True)
        return 1

    run_output_dir = build_run_output_dir(args, experiments)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"Writing run artifacts to {run_output_dir}.")
    log_progress(f"Filtered down to {len(experiments)} experiment runs.")
    print_selected_experiments(experiments)
    if args.dry_run:
        log_progress("Dry run requested. No training, evaluation, or plotting will be executed.")
        return 0

    failed_training: list[str] = []
    failed_evaluation: list[str] = []
    completed = 0
    training_logs_dir = run_output_dir / "logs"

    states = resolve_experiment_states(
        manifest=manifest,
        experiments=experiments,
        force_train=args.force_train,
        skip_train=args.skip_train,
        failed_training=failed_training,
    )
    run_training_jobs(
        manifest=manifest,
        states=states,
        enable_wandb=args.enable_wandb,
        parallel_jobs=args.parallel_jobs,
        logs_dir=training_logs_dir,
        failed_training=failed_training,
    )

    for state in states:
        index = state.index
        experiment = state.experiment
        log_progress(
            f"Starting experiment {index}/{len(experiments)}: {experiment['config_stem']}"
        )
        print(f"\n>>> {experiment['config_stem']} <<<", flush=True)
        if state.training_failed:
            continue
        checkpoint_path = state.checkpoint_path

        if checkpoint_path is None:
            print(
                "Training completed without a checkpoint, skipping evaluation.",
                flush=True,
            )
            failed_training.append(experiment["config_stem"])
            continue

        if args.skip_eval:
            completed += 1
            log_progress(
                f"Completed experiment {index}/{len(experiments)} with training/checkpoint only."
            )
            continue

        existing_results = get_results_file(manifest, experiment)
        if existing_results is not None:
            copy_experiment_results(manifest, experiment, run_output_dir)
            if not (args.force_eval or args.force_train):
                print(
                    f"Results already exist at {existing_results}, skipping evaluation.",
                    flush=True,
                )
                completed += 1
                log_progress(
                    f"Completed experiment {index}/{len(experiments)} using existing evaluation results."
                )
                continue

        log_progress(f"Launching evaluation from checkpoint {checkpoint_path}.")
        if not evaluate_experiment(
            manifest=manifest,
            experiment=experiment,
            checkpoint_path=checkpoint_path,
            episodes=args.episodes,
            enable_wandb=args.enable_wandb,
            run_output_dir=run_output_dir,
        ):
            failed_evaluation.append(experiment["config_stem"])
            continue

        completed += 1
        log_progress(f"Completed experiment {index}/{len(experiments)} successfully.")

    if not args.skip_plot:
        log_progress("Aggregating results and generating plots.")
        from experiments.plotter import parse_results, plot_results, save_tables

        plot_df = parse_results(
            manifest=manifest,
            environments=args.only_env,
            methods=args.only_method,
            num_samples=args.num_samples,
            seeds=args.seeds,
            num_seeds=args.num_seeds,
        )
        save_tables(plot_df, run_output_dir)
        plot_results(plot_df, run_output_dir)
        log_progress("Plot generation finished.")

    if failed_training:
        print("\nTraining failures:", flush=True)
        for config_stem in failed_training:
            print(f" - {config_stem}", flush=True)
    if failed_evaluation:
        print("\nEvaluation failures:", flush=True)
        for config_stem in failed_evaluation:
            print(f" - {config_stem}", flush=True)

    elapsed = datetime.now(timezone.utc) - start_time
    log_progress(
        "Run summary: "
        f"completed={completed}, "
        f"training_failures={len(failed_training)}, "
        f"evaluation_failures={len(failed_evaluation)}, "
        f"elapsed={elapsed}."
    )
    return 0 if not failed_training and not failed_evaluation else 1


if __name__ == "__main__":
    sys.exit(main())
