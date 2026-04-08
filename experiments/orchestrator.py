from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

from experiments.common import (
    build_command_env,
    filter_experiments,
    find_checkpoint,
    find_latest_rollout_results,
    launch_command,
    load_manifest,
    resolve_manifest_path,
)
from oss_configs.generate_configs_human import load_yaml


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-aware experiment orchestrator for explicit config lists and manifest-based "
            "experiment batches."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to a YAML or JSON orchestration plan.",
    )
    parser.add_argument(
        "--only_job",
        action="append",
        default=[],
        help="Restrict execution to one named job group from the plan. Pass multiple times if needed.",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Restrict execution to resolved runs whose names contain this substring. Pass multiple times if needed.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved runs and scheduler slots without launching anything.",
    )
    parser.add_argument(
        "--calibrate_slots",
        action="store_true",
        help="Run a train-only throughput sweep over slots_per_gpu candidates using temporary overridden configs.",
    )
    parser.add_argument(
        "--slot_candidates",
        type=int,
        nargs="+",
        default=None,
        help="slots_per_gpu values to benchmark during calibration. Defaults to plan defaults or 1 2 3 4.",
    )
    parser.add_argument(
        "--calibration_steps",
        type=int,
        default=None,
        help="Temporary trainer.max_steps to use for each calibration run. Defaults to plan defaults or 500.",
    )
    parser.add_argument(
        "--calibration_repeats",
        type=int,
        default=None,
        help="Number of repeats to run per slots_per_gpu candidate. Defaults to plan defaults or 2.",
    )
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_settings(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def qualify_run_name(job_name: str, run_name: str) -> str:
    if run_name.startswith(f"{job_name}/"):
        return run_name
    return f"{job_name}/{run_name}"


def resolve_repo_path(path_value: str | Path | None, repo_root: Path) -> Path | None:
    if path_value is None:
        return None
    path_str = str(path_value).replace("$REPO_DIRECTORY", str(repo_root))
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_wandb_env(enable_wandb: bool) -> dict[str, str]:
    if enable_wandb:
        return {}
    return {
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled",
    }


def load_plan(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return load_yaml(path.resolve())


def load_manifest_from_path(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_manifest(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_checkpoint_dir(config_data: dict[str, Any], repo_root: Path) -> Path:
    dirpath = (
        config_data.get("callbacks", {})
        .get("checkpoint_callback_kwargs", {})
        .get("dirpath")
    )
    if dirpath:
        resolved = resolve_repo_path(dirpath, repo_root)
        assert resolved is not None
        return resolved
    experiment_name = config_data.get("experiment_name", "manual_experiment")
    return (repo_root / "checkpoints" / experiment_name).resolve()


def infer_results_dir(
    config_data: dict[str, Any],
    config_path: Path,
    repo_root: Path,
    override: str | Path | None,
) -> Path:
    if override is not None:
        resolved = resolve_repo_path(override, repo_root)
        assert resolved is not None
        return resolved
    experiment_name = config_data.get("experiment_name", config_path.stem)
    return (repo_root / "results" / "manual" / experiment_name).resolve()


def resolve_reference_recording(
    manifest: dict[str, Any],
    experiment: dict[str, Any],
    checkpoint_path: Path,
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


def find_checkpoint_in_dir(checkpoint_dir: Path) -> Path | None:
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    candidates = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob("step=*.ckpt"))
    if candidates:
        return candidates[-1]
    return None


@dataclass
class GpuSlot:
    gpu_id: int
    slot_id: int

    @property
    def label(self) -> str:
        return f"gpu{self.gpu_id}-slot{self.slot_id}"


@dataclass
class RunSpec:
    name: str
    config_path: Path
    phases: tuple[str, ...]
    agent: str | None
    toy_config: Path | None
    checkpoint_dir: Path
    results_dir: Path
    episodes: int
    enable_wandb: bool
    force_train: bool
    force_eval: bool
    save_results: bool
    train_args: list[str] = field(default_factory=list)
    eval_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None
    manifest_experiment: dict[str, Any] | None = None

    @property
    def stable_results_file(self) -> Path:
        return self.results_dir / "results.json"


@dataclass
class RunState:
    spec: RunSpec
    pending_phases: deque[str]
    checkpoint_path: Path | None = None
    failed: bool = False
    completed_phases: list[str] = field(default_factory=list)


@dataclass
class ActiveProcess:
    state: RunState
    phase: str
    slot: GpuSlot
    process: Any
    log_handle: TextIO
    log_path: Path


@dataclass
class CalibrationTrial:
    slots_per_gpu: int
    repeat: int
    duration_seconds: float
    successful_runs: int
    failed_runs: int
    successful_steps: int
    steps_per_second: float
    runs_per_hour: float
    output_root: Path


def build_config_run_specs(
    repo_root: Path,
    job_name: str,
    job_settings: dict[str, Any],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    configs = as_list(job_settings.get("configs"))
    if not configs:
        raise ValueError(f"Job '{job_name}' must specify at least one config path.")

    phases = tuple(as_list(job_settings.get("phases", ["train", "eval"])))
    train_args = [str(arg) for arg in as_list(job_settings.get("train_args"))]
    eval_args = [str(arg) for arg in as_list(job_settings.get("eval_args"))]
    env_overrides = {
        str(key): str(value) for key, value in job_settings.get("env", {}).items()
    }

    for item in configs:
        item_settings: dict[str, Any]
        if isinstance(item, str):
            item_settings = {"path": item}
        else:
            item_settings = dict(item)

        config_path = resolve_repo_path(item_settings["path"], repo_root)
        assert config_path is not None
        config_data = load_yaml(config_path)
        config_stem = config_path.stem
        run_suffix = str(item_settings.get("name", config_stem))
        run_name = qualify_run_name(job_name, run_suffix)
        toy_config = resolve_repo_path(
            item_settings.get("toy_config", job_settings.get("toy_config")),
            repo_root,
        )
        agent = item_settings.get("agent", job_settings.get("agent"))
        if agent is None:
            agent = config_data.get("agent", {}).get("type")

        checkpoint_dir = infer_checkpoint_dir(config_data, repo_root)
        results_dir = infer_results_dir(
            config_data=config_data,
            config_path=config_path,
            repo_root=repo_root,
            override=item_settings.get("results_dir", job_settings.get("results_dir")),
        )

        specs.append(
            RunSpec(
                name=run_name,
                config_path=config_path,
                phases=phases,
                agent=agent,
                toy_config=toy_config,
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                episodes=int(item_settings.get("episodes", job_settings.get("episodes", 50))),
                enable_wandb=bool(
                    item_settings.get(
                        "enable_wandb", job_settings.get("enable_wandb", False)
                    )
                ),
                force_train=bool(
                    item_settings.get(
                        "force_train", job_settings.get("force_train", False)
                    )
                ),
                force_eval=bool(
                    item_settings.get(
                        "force_eval", job_settings.get("force_eval", False)
                    )
                ),
                save_results=bool(
                    item_settings.get(
                        "save_results", job_settings.get("save_results", True)
                    )
                ),
                train_args=train_args
                + [str(arg) for arg in as_list(item_settings.get("train_args"))],
                eval_args=eval_args
                + [str(arg) for arg in as_list(item_settings.get("eval_args"))],
                env_overrides={
                    **env_overrides,
                    **{
                        str(key): str(value)
                        for key, value in item_settings.get("env", {}).items()
                    },
                },
            )
        )
    return specs


def build_manifest_run_specs(
    repo_root: Path,
    job_name: str,
    job_settings: dict[str, Any],
) -> list[RunSpec]:
    manifest_path = resolve_repo_path(job_settings.get("manifest"), repo_root)
    if manifest_path is None:
        raise ValueError(f"Manifest job '{job_name}' must provide a manifest path.")
    manifest = load_manifest_from_path(manifest_path)
    filters = job_settings.get("filters", {})
    experiments = filter_experiments(
        manifest=manifest,
        environments=filters.get("environments"),
        methods=filters.get("methods"),
        num_samples=filters.get("num_samples"),
        seeds=filters.get("seeds"),
        num_seeds=filters.get("num_seeds"),
    )
    if not experiments:
        raise ValueError(f"Manifest job '{job_name}' matched no experiments.")

    phases = tuple(as_list(job_settings.get("phases", ["train", "eval"])))
    train_args = [str(arg) for arg in as_list(job_settings.get("train_args"))]
    eval_args = [str(arg) for arg in as_list(job_settings.get("eval_args"))]
    env_overrides = {
        str(key): str(value) for key, value in job_settings.get("env", {}).items()
    }
    specs: list[RunSpec] = []
    for experiment in experiments:
        specs.append(
            RunSpec(
                name=qualify_run_name(job_name, experiment["config_stem"]),
                config_path=resolve_manifest_path(manifest, experiment["config_path"]),
                phases=phases,
                agent=experiment["eval_agent"],
                toy_config=resolve_manifest_path(manifest, experiment["toy_config"]),
                checkpoint_dir=resolve_manifest_path(
                    manifest, experiment["checkpoint_dir"]
                ),
                results_dir=resolve_manifest_path(manifest, experiment["results_dir"]),
                episodes=int(job_settings.get("episodes", 50)),
                enable_wandb=bool(job_settings.get("enable_wandb", False)),
                force_train=bool(job_settings.get("force_train", False)),
                force_eval=bool(job_settings.get("force_eval", False)),
                save_results=bool(job_settings.get("save_results", True)),
                train_args=train_args,
                eval_args=eval_args,
                env_overrides=env_overrides,
                manifest=manifest,
                manifest_experiment=experiment,
            )
        )
    return specs


def build_run_specs(
    plan: dict[str, Any],
    repo_root: Path,
    selected_jobs: set[str] | None = None,
) -> tuple[list[RunSpec], dict[str, Any]]:
    defaults = dict(plan.get("defaults", {}))
    jobs = as_list(plan.get("jobs"))
    if not jobs:
        raise ValueError("Plan must contain at least one job entry.")

    all_specs: list[RunSpec] = []
    for idx, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Job at index {idx} must be a mapping, got {type(raw_job)}.")
        job_name = str(raw_job.get("name", f"job{idx}"))
        if selected_jobs is not None and job_name not in selected_jobs:
            continue
        job_settings = merge_settings(defaults, raw_job)

        if "configs" in job_settings:
            specs = build_config_run_specs(repo_root, job_name, job_settings)
        elif "manifest" in job_settings:
            specs = build_manifest_run_specs(repo_root, job_name, job_settings)
        else:
            raise ValueError(
                f"Job '{job_name}' must specify either 'configs' or 'manifest'."
            )
        all_specs.extend(specs)
    return all_specs, defaults


def sanitize_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def filter_run_specs(
    specs: list[RunSpec], only_job: list[str], match_filters: list[str]
) -> list[RunSpec]:
    selected = specs
    if only_job:
        allowed = set(only_job)
        selected = [
            spec for spec in selected if spec.name.split("/", 1)[0] in allowed
        ]
    if match_filters:
        lowered = [value.lower() for value in match_filters]
        selected = [
            spec
            for spec in selected
            if any(value in spec.name.lower() for value in lowered)
        ]
    return selected


def detect_results(results_dir: Path) -> Path | None:
    stable = results_dir / "results.json"
    if stable.exists():
        return stable
    return find_latest_rollout_results(results_dir)


def promote_results(results_dir: Path) -> Path | None:
    latest = find_latest_rollout_results(results_dir)
    if latest is None:
        return None
    stable = results_dir / "results.json"
    if latest.resolve() != stable.resolve():
        shutil.copy2(latest, stable)
    source_marker = results_dir / "results_source.txt"
    source_marker.write_text(f"{latest.parent.name}\n", encoding="utf-8")
    return stable


def resolve_existing_checkpoint(spec: RunSpec) -> Path | None:
    if spec.manifest is not None and spec.manifest_experiment is not None:
        checkpoint = find_checkpoint(spec.manifest, spec.manifest_experiment)
        if checkpoint is not None:
            return checkpoint
    return find_checkpoint_in_dir(spec.checkpoint_dir)


def apply_calibration_overrides(
    config_data: dict[str, Any],
    run_name: str,
    checkpoint_dir: Path,
    calibration_steps: int,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(config_data))
    updated["experiment_name"] = run_name

    trainer_config = updated.setdefault("pytorch_lightning", {}).setdefault("trainer", {})
    trainer_config["max_steps"] = calibration_steps

    callbacks_config = updated.setdefault("callbacks", {})
    checkpoint_kwargs = callbacks_config.setdefault("checkpoint_callback_kwargs", {})
    checkpoint_kwargs["dirpath"] = str(checkpoint_dir)
    checkpoint_kwargs["every_n_train_steps"] = calibration_steps
    checkpoint_kwargs["every_n_epochs"] = 0
    checkpoint_kwargs["save_last"] = True
    checkpoint_kwargs.pop("every_n_steps_custom", None)

    wandb_config = updated.get("wandb")
    if isinstance(wandb_config, dict):
        wandb_config["offline"] = True
        wandb_config["train_name"] = run_name
        wandb_config["eval_name"] = f"{run_name}/eval"
        wandb_config["train_group"] = "slot_calibration"
        wandb_config["eval_group"] = "slot_calibration"

    return updated


def build_calibration_specs(
    specs: list[RunSpec],
    trial_root: Path,
    slots_per_gpu: int,
    repeat: int,
    calibration_steps: int,
) -> list[RunSpec]:
    calibration_specs: list[RunSpec] = []
    config_dir = trial_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        config_data = load_yaml(spec.config_path.resolve())
        run_slug = sanitize_name(spec.name)
        calibration_name = f"{run_slug}__slots{slots_per_gpu}__repeat{repeat}"
        checkpoint_dir = trial_root / "checkpoints" / run_slug
        results_dir = trial_root / "results" / run_slug
        calibration_config_path = config_dir / f"{run_slug}.yaml"
        overridden = apply_calibration_overrides(
            config_data=config_data,
            run_name=calibration_name,
            checkpoint_dir=checkpoint_dir,
            calibration_steps=calibration_steps,
        )
        calibration_config_path.write_text(
            yaml.safe_dump(overridden, sort_keys=False),
            encoding="utf-8",
        )
        calibration_specs.append(
            replace(
                spec,
                config_path=calibration_config_path,
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                phases=("train",),
                enable_wandb=False,
                force_train=True,
                force_eval=False,
                save_results=False,
            )
        )
    return calibration_specs


def build_training_command(spec: RunSpec) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "pidm_imitation.agents.supervised_learning.train",
        "--config",
        str(spec.config_path),
        "--new",
        *spec.train_args,
    ]


def build_evaluation_command(spec: RunSpec, checkpoint_path: Path) -> list[str]:
    if spec.toy_config is None:
        raise ValueError(f"Run '{spec.name}' requires toy_config for evaluation.")
    if spec.agent is None:
        raise ValueError(f"Run '{spec.name}' requires an agent for evaluation.")

    command = [
        sys.executable,
        "-u",
        "-m",
        "pidm_imitation.toy_evaluate_model",
        "--config",
        str(spec.config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--toy_config",
        str(spec.toy_config),
        "--agent",
        spec.agent,
        "--episodes",
        str(spec.episodes),
        "--output_dir",
        str(spec.results_dir),
    ]
    if spec.save_results:
        command.append("--save_results")
    if spec.enable_wandb:
        command.append("--enable_wandb")
    if spec.manifest is not None and spec.manifest_experiment is not None:
        resolved_recording = resolve_reference_recording(
            manifest=spec.manifest,
            experiment=spec.manifest_experiment,
            checkpoint_path=checkpoint_path,
        )
        if resolved_recording is not None:
            recording_path, recording_video_path = resolved_recording
            command.extend(
                [
                    "--recording",
                    str(recording_path),
                    "--recording_video",
                    str(recording_video_path),
                ]
            )
    command.extend(spec.eval_args)
    return command


def build_phase_env(spec: RunSpec, slot: GpuSlot) -> dict[str, str]:
    env = {
        "CUDA_VISIBLE_DEVICES": str(slot.gpu_id),
        **build_wandb_env(spec.enable_wandb),
        **spec.env_overrides,
    }
    return env


def create_run_state(spec: RunSpec) -> RunState:
    pending_phases: deque[str] = deque()
    checkpoint_path = resolve_existing_checkpoint(spec)
    if "train" in spec.phases:
        if spec.force_train or checkpoint_path is None:
            pending_phases.append("train")
    elif checkpoint_path is None and "eval" in spec.phases:
        raise FileNotFoundError(
            f"Run '{spec.name}' requested eval without training, but no checkpoint was found in {spec.checkpoint_dir}."
        )

    if "eval" in spec.phases:
        existing_results = detect_results(spec.results_dir)
        should_eval = spec.force_eval or existing_results is None or "train" in pending_phases
        if should_eval:
            pending_phases.append("eval")

    return RunState(spec=spec, pending_phases=pending_phases, checkpoint_path=checkpoint_path)


def launch_phase(
    state: RunState,
    phase: str,
    slot: GpuSlot,
    logs_dir: Path,
) -> ActiveProcess:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{state.spec.name.replace('/', '__')}.{phase}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        if phase == "train":
            command = build_training_command(state.spec)
        elif phase == "eval":
            checkpoint_path = state.checkpoint_path or resolve_existing_checkpoint(state.spec)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"Run '{state.spec.name}' could not find a checkpoint for evaluation in {state.spec.checkpoint_dir}."
                )
            command = build_evaluation_command(state.spec, checkpoint_path)
        else:
            raise ValueError(f"Unsupported phase '{phase}'.")

        process = launch_command(
            command,
            env_overrides=build_phase_env(state.spec, slot),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise

    return ActiveProcess(
        state=state,
        phase=phase,
        slot=slot,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
    )


def complete_phase(job: ActiveProcess) -> tuple[bool, str]:
    spec = job.state.spec
    if job.phase == "train":
        checkpoint_path = resolve_existing_checkpoint(spec)
        if checkpoint_path is None:
            return False, f"Training finished without a checkpoint in {spec.checkpoint_dir}."
        job.state.checkpoint_path = checkpoint_path
        job.state.completed_phases.append("train")
        return True, f"Training completed -> {checkpoint_path}"

    if job.phase == "eval":
        promoted = promote_results(spec.results_dir) if spec.save_results else detect_results(spec.results_dir)
        if promoted is None and spec.save_results:
            return False, f"Evaluation finished without stable results in {spec.results_dir}."
        job.state.completed_phases.append("eval")
        return True, f"Evaluation completed -> {spec.results_dir}"

    return False, f"Unknown phase '{job.phase}'."


def schedule_runs(
    run_states: list[RunState],
    slots: list[GpuSlot],
    logs_dir: Path,
    poll_interval: float,
) -> int:
    pending = deque(state for state in run_states if state.pending_phases)
    active: list[ActiveProcess] = []
    available_slots = deque(slots)
    failures = 0
    launched = 0

    while pending or active:
        while pending and available_slots:
            state = pending.popleft()
            phase = state.pending_phases[0]
            slot = available_slots.popleft()
            launched += 1
            log_progress(
                f"Launching {launched}: {state.spec.name} [{phase}] on {slot.label}"
            )
            try:
                job = launch_phase(state=state, phase=phase, slot=slot, logs_dir=logs_dir)
            except Exception as exc:
                failures += 1
                state.failed = True
                available_slots.append(slot)
                log_progress(f"Failed to launch {state.spec.name} [{phase}]: {exc}")
                continue
            active.append(job)

        finished_any = False
        for job in list(active):
            returncode = job.process.poll()
            if returncode is None:
                continue
            finished_any = True
            active.remove(job)
            available_slots.append(job.slot)
            job.log_handle.close()

            if returncode != 0:
                failures += 1
                job.state.failed = True
                log_progress(
                    f"{job.state.spec.name} [{job.phase}] failed with return code {returncode}. See {job.log_path}"
                )
                continue

            ok, message = complete_phase(job)
            if not ok:
                failures += 1
                job.state.failed = True
                log_progress(f"{job.state.spec.name} [{job.phase}] failed: {message}")
                continue

            finished_phase = job.state.pending_phases.popleft()
            assert finished_phase == job.phase
            log_progress(f"{job.state.spec.name} [{job.phase}] succeeded: {message}")
            if job.state.pending_phases:
                pending.append(job.state)

        if active and not finished_any:
            time.sleep(poll_interval)

    return failures


def print_run_plan(specs: list[RunSpec], slots: list[GpuSlot]) -> None:
    print(f"Resolved {len(specs)} runs.", flush=True)
    print(
        f"Scheduler slots: {', '.join(slot.label for slot in slots)}",
        flush=True,
    )
    for spec in specs:
        phases = ",".join(spec.phases)
        print(
            f" - {spec.name}: phases={phases} config={spec.config_path} "
            f"checkpoint_dir={spec.checkpoint_dir} results_dir={spec.results_dir}",
            flush=True,
        )


def print_calibration_plan(
    specs: list[RunSpec],
    gpu_ids: list[int],
    slot_candidates: list[int],
    calibration_steps: int,
    calibration_repeats: int,
) -> None:
    print(f"Calibration base runs: {len(specs)}", flush=True)
    print(f"Calibration GPUs: {', '.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
    print(
        f"Calibration slots_per_gpu candidates: {', '.join(str(candidate) for candidate in slot_candidates)}",
        flush=True,
    )
    print(
        f"Calibration steps per run: {calibration_steps}; repeats per candidate: {calibration_repeats}",
        flush=True,
    )
    for spec in specs:
        print(f" - {spec.name}: config={spec.config_path}", flush=True)


def write_run_plan(
    output_root: Path,
    plan_path: Path,
    defaults: dict[str, Any],
    specs: list[RunSpec],
    slots: list[GpuSlot],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, output_root / "plan.yaml")
    payload = {
        "defaults": defaults,
        "slots": [slot.label for slot in slots],
        "runs": [
            {
                "name": spec.name,
                "config_path": str(spec.config_path),
                "phases": list(spec.phases),
                "agent": spec.agent,
                "toy_config": str(spec.toy_config) if spec.toy_config is not None else None,
                "checkpoint_dir": str(spec.checkpoint_dir),
                "results_dir": str(spec.results_dir),
                "episodes": spec.episodes,
                "enable_wandb": spec.enable_wandb,
                "force_train": spec.force_train,
                "force_eval": spec.force_eval,
                "save_results": spec.save_results,
                "train_args": spec.train_args,
                "eval_args": spec.eval_args,
                "env_overrides": spec.env_overrides,
            }
            for spec in specs
        ],
    }
    (output_root / "resolved_runs.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_slots(defaults: dict[str, Any]) -> list[GpuSlot]:
    gpu_ids = [int(value) for value in as_list(defaults.get("gpus", [0]))]
    return build_slots_for_gpu_ids(gpu_ids, int(defaults.get("slots_per_gpu", 1)))


def build_slots_for_gpu_ids(gpu_ids: list[int], slots_per_gpu: int) -> list[GpuSlot]:
    slots_per_gpu = int(slots_per_gpu)
    if slots_per_gpu <= 0:
        raise ValueError("slots_per_gpu must be positive.")
    if not gpu_ids:
        raise ValueError("At least one GPU id must be configured.")
    return [
        GpuSlot(gpu_id=gpu_id, slot_id=slot_id)
        for gpu_id in gpu_ids
        for slot_id in range(slots_per_gpu)
    ]


def build_slot_candidates(
    args: argparse.Namespace, defaults: dict[str, Any]
) -> list[int]:
    raw_candidates = args.slot_candidates
    if raw_candidates is None:
        raw_candidates = as_list(defaults.get("slot_candidates", [1, 2, 3, 4]))
    candidates = sorted({int(value) for value in raw_candidates})
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError("slot_candidates must contain positive integers.")
    return candidates


def write_calibration_summary(
    output_root: Path,
    trials: list[CalibrationTrial],
) -> None:
    calibration_root = output_root / "slot_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    trial_rows = [
        {
            "slots_per_gpu": trial.slots_per_gpu,
            "repeat": trial.repeat,
            "duration_seconds": trial.duration_seconds,
            "successful_runs": trial.successful_runs,
            "failed_runs": trial.failed_runs,
            "successful_steps": trial.successful_steps,
            "steps_per_second": trial.steps_per_second,
            "runs_per_hour": trial.runs_per_hour,
            "output_root": str(trial.output_root),
        }
        for trial in trials
    ]
    (calibration_root / "trials.json").write_text(
        json.dumps(trial_rows, indent=2),
        encoding="utf-8",
    )
    with (calibration_root / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    aggregates: dict[int, dict[str, float]] = {}
    for slots_per_gpu in sorted({trial.slots_per_gpu for trial in trials}):
        slot_trials = [trial for trial in trials if trial.slots_per_gpu == slots_per_gpu]
        count = len(slot_trials)
        aggregates[slots_per_gpu] = {
            "slots_per_gpu": float(slots_per_gpu),
            "num_trials": float(count),
            "mean_duration_seconds": sum(t.duration_seconds for t in slot_trials) / count,
            "mean_successful_runs": sum(t.successful_runs for t in slot_trials) / count,
            "mean_failed_runs": sum(t.failed_runs for t in slot_trials) / count,
            "mean_successful_steps": sum(t.successful_steps for t in slot_trials) / count,
            "mean_steps_per_second": sum(t.steps_per_second for t in slot_trials) / count,
            "mean_runs_per_hour": sum(t.runs_per_hour for t in slot_trials) / count,
        }

    aggregate_rows = [aggregates[key] for key in sorted(aggregates)]
    (calibration_root / "summary.json").write_text(
        json.dumps(aggregate_rows, indent=2),
        encoding="utf-8",
    )
    with (calibration_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)


def run_slot_calibration(
    specs: list[RunSpec],
    defaults: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> int:
    gpu_ids = [int(value) for value in as_list(defaults.get("gpus", [0]))]
    slot_candidates = build_slot_candidates(args, defaults)
    calibration_steps = int(
        args.calibration_steps
        if args.calibration_steps is not None
        else defaults.get("calibration_steps", 500)
    )
    calibration_repeats = int(
        args.calibration_repeats
        if args.calibration_repeats is not None
        else defaults.get("calibration_repeats", 2)
    )
    if calibration_steps <= 0:
        raise ValueError("calibration_steps must be positive.")
    if calibration_repeats <= 0:
        raise ValueError("calibration_repeats must be positive.")

    print_calibration_plan(
        specs=specs,
        gpu_ids=gpu_ids,
        slot_candidates=slot_candidates,
        calibration_steps=calibration_steps,
        calibration_repeats=calibration_repeats,
    )
    if args.dry_run:
        log_progress("Dry run requested; no calibration trials launched.")
        return 0

    trials: list[CalibrationTrial] = []
    for slots_per_gpu in slot_candidates:
        for repeat in range(1, calibration_repeats + 1):
            trial_root = (
                output_root
                / "slot_calibration"
                / f"slots_{slots_per_gpu}"
                / f"repeat_{repeat}"
            )
            trial_specs = build_calibration_specs(
                specs=specs,
                trial_root=trial_root,
                slots_per_gpu=slots_per_gpu,
                repeat=repeat,
                calibration_steps=calibration_steps,
            )
            run_states = [create_run_state(spec) for spec in trial_specs]
            slots = build_slots_for_gpu_ids(gpu_ids, slots_per_gpu)
            log_progress(
                f"Calibration trial start: slots_per_gpu={slots_per_gpu} repeat={repeat} "
                f"runs={len(trial_specs)} steps_per_run={calibration_steps}"
            )
            start_time = time.perf_counter()
            failures = schedule_runs(
                run_states=run_states,
                slots=slots,
                logs_dir=trial_root / "logs",
                poll_interval=float(defaults.get("poll_interval", 1.0)),
            )
            duration_seconds = time.perf_counter() - start_time
            successful_runs = sum(
                1
                for state in run_states
                if not state.failed and "train" in state.completed_phases
            )
            successful_steps = successful_runs * calibration_steps
            steps_per_second = (
                successful_steps / duration_seconds if duration_seconds > 0 else 0.0
            )
            runs_per_hour = (
                successful_runs * 3600.0 / duration_seconds
                if duration_seconds > 0
                else 0.0
            )
            trial = CalibrationTrial(
                slots_per_gpu=slots_per_gpu,
                repeat=repeat,
                duration_seconds=duration_seconds,
                successful_runs=successful_runs,
                failed_runs=failures,
                successful_steps=successful_steps,
                steps_per_second=steps_per_second,
                runs_per_hour=runs_per_hour,
                output_root=trial_root,
            )
            trials.append(trial)
            log_progress(
                f"Calibration trial done: slots_per_gpu={slots_per_gpu} repeat={repeat} "
                f"steps_per_second={steps_per_second:.2f} runs_per_hour={runs_per_hour:.2f} "
                f"successful_runs={successful_runs}/{len(run_states)}"
            )

    write_calibration_summary(output_root=output_root, trials=trials)
    valid_trials = [trial for trial in trials if trial.failed_runs == 0]
    ranked_trials = valid_trials if valid_trials else trials
    best_trial = max(ranked_trials, key=lambda trial: trial.steps_per_second)
    log_progress(
        f"Best calibration result: slots_per_gpu={best_trial.slots_per_gpu} "
        f"steps_per_second={best_trial.steps_per_second:.2f} "
        f"runs_per_hour={best_trial.runs_per_hour:.2f} "
        f"failures={best_trial.failed_runs}"
    )
    return 0 if all(trial.failed_runs == 0 for trial in trials) else 1


def build_output_root(defaults: dict[str, Any], repo_root: Path) -> Path:
    output_root = resolve_repo_path(defaults.get("output_root", "outputs/orchestrator"), repo_root)
    assert output_root is not None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return output_root / timestamp


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    plan = load_plan(args.plan.resolve())
    selected_jobs = set(args.only_job) if args.only_job else None
    specs, defaults = build_run_specs(plan, repo_root, selected_jobs)
    specs = filter_run_specs(specs, args.only_job, args.match)
    if not specs:
        print("No runs matched the requested filters.", flush=True)
        return 1

    slots = build_slots(defaults)
    output_root = build_output_root(defaults, repo_root)
    logs_dir = output_root / "logs"

    print_run_plan(specs, slots)

    write_run_plan(
        output_root=output_root,
        plan_path=args.plan,
        defaults=defaults,
        specs=specs,
        slots=slots,
    )
    if args.calibrate_slots:
        return run_slot_calibration(
            specs=specs,
            defaults=defaults,
            args=args,
            output_root=output_root,
        )
    if args.dry_run:
        log_progress("Dry run requested; no processes launched.")
        return 0

    run_states = [create_run_state(spec) for spec in specs]

    failures = schedule_runs(
        run_states=run_states,
        slots=slots,
        logs_dir=logs_dir,
        poll_interval=float(defaults.get("poll_interval", 1.0)),
    )
    completed = sum(
        1 for state in run_states if not state.failed and not state.pending_phases
    )
    log_progress(
        f"Orchestration summary: completed={completed}/{len(run_states)} failures={failures} logs={logs_dir}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
