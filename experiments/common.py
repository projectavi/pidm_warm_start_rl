from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, IO, Iterable, Mapping

DEFAULT_MANIFEST_NAME = "manifest.json"


def build_command_env(
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["PAGER"] = "cat"
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return env


def launch_command(
    command: list[str],
    cwd: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
) -> subprocess.Popen[Any]:
    print(f"Running: {' '.join(command)}", flush=True)
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=build_command_env(env_overrides),
        stdout=stdout,
        stderr=stderr,
    )


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> int:
    print(f"Running: {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=build_command_env(env_overrides),
    )
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}", flush=True)
    return result.returncode


def load_manifest(config_dir: str | Path) -> dict[str, Any]:
    config_dir = Path(config_dir)
    manifest_path = config_dir / DEFAULT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find manifest at {manifest_path}. Regenerate configs first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_manifest_path(manifest: dict[str, Any], path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(manifest["repo_root"])
    return (repo_root / path).resolve()


def filter_experiments(
    manifest: dict[str, Any],
    environments: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    num_samples: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    num_seeds: int | None = None,
) -> list[dict[str, Any]]:
    environment_filter = set(environments or [])
    method_filter = set(methods or [])
    sample_filter = set(num_samples or [])
    seed_filter = set(seeds or [])

    filtered: list[dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        if environment_filter and experiment["environment"] not in environment_filter:
            continue
        if method_filter and experiment["method"] not in method_filter:
            continue
        if sample_filter and experiment["num_train_samples"] not in sample_filter:
            continue
        if seed_filter and experiment["seed"] not in seed_filter:
            continue
        if num_seeds is not None and experiment["seed"] >= num_seeds:
            continue
        filtered.append(experiment)
    return filtered


def find_checkpoint(manifest: dict[str, Any], experiment: dict[str, Any]) -> Path | None:
    checkpoint_dir = resolve_manifest_path(manifest, experiment["checkpoint_dir"])
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    candidates = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob("step=*.ckpt"))
    if candidates:
        return candidates[-1]
    return None


def find_latest_rollout_results(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = [
        candidate
        for candidate in results_dir.glob("rollout*/results.json")
        if candidate.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def get_results_file(manifest: dict[str, Any], experiment: dict[str, Any]) -> Path | None:
    results_dir = resolve_manifest_path(manifest, experiment["results_dir"])
    stable_results = results_dir / "results.json"
    if stable_results.exists():
        return stable_results
    return find_latest_rollout_results(results_dir)


def promote_latest_results(
    manifest: dict[str, Any], experiment: dict[str, Any]
) -> Path | None:
    results_dir = resolve_manifest_path(manifest, experiment["results_dir"])
    latest_results = find_latest_rollout_results(results_dir)
    if latest_results is None:
        return None

    stable_results = results_dir / "results.json"
    if latest_results.resolve() != stable_results.resolve():
        shutil.copy2(latest_results, stable_results)
    source_marker = results_dir / "results_source.txt"
    source_marker.write_text(f"{latest_results.parent.name}\n", encoding="utf-8")
    return stable_results
