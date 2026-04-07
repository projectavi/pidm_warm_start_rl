from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SUITE_PATH = Path(__file__).with_name("experiment_suite.yaml")
DEFAULT_MANIFEST_NAME = "manifest.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


def log_progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experiment configs and a manifest for the toy PIDM suite."
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
        help="Override the output config directory from the suite file.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default=DEFAULT_MANIFEST_NAME,
        help="Name of the manifest file written into the config directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the suite batch size while keeping the template structure unchanged.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override trainer.max_steps in generated configs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override the dataloader worker count in generated configs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at {path}, got {type(data)}.")
    return data


def resolve_repo_root(suite_path: Path) -> Path:
    return suite_path.resolve().parent.parent


def resolve_output_dir(
    suite_path: Path, suite_data: dict[str, Any], config_dir_override: Path | None
) -> Path:
    repo_root = resolve_repo_root(suite_path)
    if config_dir_override is not None:
        output_dir = config_dir_override
    else:
        output_dir = Path(suite_data.get("config_dir", "toy_configs"))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return output_dir.resolve()


def expand_seeds(seed_config: Any) -> list[int]:
    if isinstance(seed_config, int):
        return list(range(seed_config))
    if isinstance(seed_config, list):
        return [int(seed) for seed in seed_config]
    if isinstance(seed_config, dict):
        start = int(seed_config.get("start", 0))
        stop = int(seed_config["stop"])
        step = int(seed_config.get("step", 1))
        return list(range(start, stop, step))
    raise ValueError(f"Unsupported seeds specification: {seed_config!r}")


def get_validation_sample_count(
    suite_data: dict[str, Any], num_train_samples: int
) -> int:
    validation_config = suite_data.get("validation_samples", {})
    if isinstance(validation_config, int):
        return validation_config
    if not isinstance(validation_config, dict):
        raise ValueError(
            "validation_samples must be an int or mapping with default/overrides."
        )

    overrides = validation_config.get("overrides", {})
    if str(num_train_samples) in overrides:
        return int(overrides[str(num_train_samples)])
    if num_train_samples in overrides:
        return int(overrides[num_train_samples])
    return int(validation_config.get("default", 0))


def resolve_template_path(
    suite_path: Path, method_spec: dict[str, Any], env_name: str
) -> Path:
    template_by_env = method_spec.get("template_by_env")
    if isinstance(template_by_env, dict):
        template_path = template_by_env.get(env_name)
        if template_path is None:
            raise KeyError(
                f"Method is missing a template for environment '{env_name}'."
            )
    else:
        template_path = method_spec.get("template")
    if template_path is None:
        raise KeyError("Method must define either template or template_by_env.")
    return (suite_path.parent / template_path).resolve()


def normalize_template_values(values: dict[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (values or {}).items():
        token = key if key.startswith("<") and key.endswith(">") else f"<{key}>"
        normalized[token] = str(value)
    return normalized


def render_template(template_text: str, replacements: dict[str, str]) -> str:
    rendered_lines: list[str] = []
    for line in template_text.splitlines():
        for token, value in replacements.items():
            if token in line:
                indentation = line[: line.index(token)]
                line = line.replace(token, value.replace("\n", f"\n{indentation}"))
        rendered_lines.append(line)
    return "\n".join(rendered_lines) + "\n"


def build_replacements(
    suite_template_values: dict[str, str],
    env_name: str,
    env_spec: dict[str, Any],
    method_name: str,
    method_spec: dict[str, Any],
    num_train_samples: int,
    num_validation_samples: int,
    seed: int,
) -> dict[str, str]:
    dataset_name = env_spec["dataset_name"]
    replacements = {
        "<ENV>": env_name,
        "<ENV_FULL>": dataset_name,
        "<METHOD>": method_name,
        "<NUM_TRAIN_SAMPLES>": str(num_train_samples),
        "<NUM_VALIDATION_SAMPLES>": str(num_validation_samples),
        "<SEED>": str(seed),
    }
    replacements.update(suite_template_values)
    replacements.update(normalize_template_values(env_spec.get("template_values")))
    replacements.update(normalize_template_values(method_spec.get("template_values")))
    return replacements


def build_config_stem(
    env_name: str, method_name: str, num_train_samples: int, seed: int
) -> str:
    return f"{env_name}_{num_train_samples}_samples_seed_{seed}_{method_name}"


def generate_suite(
    suite_path: Path = DEFAULT_SUITE_PATH,
    config_dir_override: Path | None = None,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    batch_size_override: int | None = None,
    max_steps_override: int | None = None,
    num_workers_override: int | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    suite_path = suite_path.resolve()
    suite_data = load_yaml(suite_path)
    repo_root = resolve_repo_root(suite_path)
    config_dir = resolve_output_dir(suite_path, suite_data, config_dir_override)
    config_dir.mkdir(parents=True, exist_ok=True)

    environments = suite_data.get("environments", {})
    methods = suite_data.get("methods", {})
    if not environments:
        raise ValueError("Suite file must define at least one environment.")
    if not methods:
        raise ValueError("Suite file must define at least one method.")

    seeds = expand_seeds(suite_data.get("seeds", {"start": 0, "stop": 20}))
    num_train_samples_values = [
        int(sample) for sample in suite_data.get("num_train_samples", [])
    ]
    if not num_train_samples_values:
        raise ValueError("Suite file must define num_train_samples.")

    results_root = Path(suite_data.get("results_dir", "results"))
    suite_template_values = normalize_template_values(
        suite_data.get("template_values")
    )
    if batch_size_override is not None:
        suite_template_values["<BATCH_SIZE>"] = str(batch_size_override)
        log_progress(f"Overriding batch size to {batch_size_override}.")
    if num_workers_override is not None:
        suite_template_values["<NUM_WORKERS>"] = str(num_workers_override)
        log_progress(f"Overriding num_workers to {num_workers_override}.")
    if max_steps_override is not None:
        log_progress(f"Overriding trainer.max_steps to {max_steps_override}.")
    total_jobs = (
        len(environments)
        * len(methods)
        * len(seeds)
        * len(num_train_samples_values)
    )
    log_progress(
        f"Generating suite from {suite_path.name}: "
        f"{len(environments)} environments, {len(methods)} methods, "
        f"{len(seeds)} seeds, {len(num_train_samples_values)} sample counts "
        f"({total_jobs} configs expected)."
    )

    experiments: list[dict[str, Any]] = []
    generated_files = 0

    for env_name, env_spec in environments.items():
        dataset_name = env_spec["dataset_name"]
        toy_config = env_spec["toy_config"]
        environment_label = env_spec.get(
            "display_name", env_name.replace("_", " ").title()
        )
        env_expected = len(methods) * len(seeds) * len(num_train_samples_values)
        log_progress(
            f"Generating configs for environment '{env_name}' "
            f"({environment_label}, {env_expected} configs)."
        )

        for num_train_samples in num_train_samples_values:
            num_validation_samples = get_validation_sample_count(
                suite_data, num_train_samples
            )
            log_progress(
                f"  sample_count={num_train_samples}, validation_count={num_validation_samples}"
            )
            for seed in seeds:
                for method_name, method_spec in methods.items():
                    if method_spec.get("enabled", True) is False:
                        continue

                    template_path = resolve_template_path(
                        suite_path=suite_path,
                        method_spec=method_spec,
                        env_name=env_name,
                    )
                    with template_path.open("r", encoding="utf-8") as handle:
                        template_text = handle.read()

                    replacements = build_replacements(
                        suite_template_values=suite_template_values,
                        env_name=env_name,
                        env_spec=env_spec,
                        method_name=method_name,
                        method_spec=method_spec,
                        num_train_samples=num_train_samples,
                        num_validation_samples=num_validation_samples,
                        seed=seed,
                    )
                    rendered_config = render_template(template_text, replacements)
                    config_data = yaml.safe_load(rendered_config)
                    if max_steps_override is not None:
                        trainer_config = config_data.setdefault(
                            "pytorch_lightning", {}
                        ).setdefault("trainer", {})
                        trainer_config["max_steps"] = int(max_steps_override)
                        rendered_config = yaml.safe_dump(
                            config_data,
                            sort_keys=False,
                        )

                    config_stem = build_config_stem(
                        env_name=env_name,
                        method_name=method_name,
                        num_train_samples=num_train_samples,
                        seed=seed,
                    )
                    config_path = config_dir / f"{config_stem}.yaml"
                    config_path.write_text(rendered_config, encoding="utf-8")
                    generated_files += 1

                    checkpoint_dir = (
                        config_data.get("callbacks", {})
                        .get("checkpoint_callback_kwargs", {})
                        .get("dirpath")
                    )
                    if checkpoint_dir is None:
                        raise KeyError(
                            f"Config {config_path} is missing callbacks.checkpoint_callback_kwargs.dirpath."
                        )

                    agent_type = config_data.get("agent", {}).get("type")
                    if agent_type is None:
                        raise KeyError(f"Config {config_path} is missing agent.type.")

                    results_dir = results_root / config_stem
                    experiments.append(
                        {
                            "config_stem": config_stem,
                            "config_path": str(config_path.relative_to(repo_root)),
                            "experiment_name": config_data.get(
                                "experiment_name", config_stem
                            ),
                            "environment": env_name,
                            "environment_label": environment_label,
                            "dataset_name": dataset_name,
                            "dataset_dir": f"datasets/{dataset_name}",
                            "toy_config": toy_config,
                            "method": method_name,
                            "method_label": method_spec.get(
                                "display_name", method_name.upper()
                            ),
                            "seed": seed,
                            "num_train_samples": num_train_samples,
                            "num_validation_samples": num_validation_samples,
                            "agent_type": agent_type,
                            "eval_agent": method_spec.get("eval_agent", agent_type),
                            "template_path": str(
                                template_path.relative_to(repo_root)
                            ),
                            "checkpoint_dir": checkpoint_dir,
                            "results_dir": str(results_dir),
                            "results_file": str(results_dir / "results.json"),
                        }
                    )

            log_progress(
                f"  completed sample_count={num_train_samples}; generated {generated_files}/{total_jobs} configs so far."
            )

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_path": str(suite_path),
        "repo_root": str(repo_root),
        "config_dir": str(config_dir.relative_to(repo_root)),
        "results_dir": str(results_root),
        "methods": sorted(methods.keys()),
        "environments": list(environments.keys()),
        "experiments": experiments,
    }

    manifest_path = config_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = datetime.now(timezone.utc) - started_at
    log_progress(f"Generated {generated_files} configuration files in {config_dir}.")
    log_progress(f"Wrote manifest to {manifest_path}.")
    log_progress(f"Config generation completed in {elapsed}.")
    return manifest


def main() -> None:
    args = parse_args()
    generate_suite(
        suite_path=args.suite,
        config_dir_override=args.config_dir,
        manifest_name=args.manifest_name,
        batch_size_override=args.batch_size,
        max_steps_override=args.max_steps,
        num_workers_override=args.num_workers,
    )


if __name__ == "__main__":
    main()
