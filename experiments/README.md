# PIDM Experiment Suite

This directory now provides a manifest-driven workflow for the toy navigation experiments:

1. `oss_configs/generate_configs_human.py` generates the exact OSS baseline configs and writes `toy_configs/manifest.json`.
2. `experiments/runner.py` consumes that manifest to train missing checkpoints, evaluate them, and optionally generate plots.
3. `experiments/plotter.py` reads the same manifest to aggregate results without guessing metadata from directory names.

## Quick Start

Run the full paper baseline sweep and create summaries:

```bash
python experiments/runner.py --num_seeds 20
```

The exact OSS baseline path keeps the paper batch size of `4096`. To regenerate the same suite with a larger batch size for your GPU:

```bash
python experiments/runner.py --batch_size 8192
```

To regenerate the suite with more dataloader workers as well:

```bash
python experiments/runner.py --batch_size 8192 --num_workers 8
```

For a quick proof-of-life run that is intended to finish in a few minutes instead of launching the full paper-scale setup:

```bash
python experiments/runner.py --smoke_test
```

The smoke-test preset defaults to:

- `multiroom`
- `50` training samples
- `1` seed
- `3` evaluation episodes
- `100` training steps per config

Useful filters:

```bash
python experiments/runner.py --only_env multiroom
python experiments/runner.py --only_method bc --num_samples 50
python experiments/runner.py --skip_train --skip_plot
python experiments/runner.py --dry_run --only_method idm --num_samples 10
python experiments/runner.py --only_env multiroom --num_samples 50 --num_seeds 1 --max_steps 500 --force_train
```

Generate plots again without re-running experiments:

```bash
python experiments/plotter.py
```

Outputs:

- `toy_configs/manifest.json`: the generated experiment inventory.
- `outputs/<run_name>/experiment_results_summary.csv`: one row per evaluated config/seed.
- `outputs/<run_name>/experiment_results_aggregated.csv`: grouped mean and standard error.
- `outputs/<run_name>/success_rate.png`: per-metric curve image.
- `outputs/<run_name>/avg_goal_completion_count.png`: per-metric curve image.
- `outputs/<run_name>/avg_final_goal_distance.png`: per-metric curve image.
- `outputs/<run_name>/avg_telemetry_distance.png`: per-metric curve image.
- `outputs/<run_name>/avg_episode_return.png`: per-metric curve image.
- `outputs/<run_name>/results/<config_stem>/`: stable `results.json`, `results_source.txt`, and `rollout_last_*` artifacts including `rollout_trajectories.jpg`.

Each run gets its own output directory under `outputs/` unless you pass `--output_dir`.

## Adding a New Method

To compare a new method against PIDM and BC under the same seeds, sample counts, and environments:

1. Implement and register the model in training and evaluation code.
   - Training registry: `pidm_imitation/agents/supervised_learning/model_factory.py`
   - Evaluation registry: `pidm_imitation/evaluation/toy_agents.py`
2. Add one template or per-environment templates under `oss_configs/templates/`.
3. Add the method to [`oss_configs/experiment_suite.yaml`](/home/martyna/Documents/Avi/pidm_warm_start_rl/oss_configs/experiment_suite.yaml).
   - Set `display_name`.
   - Set `eval_agent`.
   - Point `template` or `template_by_env` at the new YAML template files.
4. Re-run `python experiments/runner.py`.

The runner will generate configs for the new method with the same environment list, seeds, and training sample counts as the OSS baselines.
