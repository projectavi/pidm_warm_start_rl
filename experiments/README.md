# PIDM Experiment Suite

This directory now provides two complementary workflows for the toy navigation experiments:

1. `oss_configs/generate_configs_human.py` generates the exact OSS baseline configs and writes `toy_configs/manifest.json`.
2. `experiments/runner.py` consumes that manifest to train missing checkpoints, evaluate them, and optionally generate plots.
3. `experiments/orchestrator.py` runs an explicit plan file with GPU-aware scheduling for direct config lists or manifest-filtered batches.
4. `experiments/plotter.py` reads the same manifest to aggregate results without guessing metadata from directory names.

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

To keep small-model trainings busier by running multiple jobs side by side:

```bash
python experiments/runner.py --batch_size 8192 --num_workers 8 --parallel_jobs 2
```

Parallel training writes one log file per experiment under `outputs/<run_name>/logs/`.

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

## GPU-Aware Orchestration

Use `experiments/orchestrator.py` when you want a reusable execution plan instead of one-off CLI filters. The orchestrator is designed for:

- explicit hand-picked config lists
- manifest-driven subsets of the suite
- multiple concurrent runs on a single GPU
- concurrent runs spread across multiple GPUs

The scheduler model is:

- `gpus`: which GPU ids are available to the orchestrator
- `slots_per_gpu`: how many concurrent jobs to allow on each GPU

So `gpus: [0, 1]` and `slots_per_gpu: 2` creates four execution slots:

- `gpu0-slot0`
- `gpu0-slot1`
- `gpu1-slot0`
- `gpu1-slot1`

Each launched subprocess receives the appropriate `CUDA_VISIBLE_DEVICES` value automatically.

Dry-run an example plan:

```bash
python experiments/orchestrator.py --plan experiments/plans/ssidm_compare.yaml --dry_run
```

Run only one named job from the plan:

```bash
python experiments/orchestrator.py \
  --plan experiments/plans/ssidm_compare.yaml \
  --only_job nonlinear-smokes
```

Restrict further to resolved runs whose names contain a substring:

```bash
python experiments/orchestrator.py \
  --plan experiments/plans/ssidm_compare.yaml \
  --only_job nonlinear-smokes \
  --match prenorm
```

The example plan shows both supported job types:

- `configs`: explicit config paths with optional per-config `toy_config`, `agent`, `episodes`, `train_args`, `eval_args`, and `env`
- `manifest`: a generated `manifest.json` plus suite-style filters like `environments`, `methods`, `num_samples`, `seeds`, and `num_seeds`

Outputs for each orchestration run go under `outputs/orchestrator/<timestamp>/` by default:

- `plan.yaml`: the copied execution plan
- `resolved_runs.json`: the fully expanded run inventory
- `logs/*.train.log`: per-run training logs
- `logs/*.eval.log`: per-run evaluation logs

Use the orchestrator when you want to design a batch once and re-run the same plan later without reconstructing CLI filters by hand. Use `runner.py` when you want the existing single-command “generate the suite and run it” path.

### Calibrating `slots_per_gpu`

The orchestrator can now benchmark `slots_per_gpu` empirically instead of guessing. Calibration mode:

- expands the selected runs from your plan
- creates temporary train-only configs under `outputs/orchestrator/.../slot_calibration/...`
- overrides `trainer.max_steps` for a short benchmark
- gives every calibration run its own temporary checkpoint directory
- disables WandB for the benchmark subprocesses
- reports aggregate throughput by `slots_per_gpu`

Example dry run:

```bash
python experiments/orchestrator.py \
  --plan experiments/plans/ssidm_compare.yaml \
  --calibrate_slots \
  --slot_candidates 1 2 3 \
  --calibration_steps 500 \
  --calibration_repeats 2 \
  --only_job nonlinear-all-test \
  --match pssidm \
  --dry_run
```

Run a real calibration sweep:

```bash
python experiments/orchestrator.py \
  --plan experiments/plans/ssidm_compare.yaml \
  --calibrate_slots \
  --slot_candidates 1 2 3 \
  --calibration_steps 500 \
  --calibration_repeats 2 \
  --only_job nonlinear-all-test
```

Useful guidance:

- calibrate on a representative subset, not the full paper sweep
- measure `train` only first; eval and WandB add noise
- use enough steps that startup overhead is not dominant
- repeat each candidate at least twice

Calibration outputs are written under `outputs/orchestrator/<timestamp>/slot_calibration/`:

- `trials.json` and `trials.csv`: one row per `(slots_per_gpu, repeat)` trial
- `summary.json` and `summary.csv`: mean throughput grouped by `slots_per_gpu`
- `slots_<n>/repeat_<k>/configs/*.yaml`: generated temporary configs
- `slots_<n>/repeat_<k>/logs/*.train.log`: training logs for each calibration run

The main throughput metric is `mean_steps_per_second`. The orchestrator also reports `mean_runs_per_hour` and prefers zero-failure candidates when choosing the best result.

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
