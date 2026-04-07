from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import (
    filter_experiments,
    get_results_file,
    load_manifest,
)

DEFAULT_SUMMARY_CSV = "experiment_results_summary.csv"
DEFAULT_AGGREGATED_CSV = "experiment_results_aggregated.csv"
DEFAULT_METRICS = [
    "success_rate",
    "avg_goal_completion_count",
    "avg_final_goal_distance",
    "avg_telemetry_distance",
    "avg_episode_return",
]
LOWER_IS_BETTER = {
    "avg_final_goal_distance",
    "avg_telemetry_distance",
}
METRIC_LABELS = {
    "success_rate": "Success Rate",
    "avg_goal_completion_count": "Goal Completion Count",
    "avg_final_goal_distance": "Final Goal Distance",
    "avg_telemetry_distance": "Telemetry Distance",
    "avg_episode_return": "Episode Return",
    "avg_step_count": "Step Count",
    "avg_episode_time": "Episode Time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from the generated suite manifest."
    )
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=Path("toy_configs"),
        help="Directory containing the generated configs and manifest.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the summary CSVs and metric images will be written.",
    )
    parser.add_argument(
        "--only_env",
        action="append",
        default=[],
        help="Restrict aggregation to one environment. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--only_method",
        action="append",
        default=[],
        help="Restrict aggregation to one method. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        action="append",
        default=[],
        help="Restrict aggregation to one training sample count. Pass multiple times for more than one.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Restrict aggregation to specific seeds.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Restrict aggregation to seeds less than this value.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot. Defaults to the core dev-time diagnostics.",
    )
    return parser.parse_args()


def parse_results(
    manifest: dict,
    environments: list[str] | None = None,
    methods: list[str] | None = None,
    num_samples: list[int] | None = None,
    seeds: list[int] | None = None,
    num_seeds: int | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    metric_keys = list(metrics or DEFAULT_METRICS)
    experiments = filter_experiments(
        manifest=manifest,
        environments=environments,
        methods=methods,
        num_samples=num_samples,
        seeds=seeds,
        num_seeds=num_seeds,
    )

    for experiment in experiments:
        results_file = get_results_file(manifest, experiment)
        if results_file is None:
            continue

        with results_file.open("r", encoding="utf-8") as handle:
            result_metrics = json.load(handle)

        row = {
            "config_stem": experiment["config_stem"],
            "environment": experiment["environment"],
            "environment_label": experiment["environment_label"],
            "method": experiment["method"],
            "method_label": experiment["method_label"],
            "num_train_samples": experiment["num_train_samples"],
            "seed": experiment["seed"],
            "results_file": str(results_file),
            "avg_step_count": result_metrics.get("avg_step_count"),
            "avg_episode_time": result_metrics.get("avg_episode_time"),
        }
        for metric in metric_keys:
            if metric == "success_rate":
                row[metric] = result_metrics.get("avg_success", 0.0)
            elif metric in result_metrics:
                row[metric] = result_metrics.get(metric)
            else:
                row[metric] = None
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in df.columns
        if column
        not in {
            "config_stem",
            "environment",
            "environment_label",
            "method",
            "method_label",
            "num_train_samples",
            "seed",
            "results_file",
        }
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    grouped = (
        df.groupby(
            [
                "environment",
                "environment_label",
                "method",
                "method_label",
                "num_train_samples",
            ],
            as_index=False,
        )
        .agg(
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
            **{
                f"sem_{column}": (column, "sem")
                for column in metric_columns
            },
            num_runs=("seed", "count"),
        )
        .sort_values(["environment", "method", "num_train_samples"])
    )
    for column in metric_columns:
        grouped[f"sem_{column}"] = grouped[f"sem_{column}"].fillna(0.0)
    return grouped


def save_tables(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / DEFAULT_SUMMARY_CSV
    aggregated_csv = output_dir / DEFAULT_AGGREGATED_CSV

    df.to_csv(summary_csv, index=False)
    aggregate_results(df).to_csv(aggregated_csv, index=False)
    print(f"Saved per-run summary to {summary_csv}")
    print(f"Saved aggregated summary to {aggregated_csv}")
    return summary_csv, aggregated_csv


def plot_results(
    df: pd.DataFrame, output_dir: Path, metrics: list[str] | None = None
) -> list[Path]:
    if df.empty:
        print("No evaluation results found to plot.")
        return []

    metric_keys = list(metrics or DEFAULT_METRICS)
    for metric in metric_keys:
        if metric not in df.columns:
            raise KeyError(f"Requested metric '{metric}' is not available in the results dataframe.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    envs = list(dict.fromkeys(df["environment"]))
    written_files: list[Path] = []
    for metric_index, metric in enumerate(metric_keys):
        fig, axes = plt.subplots(
            1,
            len(envs),
            figsize=(5 * len(envs), 4.5),
            squeeze=False,
            sharey=True,
        )
        first_ax = None
        for env_index, env_name in enumerate(envs):
            ax = axes[0][env_index]
            if first_ax is None:
                first_ax = ax
            env_df = df[df["environment"] == env_name].sort_values("num_train_samples")
            for method_label, method_df in env_df.groupby("method_label", sort=False):
                grouped = (
                    method_df.groupby("num_train_samples", as_index=False)
                    .agg(
                        mean_value=(metric, "mean"),
                        sem_value=(metric, "sem"),
                    )
                    .sort_values("num_train_samples")
                )
                grouped["sem_value"] = grouped["sem_value"].fillna(0.0)
                ax.errorbar(
                    grouped["num_train_samples"],
                    grouped["mean_value"],
                    yerr=grouped["sem_value"],
                    marker="o",
                    capsize=4,
                    label=method_label,
                )
            label = env_df["environment_label"].iloc[0]
            ax.set_title(label)
            metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            if metric in LOWER_IS_BETTER:
                metric_label += " (lower is better)"
            ax.set_ylabel(metric_label)
            ax.set_xlabel("Number of Training Demonstrations")
            if metric in LOWER_IS_BETTER:
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)

        if first_ax is not None:
            handles, labels = first_ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
            for ax in axes.flat:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        metric_file = output_dir / f"{metric}.png"
        plt.savefig(metric_file)
        plt.close(fig)
        print(f"Plot saved to {metric_file}")
        written_files.append(metric_file)

    return written_files


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.config_dir)
    df = parse_results(
        manifest=manifest,
        environments=args.only_env,
        methods=args.only_method,
        num_samples=args.num_samples,
        seeds=args.seeds,
        num_seeds=args.num_seeds,
        metrics=args.metrics,
    )
    save_tables(df, args.output_dir)
    plot_results(df, args.output_dir, metrics=args.metrics)


if __name__ == "__main__":
    main()
