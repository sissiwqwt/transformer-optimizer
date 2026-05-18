# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIRS = [
    REPO_ROOT / "results" / "cuda_base_test",
    REPO_ROOT / "results" / "cpu_base_test",
]
REQUIRED_COLUMNS = {
    "dataset",
    "press",
    "compression_ratio",
    "ppl",
    "throughput_tokens_s",
}


def resolve_csv_path(path: str | Path) -> Path:
    csv_path = Path(path).expanduser()
    if csv_path.is_absolute() and csv_path.exists():
        return csv_path

    repo_path = REPO_ROOT / csv_path
    if repo_path.exists():
        return repo_path

    for results_dir in DEFAULT_RESULTS_DIRS:
        candidate = results_dir / csv_path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find CSV file: {path}")


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    for column in ("compression_ratio", "effective_compression_ratio", "keep_ratio", "ppl", "throughput_tokens_s"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["compression_ratio", "ppl", "throughput_tokens_s"])
    return df.sort_values(["dataset", "press", "compression_ratio"])


def format_title(metric: str, dataset: str, df: pd.DataFrame) -> str:
    details = []
    for column in ("model", "device", "context_tokens", "target_tokens", "samples"):
        if column in df.columns and df[column].nunique(dropna=True) == 1:
            details.append(f"{column}={df[column].dropna().iloc[0]}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{metric} on {dataset}{suffix}"


def finish_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_metric_curve(df: pd.DataFrame, dataset: str, metric: str, ylabel: str, output_path: Path) -> None:
    plot_df = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for press, group in plot_df.groupby("press", sort=True):
        group = group.sort_values("compression_ratio")
        ax.plot(group["compression_ratio"], group[metric], marker="o", linewidth=2, label=press)

    ax.set_title(format_title(ylabel, dataset, plot_df))
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Press")
    finish_plot(fig, output_path)


def plot_tradeoff(df: pd.DataFrame, dataset: str, output_path: Path) -> None:
    plot_df = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for press, group in plot_df.groupby("press", sort=True):
        ax.scatter(group["ppl"], group["throughput_tokens_s"], s=70, label=press)
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['compression_ratio']:.1f}",
                (row["ppl"], row["throughput_tokens_s"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(format_title("PPL vs Throughput", dataset, plot_df))
    ax.set_xlabel("PPL (lower is better)")
    ax.set_ylabel("Throughput tokens/s (higher is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Press")
    finish_plot(fig, output_path)


def plot_relative_to_none(df: pd.DataFrame, dataset: str, output_path: Path) -> None:
    plot_df = df[df["dataset"] == dataset].copy()
    baseline = plot_df[plot_df["press"] == "none"][["compression_ratio", "ppl", "throughput_tokens_s"]]
    if baseline.empty:
        return

    baseline = baseline.rename(
        columns={
            "ppl": "baseline_ppl",
            "throughput_tokens_s": "baseline_throughput_tokens_s",
        }
    )
    plot_df = plot_df.merge(baseline, on="compression_ratio", how="left")
    plot_df["ppl_ratio_to_none"] = plot_df["ppl"] / plot_df["baseline_ppl"]
    plot_df["throughput_ratio_to_none"] = plot_df["throughput_tokens_s"] / plot_df["baseline_throughput_tokens_s"]
    plot_df = plot_df[plot_df["press"] != "none"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for press, group in plot_df.groupby("press", sort=True):
        group = group.sort_values("compression_ratio")
        axes[0].plot(group["compression_ratio"], group["ppl_ratio_to_none"], marker="o", linewidth=2, label=press)
        axes[1].plot(
            group["compression_ratio"],
            group["throughput_ratio_to_none"],
            marker="o",
            linewidth=2,
            label=press,
        )

    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("PPL ratio to no compression")
    axes[0].set_xlabel("Compression ratio")
    axes[0].set_ylabel("PPL / none PPL")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Throughput ratio to no compression")
    axes[1].set_xlabel("Compression ratio")
    axes[1].set_ylabel("Throughput / none throughput")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(title="Press")

    finish_plot(fig, output_path)


def plot_results(csv_path: Path, output_dir: Path) -> None:
    df = load_results(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].dropna().unique()):
        safe_dataset = str(dataset).replace("/", "_")
        plot_metric_curve(
            df=df,
            dataset=dataset,
            metric="ppl",
            ylabel="PPL",
            output_path=output_dir / f"{csv_path.stem}_{safe_dataset}_ppl.png",
        )
        plot_metric_curve(
            df=df,
            dataset=dataset,
            metric="throughput_tokens_s",
            ylabel="Throughput tokens/s",
            output_path=output_dir / f"{csv_path.stem}_{safe_dataset}_throughput.png",
        )
        plot_tradeoff(
            df=df,
            dataset=dataset,
            output_path=output_dir / f"{csv_path.stem}_{safe_dataset}_tradeoff.png",
        )
        plot_relative_to_none(
            df=df,
            dataset=dataset,
            output_path=output_dir / f"{csv_path.stem}_{safe_dataset}_relative_to_none.png",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pythia_kvpress_test compression-ratio sweep results.")
    parser.add_argument(
        "--csv",
        default="pythia_kvpress_test_results.csv",
        help="CSV path. Can be absolute, repo-relative, or a filename in results/cuda_base_test or results/cpu_base_test.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated PNGs. Defaults to <csv parent>/figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else csv_path.parent / "figures"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    plot_results(csv_path, output_dir)


if __name__ == "__main__":
    main()
