# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_NAME = "uncertainty_aware"
RESULTS_DIRS = {
    "uncertainty_aware": REPO_ROOT / "results" / "uncertainty_aware",
    "pythia_kvpress_test": REPO_ROOT / "results" / "pythia_kvpress_test",
}
REQUIRED_COLUMNS = {"dataset", "press", "ppl"}


def list_csv_files(results_dir: Path) -> list[Path]:
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    return csv_files


def resolve_results_dirs(results_dir: str) -> list[Path]:
    if results_dir == "both":
        return list(RESULTS_DIRS.values())

    if results_dir in RESULTS_DIRS:
        return [RESULTS_DIRS[results_dir]]

    path = Path(results_dir).expanduser()
    if path.is_absolute():
        return [path]

    direct_path = REPO_ROOT / path
    if direct_path.exists():
        return [direct_path]

    return [REPO_ROOT / "results" / results_dir]


def resolve_csv_paths(csv_path: str | None, results_dirs: list[Path]) -> list[Path]:
    if csv_path is None:
        csv_paths = []
        for results_dir in results_dirs:
            csv_paths.extend(list_csv_files(results_dir))
        return csv_paths

    path = Path(csv_path).expanduser()
    if path.is_absolute():
        return [path]

    direct_path = REPO_ROOT / path
    if direct_path.exists():
        return [direct_path]

    csv_paths = [results_dir / path for results_dir in results_dirs if (results_dir / path).exists()]
    if csv_paths:
        return csv_paths

    return [results_dirs[0] / path]


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_ppl_comparison.png")


def resolve_output_path(output: str | None, csv_paths: list[Path], csv_path: Path) -> Path:
    if output is None:
        return default_output_path(csv_path)

    output_path = Path(output).expanduser()
    if len(csv_paths) == 1:
        return output_path

    return output_path / default_output_path(csv_path).name


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ppl"] = pd.to_numeric(df["ppl"], errors="raise")
    return df


def plot_ppl(df: pd.DataFrame, csv_path: Path, output_path: Path, use_log_scale: bool) -> None:
    pivot = df.pivot_table(index="press", columns="dataset", values="ppl", aggfunc="mean")

    preferred_order = ["none", "adakv", "knorm", "snapkv", "streamingllm"]
    ordered_presses = [press for press in preferred_order if press in pivot.index]
    ordered_presses.extend([press for press in pivot.index if press not in ordered_presses])
    pivot = pivot.loc[ordered_presses]

    ax = pivot.plot(kind="bar", figsize=(10, 5.5), width=0.78)
    ax.set_title("Pythia KVPress PPL Comparison")
    ax.set_xlabel("Press")
    ax.set_ylabel("Perplexity")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Dataset")

    if use_log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Perplexity (log scale)")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)

    metadata = []
    for column in ("model", "compression_ratio", "samples", "context_tokens", "target_tokens"):
        if column in df.columns and df[column].nunique(dropna=False) == 1:
            metadata.append(f"{column}={df[column].iloc[0]}")
    if metadata:
        ax.text(
            0.0,
            -0.24,
            " | ".join(metadata),
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )

    ax.text(
        1.0,
        -0.24,
        f"source: {csv_path.name}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Pythia PPL comparison between presses from result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_NAME,
        help=(
            "Results directory to read. Accepts uncertainty_aware, pythia_kvpress_test, both, "
            "an absolute path, a repo-relative path, or a directory name under results/. "
            "Defaults to uncertainty_aware."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV file to plot. Accepts an absolute path, a repo-relative path, or a filename in "
            "the selected results directory. Defaults to all CSV files in the selected directory."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <csv_stem>_ppl_comparison.png beside the CSV.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis. By default a log y-axis is used because PPL values may vary widely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dirs = resolve_results_dirs(args.results_dir)
    csv_paths = resolve_csv_paths(args.csv, results_dirs)

    for csv_path in csv_paths:
        output_path = resolve_output_path(args.output, csv_paths, csv_path)
        df = load_results(csv_path)
        plot_ppl(df=df, csv_path=csv_path, output_path=output_path, use_log_scale=not args.linear)

        print(f"Wrote PPL comparison plot to {output_path}")


if __name__ == "__main__":
    main()
