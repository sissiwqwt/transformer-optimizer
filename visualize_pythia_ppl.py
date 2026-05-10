# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "pythia_kvpress_test"
REQUIRED_COLUMNS = {"dataset", "press", "ppl"}


def latest_csv(results_dir: Path) -> Path:
    csv_files = sorted(results_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    return csv_files[0]


def resolve_csv_path(csv_path: str | None) -> Path:
    if csv_path is None:
        return latest_csv(RESULTS_DIR)

    path = Path(csv_path).expanduser()
    if path.is_absolute():
        return path

    direct_path = REPO_ROOT / path
    if direct_path.exists():
        return direct_path

    return RESULTS_DIR / path


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_ppl_comparison.png")


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
        description="Visualize PPL comparison between presses from one pythia_kvpress_test CSV."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV file to plot. Accepts an absolute path, a repo-relative path, or a filename in "
            "results/pythia_kvpress_test. Defaults to the newest CSV in that directory."
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
    csv_path = resolve_csv_path(args.csv)
    output_path = Path(args.output).expanduser() if args.output else default_output_path(csv_path)

    df = load_results(csv_path)
    plot_ppl(df=df, csv_path=csv_path, output_path=output_path, use_log_scale=not args.linear)

    print(f"Wrote PPL comparison plot to {output_path}")


if __name__ == "__main__":
    main()
