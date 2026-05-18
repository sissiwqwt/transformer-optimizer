# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "uncertainty_aware"
TMP_RESULTS_DIR = REPO_ROOT / "results" / "uncertainty_aware_tmp"
DATASET_FIGURE_DIRS = {
    "wikitext": "figure_wiki",
    "pg19": "figure_pg19",
}
DATASET_PPL_COLUMNS = {
    "wikitext": "wikitext_ppl",
    "pg19": "pg19_ppl",
}
METADATA_COLUMNS = {
    "model",
    "device",
    "dtype",
    "samples",
    "samples_per_dataset",
    "context_tokens",
    "target_tokens",
    "max_new_tokens",
    "latency_datasets",
}


def resolve_results_dir(raw_path: str | None, use_tmp: bool) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    if use_tmp:
        return TMP_RESULTS_DIR

    return DEFAULT_RESULTS_DIR


def list_csv_files(results_dir: Path) -> list[Path]:
    csv_paths = sorted(
        path
        for path in results_dir.rglob("*.csv")
        if not any(part.startswith("figure_") for part in path.relative_to(results_dir).parts)
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {results_dir}")
    return csv_paths


def infer_dataset(csv_path: Path, df: pd.DataFrame) -> str:
    path_parts = {part.lower() for part in csv_path.parts}
    for dataset in DATASET_FIGURE_DIRS:
        if dataset in path_parts:
            return dataset

    if "dataset" in df.columns:
        values = {str(value).lower() for value in df["dataset"].dropna().unique()}
        for dataset in DATASET_FIGURE_DIRS:
            if dataset in values:
                return dataset

    if "metric" in df.columns:
        metrics = " ".join(str(value).lower() for value in df["metric"].dropna().unique())
        for dataset in DATASET_FIGURE_DIRS:
            if dataset in metrics:
                return dataset

    for dataset, column in DATASET_PPL_COLUMNS.items():
        if column in df.columns and df[column].notna().any():
            return dataset

    raise ValueError(f"Cannot infer dataset for {csv_path}")


def parse_suffix_number(value: str, prefix: str) -> float:
    suffix = value.removeprefix(prefix)
    return float(suffix.replace("_", "."))


def metadata_text(df: pd.DataFrame) -> str:
    metadata = []
    for column in METADATA_COLUMNS:
        if column in df.columns and df[column].nunique(dropna=False) == 1:
            metadata.append(f"{column}={df[column].iloc[0]}")
    return " | ".join(metadata)


def finish_plot(fig: plt.Figure, csv_path: Path, output_path: Path, df: pd.DataFrame) -> None:
    metadata = metadata_text(df)
    if metadata:
        fig.text(0.01, 0.01, metadata, fontsize=8, ha="left", va="bottom")
    fig.text(0.99, 0.01, f"source: {csv_path.name}", fontsize=8, ha="right", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_main_table(df: pd.DataFrame, dataset: str, csv_path: Path, output_path: Path) -> None:
    ppl_column = DATASET_PPL_COLUMNS[dataset]
    plot_df = df.copy()
    plot_df[ppl_column] = pd.to_numeric(plot_df[ppl_column], errors="coerce")
    plot_df["throughput_tokens_s"] = pd.to_numeric(plot_df.get("throughput_tokens_s"), errors="coerce")
    plot_df = plot_df.dropna(subset=[ppl_column])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(plot_df["method"], plot_df[ppl_column], color="#4C78A8")
    axes[0].set_title(f"{dataset} perplexity")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("PPL")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].tick_params(axis="x", rotation=30)

    throughput_df = plot_df.dropna(subset=["throughput_tokens_s"])
    axes[1].bar(throughput_df["method"], throughput_df["throughput_tokens_s"], color="#F58518")
    axes[1].set_title("Throughput")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Tokens/s")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].tick_params(axis="x", rotation=30)

    finish_plot(fig, csv_path, output_path, df)


def plot_wide_ablation(
    df: pd.DataFrame,
    dataset: str,
    csv_path: Path,
    output_path: Path,
    prefix: str,
    xlabel: str,
) -> None:
    value_columns = [column for column in df.columns if column.startswith(prefix)]
    if not value_columns:
        raise ValueError(f"No columns with prefix {prefix!r} in {csv_path}")

    id_column = "method" if "method" in df.columns else "base_press"
    long_df = df.melt(
        id_vars=[id_column],
        value_vars=value_columns,
        var_name=xlabel,
        value_name="ppl",
    )
    long_df[xlabel] = long_df[xlabel].map(lambda value: parse_suffix_number(value, prefix))
    long_df["ppl"] = pd.to_numeric(long_df["ppl"], errors="coerce")
    long_df = long_df.dropna(subset=["ppl"]).sort_values(xlabel)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, group in long_df.groupby(id_column, sort=False):
        ax.plot(group[xlabel], group["ppl"], marker="o", linewidth=2, label=name)
    ax.set_title(f"{dataset} {xlabel} ablation")
    ax.set_xlabel(xlabel.replace("_", " ").title())
    ax.set_ylabel("PPL")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title=id_column, fontsize=8)

    finish_plot(fig, csv_path, output_path, df)


def plot_normalization(df: pd.DataFrame, dataset: str, csv_path: Path, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["value"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(plot_df["variant"], plot_df["value"], color="#54A24B")
    ax.set_title(f"{dataset} normalization ablation")
    ax.set_xlabel("Variant")
    ax.set_ylabel("PPL")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)

    finish_plot(fig, csv_path, output_path, df)


def plot_selection_difference(df: pd.DataFrame, dataset: str, csv_path: Path, output_path: Path) -> None:
    plot_df = df.copy()
    for column in ("kept_set_difference", "ppl_change"):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["kept_set_difference", "ppl_change"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(plot_df["kept_set_difference"], plot_df["ppl_change"], color="#B279A2", s=70)
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["base_press"]),
            (row["kept_set_difference"], row["ppl_change"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"{dataset} selection difference vs PPL change")
    ax.set_xlabel("Kept-set difference")
    ax.set_ylabel("UA PPL - base PPL")
    ax.grid(linestyle="--", alpha=0.35)

    finish_plot(fig, csv_path, output_path, df)


def plot_generic(df: pd.DataFrame, dataset: str, csv_path: Path, output_path: Path) -> None:
    numeric_df = df.drop(columns=list(METADATA_COLUMNS.intersection(df.columns)), errors="ignore")
    numeric_df = numeric_df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError(f"No numeric columns to plot in {csv_path}")

    label_column = next((column for column in ("method", "base_press", "variant", "metric") if column in df.columns), None)
    labels = df[label_column].astype(str) if label_column else df.index.astype(str)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    numeric_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(f"{dataset} {csv_path.stem}")
    ax.set_xlabel(label_column or "row")
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    finish_plot(fig, csv_path, output_path, df)


def plot_csv(csv_path: Path, results_dir: Path) -> Path:
    df = pd.read_csv(csv_path)
    dataset = infer_dataset(csv_path, df)
    output_dir = results_dir / DATASET_FIGURE_DIRS[dataset]
    output_path = output_dir / f"{csv_path.stem}.png"

    if csv_path.stem == "main_table_results" and DATASET_PPL_COLUMNS[dataset] in df.columns:
        plot_main_table(df, dataset, csv_path, output_path)
    elif csv_path.stem == "lambda_ablation_results":
        plot_wide_ablation(df, dataset, csv_path, output_path, prefix="lambda_", xlabel="lambda")
    elif csv_path.stem == "compression_ratio_ablation_results":
        plot_wide_ablation(df, dataset, csv_path, output_path, prefix="ratio_", xlabel="compression_ratio")
    elif csv_path.stem == "normalization_ablation_results" and {"variant", "value"}.issubset(df.columns):
        plot_normalization(df, dataset, csv_path, output_path)
    elif csv_path.stem == "selection_difference_results":
        plot_selection_difference(df, dataset, csv_path, output_path)
    else:
        plot_generic(df, dataset, csv_path, output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CSV files from uncertainty-aware experiment results.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "CSV root directory. Defaults to results/uncertainty_aware. "
            "Can be set to results/uncertainty_aware_tmp or an absolute path."
        ),
    )
    parser.add_argument(
        "--tmp",
        action="store_true",
        help="Read CSV files from results/uncertainty_aware_tmp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir, args.tmp)
    csv_paths = list_csv_files(results_dir)

    for csv_path in csv_paths:
        output_path = plot_csv(csv_path, results_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
