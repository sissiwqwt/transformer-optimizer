# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from kvpress import AnchorDedupPress
from pythia_kvpress_test import (
    MODEL_NAME,
    RESULTS_OUTPUT_DIR,
    NoPress,
    clear_memory,
    collect_token_windows,
    evaluate_ppl,
    evaluate_throughput,
    resolve_output_csv_path,
)


DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "pythia_anchor_dedup_results.csv"


@dataclass
class AnchorDedupResult:
    dataset: str
    press: str
    model: str
    compression_ratio: float
    ppl: float
    throughput_tokens_s: float
    avg_latency_s: float
    samples: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int
    n_sink: int
    chunk_size: int
    window_size: int
    anchor_bonus: float
    dedup_strength: float
    max_anchor_ratio: float
    normalize_scores: bool


def build_press(args, name: str):
    if name == "none":
        return NoPress()

    if name == "anchor_dedup":
        return AnchorDedupPress(
            compression_ratio=args.compression_ratio,
            n_sink=args.n_sink,
            chunk_size=args.chunk_size,
            window_size=args.window_size,
            anchor_bonus=args.anchor_bonus,
            dedup_strength=args.dedup_strength,
            max_anchor_ratio=args.max_anchor_ratio,
            normalize_scores=not args.no_normalize_scores,
        )

    raise ValueError(f"Unknown press: {name}")


def run(args) -> list[AnchorDedupResult]:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    gen_pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    results = []

    for dataset_name in args.datasets:
        print(f"\nDataset: {dataset_name}")

        token_windows = collect_token_windows(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=args.split,
            num_samples=args.num_samples,
            context_tokens=args.context_tokens,
            target_tokens=args.target_tokens,
            local_pg19_txt=args.local_pg19_txt,
        )
        contexts = [context for _, _, context in token_windows]

        for press_name in args.presses:
            clear_memory()
            press = build_press(args, press_name)
            ppl = evaluate_ppl(
                model=model,
                tokenizer=tokenizer,
                token_windows=token_windows,
                press=press,
                dataset_name=dataset_name,
                press_name=press_name,
            )

            clear_memory()
            press = build_press(args, press_name)
            avg_latency, throughput = evaluate_throughput(
                gen_pipe=gen_pipe,
                contexts=contexts,
                press=press,
                dataset_name=dataset_name,
                press_name=press_name,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
            )

            print(
                f"{press_name:>12}: ppl={ppl:.4f}, "
                f"throughput={throughput:.2f} tokens/s, "
                f"avg_latency={avg_latency:.4f}s"
            )

            results.append(
                AnchorDedupResult(
                    dataset=dataset_name,
                    press=press_name,
                    model=args.model,
                    compression_ratio=args.compression_ratio,
                    ppl=ppl,
                    throughput_tokens_s=throughput,
                    avg_latency_s=avg_latency,
                    samples=len(token_windows),
                    context_tokens=args.context_tokens,
                    target_tokens=args.target_tokens,
                    max_new_tokens=args.max_new_tokens,
                    n_sink=args.n_sink,
                    chunk_size=args.chunk_size,
                    window_size=args.window_size,
                    anchor_bonus=args.anchor_bonus,
                    dedup_strength=args.dedup_strength,
                    max_anchor_ratio=args.max_anchor_ratio,
                    normalize_scores=not args.no_normalize_scores,
                )
            )

    return results


def write_csv(results: list[AnchorDedupResult], output_csv: str | Path) -> Path:
    fieldnames = list(AnchorDedupResult.__dataclass_fields__)
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AnchorDedupPress on Pythia PPL and throughput.")

    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext", "pg19"],
        choices=["wikitext", "pg19"],
    )
    parser.add_argument(
        "--presses",
        nargs="+",
        default=["none", "anchor_dedup"],
        choices=["none", "anchor_dedup"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--anchor-bonus", type=float, default=8.0)
    parser.add_argument("--dedup-strength", type=float, default=1.0)
    parser.add_argument("--max-anchor-ratio", type=float, default=0.25)
    parser.add_argument("--no-normalize-scores", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="CSV output path. Bare filenames are saved under results/pythia_kvpress_test.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    results = run(args)
    output_path = write_csv(results=results, output_csv=args.output_csv)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
