# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from kvpress import AdaKVPress, KnormPress, SnapKVPress, StreamingLLMPress


MODEL_NAME = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."


@dataclass
class BenchmarkResult:
    dataset: str
    press: str
    compression_ratio: float
    samples: int
    context_tokens: int
    max_new_tokens: int
    avg_latency_s: float
    throughput_tokens_s: float
    speedup_vs_none: float | None = None


def iter_nonempty_texts(dataset_name: str, split: str, local_pg19_txt: str | None = None) -> Iterable[str]:
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        for row in dataset:
            text = row["text"].strip()
            if text:
                yield text
        return

    if dataset_name == "pg19":
        if local_pg19_txt:
            text = Path(local_pg19_txt).read_text(encoding="utf-8").strip()
            if text:
                yield text
            return

        for hub_name in ("emozilla/pg19", "pg19"):
            try:
                dataset = load_dataset(hub_name, split=split, streaming=True)
                for row in dataset:
                    text = row.get("text", "").strip()
                    if text:
                        yield text
                return
            except Exception as exc:
                print(f"[WARN] Could not load {hub_name}: {exc}")

    raise ValueError(f"Unsupported dataset or no data found: {dataset_name}")


def build_contexts(
    tokenizer, dataset_name: str, split: str, num_samples: int, context_tokens: int, local_pg19_txt=None
):
    contexts = []
    buffer = []
    buffer_token_count = 0

    for text in iter_nonempty_texts(dataset_name, split, local_pg19_txt):
        buffer.append(text)
        buffer_token_count += len(tokenizer.encode(text, add_special_tokens=False))

        if buffer_token_count < context_tokens:
            continue

        joined = "\n".join(buffer)
        ids = tokenizer.encode(joined, add_special_tokens=False)[:context_tokens]
        contexts.append(tokenizer.decode(ids, skip_special_tokens=True))
        buffer = []
        buffer_token_count = 0

        if len(contexts) >= num_samples:
            break

    if not contexts:
        raise RuntimeError(f"No usable contexts collected for {dataset_name}.")

    return contexts


def build_press(name: str, compression_ratio: float, snapkv_window_size: int):
    if name == "none":
        return None
    if name == "adakv":
        return AdaKVPress(SnapKVPress(compression_ratio=compression_ratio, window_size=snapkv_window_size))
    if name == "streaming_llm":
        return StreamingLLMPress(compression_ratio=compression_ratio)
    if name == "knorm":
        return KnormPress(compression_ratio=compression_ratio)
    if name == "snapkv":
        return SnapKVPress(compression_ratio=compression_ratio, window_size=snapkv_window_size)
    raise ValueError(f"Unknown press: {name}")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_generation(gen_pipe, contexts, press_name, press, max_new_tokens: int, warmup: int):
    question = QUESTION

    for context in contexts[:warmup]:
        gen_pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)

    latencies = []
    for context in contexts:
        synchronize()
        start = time.perf_counter()
        gen_pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)
        synchronize()
        latencies.append(time.perf_counter() - start)

    avg_latency = sum(latencies) / len(latencies)
    throughput = (len(contexts) * max_new_tokens) / sum(latencies)
    print(f"{press_name:>6}: avg_latency={avg_latency:.4f}s, throughput={throughput:.2f} tokens/s")
    return avg_latency, throughput


def run_benchmark(args):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
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
        contexts = build_contexts(
            tokenizer,
            dataset_name,
            args.split,
            args.num_samples,
            args.context_tokens,
            args.local_pg19_txt,
        )

        baseline_throughput = None
        for press_name in args.presses:
            clear_memory()
            press = build_press(press_name, args.compression_ratio, args.snapkv_window_size)
            avg_latency, throughput = time_generation(
                gen_pipe,
                contexts,
                press_name,
                press,
                args.max_new_tokens,
                args.warmup,
            )

            if press_name == "none":
                baseline_throughput = throughput

            speedup = throughput / baseline_throughput if baseline_throughput else None
            results.append(
                BenchmarkResult(
                    dataset=dataset_name,
                    press=press_name,
                    compression_ratio=0.0 if press_name == "none" else args.compression_ratio,
                    samples=len(contexts),
                    context_tokens=args.context_tokens,
                    max_new_tokens=args.max_new_tokens,
                    avg_latency_s=avg_latency,
                    throughput_tokens_s=throughput,
                    speedup_vs_none=speedup,
                )
            )

    return results


def write_csv(results: list[BenchmarkResult], output_path: str):
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def print_summary(results: list[BenchmarkResult]):
    print("\nSummary")
    print("dataset,press,throughput_tokens_s,avg_latency_s,speedup_vs_none")
    for result in results:
        speedup = "" if result.speedup_vs_none is None else f"{result.speedup_vs_none:.3f}"
        print(
            f"{result.dataset},{result.press},{result.throughput_tokens_s:.2f}," f"{result.avg_latency_s:.4f},{speedup}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KVPress throughput on Pythia-70M.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument(
        "--presses",
        nargs="+",
        default=["none", "adakv", "streaming_llm", "knorm", "snapkv"],
        choices=["none", "adakv", "streaming_llm"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--snapkv-window-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument("--output-csv", default="pythia70m_throughput_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_benchmark(args)
    print_summary(results)
    write_csv(results, args.output_csv)
    print(f"\nWrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
