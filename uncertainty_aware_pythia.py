# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline

from kvpress import KnormPress, UncertaintyAwarePress

DEFAULT_MODEL = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."


@dataclass
class EvaluationResult:
    dataset: str
    press: str
    compression_ratio: float
    samples: int
    context_tokens: int
    ppl: float
    throughput_tokens_s: float


def iter_nonempty_texts(dataset_name: str, split: str, local_pg19_txt: str | None) -> Iterable[str]:
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
    tokenizer,
    dataset_name: str,
    split: str,
    num_samples: int,
    context_tokens: int,
    local_pg19_txt: str | None,
) -> list[str]:
    contexts = []
    buffer = []
    buffer_token_count = 0

    for text in iter_nonempty_texts(dataset_name, split, local_pg19_txt):
        buffer.append(text)
        buffer_token_count += len(tokenizer.encode(text, add_special_tokens=False))
        if buffer_token_count < context_tokens:
            continue

        token_ids = tokenizer.encode("\n".join(buffer), add_special_tokens=False)[:context_tokens]
        contexts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
        buffer = []
        buffer_token_count = 0

        if len(contexts) >= num_samples:
            break

    if not contexts:
        raise RuntimeError(f"No usable contexts collected for {dataset_name}.")

    return contexts


def build_press(name: str, compression_ratio: float, uncertainty_weight: float):
    if name == "none":
        return None
    if name == "knorm":
        return KnormPress(compression_ratio=compression_ratio)
    if name == "uncertainty_head_var":
        return UncertaintyAwarePress(
            compression_ratio=compression_ratio,
            press=KnormPress(),
            uncertainty_weight=uncertainty_weight,
        )
    raise ValueError(f"Unknown press: {name}")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_ppl(model, tokenizer, contexts: list[str], press) -> float:
    model.eval()
    nll_sum = 0.0
    token_count = 0

    for context in contexts:
        input_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if seq_len < 4:
            continue

        prefix_len = max(2, seq_len // 2)
        prefix_ids = input_ids[:, :prefix_len]
        continuation_ids = input_ids[:, prefix_len:]
        if continuation_ids.shape[1] < 2:
            continue
        cache = DynamicCache()

        if press is None:
            model(input_ids=prefix_ids, past_key_values=cache)
        else:
            with press(model):
                model(input_ids=prefix_ids, past_key_values=cache)

        outputs = model(input_ids=continuation_ids, labels=continuation_ids, past_key_values=cache)
        n_tokens = continuation_ids.shape[1] - 1
        nll_sum += outputs.loss.item() * n_tokens
        token_count += n_tokens

    return math.exp(nll_sum / token_count) if token_count else float("nan")


def evaluate_throughput(gen_pipe, contexts: list[str], press, max_new_tokens: int, warmup: int) -> float:
    for context in contexts[:warmup]:
        gen_pipe(context, question=QUESTION, press=press, max_new_tokens=max_new_tokens)

    total_time = 0.0
    for context in contexts:
        synchronize()
        start = time.perf_counter()
        gen_pipe(context, question=QUESTION, press=press, max_new_tokens=max_new_tokens)
        synchronize()
        total_time += time.perf_counter() - start

    return (len(contexts) * max_new_tokens) / total_time


def run_evaluation(args) -> list[EvaluationResult]:
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

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
        contexts = build_contexts(
            tokenizer,
            dataset_name,
            args.split,
            args.num_samples,
            args.context_tokens,
            args.local_pg19_txt,
        )
        print(f"\nDataset: {dataset_name} ({len(contexts)} samples)")

        for press_name in args.presses:
            clear_memory()
            compression_ratio = 0.0 if press_name == "none" else args.compression_ratio
            press = build_press(press_name, args.compression_ratio, args.uncertainty_weight)

            ppl = evaluate_ppl(model, tokenizer, contexts, press)
            throughput = evaluate_throughput(gen_pipe, contexts, press, args.max_new_tokens, args.warmup)
            print(f"{press_name}: ppl={ppl:.4f}, throughput={throughput:.2f} tokens/s")

            results.append(
                EvaluationResult(
                    dataset=dataset_name,
                    press=press_name,
                    compression_ratio=compression_ratio,
                    samples=len(contexts),
                    context_tokens=args.context_tokens,
                    ppl=ppl,
                    throughput_tokens_s=throughput,
                )
            )

    return results


def write_csv(results: list[EvaluationResult], output_csv: str):
    fieldnames = list(EvaluationResult.__dataclass_fields__)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate head-wise variance uncertainty press on Pythia.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument(
        "--presses",
        nargs="+",
        default=["none", "knorm", "uncertainty_head_var"],
        choices=["none", "knorm", "uncertainty_head_var"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument("--output-csv", default="uncertainty_aware_pythia_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_evaluation(args)
    write_csv(results, args.output_csv)
    print(f"\nWrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
