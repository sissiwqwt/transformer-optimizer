# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# How to run:
#   python uncertainty_aware_pythia.py
#   You can paste part of the pg19 data in ~/data/<dataname>.txt and specify --local-pg19-txt ~/data/<dataname>.txt to avoid streaming issues.

import argparse
import csv
import gc
import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from kvpress import KnormPress, UncertaintyAwarePress

DEFAULT_MODEL = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."


def find_repo_root() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "uncertainty_aware"
DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "pythia_results.csv"


def resolve_repo_path(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return REPO_ROOT / resolved_path


def resolve_output_csv_path(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    if output_path.parent == Path("."):
        return RESULTS_OUTPUT_DIR / output_path
    return REPO_ROOT / output_path


@dataclass
class EvaluationResult:
    dataset: str
    press: str
    compression_ratio: float
    samples: int
    context_tokens: int
    target_tokens: int
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
            text = resolve_repo_path(local_pg19_txt).read_text(encoding="utf-8").strip()
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


def collect_token_windows(
    tokenizer,
    dataset_name: str,
    split: str,
    num_samples: int,
    context_tokens: int,
    target_tokens: int,
    local_pg19_txt: str | None,
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    windows = []
    required_tokens = context_tokens + target_tokens
    buffer: list[int] = []

    for text in iter_nonempty_texts(dataset_name, split, local_pg19_txt):
        buffer.extend(tokenizer.encode(text + "\n", add_special_tokens=False))
        if len(buffer) < required_tokens:
            continue

        token_ids = buffer[:required_tokens]
        context_ids = torch.tensor([token_ids[:context_tokens]], dtype=torch.long)
        target_ids = torch.tensor([token_ids[context_tokens:]], dtype=torch.long)
        context_text = tokenizer.decode(token_ids[:context_tokens], skip_special_tokens=True)

        windows.append((context_ids, target_ids, context_text))
        buffer = buffer[required_tokens:]

        if len(windows) >= num_samples:
            break

    if not windows:
        raise RuntimeError(
            f"No usable {dataset_name} samples with at least {required_tokens} tokens. "
            "Lower --context-tokens/--target-tokens or provide --local-pg19-txt."
        )

    return windows


def _build_position_kwargs(model, start_pos: int, seq_len: int, device) -> dict:
    forward_params = inspect.signature(model.forward).parameters
    position_ids = torch.arange(
        start_pos,
        start_pos + seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    kwargs = {"position_ids": position_ids}
    if "cache_position" in forward_params:
        kwargs["cache_position"] = position_ids.squeeze(0)
    return kwargs


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
def evaluate_ppl(model, token_windows, press) -> float:
    """
    Compute PPL on target tokens conditioned on context tokens.

    This matches pythia_kvpress_test.py:
    p(y_0 | context) is scored from the final context logits, then
    p(y_i | context, y_<i) is scored by feeding previous target tokens
    through the compressed KV cache with absolute positions preserved.
    """
    model.eval()
    nll_sum = 0.0
    token_count = 0

    for context_ids, target_ids, _ in token_windows:
        context_ids = context_ids.to(model.device)
        target_ids = target_ids.to(model.device)

        # Prefill context. If press is not None, KV cache is compressed here.
        if press is None:
            context_outputs = model(input_ids=context_ids, use_cache=True)
        else:
            with press(model):
                context_outputs = model(input_ids=context_ids, use_cache=True)

        cache = context_outputs.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return past_key_values. Make sure use_cache=True is supported.")

        first_token_logits = context_outputs.logits[:, -1, :]
        first_token_loss = F.cross_entropy(
            first_token_logits,
            target_ids[:, 0],
            reduction="sum",
        )
        nll_sum += first_token_loss.item()
        token_count += 1

        if target_ids.shape[1] > 1:
            continuation_ids = target_ids[:, :-1]
            continuation_labels = target_ids[:, 1:]
            position_kwargs = _build_position_kwargs(
                model=model,
                start_pos=context_ids.shape[1],
                seq_len=continuation_ids.shape[1],
                device=model.device,
            )

            outputs = model(
                input_ids=continuation_ids,
                past_key_values=cache,
                use_cache=True,
                **position_kwargs,
            )
            loss = F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.shape[-1]),
                continuation_labels.reshape(-1),
                reduction="sum",
            )
            nll_sum += loss.item()
            token_count += continuation_labels.numel()

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
        token_windows = collect_token_windows(
            tokenizer,
            dataset_name,
            args.split,
            args.num_samples,
            args.context_tokens,
            args.target_tokens,
            args.local_pg19_txt,
        )
        contexts = [context for _, _, context in token_windows]
        print(f"\nDataset: {dataset_name} ({len(contexts)} samples)")

        for press_name in args.presses:
            clear_memory()
            compression_ratio = 0.0 if press_name == "none" else args.compression_ratio
            press = build_press(press_name, args.compression_ratio, args.uncertainty_weight)

            ppl = evaluate_ppl(model, token_windows, press)
            throughput = evaluate_throughput(gen_pipe, contexts, press, args.max_new_tokens, args.warmup)
            print(f"{press_name}: ppl={ppl:.4f}, throughput={throughput:.2f} tokens/s")

            results.append(
                EvaluationResult(
                    dataset=dataset_name,
                    press=press_name,
                    compression_ratio=compression_ratio,
                    samples=len(contexts),
                    context_tokens=args.context_tokens,
                    target_tokens=args.target_tokens,
                    ppl=ppl,
                    throughput_tokens_s=throughput,
                )
            )

    return results


def write_csv(results: list[EvaluationResult], output_csv: str | Path) -> Path:
    fieldnames = list(EvaluationResult.__dataclass_fields__)
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


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
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=(
            "CSV output path. Passing only a filename, e.g. results.csv, saves to "
            f"{RESULTS_OUTPUT_DIR / '<filename>.csv'}."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_evaluation(args)
    output_path = write_csv(results, args.output_csv)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
