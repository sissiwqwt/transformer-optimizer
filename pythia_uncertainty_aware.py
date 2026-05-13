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

from kvpress import (
    CompactorPress,
    CURPress,
    ExpectedAttentionPress,
    KeyDiffPress,
    KnormPress,
    LagKVPress,
    NonCausalAttnPress,
    SnapKVPress,
    UncertaintyAwarePress,
)

DEFAULT_MODEL = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."
BASE_PRESS_CHOICES = (
    "knorm",
    "keydiff",
    "cur",
    "lagkv",
    "snapkv",
    "expected_attention",
    "non_causal_attn",
    "compactor",
)
PRESS_CHOICES = ("none", *BASE_PRESS_CHOICES, "uncertainty_head_var")


def find_repo_root() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
# RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "uncertainty_aware"
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "cuda_uncertainty_aware"
DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "pythia_results.csv"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested --device {device_arg}, but CUDA is not available.")
        if device.index is None:
            return "cuda:0"
        return f"cuda:{device.index}"

    if device.type == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported --device value: {device_arg}. Use auto, cpu, cuda, or cuda:0.")


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


def _cache_seq_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length(0))
    return int(cache[0][0].shape[-2])


def _build_position_kwargs(model, absolute_start_pos: int, cache_start_pos: int, seq_len: int, device) -> dict:
    forward_params = inspect.signature(model.forward).parameters
    position_ids = torch.arange(
        absolute_start_pos,
        absolute_start_pos + seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    kwargs = {"position_ids": position_ids}
    if "cache_position" in forward_params:
        kwargs["cache_position"] = torch.arange(
            cache_start_pos,
            cache_start_pos + seq_len,
            device=device,
            dtype=torch.long,
        )
    return kwargs


def build_base_press(name: str, compression_ratio: float = 0.0):
    if name == "knorm":
        return KnormPress(compression_ratio=compression_ratio)
    if name == "keydiff":
        return KeyDiffPress(compression_ratio=compression_ratio)
    if name == "cur":
        return CURPress(compression_ratio=compression_ratio)
    if name == "lagkv":
        return LagKVPress(compression_ratio=compression_ratio)
    if name == "snapkv":
        return SnapKVPress(compression_ratio=compression_ratio)
    if name == "expected_attention":
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    if name == "non_causal_attn":
        return NonCausalAttnPress(compression_ratio=compression_ratio)
    if name == "compactor":
        return CompactorPress(compression_ratio=compression_ratio)
    raise ValueError(f"Unknown base press: {name}")


def build_press(name: str, compression_ratio: float, uncertainty_weight: float, uncertainty_base_press: str):
    if name == "none":
        return None
    if name in BASE_PRESS_CHOICES:
        return build_base_press(name, compression_ratio=compression_ratio)
    if name == "uncertainty_head_var":
        return UncertaintyAwarePress(
            compression_ratio=compression_ratio,
            press=build_base_press(uncertainty_base_press),
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
                absolute_start_pos=context_ids.shape[1],
                cache_start_pos=_cache_seq_length(cache),
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

    device = resolve_device(args.device)
    use_auto_device = args.device == "auto"
    use_cuda = device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Using device: {device} (requested: {args.device})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast_tokenizer)

    model_kwargs = {"dtype": dtype}
    if use_auto_device and use_cuda:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if not use_auto_device:
        model = model.to(device)
    elif not use_cuda:
        model = model.to("cpu")

    pipeline_kwargs = {}
    if use_auto_device and use_cuda:
        pipeline_kwargs["device_map"] = "auto"
    elif not use_auto_device:
        pipeline_kwargs["device"] = device

    gen_pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs)

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
            press = build_press(
                press_name,
                args.compression_ratio,
                args.uncertainty_weight,
                args.uncertainty_base_press,
            )

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
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0. Default: auto.",
    )
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument(
        "--presses",
        nargs="+",
        default=["none", "knorm", "uncertainty_head_var"],
        choices=PRESS_CHOICES,
    )
    parser.add_argument(
        "--uncertainty-base-press",
        default="knorm",
        choices=BASE_PRESS_CHOICES,
        help="Base scorer used inside UncertaintyAwarePress when --presses includes uncertainty_head_var.",
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
