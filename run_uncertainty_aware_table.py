# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# How to run:
#   python run_uncertainty_aware_table.py
#   python run_uncertainty_aware_table.py --table lambda
#   python run_uncertainty_aware_table.py --table norm --norm-base-press knorm
#   python run_uncertainty_aware_table.py --table selection
#   python run_uncertainty_aware_table.py --table ratio
#   python run_uncertainty_aware_table.py --num-samples 8 --local-pg19-txt data/pg19_sample.txt

import argparse
import contextlib
import csv
import gc
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import CURPress, KnormPress, LagKVPress, SnapKVPress, UncertaintyAwarePress
from kvpress.presses.base_press import get_model_backbone
from kvpress.presses.scorer_press import ScorerPress

DEFAULT_MODEL = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."
BASE_PRESSES = ("knorm", "snapkv", "cur", "lagkv")
DISPLAY_NAMES = {
    "none": ("None", "Dense"),
    "knorm": ("KNorm", "KNorm"),
    "ua_knorm": ("KNorm", "UA-KNorm"),
    "snapkv": ("SnapKV", "SnapKV"),
    "ua_snapkv": ("SnapKV", "UA-SnapKV"),
    "cur": ("CUR", "CUR"),
    "ua_cur": ("CUR", "UA-CUR"),
    "lagkv": ("LagKV", "LagKV"),
    "ua_lagkv": ("LagKV", "UA-LagKV"),
}


def find_repo_root() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
PROJECT_ROOT = REPO_ROOT.parent if REPO_ROOT.name == "transformer-optimizer" else REPO_ROOT
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "uncertainty_aware"
RUNS_OUTPUT_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "main_table_results.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("KVPRESS_CACHE_DIR", str(PROJECT_ROOT / "cache")))
DEFAULT_LAMBDA_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "lambda_ablation_results.csv"
DEFAULT_NORM_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "normalization_ablation_results.csv"
DEFAULT_SELECTION_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "selection_difference_results.csv"
DEFAULT_RATIO_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "compression_ratio_ablation_results.csv"


@dataclass
class TokenWindow:
    context_ids: torch.Tensor
    target_ids: torch.Tensor
    context_text: str


@dataclass
class TableResult:
    base_press: str
    method: str
    ratio: float
    wikitext_ppl: float
    pg19_ppl: float
    ttft_s: float
    tpot_s: float
    throughput_tokens_s: float
    model: str
    device: str
    dtype: str
    samples_per_dataset: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int
    uncertainty_weight: float
    latency_datasets: str


@dataclass
class LambdaAblationResult:
    base_press: str
    metric: str
    lambda_0: float
    lambda_0_25: float
    lambda_0_5: float
    lambda_1_0: float
    lambda_2_0: float
    model: str
    device: str
    dtype: str
    compression_ratio: float
    samples_per_dataset: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int


@dataclass
class NormalizationAblationResult:
    variant: str
    normalize_mean: bool
    normalize_variance: bool
    metric: str
    value: float
    base_press: str
    model: str
    device: str
    dtype: str
    compression_ratio: float
    uncertainty_weight: float
    samples_per_dataset: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int


@dataclass
class SelectionDifferenceResult:
    base_press: str
    kept_set_difference: float
    ppl_change: float
    base_ppl: float
    ua_ppl: float
    dataset: str
    model: str
    device: str
    dtype: str
    compression_ratio: float
    uncertainty_weight: float
    samples: int
    context_tokens: int
    target_tokens: int


@dataclass
class CompressionRatioAblationResult:
    method: str
    metric: str
    ratio_0_2: float
    ratio_0_4: float
    ratio_0_6: float
    ratio_0_8: float
    model: str
    device: str
    dtype: str
    uncertainty_weight: float
    samples_per_dataset: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int


@dataclass
class RecordingScorerPress(ScorerPress):
    press: object = None
    records: list[set[int]] = field(default_factory=list)

    def __post_init__(self):
        self.compression_ratio = self.press.compression_ratio
        super().__post_init__()

    def post_init_from_model(self, model):
        if hasattr(self.press, "post_init_from_model"):
            self.press.post_init_from_model(model)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.press.score(module, hidden_states, keys, values, attentions, kwargs)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        compression_ratio = self.press.compression_ratio
        if compression_ratio == 0:
            self.records.append(set(range(keys.shape[2])))
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        self.records.append(set(indices.detach().flatten().cpu().tolist()))
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()


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


def create_run_output_dir(run_dir: str | Path | None = None) -> Path:
    if run_dir:
        output_dir = Path(run_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RUNS_OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_jsonl(path: Path, table_name: str, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row)
            payload["table"] = table_name
            payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_cache_path(path: str | Path) -> Path:
    cache_path = Path(path).expanduser()
    if cache_path.is_absolute():
        return cache_path
    return PROJECT_ROOT / cache_path


def configure_cache(cache_dir: str | Path) -> Path:
    resolved_cache_dir = resolve_cache_path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(resolved_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(resolved_cache_dir / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(resolved_cache_dir / "datasets"))
    return resolved_cache_dir


def iter_nonempty_texts(
    dataset_name: str,
    split: str,
    local_pg19_txt: str | None,
    cache_dir: Path,
) -> Iterable[str]:
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=str(cache_dir))
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
                dataset = load_dataset(hub_name, split=split, streaming=True, cache_dir=str(cache_dir))
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
    cache_dir: Path,
) -> list[TokenWindow]:
    windows = []
    required_tokens = context_tokens + target_tokens
    buffer: list[int] = []

    for text in iter_nonempty_texts(dataset_name, split, local_pg19_txt, cache_dir):
        buffer.extend(tokenizer.encode(text + "\n", add_special_tokens=False))
        if len(buffer) < required_tokens:
            continue

        token_ids = buffer[:required_tokens]
        windows.append(
            TokenWindow(
                context_ids=torch.tensor([token_ids[:context_tokens]], dtype=torch.long),
                target_ids=torch.tensor([token_ids[context_tokens:]], dtype=torch.long),
                context_text=tokenizer.decode(token_ids[:context_tokens], skip_special_tokens=True),
            )
        )
        buffer = buffer[required_tokens:]

        if len(windows) >= num_samples:
            break

    if not windows:
        raise RuntimeError(
            f"No usable {dataset_name} samples with at least {required_tokens} tokens. "
            "Lower --context-tokens/--target-tokens or provide --local-pg19-txt."
        )
    return windows


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_seq_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length(0))
    return int(cache[0][0].shape[-2])


def _build_position_kwargs(model, absolute_start_pos: int, cache_start_pos: int, seq_len: int, device) -> dict:
    forward_params = inspect.signature(model.forward).parameters
    kwargs = {
        "position_ids": torch.arange(
            absolute_start_pos,
            absolute_start_pos + seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
    }
    if "cache_position" in forward_params:
        kwargs["cache_position"] = torch.arange(
            cache_start_pos,
            cache_start_pos + seq_len,
            device=device,
            dtype=torch.long,
        )
    return kwargs


def build_base_press(name: str, args, compression_ratio: float | None = None):
    ratio = args.compression_ratio if compression_ratio is None else compression_ratio
    if name == "knorm":
        return KnormPress(compression_ratio=ratio)
    if name == "snapkv":
        return SnapKVPress(
            compression_ratio=ratio,
            window_size=args.snapkv_window_size,
            kernel_size=args.snapkv_kernel_size,
        )
    if name == "cur":
        return CURPress(
            compression_ratio=ratio,
            num_sinks=args.cur_num_sinks,
            local_window_size=args.cur_local_window_size,
        )
    if name == "lagkv":
        return LagKVPress(
            compression_ratio=ratio,
            n_sink=args.lagkv_n_sink,
            lag_size=args.lagkv_lag_size,
        )
    raise ValueError(f"Unknown base press: {name}")


def build_uncertainty_press(
    base_name: str,
    args,
    uncertainty_weight: float,
    normalize_scores: bool | None = None,
    normalize_uncertainty: bool | None = None,
):
    base_press = build_base_press(base_name, args, compression_ratio=0.0)
    if normalize_scores is None:
        normalize_scores = not args.no_normalize_scores
    if normalize_uncertainty is None:
        normalize_uncertainty = not args.no_normalize_uncertainty
    return UncertaintyAwarePress(
        compression_ratio=args.compression_ratio,
        press=base_press,
        uncertainty_weight=uncertainty_weight,
        normalize_scores=normalize_scores,
        normalize_uncertainty=normalize_uncertainty,
    )


def build_method_press(method_key: str, args):
    if method_key == "none":
        return None
    if method_key in BASE_PRESSES:
        return build_base_press(method_key, args)
    if method_key.startswith("ua_"):
        base_name = method_key.removeprefix("ua_")
        return build_uncertainty_press(base_name, args, args.uncertainty_weight)
    raise ValueError(f"Unknown method: {method_key}")


def iter_method_keys(args) -> Iterable[str]:
    if args.methods:
        for method in args.methods:
            yield method
        return

    yield "none"
    for base_name in BASE_PRESSES:
        yield base_name
        yield f"ua_{base_name}"


@torch.no_grad()
def evaluate_ppl(model, token_windows: list[TokenWindow], press) -> float:
    model.eval()
    nll_sum = 0.0
    token_count = 0

    for window in token_windows:
        context_ids = window.context_ids.to(model.device)
        target_ids = window.target_ids.to(model.device)

        if press is None:
            context_outputs = model(input_ids=context_ids, use_cache=True)
        else:
            with press(model):
                context_outputs = model(input_ids=context_ids, use_cache=True)

        cache = context_outputs.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return past_key_values. Make sure use_cache=True is supported.")

        first_token_logits = context_outputs.logits[:, -1, :]
        nll_sum += F.cross_entropy(first_token_logits, target_ids[:, 0], reduction="sum").item()
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
            nll_sum += F.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.shape[-1]),
                continuation_labels.reshape(-1),
                reduction="sum",
            ).item()
            token_count += continuation_labels.numel()

    return math.exp(nll_sum / token_count) if token_count else float("nan")


def _encode_generation_inputs(tokenizer, context: str, question: str) -> tuple[torch.Tensor, torch.Tensor]:
    bos_token = getattr(tokenizer, "bos_token", "") or ""
    context_ids = tokenizer.encode(bos_token + context, return_tensors="pt", add_special_tokens=False)
    question_ids = tokenizer.encode(question + "\n", return_tensors="pt", add_special_tokens=False)
    return context_ids, question_ids


@torch.no_grad()
def measure_generation_metrics(
    model,
    tokenizer,
    contexts: list[str],
    press,
    question: str,
    max_new_tokens: int,
    warmup: int,
) -> tuple[float, float, float]:
    model.eval()

    for context in contexts[:warmup]:
        _generate_once(model, tokenizer, context, press, question, max_new_tokens, measure=False)

    ttft_values = []
    tpot_values = []
    total_tokens = 0
    total_time = 0.0

    for context in contexts:
        ttft, tpot, generated_tokens, elapsed = _generate_once(
            model,
            tokenizer,
            context,
            press,
            question,
            max_new_tokens,
            measure=True,
        )
        ttft_values.append(ttft)
        tpot_values.append(tpot)
        total_tokens += generated_tokens
        total_time += elapsed

    avg_ttft = sum(ttft_values) / len(ttft_values)
    valid_tpot = [value for value in tpot_values if not math.isnan(value)]
    avg_tpot = sum(valid_tpot) / len(valid_tpot) if valid_tpot else float("nan")
    throughput = total_tokens / total_time if total_time > 0 else float("nan")
    return avg_ttft, avg_tpot, throughput


def _generate_once(
    model,
    tokenizer,
    context: str,
    press,
    question: str,
    max_new_tokens: int,
    measure: bool,
) -> tuple[float, float, int, float]:
    context_ids, question_ids = _encode_generation_inputs(tokenizer, context, question)
    context_ids = context_ids.to(model.device)
    question_ids = question_ids.to(model.device)
    cache = DynamicCache()

    synchronize()
    start = time.perf_counter()

    press_context = press(model) if press is not None else contextlib.nullcontext()
    with press_context:
        get_model_backbone(model)(input_ids=context_ids, past_key_values=cache)

    context_length = context_ids.shape[1]
    position_ids = torch.arange(
        context_length,
        context_length + question_ids.shape[1],
        device=model.device,
        dtype=torch.long,
    ).unsqueeze(0)
    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )
    generated_ids = [outputs.logits[0, -1].argmax()]

    synchronize()
    first_token_time = time.perf_counter()
    after_first_elapsed = 0.0

    should_stop_token_ids = model.generation_config.eos_token_id
    if not isinstance(should_stop_token_ids, list):
        should_stop_token_ids = [should_stop_token_ids]

    position_ids = position_ids[:, -1:] + 1
    for index in range(max_new_tokens - 1):
        step_start = time.perf_counter()
        outputs = model(
            input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
            past_key_values=cache,
            position_ids=position_ids + index,
        )
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        synchronize()
        after_first_elapsed += time.perf_counter() - step_start
        if new_id.item() in should_stop_token_ids:
            break

    synchronize()
    end = time.perf_counter()

    if not measure:
        return float("nan"), float("nan"), len(generated_ids), end - start

    generated_tokens = len(generated_ids)
    ttft = first_token_time - start
    tpot = after_first_elapsed / (generated_tokens - 1) if generated_tokens > 1 else float("nan")
    return ttft, tpot, generated_tokens, end - start


def format_metric(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def print_parameters(args, device: str, dtype: torch.dtype, cache_dir: Path):
    print("\nExperiment parameters")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"compression_ratio: {args.compression_ratio}")
    print(f"uncertainty_weight: {args.uncertainty_weight}")
    print(f"datasets: {', '.join(args.datasets)}")
    print(f"latency_datasets: {', '.join(args.latency_datasets)}")
    print(f"split: {args.split}")
    print(f"num_samples: {args.num_samples}")
    print(f"context_tokens: {args.context_tokens}")
    print(f"target_tokens: {args.target_tokens}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"warmup: {args.warmup}")
    print(f"cache_dir: {cache_dir}")
    print(f"local_pg19_txt: {args.local_pg19_txt or '--'}")
    print(f"snapkv_window_size: {args.snapkv_window_size}")
    print(f"snapkv_kernel_size: {args.snapkv_kernel_size}")
    print(f"cur_num_sinks: {args.cur_num_sinks}")
    print(f"cur_local_window_size: {args.cur_local_window_size}")
    print(f"lagkv_n_sink: {args.lagkv_n_sink}")
    print(f"lagkv_lag_size: {args.lagkv_lag_size}")


def print_results_table(results: list[TableResult]):
    headers = [
        "Base Press",
        "Method",
        "Ratio",
        "WikiText-2 PPL",
        "PG-19 PPL",
        "TTFT",
        "TPOT",
        "Throughput",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.base_press,
                result.method,
                f"{result.ratio:.1f}",
                format_metric(result.wikitext_ppl),
                format_metric(result.pg19_ppl),
                format_metric(result.ttft_s),
                format_metric(result.tpot_s),
                format_metric(result.throughput_tokens_s, digits=2),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\nMain results")
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def lambda_column_name(lambda_value: float) -> str:
    if math.isclose(lambda_value, 0.0):
        return "lambda_0"
    if math.isclose(lambda_value, round(lambda_value)):
        return f"lambda_{int(round(lambda_value))}_0"
    return f"lambda_{lambda_value:g}".replace(".", "_")


def ratio_column_name(ratio: float) -> str:
    if math.isclose(ratio, round(ratio)):
        return f"ratio_{int(round(ratio))}_0"
    return f"ratio_{ratio:g}".replace(".", "_")


def print_lambda_ablation_table(results: list[LambdaAblationResult], lambda_values: list[float]):
    headers = ["Base Press", *[f"lambda={value:g}" for value in lambda_values]]
    rows = []
    for result in results:
        result_values = result.__dict__
        rows.append(
            [
                result.base_press,
                *[format_metric(result_values[lambda_column_name(value)]) for value in lambda_values],
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    metric = results[0].metric if results else "unknown"
    print(f"\nLambda ablation ({metric})")
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def print_compression_ratio_ablation_table(
    results: list[CompressionRatioAblationResult],
    ratio_values: list[float],
):
    headers = ["Method", *[f"r={value:g}" for value in ratio_values]]
    rows = []
    for result in results:
        result_values = result.__dict__
        rows.append(
            [
                result.method,
                *[format_metric(result_values[ratio_column_name(value)]) for value in ratio_values],
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    metric = results[0].metric if results else "unknown"
    print(f"\nCompression ratio ablation ({metric})")
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def print_normalization_ablation_table(results: list[NormalizationAblationResult]):
    metric = results[0].metric if results else "unknown"
    value_header = "PPL" if metric in {"wikitext_ppl", "pg19_ppl"} else metric
    headers = ["Variant", "Normalize Mean", "Normalize Variance", value_header]
    rows = []
    for result in results:
        rows.append(
            [
                result.variant,
                "yes" if result.normalize_mean else "no",
                "yes" if result.normalize_variance else "no",
                format_metric(result.value),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    base_press = results[0].base_press if results else "unknown"
    print(f"\nNormalization ablation ({base_press}, {metric})")
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def print_selection_difference_table(results: list[SelectionDifferenceResult]):
    headers = ["Base Press", "Kept-set Difference", "PPL Change"]
    rows = []
    for result in results:
        rows.append(
            [
                result.base_press,
                format_metric(result.kept_set_difference),
                format_metric(result.ppl_change),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    dataset = results[0].dataset if results else "unknown"
    print(f"\nSelection difference analysis ({dataset})")
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def write_csv(results: list[TableResult], output_csv: str | Path) -> Path:
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TableResult.__dataclass_fields__)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def write_lambda_csv(results: list[LambdaAblationResult], output_csv: str | Path) -> Path:
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(LambdaAblationResult.__dataclass_fields__)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def write_ratio_csv(results: list[CompressionRatioAblationResult], output_csv: str | Path) -> Path:
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CompressionRatioAblationResult.__dataclass_fields__)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def write_normalization_csv(results: list[NormalizationAblationResult], output_csv: str | Path) -> Path:
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(NormalizationAblationResult.__dataclass_fields__)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def write_selection_csv(results: list[SelectionDifferenceResult], output_csv: str | Path) -> Path:
    output_path = resolve_output_csv_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SelectionDifferenceResult.__dataclass_fields__)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def run(args) -> list[TableResult]:
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

    cache_dir = configure_cache(args.cache_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = str(model.device)
    print_parameters(args, device=device, dtype=dtype, cache_dir=cache_dir)

    dataset_windows = {}
    for dataset_name in args.datasets:
        dataset_windows[dataset_name] = collect_token_windows(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=args.split,
            num_samples=args.num_samples,
            context_tokens=args.context_tokens,
            target_tokens=args.target_tokens,
            local_pg19_txt=args.local_pg19_txt,
            cache_dir=cache_dir,
        )
        print(f"Collected {len(dataset_windows[dataset_name])} samples for {dataset_name}.")

    latency_contexts = []
    for dataset_name in args.latency_datasets:
        if dataset_name not in dataset_windows:
            dataset_windows[dataset_name] = collect_token_windows(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                split=args.split,
                num_samples=args.num_samples,
                context_tokens=args.context_tokens,
                target_tokens=args.target_tokens,
                local_pg19_txt=args.local_pg19_txt,
                cache_dir=cache_dir,
            )
        latency_contexts.extend(window.context_text for window in dataset_windows[dataset_name])

    results = []
    for method_key in iter_method_keys(args):
        clear_memory()
        base_press, method = DISPLAY_NAMES[method_key]
        ratio = 0.0 if method_key == "none" else args.compression_ratio
        print(f"\nEvaluating {method}...")

        ppls = {}
        for dataset_name in args.datasets:
            press = build_method_press(method_key, args)
            ppls[dataset_name] = evaluate_ppl(model, dataset_windows[dataset_name], press)
            print(f"  {dataset_name} ppl={ppls[dataset_name]:.4f}")
            clear_memory()

        press = build_method_press(method_key, args)
        ttft, tpot, throughput = measure_generation_metrics(
            model=model,
            tokenizer=tokenizer,
            contexts=latency_contexts,
            press=press,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
        )
        print(f"  ttft={ttft:.4f}s, tpot={tpot:.4f}s, throughput={throughput:.2f} tokens/s")

        results.append(
            TableResult(
                base_press=base_press,
                method=method,
                ratio=ratio,
                wikitext_ppl=ppls.get("wikitext", float("nan")),
                pg19_ppl=ppls.get("pg19", float("nan")),
                ttft_s=ttft,
                tpot_s=tpot,
                throughput_tokens_s=throughput,
                model=args.model,
                device=device,
                dtype=str(dtype),
                samples_per_dataset=args.num_samples,
                context_tokens=args.context_tokens,
                target_tokens=args.target_tokens,
                max_new_tokens=args.max_new_tokens,
                uncertainty_weight=args.uncertainty_weight,
                latency_datasets=",".join(args.latency_datasets),
            )
        )

    return results


def validate_lambda_values(lambda_values: list[float]):
    expected_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    if len(lambda_values) != len(expected_values) or any(
        not math.isclose(actual, expected) for actual, expected in zip(lambda_values, expected_values)
    ):
        raise ValueError(
            "This lambda ablation table uses fixed columns: "
            "--lambda-values 0 0.25 0.5 1.0 2.0."
        )


def validate_ratio_values(ratio_values: list[float]):
    expected_values = [0.2, 0.4, 0.6, 0.8]
    if len(ratio_values) != len(expected_values) or any(
        not math.isclose(actual, expected) for actual, expected in zip(ratio_values, expected_values)
    ):
        raise ValueError(
            "This compression-ratio ablation table uses fixed columns: "
            "--ratio-values 0.2 0.4 0.6 0.8."
        )


def collect_required_windows(args, tokenizer, cache_dir: Path, metric: str) -> dict[str, list[TokenWindow]]:
    required_datasets = set()
    if metric == "wikitext_ppl":
        required_datasets.add("wikitext")
    elif metric == "pg19_ppl":
        required_datasets.add("pg19")
    elif metric in {"ttft", "tpot", "throughput"}:
        required_datasets.update(args.latency_datasets)

    dataset_windows = {}
    for dataset_name in sorted(required_datasets):
        dataset_windows[dataset_name] = collect_token_windows(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=args.split,
            num_samples=args.num_samples,
            context_tokens=args.context_tokens,
            target_tokens=args.target_tokens,
            local_pg19_txt=args.local_pg19_txt,
            cache_dir=cache_dir,
        )
        print(f"Collected {len(dataset_windows[dataset_name])} samples for {dataset_name}.")
    return dataset_windows


def evaluate_metric(
    args,
    model,
    tokenizer,
    dataset_windows: dict[str, list[TokenWindow]],
    press,
    metric: str,
) -> float:
    if metric == "wikitext_ppl":
        return evaluate_ppl(model, dataset_windows["wikitext"], press)
    if metric == "pg19_ppl":
        return evaluate_ppl(model, dataset_windows["pg19"], press)

    latency_contexts = []
    for dataset_name in args.latency_datasets:
        latency_contexts.extend(window.context_text for window in dataset_windows[dataset_name])

    ttft, tpot, throughput = measure_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        contexts=latency_contexts,
        press=press,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )
    if metric == "ttft":
        return ttft
    if metric == "tpot":
        return tpot
    if metric == "throughput":
        return throughput
    raise ValueError(f"Unknown metric: {metric}")


def run_lambda_ablation(args) -> list[LambdaAblationResult]:
    validate_lambda_values(args.lambda_values)
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

    cache_dir = configure_cache(args.cache_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = str(model.device)
    print_parameters(args, device=device, dtype=dtype, cache_dir=cache_dir)
    print(f"lambda_metric: {args.lambda_metric}")
    print(f"lambda_values: {', '.join(str(value) for value in args.lambda_values)}")

    dataset_windows = collect_required_windows(args, tokenizer, cache_dir, args.lambda_metric)
    results = []

    for base_name in BASE_PRESSES:
        method_key = f"ua_{base_name}"
        _, method = DISPLAY_NAMES[method_key]
        row_values = {
            "lambda_0": float("nan"),
            "lambda_0_25": float("nan"),
            "lambda_0_5": float("nan"),
            "lambda_1_0": float("nan"),
            "lambda_2_0": float("nan"),
        }

        print(f"\nEvaluating {method} lambda ablation...")
        for lambda_value in args.lambda_values:
            clear_memory()
            press = build_uncertainty_press(base_name, args, lambda_value)
            value = evaluate_metric(args, model, tokenizer, dataset_windows, press, args.lambda_metric)
            row_values[lambda_column_name(lambda_value)] = value
            print(f"  lambda={lambda_value:g}: {args.lambda_metric}={value:.4f}")

        results.append(
            LambdaAblationResult(
                base_press=method,
                metric=args.lambda_metric,
                model=args.model,
                device=device,
                dtype=str(dtype),
                compression_ratio=args.compression_ratio,
                samples_per_dataset=args.num_samples,
                context_tokens=args.context_tokens,
                target_tokens=args.target_tokens,
                max_new_tokens=args.max_new_tokens,
                **row_values,
            )
        )

    return results


def run_compression_ratio_ablation(args) -> list[CompressionRatioAblationResult]:
    validate_ratio_values(args.ratio_values)
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

    cache_dir = configure_cache(args.cache_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = str(model.device)
    print_parameters(args, device=device, dtype=dtype, cache_dir=cache_dir)
    print(f"ratio_metric: {args.ratio_metric}")
    print(f"ratio_values: {', '.join(str(value) for value in args.ratio_values)}")

    dataset_windows = collect_required_windows(args, tokenizer, cache_dir, args.ratio_metric)
    original_compression_ratio = args.compression_ratio
    results = []

    try:
        for method_key in args.ratio_methods:
            _, method = DISPLAY_NAMES[method_key]
            row_values = {
                "ratio_0_2": float("nan"),
                "ratio_0_4": float("nan"),
                "ratio_0_6": float("nan"),
                "ratio_0_8": float("nan"),
            }

            print(f"\nEvaluating {method} compression-ratio ablation...")
            for ratio_value in args.ratio_values:
                clear_memory()
                args.compression_ratio = ratio_value
                press = build_method_press(method_key, args)
                value = evaluate_metric(args, model, tokenizer, dataset_windows, press, args.ratio_metric)
                row_values[ratio_column_name(ratio_value)] = value
                print(f"  r={ratio_value:g}: {args.ratio_metric}={value:.4f}")

            results.append(
                CompressionRatioAblationResult(
                    method=method,
                    metric=args.ratio_metric,
                    model=args.model,
                    device=device,
                    dtype=str(dtype),
                    uncertainty_weight=args.uncertainty_weight,
                    samples_per_dataset=args.num_samples,
                    context_tokens=args.context_tokens,
                    target_tokens=args.target_tokens,
                    max_new_tokens=args.max_new_tokens,
                    **row_values,
                )
            )
    finally:
        args.compression_ratio = original_compression_ratio

    return results


def run_normalization_ablation(args) -> list[NormalizationAblationResult]:
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

    cache_dir = configure_cache(args.cache_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = str(model.device)
    print_parameters(args, device=device, dtype=dtype, cache_dir=cache_dir)
    print(f"norm_base_press: {args.norm_base_press}")
    print(f"norm_metric: {args.norm_metric}")

    dataset_windows = collect_required_windows(args, tokenizer, cache_dir, args.norm_metric)
    variants = [
        ("Full UA", True, True),
        ("w/o mean normalization", False, True),
        ("w/o variance normalization", True, False),
        ("Raw combination", False, False),
    ]

    results = []
    _, method = DISPLAY_NAMES[f"ua_{args.norm_base_press}"]
    for variant, normalize_mean, normalize_variance in variants:
        clear_memory()
        press = build_uncertainty_press(
            args.norm_base_press,
            args,
            args.uncertainty_weight,
            normalize_scores=normalize_mean,
            normalize_uncertainty=normalize_variance,
        )
        value = evaluate_metric(args, model, tokenizer, dataset_windows, press, args.norm_metric)
        print(
            f"  {variant}: normalize_mean={normalize_mean}, "
            f"normalize_variance={normalize_variance}, {args.norm_metric}={value:.4f}"
        )
        results.append(
            NormalizationAblationResult(
                variant=variant,
                normalize_mean=normalize_mean,
                normalize_variance=normalize_variance,
                metric=args.norm_metric,
                value=value,
                base_press=method,
                model=args.model,
                device=device,
                dtype=str(dtype),
                compression_ratio=args.compression_ratio,
                uncertainty_weight=args.uncertainty_weight,
                samples_per_dataset=args.num_samples,
                context_tokens=args.context_tokens,
                target_tokens=args.target_tokens,
                max_new_tokens=args.max_new_tokens,
            )
        )

    return results


@torch.no_grad()
def collect_kept_layer_sets(model, token_windows: list[TokenWindow], press) -> list[list[set[int]]]:
    sample_layer_sets = []
    for window in token_windows:
        recorder = RecordingScorerPress(press=press)
        context_ids = window.context_ids.to(model.device)
        with recorder(model):
            model(input_ids=context_ids, use_cache=True)

        sample_layer_sets.append(recorder.records)
    return sample_layer_sets


def average_kept_set_difference(base_sets: list[list[set[int]]], ua_sets: list[list[set[int]]]) -> float:
    differences = []
    for base_sample_sets, ua_sample_sets in zip(base_sets, ua_sets):
        for base_set, ua_set in zip(base_sample_sets, ua_sample_sets):
            if not base_set:
                continue
            differences.append(1.0 - (len(base_set.intersection(ua_set)) / len(base_set)))
    return sum(differences) / len(differences) if differences else float("nan")


def run_selection_difference(args) -> list[SelectionDifferenceResult]:
    if not args.model.startswith("EleutherAI/pythia-"):
        raise ValueError("Pythia is the only model allowed for this evaluation script.")

    cache_dir = configure_cache(args.cache_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = str(model.device)
    print_parameters(args, device=device, dtype=dtype, cache_dir=cache_dir)
    print(f"selection_dataset: {args.selection_dataset}")

    token_windows = collect_token_windows(
        tokenizer=tokenizer,
        dataset_name=args.selection_dataset,
        split=args.split,
        num_samples=args.num_samples,
        context_tokens=args.context_tokens,
        target_tokens=args.target_tokens,
        local_pg19_txt=args.local_pg19_txt,
        cache_dir=cache_dir,
    )
    print(f"Collected {len(token_windows)} samples for {args.selection_dataset}.")

    results = []
    for base_name in BASE_PRESSES:
        base_label, _ = DISPLAY_NAMES[base_name]
        print(f"\nEvaluating {base_label} selection difference...")

        clear_memory()
        base_press = build_base_press(base_name, args)
        base_ppl = evaluate_ppl(model, token_windows, base_press)

        clear_memory()
        ua_press = build_uncertainty_press(base_name, args, args.uncertainty_weight)
        ua_ppl = evaluate_ppl(model, token_windows, ua_press)

        clear_memory()
        base_sets = collect_kept_layer_sets(model, token_windows, build_base_press(base_name, args))
        clear_memory()
        ua_sets = collect_kept_layer_sets(
            model,
            token_windows,
            build_uncertainty_press(base_name, args, args.uncertainty_weight),
        )
        kept_difference = average_kept_set_difference(base_sets, ua_sets)
        ppl_change = ua_ppl - base_ppl
        print(
            f"  delta_keep={kept_difference:.4f}, "
            f"base_ppl={base_ppl:.4f}, ua_ppl={ua_ppl:.4f}, ppl_change={ppl_change:.4f}"
        )

        results.append(
            SelectionDifferenceResult(
                base_press=base_label,
                kept_set_difference=kept_difference,
                ppl_change=ppl_change,
                base_ppl=base_ppl,
                ua_ppl=ua_ppl,
                dataset=args.selection_dataset,
                model=args.model,
                device=device,
                dtype=str(dtype),
                compression_ratio=args.compression_ratio,
                uncertainty_weight=args.uncertainty_weight,
                samples=len(token_windows),
                context_tokens=args.context_tokens,
                target_tokens=args.target_tokens,
            )
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print main-table metrics for UncertaintyAwarePress on Pythia."
    )
    parser.add_argument(
        "--table",
        default="main",
        choices=["main", "lambda", "ratio", "norm", "selection", "all"],
        help="Which result table to print.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument(
        "--latency-datasets",
        nargs="+",
        default=["wikitext", "pg19"],
        choices=["wikitext", "pg19"],
        help="Datasets whose sampled contexts are pooled for TTFT, TPOT, and throughput.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list(DISPLAY_NAMES),
        help="Subset of table methods to evaluate. Default evaluates all rows.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
        help="Lambda ablation columns. Keep the default values to match the paper table.",
    )
    parser.add_argument(
        "--lambda-metric",
        default="wikitext_ppl",
        choices=["wikitext_ppl", "pg19_ppl", "ttft", "tpot", "throughput"],
        help="Metric printed in the lambda ablation cells.",
    )
    parser.add_argument(
        "--ratio-values",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Compression-ratio ablation columns. Keep the default values to match the paper table.",
    )
    parser.add_argument(
        "--ratio-methods",
        nargs="+",
        default=["ua_knorm", "ua_snapkv", "ua_cur", "ua_lagkv"],
        choices=[name for name in DISPLAY_NAMES if name != "none"],
        help="Methods to evaluate in the compression-ratio ablation table.",
    )
    parser.add_argument(
        "--ratio-metric",
        default="wikitext_ppl",
        choices=["wikitext_ppl", "pg19_ppl", "ttft", "tpot", "throughput"],
        help="Metric printed in the compression-ratio ablation cells.",
    )
    parser.add_argument(
        "--norm-base-press",
        default="knorm",
        choices=list(BASE_PRESSES),
        help="Base scorer wrapped by UA for the normalization ablation table.",
    )
    parser.add_argument(
        "--norm-metric",
        default="wikitext_ppl",
        choices=["wikitext_ppl", "pg19_ppl", "ttft", "tpot", "throughput"],
        help="Metric printed in the normalization ablation PPL/value column.",
    )
    parser.add_argument(
        "--selection-dataset",
        default="wikitext",
        choices=["wikitext", "pg19"],
        help="Dataset used for retained-token set difference and PPL change.",
    )
    parser.add_argument("--no-normalize-scores", action="store_true")
    parser.add_argument("--no-normalize-uncertainty", action="store_true")
    parser.add_argument("--snapkv-window-size", type=int, default=64)
    parser.add_argument("--snapkv-kernel-size", type=int, default=5)
    parser.add_argument("--cur-num-sinks", type=int, default=4)
    parser.add_argument("--cur-local-window-size", type=int, default=16)
    parser.add_argument("--lagkv-n-sink", type=int, default=4)
    parser.add_argument("--lagkv-lag-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--lambda-output-csv", default=str(DEFAULT_LAMBDA_OUTPUT_CSV))
    parser.add_argument("--ratio-output-csv", default=str(DEFAULT_RATIO_OUTPUT_CSV))
    parser.add_argument("--norm-output-csv", default=str(DEFAULT_NORM_OUTPUT_CSV))
    parser.add_argument("--selection-output-csv", default=str(DEFAULT_SELECTION_OUTPUT_CSV))
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory for per-run JSONL logs. Default creates results/TIMESTAMP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_output_dir = create_run_output_dir(args.run_dir)
    run_jsonl = run_output_dir / "results.jsonl"
    print(f"\nPer-run JSONL: {run_jsonl}")

    if args.table in {"main", "all"}:
        results = run(args)
        print_results_table(results)
        output_path = write_csv(results, args.output_csv)
        append_jsonl(run_jsonl, "main", results)
        print(f"\nWrote main results to {output_path}")

    if args.table in {"lambda", "all"}:
        lambda_results = run_lambda_ablation(args)
        print_lambda_ablation_table(lambda_results, args.lambda_values)
        lambda_output_path = write_lambda_csv(lambda_results, args.lambda_output_csv)
        append_jsonl(run_jsonl, "lambda_ablation", lambda_results)
        print(f"\nWrote lambda ablation results to {lambda_output_path}")

    if args.table in {"ratio", "all"}:
        ratio_results = run_compression_ratio_ablation(args)
        print_compression_ratio_ablation_table(ratio_results, args.ratio_values)
        ratio_output_path = write_ratio_csv(ratio_results, args.ratio_output_csv)
        append_jsonl(run_jsonl, "compression_ratio_ablation", ratio_results)
        print(f"\nWrote compression-ratio ablation results to {ratio_output_path}")

    if args.table in {"norm", "all"}:
        norm_results = run_normalization_ablation(args)
        print_normalization_ablation_table(norm_results)
        norm_output_path = write_normalization_csv(norm_results, args.norm_output_csv)
        append_jsonl(run_jsonl, "normalization_ablation", norm_results)
        print(f"\nWrote normalization ablation results to {norm_output_path}")

    if args.table in {"selection", "all"}:
        selection_results = run_selection_difference(args)
        print_selection_difference_table(selection_results)
        selection_output_path = write_selection_csv(selection_results, args.selection_output_csv)
        append_jsonl(run_jsonl, "selection_difference", selection_results)
        print(f"\nWrote selection difference results to {selection_output_path}")


if __name__ == "__main__":
    main()
