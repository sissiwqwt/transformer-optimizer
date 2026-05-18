# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import AdaKVPress, KnormPress, SnapKVPress, StreamingLLMPress


MODEL_NAME = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."
DEFAULT_COMPRESSION_RATIOS = [0.2, 0.4, 0.5, 0.6, 0.8]


def find_repo_root() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
# RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "pythia_kvpress_test"
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "cuda_base_test"
CPU_RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "cpu_base_test"
DEFAULT_OUTPUT_CSV_NAME = "pythia_kvpress_test_results.csv"
DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / DEFAULT_OUTPUT_CSV_NAME


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


def get_results_output_dir(device: str) -> Path:
    if device == "cpu":
        return CPU_RESULTS_OUTPUT_DIR
    return RESULTS_OUTPUT_DIR


def resolve_output_csv_path(path: str | Path, output_dir: Path = RESULTS_OUTPUT_DIR) -> Path:
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    if output_path.parent == Path("."):
        return output_dir / output_path
    return REPO_ROOT / output_path


@dataclass
class Result:
    dataset: str
    press: str
    model: str
    compression_ratio: float
    effective_compression_ratio: float
    keep_ratio: float
    device: str
    dtype: str
    ppl: float
    throughput_tokens_s: float
    avg_latency_s: float
    samples: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int
    warmup: int
    throughput_samples: int


class NoPress:
    """Baseline press: no KV compression."""

    def __call__(self, model):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def iter_texts(dataset_name: str, split: str, local_pg19_txt: str | None = None) -> Iterable[str]:
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
        return

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def collect_token_windows(
    tokenizer,
    dataset_name: str,
    split: str,
    num_samples: int,
    context_tokens: int,
    target_tokens: int,
    local_pg19_txt: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    windows = []
    required_tokens = context_tokens + target_tokens
    buffer: list[int] = []

    for text in iter_texts(dataset_name, split, local_pg19_txt):
        # Add a newline to avoid unnaturally joining separate documents/paragraphs.
        buffer.extend(tokenizer.encode(text + "\n", add_special_tokens=False))

        if len(buffer) < required_tokens:
            continue

        token_ids = buffer[:required_tokens]

        context_ids = torch.tensor([token_ids[:context_tokens]], dtype=torch.long)
        target_ids = torch.tensor([token_ids[context_tokens:]], dtype=torch.long)
        context_text = tokenizer.decode(token_ids[:context_tokens], skip_special_tokens=True)

        windows.append((context_ids, target_ids, context_text))

        # Non-overlapping windows.
        buffer = buffer[required_tokens:]

        if len(windows) >= num_samples:
            break

    if not windows:
        raise RuntimeError(
            f"No usable {dataset_name} samples with at least {required_tokens} tokens. "
            "Lower --context-tokens/--target-tokens or provide --local-pg19-txt."
        )

    return windows


def build_press(name: str, compression_ratio: float, snapkv_window_size: int):
    if name == "none":
        return NoPress()

    if name == "adakv":
        return AdaKVPress(
            SnapKVPress(
                compression_ratio=compression_ratio,
                window_size=snapkv_window_size,
            )
        )

    if name == "knorm":
        return KnormPress(compression_ratio=compression_ratio)

    if name == "snapkv":
        return SnapKVPress(
            compression_ratio=compression_ratio,
            window_size=snapkv_window_size,
        )

    if name == "streamingllm":
        return StreamingLLMPress(compression_ratio=compression_ratio)

    raise ValueError(f"Unknown press: {name}")


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
    """
    Important for compressed KV cache.

    After KV compression, cache length is shorter than the original context length.
    The rotary position_ids should stay on the original absolute timeline, while
    cache_position should follow the physical cache indices used for masking/cache updates.

    Example:
      original context length = 1024
      compressed cache length = 512
      position_ids should start at 1024
      cache_position should start at 512
    """
    forward_params = inspect.signature(model.forward).parameters

    position_ids = torch.arange(
        absolute_start_pos,
        absolute_start_pos + seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    kwargs = {"position_ids": position_ids}

    # Newer Transformers models may use cache_position.
    if "cache_position" in forward_params:
        kwargs["cache_position"] = torch.arange(
            cache_start_pos,
            cache_start_pos + seq_len,
            device=device,
            dtype=torch.long,
        )

    return kwargs


def _preview_text(text: str, max_chars: int = 1200) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@torch.no_grad()
def evaluate_ppl(model, tokenizer, token_windows, press, dataset_name: str, press_name: str) -> float:
    """
    Computes causal LM perplexity on target tokens conditioned on context tokens.

    Scoring layout:
      context: x_0 ... x_{C-1}
      target:  y_0 ... y_{T-1}

    Loss terms:
      p(y_0 | context)
      p(y_1 | context, y_0)
      ...
      p(y_{T-1} | context, y_0 ... y_{T-2})
    """
    model.eval()

    nll_sum = 0.0
    token_count = 0
    printed_check = False

    for context_ids, target_ids, context_text in token_windows:
        context_ids = context_ids.to(model.device)
        target_ids = target_ids.to(model.device)

        # Prefill context. If press is not NoPress, KV cache will be compressed here.
        with press(model):
            context_outputs = model(
                input_ids=context_ids,
                use_cache=True,
            )

        cache = context_outputs.past_key_values

        if cache is None:
            raise RuntimeError("Model did not return past_key_values. " "Make sure the model supports use_cache=True.")

        # First target token is predicted by the last context logits.
        first_token_logits = context_outputs.logits[:, -1, :]
        first_token_loss = F.cross_entropy(
            first_token_logits,
            target_ids[:, 0],
            reduction="sum",
        )
        predicted_tokens = [first_token_logits.argmax(dim=-1, keepdim=True)]

        nll_sum += first_token_loss.item()
        token_count += 1

        # Remaining target tokens are predicted using previous target tokens.
        if target_ids.shape[1] > 1:
            continuation_ids = target_ids[:, :-1]
            continuation_labels = target_ids[:, 1:]

            seq_len = continuation_ids.shape[1]

            position_kwargs = _build_position_kwargs(
                model=model,
                absolute_start_pos=context_ids.shape[1],
                cache_start_pos=_cache_seq_length(cache),
                seq_len=seq_len,
                device=model.device,
            )

            outputs = model(
                input_ids=continuation_ids,
                past_key_values=cache,
                use_cache=True,
                **position_kwargs,
            )

            logits = outputs.logits
            predicted_tokens.append(logits.argmax(dim=-1))

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                continuation_labels.reshape(-1),
                reduction="sum",
            )

            nll_sum += loss.item()
            token_count += continuation_labels.numel()

        # if not printed_check:
        #     predicted_target_ids = torch.cat(predicted_tokens, dim=1)
        #     target_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        #     predicted_text = tokenizer.decode(predicted_target_ids[0], skip_special_tokens=True)

        #     print(f"\n[PPL check] dataset={dataset_name} press={press_name}")
        #     print(f"Input context:\n{_preview_text(context_text)}")
        #     print(f"Target context:\n{_preview_text(target_text)}")
        #     print(f"Model predicted context:\n{_preview_text(predicted_text)}")
        #     printed_check = True

    return math.exp(nll_sum / token_count)


@torch.no_grad()
def evaluate_throughput(
    model,
    tokenizer,
    token_windows,
    press,
    dataset_name: str,
    press_name: str,
    max_new_tokens: int,
    warmup: int,
    throughput_samples: int = 3,
) -> tuple[float, float]:
    if throughput_samples <= 0:
        raise ValueError(f"throughput_samples must be positive, got {throughput_samples}.")

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")

    model.eval()
    question_ids = tokenizer.encode(QUESTION + "\n", return_tensors="pt", add_special_tokens=False).to(model.device)

    def decode_once(context_ids: torch.Tensor, timed: bool) -> float:
        context_ids = context_ids.to(model.device)

        # Build the compressed context cache outside the measured decode window.
        with press(model):
            context_outputs = model(
                input_ids=context_ids,
                use_cache=True,
            )

        cache = context_outputs.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return past_key_values. " "Make sure the model supports use_cache=True.")

        question_position_kwargs = _build_position_kwargs(
            model=model,
            absolute_start_pos=context_ids.shape[1],
            cache_start_pos=_cache_seq_length(cache),
            seq_len=question_ids.shape[1],
            device=model.device,
        )
        question_outputs = model(
            input_ids=question_ids,
            past_key_values=cache,
            use_cache=True,
            **question_position_kwargs,
        )

        next_token = question_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        absolute_position = context_ids.shape[1] + question_ids.shape[1]

        if timed:
            synchronize()
            start = time.perf_counter()

        for step in range(max_new_tokens):
            position_kwargs = _build_position_kwargs(
                model=model,
                absolute_start_pos=absolute_position + step,
                cache_start_pos=_cache_seq_length(cache),
                seq_len=1,
                device=model.device,
            )
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                **position_kwargs,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if not timed:
            return 0.0

        synchronize()
        return time.perf_counter() - start

    warmup_windows = token_windows[:warmup]
    for context_ids, _, _ in warmup_windows:
        decode_once(context_ids, timed=False)

    latencies = []

    for _ in range(throughput_samples):
        for context_ids, _, _ in token_windows:
            latencies.append(decode_once(context_ids, timed=True))

    total_latency = sum(latencies)
    avg_latency = total_latency / len(latencies)
    throughput = (len(token_windows) * throughput_samples * max_new_tokens) / total_latency

    return avg_latency, throughput


def get_press_dtype(press_name: str, default_dtype: torch.dtype, use_cuda: bool) -> torch.dtype:
    if use_cuda and press_name == "adakv":
        return torch.float32
    return default_dtype


def set_model_dtype(model, dtype: torch.dtype) -> None:
    current_dtype = next(param.dtype for param in model.parameters() if param.is_floating_point())
    if current_dtype == dtype:
        return
    model.to(dtype=dtype)


def effective_compression_ratio(press_name: str, compression_ratio: float) -> float:
    if press_name == "none":
        return 0.0
    return compression_ratio


def get_compression_ratios(args) -> list[float]:
    ratios = [args.compression_ratio] if args.compression_ratio is not None else args.compression_ratios
    for ratio in ratios:
        if not 0 <= ratio < 1:
            raise ValueError(f"Compression ratio must be in [0, 1), got {ratio}.")
    return ratios


def run(args) -> list[Result]:
    device = resolve_device(args.device)
    use_auto_device = args.device == "auto"
    use_cuda = device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    compression_ratios = get_compression_ratios(args)

    print(f"Using device: {device} (requested: {args.device})")
    print(f"Compression ratios: {', '.join(str(ratio) for ratio in compression_ratios)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
    )

    model_kwargs = {"torch_dtype": dtype}
    if use_auto_device and use_cuda:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if not use_auto_device:
        model = model.to(device)
    elif not use_cuda:
        model = model.to("cpu")

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

        none_baseline_result = None

        for compression_ratio in compression_ratios:
            print(f"\nCompression ratio: {compression_ratio}")

            for press_name in args.presses:
                if press_name == "none" and none_baseline_result is not None:
                    result = Result(
                        dataset=dataset_name,
                        press=press_name,
                        model=args.model,
                        compression_ratio=compression_ratio,
                        effective_compression_ratio=0.0,
                        keep_ratio=1.0,
                        device=none_baseline_result.device,
                        dtype=none_baseline_result.dtype,
                        ppl=none_baseline_result.ppl,
                        throughput_tokens_s=none_baseline_result.throughput_tokens_s,
                        avg_latency_s=none_baseline_result.avg_latency_s,
                        samples=none_baseline_result.samples,
                        context_tokens=none_baseline_result.context_tokens,
                        target_tokens=none_baseline_result.target_tokens,
                        max_new_tokens=none_baseline_result.max_new_tokens,
                        warmup=none_baseline_result.warmup,
                        throughput_samples=none_baseline_result.throughput_samples,
                    )
                    results.append(result)
                    print(
                        f"{press_name:>12}: ppl={result.ppl:.4f}, "
                        f"throughput={result.throughput_tokens_s:.2f} tokens/s, "
                        f"avg_latency={result.avg_latency_s:.4f}s, "
                        f"throughput_samples={result.throughput_samples} "
                        f"(reused none baseline)"
                    )
                    continue

                clear_memory()
                press_dtype = get_press_dtype(press_name, dtype, use_cuda)
                set_model_dtype(model, press_dtype)

                press = build_press(
                    name=press_name,
                    compression_ratio=compression_ratio,
                    snapkv_window_size=args.snapkv_window_size,
                )

                ppl = evaluate_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    token_windows=token_windows,
                    press=press,
                    dataset_name=dataset_name,
                    press_name=press_name,
                )

                clear_memory()

                press = build_press(
                    name=press_name,
                    compression_ratio=compression_ratio,
                    snapkv_window_size=args.snapkv_window_size,
                )

                avg_latency, throughput = evaluate_throughput(
                    model=model,
                    tokenizer=tokenizer,
                    token_windows=token_windows,
                    press=press,
                    dataset_name=dataset_name,
                    press_name=press_name,
                    max_new_tokens=args.max_new_tokens,
                    warmup=args.warmup,
                    throughput_samples=args.throughput_samples,
                )

                print(
                    f"{press_name:>12}: ppl={ppl:.4f}, "
                    f"throughput={throughput:.2f} tokens/s, "
                    f"avg_latency={avg_latency:.4f}s, "
                    f"throughput_samples={args.throughput_samples}"
                )

                effective_ratio = effective_compression_ratio(press_name, compression_ratio)
                result = Result(
                    dataset=dataset_name,
                    press=press_name,
                    model=args.model,
                    compression_ratio=compression_ratio,
                    effective_compression_ratio=effective_ratio,
                    keep_ratio=1 - effective_ratio,
                    device=device,
                    dtype=str(press_dtype).replace("torch.", ""),
                    ppl=ppl,
                    throughput_tokens_s=throughput,
                    avg_latency_s=avg_latency,
                    samples=len(token_windows),
                    context_tokens=args.context_tokens,
                    target_tokens=args.target_tokens,
                    max_new_tokens=args.max_new_tokens,
                    warmup=args.warmup,
                    throughput_samples=args.throughput_samples,
                )
                results.append(result)

                if press_name == "none":
                    none_baseline_result = result

    return results


def write_csv(results: list[Result], output_csv: str | Path, output_dir: Path = RESULTS_OUTPUT_DIR) -> Path:
    fieldnames = list(Result.__dataclass_fields__)
    output_path = resolve_output_csv_path(output_csv, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.__dict__)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KVPress methods on causal LM perplexity and throughput.")

    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0. Default: auto.",
    )
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
        default=["none", "adakv", "knorm", "snapkv", "streamingllm"],
        choices=["none", "adakv", "knorm", "snapkv", "streamingllm"],
    )

    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--throughput-samples",
        type=int,
        default=3,
        help="Number of measured throughput passes over all sampled contexts. Default: 3.",
    )
    parser.add_argument(
        "--compression-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_COMPRESSION_RATIOS,
        help="Compression ratios to sweep. Default: 0.2 0.4 0.5 0.6 0.8.",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help="Optional single-ratio override for backwards-compatible one-off runs.",
    )
    parser.add_argument("--snapkv-window-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--local-pg19-txt", default="data/pg19_sample.txt")
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV_NAME,
        help=(
            "CSV output path. Passing only a filename, e.g. results.csv, saves to "
            f"{RESULTS_OUTPUT_DIR / '<filename>.csv'} for CUDA and "
            f"{CPU_RESULTS_OUTPUT_DIR / '<filename>.csv'} for CPU."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results = run(args)
    output_dir = get_results_output_dir(resolve_device(args.device))

    output_path = write_csv(
        results=results,
        output_csv=args.output_csv,
        output_dir=output_dir,
    )

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
