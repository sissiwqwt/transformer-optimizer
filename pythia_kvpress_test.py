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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from kvpress import AdaKVPress, KnormPress, SnapKVPress, StreamingLLMPress


MODEL_NAME = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."


def find_repo_root() -> Path:
    for path in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
RESULTS_OUTPUT_DIR = REPO_ROOT / "results" / "pythia_kvpress_test"
DEFAULT_OUTPUT_CSV = RESULTS_OUTPUT_DIR / "pythia_kvpress_test_results.csv"


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
class Result:
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
    gen_pipe,
    contexts: list[str],
    press,
    dataset_name: str,
    press_name: str,
    max_new_tokens: int,
    warmup: int,
) -> tuple[float, float]:
    for context in contexts[:warmup]:
        gen_pipe(
            context,
            question=QUESTION,
            press=press,
            max_new_tokens=max_new_tokens,
        )

    latencies = []
    printed_check = False

    for context in contexts:
        synchronize()
        start = time.perf_counter()

        result = gen_pipe(
            context,
            question=QUESTION,
            press=press,
            max_new_tokens=max_new_tokens,
        )

        synchronize()
        latencies.append(time.perf_counter() - start)

        # if not printed_check:
        #     print(f"\n[Throughput check] dataset={dataset_name} press={press_name}")
        #     print(f"Question:{QUESTION}")
        #     print(f"Answer:\n{_preview_text(result['answer'])}")
        #     printed_check = True

    total_latency = sum(latencies)
    avg_latency = total_latency / len(latencies)
    throughput = (len(contexts) * max_new_tokens) / total_latency

    return avg_latency, throughput


def run(args) -> list[Result]:
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

            press = build_press(
                name=press_name,
                compression_ratio=args.compression_ratio,
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
                compression_ratio=args.compression_ratio,
                snapkv_window_size=args.snapkv_window_size,
            )

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
                Result(
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
                )
            )

    return results


def write_csv(results: list[Result], output_csv: str | Path) -> Path:
    fieldnames = list(Result.__dataclass_fields__)
    output_path = resolve_output_csv_path(output_csv)
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
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--snapkv-window-size", type=int, default=64)
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

    results = run(args)

    output_path = write_csv(
        results=results,
        output_csv=args.output_csv,
    )

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
