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

from kvpress import BasePress, KVzapPress, KnormPress


MODEL_NAME = "EleutherAI/pythia-70m"
QUESTION = "\nSummarize the passage in one sentence."
KVZAP_UNAVAILABLE_HINTS = (
    "is not a local folder and is not a valid model identifier",
    "Repository Not Found",
    "401 Client Error",
    "401 Unauthorized",
)


@dataclass
class KnormKVzapSequentialPress(BasePress):
    """
    Apply Knorm first, then KVzap on the Knorm-retained token subset.

    The generic ComposedPress is not appropriate for this pair in Knorm -> KVzap
    order because KVzap scores are predicted from original hidden states. This
    class keeps the original indices retained by Knorm and gathers KVzap scores
    at those positions before the second pruning stage.
    """

    knorm_compression_ratio: float = 0.25
    kvzap_compression_ratio: float = 0.25
    kvzap_model_type: str = "mlp"

    def __post_init__(self):
        self.knorm = KnormPress(compression_ratio=self.knorm_compression_ratio)
        self.kvzap = KVzapPress(compression_ratio=self.kvzap_compression_ratio, model_type=self.kvzap_model_type)
        self.compression_ratio = 1 - (1 - self.knorm_compression_ratio) * (1 - self.kvzap_compression_ratio)

    def post_init_from_model(self, model):
        self.knorm.post_init_from_model(model)
        self.kvzap.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        k_len = keys.shape[2]
        knorm_n_kept = int(k_len * (1 - self.knorm_compression_ratio))
        knorm_scores = self.knorm.score(module, hidden_states, keys, values, attentions, kwargs)
        knorm_indices = knorm_scores.topk(knorm_n_kept, dim=-1).indices
        gather_indices = knorm_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()

        kvzap_scores = self.kvzap.score(module, hidden_states, keys, values, attentions, kwargs)
        kvzap_scores = kvzap_scores.gather(2, knorm_indices)
        kvzap_n_kept = int(knorm_n_kept * (1 - self.kvzap_compression_ratio))
        kvzap_indices = kvzap_scores.topk(kvzap_n_kept, dim=-1).indices
        gather_indices = kvzap_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()


@dataclass
class Result:
    dataset: str
    press: str
    model: str
    effective_compression_ratio: float
    knorm_compression_ratio: float
    kvzap_compression_ratio: float
    ppl: float
    throughput_tokens_s: float
    avg_latency_s: float
    samples: int
    context_tokens: int
    target_tokens: int
    max_new_tokens: int


class NoPress:
    compression_ratio = 0.0

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


def build_press(name: str, knorm_ratio: float, kvzap_ratio: float, kvzap_model_type: str):
    if name == "none":
        return NoPress()
    if name == "knorm":
        return KnormPress(compression_ratio=knorm_ratio)
    if name == "kvzap":
        return KVzapPress(compression_ratio=kvzap_ratio, model_type=kvzap_model_type)
    if name == "knorm_kvzap":
        return KnormKVzapSequentialPress(
            knorm_compression_ratio=knorm_ratio,
            kvzap_compression_ratio=kvzap_ratio,
            kvzap_model_type=kvzap_model_type,
        )
    raise ValueError(f"Unknown press: {name}")


def uses_kvzap(press_name: str) -> bool:
    return "kvzap" in press_name


def is_unavailable_kvzap_error(exc: Exception) -> bool:
    message = str(exc)
    return any(hint in message for hint in KVZAP_UNAVAILABLE_HINTS)


def kvzap_repo_name(model_name: str, kvzap_model_type: str) -> str:
    return f"nvidia/KVzap-{kvzap_model_type}-{model_name.split('/')[-1]}"


def effective_compression_ratio(press) -> float:
    ratio = getattr(press, "compression_ratio", 0.0)
    return 0.0 if ratio is None else float(ratio)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_position_kwargs(model, start_pos: int, seq_len: int, device) -> dict:
    forward_params = inspect.signature(model.forward).parameters
    position_ids = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long).unsqueeze(0)
    kwargs = {"position_ids": position_ids}
    if "cache_position" in forward_params:
        kwargs["cache_position"] = position_ids.squeeze(0)
    return kwargs


@torch.no_grad()
def evaluate_ppl(model, token_windows, press) -> float:
    model.eval()
    nll_sum = 0.0
    token_count = 0

    for context_ids, target_ids, _ in token_windows:
        context_ids = context_ids.to(model.device)
        target_ids = target_ids.to(model.device)

        with press(model):
            context_outputs = model(input_ids=context_ids, use_cache=True)

        cache = context_outputs.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return past_key_values with use_cache=True.")

        first_token_logits = context_outputs.logits[:, -1, :]
        nll_sum += F.cross_entropy(first_token_logits, target_ids[:, 0], reduction="sum").item()
        token_count += 1

        if target_ids.shape[1] > 1:
            continuation_ids = target_ids[:, :-1]
            continuation_labels = target_ids[:, 1:]
            position_kwargs = build_position_kwargs(
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
            logits = outputs.logits
            nll_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                continuation_labels.reshape(-1),
                reduction="sum",
            ).item()
            token_count += continuation_labels.numel()

    return math.exp(nll_sum / token_count)


@torch.no_grad()
def evaluate_throughput(gen_pipe, contexts: list[str], press, max_new_tokens: int, warmup: int) -> tuple[float, float]:
    for context in contexts[:warmup]:
        gen_pipe(context, question=QUESTION, press=press, max_new_tokens=max_new_tokens)

    latencies = []
    for context in contexts:
        synchronize()
        start = time.perf_counter()
        gen_pipe(context, question=QUESTION, press=press, max_new_tokens=max_new_tokens)
        synchronize()
        latencies.append(time.perf_counter() - start)

    total_latency = sum(latencies)
    return total_latency / len(latencies), (len(contexts) * max_new_tokens) / total_latency


def run(args) -> list[Result]:
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
            press = build_press(press_name, args.knorm_compression_ratio, args.kvzap_compression_ratio, args.kvzap_type)
            try:
                ppl = evaluate_ppl(model, token_windows, press)
            except OSError as exc:
                if not uses_kvzap(press_name) or args.strict_kvzap or not is_unavailable_kvzap_error(exc):
                    raise
                print(
                    f"{press_name:>12}: skipped because {kvzap_repo_name(args.model, args.kvzap_type)} "
                    "is unavailable. Use a supported base model, authenticate to Hugging Face, "
                    "or train matching KVzap weights."
                )
                continue

            clear_memory()
            press = build_press(press_name, args.knorm_compression_ratio, args.kvzap_compression_ratio, args.kvzap_type)
            avg_latency, throughput = evaluate_throughput(
                gen_pipe=gen_pipe,
                contexts=contexts,
                press=press,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
            )

            print(
                f"{press_name:>12}: ppl={ppl:.4f}, throughput={throughput:.2f} tokens/s, "
                f"avg_latency={avg_latency:.4f}s"
            )
            results.append(
                Result(
                    dataset=dataset_name,
                    press=press_name,
                    model=args.model,
                    effective_compression_ratio=effective_compression_ratio(press),
                    knorm_compression_ratio=args.knorm_compression_ratio,
                    kvzap_compression_ratio=args.kvzap_compression_ratio,
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


def write_csv(results: list[Result], output_csv: str):
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Knorm -> KVzap sequential compression on Pythia.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext", "pg19"], choices=["wikitext", "pg19"])
    parser.add_argument(
        "--presses",
        nargs="+",
        default=["none", "knorm", "kvzap", "knorm_kvzap"],
        choices=["none", "knorm", "kvzap", "knorm_kvzap"],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--knorm-compression-ratio", type=float, default=0.25)
    parser.add_argument("--kvzap-compression-ratio", type=float, default=0.25)
    parser.add_argument("--kvzap-type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--strict-kvzap", action="store_true", help="Raise instead of skipping unavailable KVzap weights.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-pg19-txt", default=None)
    parser.add_argument("--output-csv", default="pythia_knorm_kvzap_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run(args)
    write_csv(results, args.output_csv)
    print(f"\nWrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
