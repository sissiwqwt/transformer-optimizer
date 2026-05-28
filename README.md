# KVPress Pythia Reproduction

This README is for Wenqian Fan's personal part. But this repository also contains our group's work on uncertainty aware kvpress innovation.

This repository contains a reproduction-oriented KV cache compression evaluation based on the original `kvpress` implementation. The KVPress methods themselves are already implemented by the upstream project; this work focuses on running a small Pythia benchmark and visualizing the resulting perplexity and throughput sweeps.
The generated evaluation report is available at [Evaluation report](report/kvpress_pythia_report.pdf).

The reproduction covers four KVPress methods, plus an uncompressed baseline:

- `adakv`: `AdaKVPress(SnapKVPress(...))`
- `knorm`: `KnormPress`
- `snapkv`: `SnapKVPress`
- `streamingllm`: `StreamingLLMPress`
- `none`: no-compression baseline used for comparison

This repository also contains `kvpress\presses\anchor_dedup_press.py`, an experimental new KVPress method developed in this reproduction work. Its current results are poor, so it is left out of the four-method sweep and not discussed as a main result.

## Environment Requirements

- Python `>=3.10`
- `uv` package manager
- PyTorch, Transformers, Datasets, Pandas, and the local `kvpress` package
- Matplotlib for figure generation
- CUDA is optional. The benchmark supports CPU and CUDA; `--device auto` selects CUDA when available and otherwise falls back to CPU.

From the repository root:

```powershell
uv sync
```

If Matplotlib is not installed in your environment, install the development dependency group as well:

```powershell
uv sync --group dev
```

Run all commands from:

```text
D:\workspace\transformer_proj\kvpress
```

## Run The KVPress Sweep

Use `pythia_kvpress_test.py` to run the actual benchmark. It evaluates `EleutherAI/pythia-70m` on WikiText and PG-19, then writes a CSV containing PPL and decode throughput.

CPU reproduction command:

```powershell
python pythia_kvpress_test.py --output-csv ctxt1024_tar_256_try_cpu.csv --device cpu
```

With `--device cpu`, a filename-only `--output-csv` is saved under:

```text
results\cpu_base_test\
```

With CUDA, filename-only outputs are saved under:

```text
results\cuda_base_test\
```

## Visualize The Sweep

Use `visualize_pythia_kvpress_sweep.py` to plot a benchmark CSV. The script looks for CSV files by absolute path, repo-relative path, or by filename inside `results\cuda_base_test\` and `results\cpu_base_test\`.

For the CPU reproduction CSV:

```powershell
python visualize_pythia_kvpress_sweep.py --csv ctxt1024_tar_256_try_cpu.csv
```

By default, figures are written to a `figures` directory next to the CSV, for example:

```text
results\cpu_base_test\figures\
```

For each dataset, the visualization script writes:

- PPL vs compression ratio
- Throughput vs compression ratio
- PPL/throughput tradeoff scatter plot
- Relative PPL and throughput compared with `none`

## Optional CLI Parameters

### `pythia_kvpress_test.py`

| Parameter | Default | Description |
| --- | --- | --- |
| `--model` | `EleutherAI/pythia-70m` | Hugging Face causal LM to evaluate. |
| `--device` | `auto` | Use `auto`, `cpu`, `cuda`, or `cuda:0`. |
| `--use-fast-tokenizer` | `False` | Enable the fast tokenizer. |
| `--datasets` | `wikitext pg19` | Datasets to evaluate. Choices: `wikitext`, `pg19`. |
| `--presses` | `none adakv knorm snapkv streamingllm` | Presses to evaluate. |
| `--split` | `test` | Dataset split. |
| `--num-samples` | `3` | Number of token windows per dataset. |
| `--context-tokens` | `1024` | Context tokens used for prefill and compression. |
| `--target-tokens` | `256` | Target tokens used for perplexity. |
| `--max-new-tokens` | `64` | Number of generated tokens used for throughput timing. |
| `--throughput-samples` | `3` | Measured throughput passes over all sampled contexts. |
| `--compression-ratios` | `0.2 0.4 0.5 0.6 0.8` | Compression ratios to sweep. |
| `--compression-ratio` | `None` | Single-ratio override for one-off runs. |
| `--snapkv-window-size` | `64` | SnapKV observation window size. |
| `--warmup` | `3` | Warmup decode passes before throughput timing. |
| `--local-pg19-txt` | `data/pg19_sample.txt` | Local PG-19 text file used before trying remote PG-19. |
| `--output-csv` | `pythia_kvpress_test_results.csv` | Output CSV path or filename. |

Example with explicit defaults:

```powershell
python pythia_kvpress_test.py `
  --model EleutherAI/pythia-70m `
  --device cpu `
  --datasets wikitext pg19 `
  --presses none adakv knorm snapkv streamingllm `
  --split test `
  --num-samples 3 `
  --context-tokens 1024 `
  --target-tokens 256 `
  --max-new-tokens 64 `
  --throughput-samples 3 `
  --compression-ratios 0.2 0.4 0.5 0.6 0.8 `
  --snapkv-window-size 64 `
  --warmup 3 `
  --local-pg19-txt data/pg19_sample.txt `
  --output-csv ctxt1024_tar_256_try_cpu.csv
```

### `visualize_pythia_kvpress_sweep.py`

| Parameter | Default | Description |
| --- | --- | --- |
| `--csv` | `pythia_kvpress_test_results.csv` | CSV path. Can be absolute, repo-relative, or a filename in `results\cuda_base_test\` or `results\cpu_base_test\`. |
| `--output-dir` | `<csv parent>\figures` | Directory for generated PNG figures. |

Example with a custom output directory:

```powershell
python visualize_pythia_kvpress_sweep.py `
  --csv results\cpu_base_test\ctxt1024_tar_256_try_cpu.csv `
  --output-dir results\cpu_base_test\figures
```

## Metrics

Perplexity is computed on `--target-tokens` target tokens after prefilling a `--context-tokens` context window. For compressed methods, prefill runs inside the KVPress hook, so target scoring uses the compressed KV cache.

Throughput is measured after prefilling and compression. The reported `throughput_tokens_s` times the greedy decode loop for `--max-new-tokens`; it does not include the one-time compression cost.
