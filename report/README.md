# Running `pythia_kvpress_test.py`

This benchmark evaluates KVPress methods on `EleutherAI/pythia-70m` using
perplexity and decode throughput.

## Setup

From the repository root:

```powershell
uv sync
```

Then run commands from:

```powershell
D:\workspace\transformer_proj\kvpress
```

## CPU Run Used For This Report

```powershell
python pythia_kvpress_test.py --output-csv ctxt1024_tar_256_try_cpu.csv --device cpu
```

With `--device cpu`, a filename-only `--output-csv` is saved under:

```text
results/cpu_base_test/
```

The reproduced report uses:

```text
results/cpu_base_test/ctxt1024_tar_256_try_cpu.csv
```

## Common Options

```powershell
python pythia_kvpress_test.py `
  --device cpu `
  --datasets wikitext pg19 `
  --presses none adakv knorm snapkv streamingllm `
  --context-tokens 1024 `
  --target-tokens 256 `
  --max-new-tokens 64 `
  --compression-ratios 0.2 0.4 0.5 0.6 0.8 `
  --throughput-samples 3 `
  --warmup 3 `
  --output-csv ctxt1024_tar_256_try_cpu.csv
```

Use `--device auto` or `--device cuda` for GPU evaluation. CUDA outputs are
saved to `results/cuda_base_test/` when `--output-csv` is only a filename.

## What The Metrics Mean

Perplexity is computed on `--target-tokens` target tokens after prefilling a
`--context-tokens` context window. For compressed methods, the prefill is run
inside the press hook, so target scoring uses the compressed KV cache.

Throughput starts after prefilling and compression. For `none`, timing starts
after prefilling only. The reported `throughput_tokens_s` measures only the
greedy decode loop for `--max-new-tokens`, so it isolates the benefit of a
smaller KV cache during decoding and does not include one-time compression cost.

## Plotting Results

After a run, generate figures with:

```powershell
python visualize_pythia_kvpress_sweep.py --csv ctxt1024_tar_256_try_cpu.csv
```

For the CPU CSV above, figures are written to:

```text
results/cpu_base_test/figures/
```

## PG-19 Data

By default the script uses:

```text
data/pg19_sample.txt
```

for PG-19 if `--local-pg19-txt` is left unchanged. To evaluate a different local
PG-19 text file:

```powershell
python pythia_kvpress_test.py --device cpu --local-pg19-txt path\to\pg19.txt
```
