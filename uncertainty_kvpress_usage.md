# Pythia KVPress 脚本运行说明

本文档说明如何运行仓库根目录下的三个脚本：

- `pythia_uncertainty_aware.py`
- `visualize_pythia_ppl.py`
- `pythia_kvpress_test.py`

所有命令都建议在仓库根目录执行：

```powershell
cd D:\workspace\transformer_proj\kvpress
uv sync
```

后续示例使用 `uv run python ...`。如果已经激活 `.venv`，也可以直接使用 `python ...`。

## 通用注意事项

- 目前评测只允许使用 Pythia 模型，默认模型是 `EleutherAI/pythia-70m`。
- 数据集只使用 `wikitext` 和 `pg19`。
- 评测指标包括 PPL 和 throughput。
- `pg19` 默认会从 Hugging Face 以 streaming 方式加载。如果网络或数据集访问不稳定，建议准备本地文本文件，并通过 `--local-pg19-txt` 指定。
- 相对路径会按仓库根目录解析。例如 `--local-pg19-txt data/pg19_sample.txt` 会读取 `D:\workspace\transformer_proj\kvpress\data\pg19_sample.txt`。
- 输出 CSV 如果只传文件名，会自动写入脚本对应的 `results/.../` 目录。

## 运行 pythia_uncertainty_aware.py

用途：在 Pythia 上评测 uncertainty-aware press，以及基础 KVPress 方法的 PPL 和 throughput。

当前可用的 KVPress 方法只按下面四类使用：

| 方法名称 | 命令行参数名 |
| --- | --- |
| knorm | `knorm` |
| cur | `cur` |
| snap | `snapkv` |
| lag | `lagkv` |

脚本里还保留了其它历史选项，但当前不要在评测命令中使用它们。

示例：

```powershell
uv run python pythia_uncertainty_aware.py `
--presses none knorm uncertainty_head_var `
--uncertainty-base-press knorm `
--local-pg19-txt data/pg19_sample.txt `
--output-csv ctxt1024_tar256_knorm.csv
```
命令含义：评测 `none`、`knorm` 的基础表现，并且评测 `uncertainty_head_var`，其中 `knorm` 作为 `uncertainty_head_var` 内部计算的基础 press。PG19 使用本地文本，输出 CSV 到 `results/uncertainty_aware/ctxt1024_tar256_knorm.csv`。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `EleutherAI/pythia-70m` | Pythia 模型名。脚本会拒绝非 `EleutherAI/pythia-*` 模型。 |
| `--datasets` | `wikitext pg19` | 可选 `wikitext`、`pg19`。 |
| `--presses` | `none knorm uncertainty_head_var` | 要评测的 press 列表。当前基础方法只用 `knorm cur snapkv lagkv`。 |
| `--uncertainty-base-press` | `knorm` | `uncertainty_head_var` 内部使用的基础 press。 |
| `--split` | `test` | 数据集 split。 |
| `--num-samples` | `3` | 每个数据集采样窗口数。调试时可设为 `1`。 |
| `--context-tokens` | `1024` | 上下文 token 数。 |
| `--target-tokens` | `256` | 计算 PPL 的目标 token 数。 |
| `--max-new-tokens` | `64` | 计算 throughput 时生成的新 token 数。 |
| `--compression-ratio` | `0.5` | KV cache 压缩比例。 |
| `--uncertainty-weight` | `1.0` | uncertainty 分数权重。 |
| `--warmup` | `1` | throughput 计时前预热样本数。 |
| `--local-pg19-txt` | 空 | 本地 PG19 文本路径。 |
| `--output-csv` | `results/uncertainty_aware/pythia_results.csv` | 输出 CSV 路径。 |
| `--use-fast-tokenizer` | 关闭 | 使用 fast tokenizer。 |

默认输出：

```text
results/uncertainty_aware/pythia_results.csv
```

如果传入 `--output-csv my_run.csv`，实际输出为：

```text
results/uncertainty_aware/my_run.csv
```

## 使用 visualize_pythia_ppl.py

用途：读取评测 CSV，生成 PPL 对比柱状图 PNG。

默认读取 `results/uncertainty_aware/` 下所有 CSV，并在每个 CSV 旁边生成图：

```powershell
uv run python visualize_pythia_ppl.py
```

指定一个 CSV：

```powershell
uv run python visualize_pythia_ppl.py --csv results/uncertainty_aware/pythia_results.csv
```

读取 `pythia_kvpress_test.py` 的结果：

```powershell
uv run python visualize_pythia_ppl.py --results-dir pythia_kvpress_test
```

同时读取两个默认结果目录：

```powershell
uv run python visualize_pythia_ppl.py --results-dir both
```

指定输出目录或输出文件：

```powershell
uv run python visualize_pythia_ppl.py `
  --csv results/uncertainty_aware/pythia_results.csv `
  --output results/uncertainty_aware/pythia_results_ppl.png
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--results-dir` | `uncertainty_aware` | 可传 `uncertainty_aware`、`pythia_kvpress_test`、`both`、绝对路径、仓库相对路径，或 `results/` 下的目录名。 |
| `--csv` | 空 | 指定 CSV。为空时读取所选结果目录下所有 CSV。 |
| `--output` | 空 | 输出 PNG。单个 CSV 时作为文件路径；多个 CSV 时作为输出目录。 |
| `--linear` | 关闭 | 使用线性 y 轴。默认使用 log y 轴。 |

输入 CSV 至少需要包含以下列：

```text
dataset, press, ppl
```

默认图像输出名为：

```text
<csv_stem>_ppl_comparison.png
```

## 使用 pythia_kvpress_test.py

用途：在 Pythia 上对若干 KVPress 方法做基础 PPL 和 throughput 测试。

注意：这个脚本当前的 `--presses` 参数只支持：

```text
none, adakv, knorm, snapkv, streamingllm
```

如果严格遵守当前可用方法只有 `knorm`、`cur`、`snap`、`lag`，那么这个脚本里只能直接使用 `knorm` 和 `snapkv`；`cur`、`lagkv` 不在该脚本当前参数列表中。

最小示例：

```powershell
uv run python pythia_kvpress_test.py --datasets wikitext --presses none knorm snapkv
```

使用本地 PG19 文本：

```powershell
uv run python pythia_kvpress_test.py `
  --datasets wikitext pg19 `
  --presses none knorm snapkv `
  --local-pg19-txt data/pg19_sample.txt `
  --output-csv pythia_kvpress_test_results.csv
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `EleutherAI/pythia-70m` | Pythia 模型名。 |
| `--datasets` | `wikitext pg19` | 可选 `wikitext`、`pg19`。 |
| `--presses` | `none adakv knorm snapkv streamingllm` | 要评测的 press 列表。当前推荐只用 `none knorm snapkv`。 |
| `--split` | `test` | 数据集 split。 |
| `--num-samples` | `3` | 每个数据集采样窗口数。 |
| `--context-tokens` | `1024` | 上下文 token 数。 |
| `--target-tokens` | `256` | 计算 PPL 的目标 token 数。 |
| `--max-new-tokens` | `64` | 计算 throughput 时生成的新 token 数。 |
| `--compression-ratio` | `0.5` | KV cache 压缩比例。 |
| `--snapkv-window-size` | `64` | `snapkv` 的窗口大小。 |
| `--warmup` | `1` | throughput 计时前预热样本数。 |
| `--local-pg19-txt` | 空 | 本地 PG19 文本路径。 |
| `--output-csv` | `results/pythia_kvpress_test/pythia_kvpress_test_results.csv` | 输出 CSV 路径。 |
| `--use-fast-tokenizer` | 关闭 | 使用 fast tokenizer。 |

默认输出：

```text
results/pythia_kvpress_test/pythia_kvpress_test_results.csv
```

生成该脚本结果图：

```powershell
uv run python visualize_pythia_ppl.py --results-dir pythia_kvpress_test
```
