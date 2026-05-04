import time
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# KVPress
from kvpress import KnormPress, SnapKVPress

MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# -------- 1) 数据准备 --------
def get_wikitext_samples(n=50, split="test"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [x["text"] for x in ds if x["text"] and len(x["text"].strip()) > 0]
    return texts[:n]

def get_pg19_sample(split="test", local_pg19_txt=None):
    """
    优先尝试Hub可直接加载的数据集；
    如果环境不支持，再回退到本地txt文件。
    """

    # 方案A：先尝试常见可直接加载的PG19镜像（不走旧script）
    candidate_loaders = [
        # 你可以按自己环境增减
        ("emozilla/pg19", None),
        ("pg19", None),  # 旧写法，可能失败
    ]

    for ds_name, ds_config in candidate_loaders:
        try:
            if ds_config is None:
                ds = load_dataset(ds_name, split=split, streaming=True)
            else:
                ds = load_dataset(ds_name, ds_config, split=split, streaming=True)
            for ex in ds:
                txt = ex.get("text", "")
                if txt and txt.strip():
                    return txt
        except Exception as e:
            print(f"[WARN] load_dataset({ds_name}) failed: {e}")

    # 方案B：本地文件回退（最稳）
    if local_pg19_txt is not None:
        with open(local_pg19_txt, "r", encoding="utf-8") as f:
            txt = f.read()
        if txt.strip():
            return txt

    raise RuntimeError(
        "PG19加载失败：请传入 local_pg19_txt，或换成可直接加载的PG19镜像数据集。"
    )

# -------- 2) PPL 评测 --------
@torch.no_grad()
def evaluate_ppl(model, tokenizer, texts, max_length=2048, chunk_length=512):
    model.eval()
    nll_sum = 0.0
    token_count = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.shape[1] < 2:
            continue

        # Score in overlapping chunks to avoid materializing
        # (batch, max_length, vocab_size) logits on CPU.
        seq_len = input_ids.shape[1]
        for start in range(0, seq_len - 1, chunk_length):
            end = min(start + chunk_length + 1, seq_len)
            chunk_ids = input_ids[:, start:end]
            if chunk_ids.shape[1] < 2:
                continue

            out = model(input_ids=chunk_ids, labels=chunk_ids)
            n_tokens = chunk_ids.shape[1] - 1
            nll_sum += out.loss.item() * n_tokens
            token_count += n_tokens

    if token_count == 0:
        return float("nan")

    avg_nll = nll_sum / token_count
    ppl = math.exp(avg_nll)
    return ppl

# -------- 3) 速度评测（生成）--------
def evaluate_speed(gen_pipe, context, question, press, max_new_tokens=64, warmup=1, runs=3):
    # warmup
    for _ in range(warmup):
        _ = gen_pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)["answer"]

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = gen_pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)["answer"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_t = sum(times) / len(times)
    # 粗略吞吐：仅按生成 token 统计（因为 max_new_tokens 固定）
    toks_per_sec = max_new_tokens / avg_t
    return {
        "avg_time_s": avg_t,
        "tokens_per_sec": toks_per_sec,
        "example_answer": out[:200],
    }

def build_press(name, ratio=0.5):
    if name == "none":
        return None
    if name == "knorm":
        return KnormPress(compression_ratio=ratio)
    if name == "snapkv":
        return SnapKVPress(compression_ratio=ratio)
    raise ValueError(name)

def main():
    print(f"Model: {MODEL_NAME}, device={DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(DEVICE)

    # KVPress 自定义 pipeline（用于速度/端到端）
    gen_pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 数据
    wiki_texts = get_wikitext_samples(n=50)
    pg19_text = get_pg19_sample(
    split="test",
    local_pg19_txt="data/pg19_sample.txt",  # 你本地准备一个长文本
)

    # pg19 用单 sample 做速度和单样本 ppl
    context_pg19 = pg19_text[:20000]  # 防止过长 OOM，可按显存调大
    question = "Summarize the main storyline in 5 bullet points."

    presses = ["none", "knorm", "snapkv"]
    
    for p in presses:
        press = build_press(p, ratio=0.5)
        print(f"\n==== Press: {p} ====")

        # PPL：wikitext（多样本）
        ppl_wiki = evaluate_ppl(model, tokenizer, wiki_texts, max_length=1024)
        print(f"[PPL] wikitext-2: {ppl_wiki:.4f}")

        # PPL：pg19（单样本）
        ppl_pg19 = evaluate_ppl(model, tokenizer, [context_pg19], max_length=2048)
        print(f"[PPL] pg19(single sample): {ppl_pg19:.4f}")

        # 速度：pg19 单样本长上下文
        spd = evaluate_speed(
            gen_pipe=gen_pipe,
            context=context_pg19,
            question=question,
            press=press,
            max_new_tokens=64,
            warmup=1,
            runs=3,
        )
        print(f"[Speed] avg_time_s={spd['avg_time_s']:.4f}, tokens/s={spd['tokens_per_sec']:.2f}")
        print(f"[Answer head] {spd['example_answer']}")

if __name__ == "__main__":
    main()
