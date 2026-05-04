from transformers import pipeline
from kvpress import KnormPress  # 训练免费方法之一
from kvpress import ThinKPress

def main():
    model = "EleutherAI/pythia-70m"
    pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        device_map="auto",
        dtype="auto",
    )

    context = (
        "The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled 'Computing Machinery and Intelligence' which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.\n"
    )
    question = "Please summarize the main points and provide 3 conclusions."

    # compression_ratio 越小，压缩越强（通常速度/显存更省，但可能影响质量）
    press = ThinKPress(compression_ratio=0.5)

    out = pipe(context, question=question, press=press)
    print(out["answer"])

if __name__ == "__main__":
    main()