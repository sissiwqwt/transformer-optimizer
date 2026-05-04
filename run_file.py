# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import DynamicCache, pipeline

from kvpress import DecodingPress, KnormPress


def main():
    model = "EleutherAI/pythia-70m"
    pipe = pipeline(
        "kv-press-text-generation",
        model=model,
        device_map="auto",
        dtype="auto",
    )

    context = (
        "Natural language processing began as a field in the 1950s, with early work on machine "
        "translation and symbolic approaches. Statistical methods became prominent in the late "
        "1980s and 1990s, and neural network methods later reshaped the field. Transformer models "
        "now support many language tasks, but long-context generation can use a large KV cache."
    )
    question = "Summarize the context in two concise sentences."

    press = DecodingPress(
        base_press=KnormPress(),
        compression_interval=4,
        target_size=32,
        hidden_states_buffer_size=0,
    )

    output = pipe(
        context,
        question=question,
        press=press,
        cache=DynamicCache(),
        max_new_tokens=40,
    )
    print(output["answer"])


if __name__ == "__main__":
    main()
