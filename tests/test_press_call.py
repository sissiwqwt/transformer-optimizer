# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from transformers import DynamicCache
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM

from kvpress import KnormPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_context_manager_adds_and_removes_hook(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.2)

    with press(unit_test_model):
        for layer in unit_test_model.model.layers:
            assert len(layer.self_attn._forward_hooks) == 1

    for layer in unit_test_model.model.layers:
        assert len(layer._forward_hooks) == 0


def test_context_manager_applies_compression(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.2)

    with press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"].to(unit_test_model.device)
        past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    seq_len = input_ids.shape[-1]

    for layer in past_key_values.layers:
        assert layer.keys.shape[2] == int(seq_len * 0.8) == past_key_values.get_seq_length()
        assert layer.values.shape[2] == int(seq_len * 0.8) == past_key_values.get_seq_length()

    input_ids = unit_test_model.dummy_inputs["input_ids"].to(unit_test_model.device)
    past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    for layer in past_key_values.layers:
        assert layer.keys.shape[2] == seq_len == past_key_values.get_seq_length()
        assert layer.values.shape[2] == seq_len == past_key_values.get_seq_length()


def test_context_manager_supports_gpt_neox():
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=128,
        )
    ).eval()
    press = KnormPress(compression_ratio=0.2)

    with press(model):
        for layer in model.gpt_neox.layers:
            assert len(layer.attention._forward_hooks) == 1

        input_ids = model.dummy_inputs["input_ids"]
        past_key_values = model(input_ids, past_key_values=DynamicCache()).past_key_values

    seq_len = input_ids.shape[-1]

    for layer in model.gpt_neox.layers:
        assert len(layer.attention._forward_hooks) == 0
    for layer in past_key_values.layers:
        assert layer.keys.shape[2] == int(seq_len * 0.8) == past_key_values.get_seq_length()
        assert layer.values.shape[2] == int(seq_len * 0.8) == past_key_values.get_seq_length()
