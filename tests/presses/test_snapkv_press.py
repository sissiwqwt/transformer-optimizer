# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from kvpress.presses.snapkv_press import SnapKVPress


def test_compute_window_attention_supports_gpt_neox_without_gqa_config():
    config = GPTNeoXConfig(
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_hidden_layers=1,
    )
    module = GPTNeoXAttention(config, layer_idx=0)
    module.head_dim = module.head_size

    bsz, seq_len, window_size = 2, 8, 3
    hidden_states = torch.randn(bsz, seq_len, config.hidden_size)
    keys = torch.randn(bsz, config.num_attention_heads, seq_len, module.head_dim)
    rotary_dim = module.rotary_ndims
    position_embeddings = (
        torch.ones(bsz, seq_len, rotary_dim),
        torch.zeros(bsz, seq_len, rotary_dim),
    )

    attn_weights = SnapKVPress.compute_window_attention(
        module,
        hidden_states,
        keys,
        window_size,
        position_embeddings,
    )

    assert not hasattr(config, "num_key_value_heads")
    assert attn_weights.shape == (
        bsz,
        config.num_attention_heads,
        window_size,
        seq_len - window_size,
    )
