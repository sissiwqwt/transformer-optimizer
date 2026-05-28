# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class AnchorDedupPress(ScorerPress):
    """
    Anchor-and-Dedup KV cache compression.

    This press combines a simple key-norm base score with two document-oriented
    signals:

    - positional anchors, which receive a large keep bonus;
    - local key redundancy, which penalizes non-anchor tokens that are similar
      to nearby keys.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_sink : int, default=4
        Number of initial tokens to mark as anchors.
    chunk_size : int, default=128
        Marks the first token of each chunk as an anchor. Set to 0 to disable.
    window_size : int, default=8
        Local radius used to compute max cosine redundancy.
    anchor_bonus : float, default=8.0
        Score bonus added to anchor tokens.
    dedup_strength : float, default=1.0
        Multiplier for the local redundancy penalty.
    max_anchor_ratio : float, default=0.25
        Maximum fraction of sequence positions that may be anchors.
    normalize_scores : bool, default=True
        Normalize base scores and redundancy scores per batch/head before combining.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4
    chunk_size: int = 128
    window_size: int = 8
    anchor_bonus: float = 8.0
    dedup_strength: float = 1.0
    max_anchor_ratio: float = 0.25
    normalize_scores: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.n_sink >= 0, "n_sink must be non-negative"
        assert self.chunk_size >= 0, "chunk_size must be non-negative"
        assert self.window_size >= 0, "window_size must be non-negative"
        assert 0 <= self.max_anchor_ratio <= 1, "max_anchor_ratio must be between 0 and 1"

    def _anchor_mask(self, k_len: int, device: torch.device) -> torch.Tensor:
        anchor_mask = torch.zeros(k_len, dtype=torch.bool, device=device)

        if self.n_sink:
            anchor_mask[: min(self.n_sink, k_len)] = True

        if self.chunk_size:
            anchor_mask[:: self.chunk_size] = True

        max_anchors = int(k_len * self.max_anchor_ratio)
        if self.n_sink > 0 and k_len > 0:
            max_anchors = max(1, max_anchors)

        if max_anchors < int(anchor_mask.sum()):
            anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).flatten()
            anchor_mask[:] = False
            anchor_mask[anchor_indices[:max_anchors]] = True

        return anchor_mask

    @staticmethod
    def _normalize(scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.mean(dim=-1, keepdim=True)
        scale = centered.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return centered / scale

    def _local_redundancy(self, keys: torch.Tensor) -> torch.Tensor:
        if self.window_size == 0 or keys.shape[2] <= 1:
            return torch.zeros_like(keys[..., 0])

        normalized_keys = F.normalize(keys.float(), p=2, dim=-1)
        redundancy = torch.full_like(normalized_keys[..., 0], -1.0)
        max_offset = min(self.window_size, keys.shape[2] - 1)

        for offset in range(1, max_offset + 1):
            similarity = (normalized_keys[:, :, :-offset] * normalized_keys[:, :, offset:]).sum(dim=-1)
            redundancy[:, :, :-offset] = torch.maximum(redundancy[:, :, :-offset], similarity)
            redundancy[:, :, offset:] = torch.maximum(redundancy[:, :, offset:], similarity)

        return redundancy.clamp_min(0).to(keys.dtype)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        base_scores = -keys.norm(dim=-1)
        redundancy = self._local_redundancy(keys)

        if self.normalize_scores:
            base_scores = self._normalize(base_scores)
            redundancy = self._normalize(redundancy)

        anchor_mask = self._anchor_mask(keys.shape[2], keys.device).view(1, 1, -1)
        scores = base_scores - self.dedup_strength * redundancy.masked_fill(anchor_mask, 0)
        scores = scores + self.anchor_bonus * anchor_mask.to(scores.dtype)

        return scores
