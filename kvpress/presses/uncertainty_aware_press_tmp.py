# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class UncertaintyAwarePressTmp(ScorerPress):
    """
    Uncertainty-aware wrapper for score-based KV cache compression.

    This press uses a base ``ScorerPress`` to obtain per-head token importance
    scores. It then estimates token uncertainty from head-wise score variance
    and keeps tokens with high mean importance or high head disagreement:

        adjusted_score = mean_head_score + uncertainty_weight * var_head_score

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    press : ScorerPress, default=KnormPress()
        Base scorer used to compute per-head token importance scores.
        Query/key/value geometry and attention-based scorers with dense
        per-KV-head scores are the best fit for the head-variance adjustment.
    uncertainty_weight : float, default=1.0
        Risk-aversion coefficient applied to the normalized head-wise variance.
    normalize_scores : bool, default=True
        Whether to min-max normalize mean importance scores before combining.
    normalize_uncertainty : bool, default=True
        Whether to min-max normalize head-wise variance before combining.
    epsilon : float, default=1e-6
        Numerical stability constant used during min-max normalization.
    """

    compression_ratio: float = 0.0
    press: ScorerPress = field(default_factory=KnormPress)
    uncertainty_weight: float = 1.0
    normalize_scores: bool = True
    normalize_uncertainty: bool = True
    epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.press, ScorerPress), "UncertaintyAwarePress requires a ScorerPress as input"
        assert self.uncertainty_weight >= 0, "uncertainty_weight must be non-negative"
        assert self.epsilon > 0, "epsilon must be positive"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    def _minmax_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        min_scores = scores.amin(dim=-1, keepdim=True)
        max_scores = scores.amax(dim=-1, keepdim=True)
        return (scores - min_scores) / (max_scores - min_scores).clamp_min(self.epsilon)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        base_scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs).float()

        base_scores = self._minmax_normalize(base_scores)

        mean_scores = base_scores.mean(dim=1, keepdim=True)
        uncertainty = base_scores.var(dim=1, keepdim=True, unbiased=False)

        adjusted_scores = mean_scores + self.uncertainty_weight * uncertainty
        return adjusted_scores.expand_as(base_scores).contiguous()
