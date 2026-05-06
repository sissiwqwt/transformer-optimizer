# Uncertainty-Aware KV Press

## 1) Motivation

Most KV-cache compression methods assign one importance score per token and prune low-score tokens in a single pass.
This is efficient, but it assumes the score is reliable enough to decide deletion immediately.
In long-context QA/RAG/document settings, that assumption often fails: some tokens look unimportant now, but may become critical for future queries.

**Core idea:** in prefill, use both **importance** and **uncertainty**.
- High-confidence low-importance tokens: prune aggressively.
- Low-confidence tokens: keep (or defer pruning) to reduce irreversible mistakes.

---

## 2) High-level Method

Let a base scorer (any `ScorerPress`) provide token scores.
For token `t`, gather multiple score views/samples and compute:

- Mean importance: `mu_t`
- Uncertainty: `unc_t` (variance-like signal)

Then define adjusted score (one common option):

```text
score_adj_t = mu_t + lambda * unc_t
```

where `lambda >= 0` controls risk aversion.

Alternative policy: keep base score unchanged and use uncertainty to create a conservative threshold (e.g., higher keep probability for high-uncertainty tokens near the pruning boundary).

---

## 3) Where This Fits in KVPress

The design naturally wraps score-based pruning:
1. Use existing scorer to produce token scores.
2. Compute uncertainty per token.
3. Adjust score or threshold.
4. Run normal top-k pruning.

This keeps compatibility with current `ScorerPress` workflows and allows reusing existing scoring methods.

---

## 4) Variance / Uncertainty Definitions

Below are practical uncertainty definitions for token `t`.

## 4.1 Head-wise variance (cheap, recommended first)

If per-head scores are available (`s_{h,t}`):

```text
unc_t = Var_h(s_{h,t})
```

Interpretation: if heads disagree strongly, confidence is low.

**Pros:** no extra forward pass, very low overhead.

---

## 4.2 Local-window variance (context heterogeneity)

Using token neighborhood `W_t = [t-w, ..., t+w]`:

```text
unc_t = Var_{j in W_t}(s_j)
```

or residual-to-smoothed-score form:

```text
unc_t = (s_t - EMA_W(s_t))^2
```

Interpretation: highly unstable local score landscape implies uncertain token utility.

**Pros:** cheap and simple.

---

## 4.3 Monte-Carlo (dropout/noise) variance

Run `M` lightweight stochastic views and measure score variance:

```text
unc_t = Var_{m=1..M}(s_t^(m))
```

Interpretation: score sensitivity to perturbation indicates uncertainty.

**Pros:** strong uncertainty signal.
**Cons:** additional compute.

---

## 4.4 Multi-view scorer variance

Use multiple scoring views (e.g., key-norm, recency, attention proxy), normalized to a comparable scale:

```text
unc_t = Var_v(s_t^(v))
```

Interpretation: disagreement across scoring principles means uncertain decision.

**Pros:** interpretable and modular.

---

## 4.5 Rank instability (decision-aware)

Instead of score variance, track rank variability across views/samples:

```text
unc_t = Var(rank_t^(m))
```

Or boundary-membership uncertainty for top-k:

```text
p_t = P(t in TopK),
unc_t = p_t * (1 - p_t)
```

Interpretation: token frequently crossing the keep/drop boundary is uncertain.

**Pros:** directly aligned with pruning decisions.

---

## 4.6 Confidence interval width

Given repeated samples:

```text
SE_t = sigma_t / sqrt(M)
CI_width_t ~ 1.96 * SE_t
```

Larger interval => higher uncertainty.

---

## 5) Practical Combined Uncertainty

A robust implementation can blend multiple uncertainty sources:

```text
unc_t = alpha * U_head_t + beta * U_window_t + gamma * U_mc_t
```

with each `U_*` normalized to `[0, 1]`.

Then:

```text
score_adj_t = mu_t + lambda * unc_t
```

or use `unc_t` to inflate keep probability near threshold.

---

## 6) Guardrails & Engineering Notes

1. **Normalize scales** before combining views (z-score or min-max).
2. **Clip/saturate uncertainty** to avoid overly conservative behavior.
3. **Set a target compression floor** so uncertainty does not collapse compression rate.
4. **Use rank-based uncertainty** if absolute score scales are unstable.
5. **Start simple:** head-wise variance + rank instability often gives strong gains at low cost.

---

## 7) Minimal Implementation Plan

1. Build `UncertaintyAwarePress` as a wrapper around a base `ScorerPress`.
2. Reuse base scorer outputs to compute `mu_t`.
3. Add at least one uncertainty term (`U_head` first).
4. Produce `score_adj` and prune with existing top-k logic.
5. Benchmark quality/latency/memory trade-offs and tune `lambda`.
