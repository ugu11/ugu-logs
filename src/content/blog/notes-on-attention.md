---
title: "Quick notes: why attention scales the way it does"
description: "A short refresher on the compute and memory cost of self-attention, and why it matters for long-context models."
date: 2026-08-20
tags: ["transformers", "notes"]
---

A few notes I keep coming back to whenever long-context models come up.

## The core cost

Self-attention computes a score between every pair of tokens in a sequence. For a
sequence of length `n`, that's `O(n^2)` pairs, and each one costs `O(d)` work for a
head dimension `d`. So the compute cost of a single attention layer is roughly:

```
O(n^2 * d)
```

The memory cost follows the same shape: the attention matrix itself is `n x n`,
which is why doubling context length quadruples the memory needed to hold it.

## Why this matters in practice

- **Training cost** grows quickly as context windows grow — this is the main
  reason long-context training runs are expensive.
- **KV-cache size at inference time** grows linearly with sequence length per
  token generated, which is a separate (and often more painful) bottleneck for
  serving.
- Techniques like sliding-window attention, linear attention, and state-space
  models (Mamba, RWKV) all exist to push against this `O(n^2)` wall in different
  ways.

## Takeaway

When evaluating "long context" claims, it's worth asking whether the
underlying mechanism actually changed the complexity class, or whether it's
just a bigger GPU and a longer training run.
