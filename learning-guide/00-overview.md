# AutoParallel: Overview

## What is AutoParallel?

AutoParallel is a PyTorch library that **automatically shards and parallelizes models for distributed training**. Instead of manually writing distributed code (FSDP wrappers, tensor parallelism annotations, collective communications), you hand AutoParallel your model and a device mesh — and it figures out the optimal parallelism strategy using **linear programming**.

```python
from autoparallel import auto_parallel

# That's it. No manual sharding annotations.
parallel_model = auto_parallel(model, mesh, sample_inputs)
```

## Why Does This Matter?

Training large models (LLMs, vision transformers, MoE architectures) requires distributing computation across many GPUs. Today, engineers must manually decide:

- Which layers to shard and how (FSDP? Tensor parallelism? Both?)
- Which dimensions to split on (batch? heads? hidden?)
- Where to insert communication collectives (all-gather, reduce-scatter, all-to-all)
- How to balance memory vs. communication vs. compute

This is **tedious, error-prone, and requires deep expertise** in distributed systems. A single wrong sharding decision can 2-10x your training time. AutoParallel replaces this manual process with an automated optimizer.

## The Core Insight

Sharding a neural network is fundamentally an **optimization problem**:

- **Decision variables**: For each tensor operation, which sharding strategy to use
- **Objective**: Minimize total training time (compute + communication)
- **Constraints**: Memory limits, tensor divisibility, flow consistency (producer and consumer must agree on placements)

AutoParallel formulates this as an **Integer Linear Program (ILP)** and solves it with the PuLP/CBC solver. This is the same class of optimization used in logistics, scheduling, and chip design — proven to find globally optimal solutions.

## Who Made It?

AutoParallel is an open-source project from **Meta** (BSD-3 license). It builds on PyTorch's DTensor (Distributed Tensor) infrastructure and is designed to integrate with PyTorch's existing distributed training ecosystem.

## Project Status

**Experimental / early development**. The API is unstable and requires PyTorch nightly (>= 2.10). It is under active development with known gaps (covered in later chapters). But the core optimization loop works and has been validated on production architectures like LLaMA-3 and DeepSeek-V3.

## What You'll Learn in This Guide

| Chapter | Topic |
|---------|-------|
| 01 | Prerequisites — what you need to know first |
| 02 | Architecture — how the code is organized |
| 03 | Parallelism strategies — what's supported and how |
| 04 | The optimization pipeline — from model to sharded graph |
| 05 | Getting started — setup and first run |
| 06 | Testing — how to validate and test |
| 07 | Examples walkthrough — real code, explained |
| 08 | Pros, cons, and gaps — honest assessment |
| 09 | Real-world usage — where and when to use it |
| 10 | Pitching AutoParallel — how to give a talk about it |
