# AutoParallel — Quick Summary

A PyTorch library that **automatically shards models for distributed training** using Integer Linear Programming. You give it a model and a GPU mesh; it finds the optimal parallelism strategy.

```python
from autoparallel import auto_parallel
parallel_model = auto_parallel(model, mesh, sample_inputs)
```

## How It Works (4 phases)

```
nn.Module ──► FX Graph ──► Sharding Options ──► ILP Solve ──► Parallel Module
              (trace)       (enumerate)         (optimize)     (apply)
```

1. **Trace**: Convert model to a joint forward+backward FX graph using TorchDynamo + AOTAutograd. Runs on meta device (no GPU memory).
2. **Enumerate**: For each graph node, list all valid sharding strategies with compute + communication costs.
3. **Optimize**: Formulate as ILP (binary variables, linear constraints). PuLP/CBC solver finds the global optimum in seconds.
4. **Apply**: Insert collectives (all-gather, reduce-scatter, etc.), compile with Inductor, build final `nn.Module` with DTensor parameters.

## Supported Parallelism

| Strategy | Status | How |
|----------|--------|-----|
| FSDP | Fully supported | Params get `Shard(0)`, all-gather before compute, reduce-scatter grads |
| Tensor Parallelism | Fully supported | Column/row parallel via `mm` decomposition |
| Data Parallelism | Fully supported | Batch sharded, params replicated |
| Hybrid (FSDP+TP) | Fully supported | Multi-dim mesh, ILP finds optimal mix |
| Pipeline Parallelism | Partial | Graph splitting exists, but manual stage assignment |
| Expert Parallelism | Manual only | Requires `local_map` wrappers |
| Context Parallelism | Disabled | Upstream PyTorch bug |
| Sequence Parallelism | Not implemented | No Ring Attention logic |

## Quick Start

```bash
# Simplest example (works with fake process group, no multi-GPU needed)
python examples/example_hf.py --model gpt2 --mesh 8

# Full demo with mixed precision + activation checkpointing
python examples/example_autoparallel.py

# Production model
python examples/example_llama3.py

# Run tests (needs CUDA)
pytest tests/
```

## Key Strengths

- **Globally optimal** — ILP solver, not heuristics
- **Zero manual sharding code** — no FSDP wrappers, no TP annotations
- **Runs on meta device** — optimize sharding for a 70B model on a laptop
- **PyTorch-native** — DTensor, FX, AOTAutograd, Inductor
- **Accurate cost models** — NCCL algorithm simulation, FLOP counting, memory bandwidth

## Key Limitations

- Requires **PyTorch nightly** (>= 2.10), not stable releases
- **Experimental** — API is unstable, not production-hardened
- **Static graphs only** — no dynamic control flow or dynamic shapes
- **~30 ops** have custom sharding rules; others fall back to all-Replicate
- **GPU-only** — hardcoded specs for NVIDIA (H100, B200, A100, etc.) and AMD (MI200X–MI355X); no TPU/CPU

## Architecture at a Glance

| File | Role |
|------|------|
| `api.py` | Main entry point, orchestrates the 4 phases |
| `optimize_sharding.py` | ILP formulation and solver |
| `apply_sharding.py` | Inserts collectives into graph |
| `shardings/propagation_rules.py` | Custom sharding rules for ~30 ops |
| `cost_models/compute_estimation.py` | FLOP + memory bandwidth cost |
| `cost_models/nccl_cost_model.py` | NCCL algorithm simulation |
| `module_construction.py` | Builds final nn.Module |
| `tracing.py` | FX tracing + decomposition table |

## The Elevator Pitch

> "AutoParallel replaces manual distributed training code with a single function call. It uses linear programming to find the optimal mix of FSDP, tensor parallelism, and data parallelism — automatically. Write your model normally; AutoParallel handles the rest."

## Competitive Context

| Tool | Difference from AutoParallel |
|------|------------------------------|
| Manual FSDP/TP | AutoParallel automates what you'd do by hand |
| Megatron-LM | Fixed patterns, manually configured; AutoParallel discovers strategies |
| DeepSpeed | Memory optimization (ZeRO); different approach |
| Alpa | Similar ILP idea but for JAX, not PyTorch |
| GSPMD | Similar idea but XLA-specific (Google) |

## Deep Dive

For the full guide, read the chapters in order:

| # | Topic |
|---|-------|
| [00](00-overview.md) | What and why |
| [01](01-prerequisites.md) | Prerequisites (PyTorch, DTensor, FX, LP) |
| [02](02-architecture.md) | Code structure and class relationships |
| [03](03-parallelism-explained.md) | All parallelism strategies explained |
| [04](04-optimization-flow.md) | The 4-phase pipeline in detail |
| [05](05-getting-started.md) | Setup, install, first run |
| [06](06-testing.md) | Test suite and CI/CD |
| [07](07-examples-walkthrough.md) | Every example explained |
| [08](08-pros-and-cons.md) | Honest pros, cons, and known gaps |
| [09](09-real-world-usage.md) | Where to use it, who benefits |
| [10](10-pitching-autoparallel.md) | How to give a talk about it |
