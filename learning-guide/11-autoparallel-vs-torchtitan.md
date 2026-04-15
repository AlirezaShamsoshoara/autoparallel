# AutoParallel vs TorchTitan

## The Short Answer

They are **not competitors**. They operate at different layers of the stack and are designed to work together:

- **AutoParallel** = the **parallelization compiler** — "How should I shard this model?"
- **TorchTitan** = the **training framework** — "How do I train this model?"

AutoParallel produces a sharded `nn.Module`. TorchTitan takes that module and runs the full training loop. AutoParallel is a compiler pass that runs **before** training begins; TorchTitan is the runtime that drives training **after** the model is sharded.

## The Analogy

Think of it like building a car:
- **AutoParallel** is the engineer who designs the optimal engine layout — where each cylinder goes, how the exhaust routes, what the compression ratio should be
- **TorchTitan** is the factory — it takes that engine design and builds the car: adds the chassis, wheels, fuel system, dashboard, and drives it off the line

You wouldn't ask the engine designer to also build the steering wheel. You wouldn't ask the factory to redesign the engine layout.

## What Each Project Handles

```
┌──────────────────────────────────────────────────────────┐
│                      TorchTitan                          │
│            (Training Framework / Runtime)                 │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              AutoParallel                           │ │
│  │         (Parallelization Compiler)                  │ │
│  │                                                     │ │
│  │  • FX graph tracing                                 │ │
│  │  • Sharding strategy enumeration                    │ │
│  │  • ILP optimization                                 │ │
│  │  • Collective insertion                             │ │
│  │  • Activation checkpointing                         │ │
│  │  • Mixed precision graph transforms                 │ │
│  │  • torch.compile / Inductor                         │ │
│  │                                                     │ │
│  │  OUTPUT: sharded nn.Module                          │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│                         ▼                                │
│  • Distributed launching (torchrun)                      │
│  • Training loop (forward → loss → backward → step)      │
│  • Optimizer (Adam, SGD, etc.)                           │
│  • Learning rate scheduling                              │
│  • Data loading & batching                               │
│  • Checkpointing (save/resume)                           │
│  • Gradient clipping                                     │
│  • Metrics & logging                                     │
│  • Fault tolerance                                       │
│  • Configuration management                              │
└──────────────────────────────────────────────────────────┘
```

## Side-by-Side Comparison

| Concern | AutoParallel | TorchTitan |
|---------|-------------|------------|
| Model sharding strategy | ILP solver finds optimal FSDP/TP/DP mix | Manual: user configures `parallelize_fn` |
| Collective insertion | Automatic (all-gather, reduce-scatter, etc.) | Manual via DTensor/FSDP wrappers |
| Training loop | None — examples do `out.backward()` ad hoc | Full loop with step counting |
| Optimizer | None — examples create `Adam` manually | Creates and manages optimizers |
| Loss function | None — examples use `out.sum()` | Proper loss computation |
| LR scheduler | Not handled | Warmup, decay, cosine, etc. |
| Data loading | None — examples use random tensors | DataLoader, dataset sharding |
| Checkpointing | Not handled | Save/load/resume training |
| Distributed launch | Fake process groups for simulation | `torchrun`, elastic launching |
| Gradient clipping | Not handled | Configurable gradient clipping |
| Metrics & logging | Internal timing only | WandB, TensorBoard, etc. |
| Fault tolerance | Not handled | TorchFT integration |
| Configuration | Python API | TOML config files |

## How They Actually Integrate

TorchTitan has a **module/plugin system**. AutoParallel plugs in as an alternative parallelization strategy:

### Without AutoParallel (TorchTitan default)

TorchTitan manually applies FSDP + TP with hand-written `parallelize_fn`:

```python
# Inside TorchTitan's parallelize_llama()
# Manual: wrap each layer explicitly
for layer in model.layers:
    layer.attn.q_proj = ColwiseParallel(layer.attn.q_proj)
    layer.attn.k_proj = ColwiseParallel(layer.attn.k_proj)
    # ... 20+ more lines of manual TP annotations
model = FSDP(model, ...)
```

### With AutoParallel (TorchTitan + AutoParallel)

AutoParallel replaces the manual `parallelize_fn` with automated optimization:

```python
# Inside TorchTitan's experiment module for AutoParallel
def parallelize_fn(model, mesh, ...):
    return auto_parallel(model, mesh, sample_inputs)
    # That's it. ILP finds the optimal strategy.
```

### The Integration Command

The CI test shows exactly how TorchTitan invokes AutoParallel:

```bash
# TorchTitan's training script, but with AutoParallel as the parallelization module
NGPU=4 ./run_train.sh \
  --module autoparallel.llama3 \                    # Use AutoParallel's strategy
  --config autoparallel_llama3_debugmodel \          # AutoParallel-specific config
  --parallelism.tensor_parallel_degree 4
```

- `--module autoparallel.llama3` tells TorchTitan to load an experiment module that calls `autoparallel.auto_parallel()` instead of TorchTitan's default manual sharding
- TorchTitan still handles everything else: launching, training loop, optimizer, checkpointing

## Where Each Project's Code Lives

| Component | Lives In |
|-----------|----------|
| Sharding engine (`api.py`, `optimize_sharding.py`, `apply_sharding.py`) | AutoParallel repo |
| Cost models, propagation rules | AutoParallel repo |
| Test models (LLaMA-3, DeepSeek-V3) | AutoParallel repo |
| CI integration test workflow | AutoParallel repo |
| TorchTitan experiment modules (`autoparallel.llama3`) | TorchTitan repo |
| TOML configs (`autoparallel_llama3_debugmodel`) | TorchTitan repo |
| `parallelize_fn` glue code | TorchTitan repo |
| Training loop, optimizer, data loading | TorchTitan repo |
| `run_train.sh` entry point | TorchTitan repo |

## Do They Help Each Other?

**Yes, they are complementary.**

### How AutoParallel helps TorchTitan:
1. **Eliminates manual sharding code** — no more per-model `parallelize_fn` with dozens of TP annotations
2. **Finds better strategies** — the ILP solver may discover hybrid strategies that manual configuration misses
3. **Reduces onboarding cost** — new models don't need a distributed systems expert to write their parallel plan
4. **Enables experimentation** — easily test different mesh shapes, compare 1D FSDP vs 2D FSDP+TP

### How TorchTitan helps AutoParallel:
1. **Provides the training runtime** — AutoParallel only produces a model; TorchTitan trains it
2. **Handles production concerns** — checkpointing, fault tolerance, logging, data loading
3. **Provides benchmarking infrastructure** — the MAST sweep scripts compare AutoParallel vs baseline
4. **Gives real-world validation** — TorchTitan integration tests prove AutoParallel works end-to-end

## Benchmark Comparison: AutoParallel vs Manual Sharding

The MAST sweep configuration (`mast/sweep.py`) benchmarks these configurations head-to-head:

| Config | Strategy | Description |
|--------|----------|-------------|
| `llama3_FSDP_compile` | Manual | TorchTitan's baseline: hand-written FSDP + compile |
| `llama3_autop_1d_compile` | AutoParallel | 1D mesh, ILP-optimized sharding |
| `llama3_autop_1d_compile_aten_bucket_reorder` | AutoParallel + bucketing | 1D mesh + collective bucketing at aten level |
| `llama3_autop_2d_compile` | AutoParallel | 2D mesh, FSDP+TP discovered by ILP |

This lets you directly compare AutoParallel's automated decisions against TorchTitan's manual expert decisions on the same model, same hardware, same training loop.

## When to Use Which

### Use TorchTitan alone (manual sharding) when:
- Your model's parallelism strategy is well-understood and stable
- You need production-grade stability (AutoParallel is experimental)
- Your model uses ops without AutoParallel sharding rules
- You're on a stable PyTorch release (AutoParallel needs nightly)

### Use TorchTitan + AutoParallel when:
- You're prototyping a new model and want to quickly find the right parallelism
- You want to explore whether your manual strategy is suboptimal
- You're scaling to a new mesh topology and need to re-derive strategies
- You want to compare automated vs manual strategies with real benchmarks

### Use AutoParallel standalone (without TorchTitan) when:
- You have your own training loop and just need the sharding step
- You're building a custom training framework
- You want to analyze sharding strategies without actually training (meta device)

## The Bigger Picture

```
                    PyTorch Ecosystem
                    ─────────────────

    ┌─────────────────────────────────────────────┐
    │              User's Model                    │
    │           (nn.Module on meta)                │
    └──────────────────┬──────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    ┌──────────────┐      ┌───────────────┐
    │ AutoParallel │      │ Manual FSDP/TP│
    │  (ILP-based) │      │  (by hand)    │
    └──────┬───────┘      └───────┬───────┘
           │                      │
           └──────────┬───────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Sharded      │
              │  nn.Module    │
              │  (DTensor)    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  TorchTitan   │
              │  (or custom   │
              │   training    │
              │   loop)       │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  Trained      │
              │  Model        │
              └───────────────┘
```

AutoParallel and TorchTitan aren't two ways to do the same thing — they're two halves of the same workflow. AutoParallel decides *what to shard*; TorchTitan decides *how to train*.
