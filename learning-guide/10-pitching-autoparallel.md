# Pitching AutoParallel

How to give a compelling talk, presentation, or pitch about AutoParallel.

## The Elevator Pitch (30 seconds)

> "AutoParallel is a PyTorch library that automatically figures out how to distribute your model across GPUs. Instead of manually writing FSDP wrappers, tensor parallelism annotations, and collective communications — you give it your model and your GPU mesh, and it uses linear programming to find the optimal sharding strategy. It's like having a distributed systems expert optimize your training setup, but in seconds."

## The One-Slide Summary

```
┌────────────────────────────────────────────────────────┐
│                    AutoParallel                        │
│     Automatic Model Parallelism via Linear Programming │
│                                                        │
│  INPUT:  nn.Module + DeviceMesh                        │
│  OUTPUT: Optimally sharded parallel module             │
│                                                        │
│  ✓ FSDP + Tensor Parallelism + Data Parallelism        │
│  ✓ Globally optimal (ILP solver, not heuristics)       │
│  ✓ Zero manual parallelism code                        │
│  ✓ PyTorch-native (DTensor, FX, AOTAutograd)           │
│  ✓ Runs on meta device (no GPUs needed for planning)   │
│                                                        │
│  parallel_model = auto_parallel(model, mesh, inputs)   │
└────────────────────────────────────────────────────────┘
```

## Talk Structure (20-minute version)

### 1. The Problem (3 min)
- Training large models requires distributing across GPUs
- Manual parallelism is hard: FSDP? TP? Both? Which layers? Which dimensions?
- One wrong decision → 2-10x slower training
- Show a before/after code comparison (50+ lines of manual sharding → 1 line)

### 2. The Insight (3 min)
- Model sharding is an optimization problem
- Decision: for each op, which sharding strategy?
- Objective: minimize total training time
- Constraints: memory, divisibility, consistency
- This is an Integer Linear Program — a well-studied problem class

### 3. How It Works (5 min)
- Show the 4-phase pipeline diagram (trace → enumerate → optimize → apply)
- Emphasize: runs on meta device (no GPUs for planning)
- Show the ILP formulation (keep it visual, minimize math)
- Mention the cost models (FLOP counting, NCCL simulation)

### 4. Demo (5 min)
- Live demo with `example_hf.py --model gpt2 --mesh 8`
- Show the sharding log output
- Point out what the optimizer decided (FSDP here, TP there)
- If time: show `example_llama3.py` for a production model

### 5. Results & Status (2 min)
- Validated on LLaMA-3, DeepSeek-V3
- Matches or beats manual sharding
- Open source (BSD-3), experimental
- Active development at Meta

### 6. What's Next (2 min)
- Context parallelism support
- Automatic pipeline stage partitioning
- Broader op coverage
- Production hardening

## Key Talking Points

### For ML Engineers
- "You write your model normally. No `FullyShardedDataParallel`, no `ColwiseParallel`. One function call."
- "It finds combinations you wouldn't try manually — like FSDP on outer dims and TP on inner dims of a 3D mesh."
- "The optimization runs without GPUs. You can explore strategies on your laptop."

### For Researchers
- "When you change your model architecture, you don't need to re-derive the parallelism strategy."
- "It handles hybrid parallelism automatically — no manual tuning of TP/FSDP ratios."

### For Systems/Infra Engineers
- "Built on PyTorch's own distributed primitives (DTensor). No custom runtime."
- "The NCCL cost model simulates actual algorithm selection — Ring, Tree, NVLS, CollNet."
- "Prefetch overlap is modeled — FSDP all-gathers that overlap with compute are discounted."

### For Managers / Decision Makers
- "Reduces distributed training from a systems engineering problem to a model definition problem."
- "Engineers spend time on model innovation, not parallel code."
- "The optimizer guarantees global optimality (subject to cost model accuracy)."

## Common Questions & Answers

**Q: How does it compare to Megatron-LM?**
A: Megatron-LM provides fixed TP/PP/DP patterns that you configure manually. AutoParallel discovers the optimal combination automatically. Megatron is more battle-tested; AutoParallel is more flexible and automated.

**Q: Does it work with custom ops?**
A: If the op has no sharding rule, it falls back to replication. You can register custom rules for your ops, or use `local_map` for manual sharding.

**Q: How long does the optimization take?**
A: Seconds for typical models. The ILP solver is fast, and graph clustering reduces problem size for repeated structures (transformer layers).

**Q: Can I override its decisions?**
A: Yes. Use `with_sharding_constraint()` to fix specific tensors' placements. The optimizer respects these as hard constraints.

**Q: Is it production-ready?**
A: Experimental. Requires PyTorch nightly. Validated on LLaMA-3 and DeepSeek-V3 but not production-hardened.

**Q: Does it support inference?**
A: Yes, inference mode is supported (forward-only, no backward graph).

## Visual Aids

### Before/After Code Comparison

**Before (manual parallelism):**
```python
# 50+ lines of boilerplate
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

model = MyTransformer()
# Manually wrap each layer
for layer in model.layers:
    layer.attn.q_proj = ColwiseParallel(layer.attn.q_proj)
    layer.attn.k_proj = ColwiseParallel(layer.attn.k_proj)
    layer.attn.v_proj = ColwiseParallel(layer.attn.v_proj)
    layer.attn.o_proj = RowwiseParallel(layer.attn.o_proj)
    layer.ffn.gate_proj = ColwiseParallel(layer.ffn.gate_proj)
    layer.ffn.up_proj = ColwiseParallel(layer.ffn.up_proj)
    layer.ffn.down_proj = RowwiseParallel(layer.ffn.down_proj)
model = FSDP(model, ...)
```

**After (AutoParallel):**
```python
from autoparallel import auto_parallel
parallel_model = auto_parallel(model, mesh, sample_inputs)
```

### The Optimization as a Diagram

```
          input   param    bias    output
          ─────   ─────    ────    ──────
Layer 1:  [R]─────[S0]─────[R]─────[S1]
            │       │        │       │
Layer 2:  [R]─────[S0]─────[R]─────[S1]
            │       │        │       │
Layer N:  [R]─────[S0]─────[R]─────[S1]

  [R]  = Replicate     (full copy on every GPU)
  [S0] = Shard(0)      (split along dim 0)
  [S1] = Shard(1)      (split along dim 1)

  Every box is a placement decision.
  The ILP solver picks the optimal [R], [S0], or [S1]
  for every tensor across all layers simultaneously.
```

## Demo Script

```bash
# 1. Show the simplest usage
echo "=== AutoParallel on GPT-2 ==="
python examples/example_hf.py --model gpt2 --mesh 8

# 2. Show a production model
echo "=== AutoParallel on LLaMA-3 ==="
python examples/example_llama3.py

# 3. Show the core demo with mixed precision
echo "=== Full API Demo ==="
python examples/example_autoparallel.py
```
