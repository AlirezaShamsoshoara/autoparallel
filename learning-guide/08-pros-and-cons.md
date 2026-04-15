# Pros, Cons, and Gaps

## Pros

### 1. Globally Optimal Sharding
Unlike manual sharding where engineers make local decisions per layer, AutoParallel solves for the **global optimum** across the entire model using ILP. This catches cross-layer interactions that humans miss.

### 2. Zero Manual Parallelism Code
No `FullyShardedDataParallel()` wrappers, no column/row parallel annotations, no collective insertions. The user writes a standard `nn.Module` and gets a parallel version.

### 3. Automatic Hybrid Parallelism Discovery
The optimizer naturally discovers the best mix of FSDP, TP, and DP. On a 2D mesh, it might choose FSDP on dimension 0 for some layers and TP on dimension 1 for others — combinations that would take significant manual experimentation.

### 4. Accurate Cost Modeling
The NCCL cost model (50+ tests) simulates 6 algorithms and 3 protocols with empirical corrections from H100/B200 benchmarks. The compute model uses FLOP counting with memory-bandwidth awareness. This is more principled than heuristic-based approaches.

### 5. Prefetch Overlap Modeling
FSDP all-gathers and gradient reduce-scatters are discounted in the cost model when they can overlap with compute, preventing the optimizer from avoiding FSDP due to naively-counted communication costs.

### 6. Meta Device / Fake Tensors
The entire optimization runs without GPU memory. Models are created on `meta` device, traced with fake tensors. You can optimize sharding for a 70B model on a laptop.

### 7. Model Identity Preservation
The parallel module preserves `isinstance` checks, user methods, properties, classmethods, weight tying, and buffer aliases. It behaves like the original module.

### 8. PyTorch-Native
Built entirely on PyTorch primitives: DTensor, FX, AOTAutograd, TorchDynamo. No custom CUDA kernels, no separate runtime. Compatible with PyTorch's ecosystem (checkpointing, profiling, Inductor).

### 9. Activation Checkpointing Integration
Automatic AC tagging with staged recomputation. The sqrt(memory) heuristic bounds peak memory while minimizing recomputation.

### 10. Debugging Tools
`explain_placement()`, `diff_solutions()`, `print_costs_for_node()` — tools to understand and debug the optimizer's decisions.

## Cons

### 1. PyTorch Nightly Required
Requires PyTorch >= 2.10 nightly. Not compatible with stable PyTorch releases. This is a significant adoption barrier.

### 2. Experimental / Unstable API
The project is explicitly "highly under development." APIs can change between versions. Not recommended for production workloads without careful version pinning.

### 3. Static Graphs Only
Uses FX tracing, which captures a single execution path. Dynamic control flow (data-dependent branching, variable-length sequences) is not supported. The `make_fx` usage is flagged as "suspicious in case of dynamic shapes."

### 4. GPU-Only (NVIDIA + AMD)
Requires NVIDIA CUDA or AMD ROCm GPUs. The cost model has specs for both NVIDIA (H100, B200, A100, etc.) and AMD (MI200X, MI300X, MI325X, MI350X, MI355X). No support for CPU-only training, TPUs, or other accelerators.

### 5. Limited Op Coverage
Only ~30 ops have custom sharding rules. Many ops silently fall back to all-Replicate, losing parallelism opportunities. If your model uses uncommon ops, you may get suboptimal sharding.

### 6. Solver Scalability
The ILP grows with model size × mesh size × number of strategies per node. Graph clustering mitigates this for repeated structures (transformer layers), but non-regular architectures could hit solver time limits.

### 7. No Automatic Pipeline Stage Partitioning
Pipeline parallelism exists but requires manual stage assignment. The optimizer doesn't decide where to cut the model.

### 8. Cost Model Accuracy
The compute cost model has acknowledged inaccuracies (uses hardcoded TFLOPs rather than actual device capabilities). The all-to-all cost has a `5x` hack multiplier. Cost estimates are approximate.

## Known Gaps

### Critical Gaps

| Gap | Impact | Where in Code |
|-----|--------|---------------|
| **kwargs not processed** | Ops using kwargs for tensor args are silently missed | `optimize_sharding.py:203` |
| **Partial placement not explored** | Reduces search space for intermediates | `propagation_rules.py:193,221` |
| **Uneven sharding** | Factory ops with shapes not divisible by mesh fail | `apply_sharding.py:203` |
| **Dynamic shapes** | `make_fx` may break with dynamic shapes | `apply_sharding.py:295` |

### Disabled Features

| Feature | Status | Reason |
|---------|--------|--------|
| Context parallelism (CP) | Actively disabled | Upstream PyTorch bug (PR #131351) |
| `_unsafe_index.Tensor` | NotImplementedError | No sharding rule |
| `randperm` | NotImplementedError | "Needs hardening" |
| Cross-mesh communication | Infinite cost | Not implemented |

### Incomplete Implementations

| Feature | Issue |
|---------|-------|
| All-reduce backward | Should be identity, is currently another all-reduce |
| `local_map` | No support for `in_grad_placements` or custom device meshes |
| SDPA AC policy | "Not working yet" |
| Parameter memory bounds | Semantics of low/high undefined |
| `fill_missing_redistribute_cost` | Function body is `...` (empty) in dtensor helpers |

### Model Architecture Gaps

| Architecture | Status |
|-------------|--------|
| Transformer (decoder-only) | Fully supported (LLaMA-3, GPT-2) |
| Transformer (encoder-decoder) | Supported (T5 via HF example) |
| Transformer (encoder-only) | Supported (BERT via HF example) |
| MoE | Requires manual `local_map`, not auto-discovered |
| Vision models (ViT, ConvNeXt) | Conv rules exist but no examples/tests |
| Multi-modal | Not supported |
| Dynamic architectures | Not supported (static graphs) |

### Device Support

Hardcoded GPU specs for:
- **NVIDIA**: H100, B200, A100, A30, A10G, T4, V100, P100
- **AMD**: MI200X, MI300X, MI325X, MI350X, MI355X

Any GPU not in this list raises `ValueError`. No support for TPUs, Intel GPUs, or CPU-only execution.
