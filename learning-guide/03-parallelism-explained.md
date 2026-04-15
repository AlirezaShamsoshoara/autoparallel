# Parallelism Strategies

AutoParallel works by placing tensors on a multi-dimensional device mesh. Each mesh dimension can implement a different parallelism strategy. The ILP optimizer finds the best combination automatically.

## Supported Parallelism Types

### 1. Fully Sharded Data Parallel (FSDP)

**What**: Parameters are sharded across GPUs on a mesh dimension. Before each operation, parameters are all-gathered; after backward, gradients are reduce-scattered.

**How AutoParallel implements it**: Parameters get `Shard(0)` placement on the FSDP mesh dimension. The optimizer inserts all-gather before forward compute and reduce-scatter after backward compute.

**Memory savings**: Each GPU only stores 1/N of parameters (N = mesh dim size).

**Communication**: All-gather (forward) + reduce-scatter (backward) per layer.

```
GPU 0: [param_shard_0] --all-gather→ [full_param] → compute → [grad] --reduce-scatter→ [grad_shard_0]
GPU 1: [param_shard_1] --all-gather→ [full_param] → compute → [grad] --reduce-scatter→ [grad_shard_1]
```

### 2. Tensor Parallelism (TP)

**What**: Individual weight matrices are split across GPUs. For linear layers, this means column-parallel or row-parallel splitting.

**How AutoParallel implements it**: The `addmm` decomposition into `mm + add` enables the optimizer to discover TP strategies. For `Y = X @ W`:
- Column parallel: Shard W on dim 1 → each GPU computes a portion of the output → all-gather result
- Row parallel: Shard W on dim 0 → each GPU computes a partial result → reduce-scatter

**Typical pattern for a Transformer FFN**:
```
Linear1 (column-parallel): Shard(1) on weight → Partial output → 
Linear2 (row-parallel): Shard(0) on weight → all-reduce/reduce-scatter output
```

### 3. Data Parallelism (DP)

**What**: The simplest strategy. Replicate the model, shard the batch.

**How AutoParallel implements it**: Activations get `Shard(0)` on the batch dimension with replicated parameters. Gradient all-reduce after backward.

**When the optimizer chooses it**: For small models where the communication cost of FSDP outweighs the memory savings, or for mesh dimensions with few GPUs.

### 4. Pipeline Parallelism (PP)

**What**: Split model layers across GPUs. GPU 0 runs layers 1-10, GPU 1 runs layers 11-20, etc.

**How AutoParallel implements it**: `AutoParallelPP` extends the base class:
- `graph_partition.py` splits the joint graph into forward and backward subgraphs
- `split_fsdp_collectives.py` separates FSDP communication for overlap
- `split_di_dw_graph.py` splits backward into input-gradient (dI) and weight-gradient (dW) for scheduling

**Current status**: Requires manual stage assignment. The optimizer doesn't auto-partition into PP stages.

### 5. Hybrid / Mixed Parallelism

**What**: Combine multiple strategies on different mesh dimensions.

**Example**: On a `(4, 8)` mesh:
- Dimension 0 (size 4): FSDP — shard parameters
- Dimension 1 (size 8): TP — shard weight columns/rows

```python
mesh = DeviceMesh("cuda", torch.arange(32).reshape(4, 8), mesh_dim_names=("dp", "tp"))
# AutoParallel discovers the optimal FSDP+TP combination
```

This is where AutoParallel shines — it finds the optimal mix automatically via the ILP.

## Partially Supported

### Expert Parallelism (EP)
For Mixture-of-Experts (MoE) models, experts can be distributed across GPUs. AutoParallel supports this through `local_map` wrappers, but requires **manual** configuration — the automatic optimizer does not discover EP strategies on its own.

See `examples/native_ds3/` for DeepSeek-V3 MoE with manual expert parallelism.

### Context Parallelism (CP)
Splitting the sequence dimension of attention across GPUs. This is **actively disabled** in the SDPA propagation rule due to correctness issues in upstream PyTorch (PR #131351). The code filters out any strategy that shards on sequence dimension 2.

## Not Supported

### Sequence Parallelism (SP)
No dedicated Ring Attention or sequence-parallel-specific logic. While the optimizer could theoretically shard on sequence dimensions for non-attention ops, there's no integrated SP implementation.

### Automatic Pipeline Stage Partitioning
The user must manually assign pipeline stages. The optimizer doesn't decide where to cut the model.

### Cross-Mesh Communication
Redistribution between different meshes returns infinite cost — it's not supported.

## How the Mesh Maps to Parallelism

```
DeviceMesh("cuda", shape=(dp, tp))

                    TP dimension (shard weights)
                    ◄────────────────────────►
                ┌────┬────┬────┬────┬────┬────┬────┬────┐
  FSDP dim      │ G0 │ G1 │ G2 │ G3 │ G4 │ G5 │ G6 │ G7 │
  (shard params)├────┼────┼────┼────┼────┼────┼────┼────┤
       │        │ G8 │ G9 │G10 │G11 │G12 │G13 │G14 │G15 │
       │        ├────┼────┼────┼────┼────┼────┼────┼────┤
       ▼        │G16 │G17 │G18 │G19 │G20 │G21 │G22 │G23 │
                ├────┼────┼────┼────┼────┼────┼────┼────┤
                │G24 │G25 │G26 │G27 │G28 │G29 │G30 │G31 │
                └────┴────┴────┴────┴────┴────┴────┴────┘

  - Each row: TP group (8 GPUs share weight shards)
  - Each column: FSDP group (4 GPUs shard parameters)
```

## Placement Types

AutoParallel uses three DTensor placement types, composable per mesh dimension:

| Placement | Meaning | Memory | Communication |
|-----------|---------|--------|---------------|
| `Replicate()` | Full copy on every GPU | 1x (no savings) | None |
| `Shard(dim)` | Split along tensor dim | 1/N | All-gather to reconstruct |
| `Partial()` | Each GPU has partial sum | 1x | Reduce to get full result |

A tensor on a 2D mesh has a placement **per dimension**:
- `[Shard(0), Replicate()]` → sharded on dim 0 across mesh dim 0, replicated on mesh dim 1
- `[Shard(0), Shard(1)]` → sharded on different tensor dims across different mesh dims
