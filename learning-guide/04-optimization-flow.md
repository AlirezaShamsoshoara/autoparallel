# The Optimization Pipeline

This chapter walks through what happens when you call `auto_parallel(model, mesh, sample_inputs)`. Understanding this flow is key to debugging, extending, and explaining AutoParallel.

## High-Level Flow

```
User Model (nn.Module)
        │
        ▼
┌──────────────────────┐
│  Phase 1: TRACING    │  Model → FX joint graph (forward + backward)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Phase 2: ENUMERATE  │  For each node: list all valid sharding options
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Phase 3: OPTIMIZE   │  ILP solver picks the globally optimal strategy
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Phase 4: APPLY      │  Insert collectives, compile, build module
└──────────────────────┘
        │
        ▼
  Parallel nn.Module (ready for distributed training)
```

## Phase 1: Tracing

**File**: `api.py:build_joint_graph()` → `tracing.py`

**Goal**: Convert the user's `nn.Module` into a flat FX graph containing both forward and backward operations.

### Step by step:

1. **Deep copy** the model (don't modify the original)
2. **Mixed precision wrapping** via `cast_parametrization.py` — inserts `autoparallel::dtype_cast` ops for bf16/fp16 casting
3. **Move to fake tensors** — `move_to_fake()` creates FakeTensors that have shapes but no data (no GPU memory needed)
4. **TorchDynamo capture** — `_dynamo_graph_capture_for_export` traces the model with sample inputs
5. **AOTAutograd** — `aot_export_joint_with_descriptors` produces the **joint graph**: forward ops followed by backward ops in a single FX graph
6. **Custom decomposition table** — key decisions:
   - `addmm` → `mm + add` (enables tensor parallelism on the inner matmul)
   - `native_layer_norm`, `softmax`, etc. are **preserved** (they have custom sharding rules)
7. **Alias insertion** — `aten.alias` nodes are added after multi-use nodes, giving the optimizer independent placement choices at each fan-out edge

### Why a joint graph?

Having forward and backward in one graph lets the optimizer reason about end-to-end costs. An FSDP all-gather in the forward has a corresponding reduce-scatter in the backward — the optimizer sees both and can balance them.

## Phase 2: Sharding Enumeration

**File**: `optimize_sharding.py:build_sharding_metadata()` → `shardings/`

**Goal**: For every FX node, compute a list of valid `OpSpec` strategies with costs.

### For each node:

```
Node: aten.mm(X, W)

Strategy 1: [Replicate, Replicate] → [Replicate]       cost: compute_only
Strategy 2: [Shard(0), Replicate] → [Shard(0)]         cost: compute_only (batch parallel)
Strategy 3: [Replicate, Shard(1)] → [Shard(1)]         cost: need all-gather on output
Strategy 4: [Shard(1), Shard(0)] → [Partial]           cost: need reduce-scatter on output
...
```

### The lookup cascade:

1. **Custom rules** (`propagation_rules.py`): ~30 ops have hand-written rules (SDPA, layer_norm, conv, factory ops, view, split, einsum, etc.)
2. **DTensor upstream** (`dtensor_sharding_helpers.py`): PyTorch's built-in sharding propagation for common ops
3. **Single-dim expansion** (`_try_single_dim_strategy`): Derive multi-dim strategies from single-dim ones
4. **Decomposition fallback** (`_try_decomp_sharding`): Decompose the op and derive strategies from sub-ops
5. **All-Replicate fallback**: When `enable_implicit_replication=True`, replicate everything as last resort

### Cost computation:

Each strategy's cost has three components:
- **Compute cost**: FLOPs ÷ device TFLOPS, bounded by memory bandwidth
- **Communication cost**: Redistribution cost from each input's current placement to the strategy's required input placement
- **Transition cost**: A small penalty (1) for each redistribution, favoring fewer collectives

## Phase 3: ILP Optimization

**File**: `optimize_sharding.py:ShardingOptimizer`

**Goal**: Find the globally optimal sharding assignment by solving an Integer Linear Program.

### The ILP formulation:

**Variables**: For each node `i`, argument `a`, output placement `o`, input placement `j`:
```
x[i, a, o, j] ∈ {0, 1}   (binary: selected or not)
```

**Objective**: Minimize total cost
```
minimize Σ c[i,a,o,j] × x[i,a,o,j]
```

**Constraints**:

1. **Uniqueness** — Each (node, arg) pair selects exactly one strategy:
```
Σ_{o,j} x[i,a,o,j] = 1    for each (i, a)
```

2. **Consistency** — All arguments of a node agree on the output placement:
```
Σ_j x[i,0,o,j] = Σ_j x[i,1,o,j]    for each output placement o
```

3. **Flow** — Producer's output matches consumer's input:
```
Σ_j x[producer,0,o,j] = Σ_j x[consumer,a,j,o]
```
(The `o` in the producer becomes the `j` in the consumer — what you produce is what I consume)

4. **Validity** — Infinite-cost (impossible) configurations are forced to zero

5. **User constraints** — Fixed input/output placements, parameter memory bounds, forward-backward consistency

### Advanced optimizations:

- **Graph clustering** (`graph_clustering.py`): Repeated subgraphs (e.g., transformer layers) share ILP variables, reducing problem size dramatically
- **Prefetch discount**: Communication costs for FSDP all-gathers and gradient reduce-scatters are discounted because they can overlap with compute

### Solving:

The PuLP library's CBC solver finds the optimal solution. For a typical LLaMA-3 model, this takes seconds.

## Phase 4: Application

**File**: `apply_sharding.py` → `module_construction.py`

**Goal**: Transform the FX graph according to the solved strategy and build the final parallel module.

### Step by step:

1. **Shard placeholders** — Create DTensors from input placeholders using the solved placements

2. **Insert collectives** — `ApplyShardingInterpreter` walks the graph:
   - If an input's placement matches what the op needs → no-op
   - If it doesn't match → insert `redistribute_local_tensor` (triggers all-gather, reduce-scatter, or all-to-all)

3. **Handle special ops**:
   - Factory ops (zeros, ones): Adjust shape arguments for sharding
   - View ops: Use `DTensor.from_local` for correct local-shape handling

4. **Trace and compile** — The interpreted graph is traced via `make_fx`, then compiled with Inductor (default) or eager mode

5. **Activation checkpointing** — Tags operations as recompute-vs-save to bound peak memory:
   - FSDP all-gathers: recompute (or save, depending on `reshard_after_forward`)
   - Compute ops: staged recomputation using `sqrt(total_memory)` heuristic

6. **Build module** — `make_parallel_module()` creates the final `nn.Module`:
   - Parameters are DTensors with solved placements
   - Model class identity is preserved (`isinstance` checks work)
   - Aliases (weight tying) are maintained
   - `init_weights()` is wrapped for DTensor-aware initialization

## Debugging the Pipeline

AutoParallel provides debugging tools at each phase:

```python
# Log the sharding decisions
ap.log_sharding(solution)

# Explain why a specific placement was/wasn't chosen for a node
optimizer.explain_placement(node, target_placement)

# Compare two solutions
optimizer.diff_solutions(solution_a, solution_b)

# Print the full cost matrix for a node
optimizer.print_costs_for_node(node)
```
