# Architecture Deep Dive

## Directory Structure

```
autoparallel/
├── __init__.py                    # Public API exports
├── api.py                         # Main entry point: AutoParallel class
├── api_pp.py                      # Pipeline parallelism extension
├── optimize_sharding.py           # ILP-based sharding optimizer (the brain)
├── apply_sharding.py              # Applies solved sharding to the graph
├── tracing.py                     # FX graph tracing and decomposition
├── collectives.py                 # Communication primitives (all_gather, etc.)
├── module_construction.py         # Builds the final parallel nn.Module
├── init_weights.py                # DTensor-aware weight initialization
├── input_validation.py            # Input/output constraint checking
├── cast_parametrization.py        # Mixed precision dtype casting
├── log_formatting.py              # Human-readable sharding logs
├── ops.py                         # Custom ops (permutation)
│
├── shardings/                     # Sharding strategy enumeration
│   ├── propagation_rules.py       # Custom rules for ~30 PyTorch ops
│   ├── placement_options.py       # Generates all valid placements per node
│   ├── dtensor_sharding_helpers.py# DTensor integration and fallbacks
│   └── ordered_sharding.py        # Fused ND→1D collective optimization
│
├── cost_models/                   # Cost estimation
│   ├── compute_estimation.py      # FLOP counting + memory bandwidth model
│   ├── collective_runtime_estimation.py  # Communication cost model
│   └── nccl_cost_model.py         # Detailed NCCL algorithm simulation
│
├── graph_passes/                  # Graph transformations
│   ├── activation_checkpointing.py# AC: recompute vs. save decisions
│   ├── graph_clustering.py        # Repeated subgraph detection for ILP
│   ├── graph_partition.py         # Forward/backward splitting for PP
│   ├── split_di_dw_graph.py       # Gradient splitting for PP scheduling
│   ├── split_fsdp_collectives.py  # FSDP collective separation for PP
│   ├── graph_utils.py             # Graph manipulation utilities
│   ├── async_tp/                  # Async tensor parallelism passes
│   └── autobucketing_inductor/    # Inductor-specific collective bucketing
│
├── tools/overlap_simulator/       # Overlap simulation tools
├── _testing/models/               # Test model definitions (LLaMA-3)
│
examples/                          # Usage examples
tests/                             # Test suite (24 test files)
```

## Key Classes and Their Relationships

```
┌──────────────────────────────────────────────────────┐
│                    AutoParallel                      │
│                     (api.py)                         │
│                                                      │
│  1. build_joint_graph()                              │
│     └── Traces model → FX joint graph                │
│                                                      │
│  2. ShardingOptimizer (optimize_sharding.py)         │
│     ├── Enumerates sharding options per node         │
│     ├── Formulates ILP with PuLP                     │
│     └── Solves for optimal strategy                  │
│                                                      │
│  3. apply_sharding_to_model()                        │
│     ├── ApplyShardingInterpreter inserts collectives │
│     └── Compiles parallelized graph                  │
│                                                      │
│  4. make_parallel_module()                           │
│     └── Wraps result as nn.Module with DTensor params│
└──────────────────────────────────────────────────────┘
         │
         │ extends
         ▼
┌─────────────────────────────────────────────────────┐
│                  AutoParallelPP                     │
│                   (api_pp.py)                       │
│  Adds pipeline parallelism:                         │
│  - Graph partitioning into stages                   │
│  - FSDP collective splitting                        │
│  - dI/dW gradient splitting                         │
└─────────────────────────────────────────────────────┘
```

## The Four Phases

### Phase 1: Tracing (`api.py` → `tracing.py`)

The model (on `meta` device, no GPU memory) is traced into a joint forward+backward FX graph:

1. Deep copy model, apply mixed precision wrapping
2. `_dynamo_graph_capture_for_export` traces via TorchDynamo
3. `aot_export_joint_with_descriptors` produces joint graph via AOTAutograd
4. Custom decomposition table preserves ops with sharding rules (e.g., `native_layer_norm` is kept; `addmm` is decomposed into `mm + add` to enable TP)
5. Alias nodes inserted at multi-use points for optimizer flexibility

### Phase 2: Sharding Enumeration (`optimize_sharding.py` → `shardings/`)

For every node in the FX graph, enumerate all valid sharding strategies:

- **Placeholders**: All combos of `Replicate()` and `Shard(d)` per mesh dim
- **Operations**: Check custom rules → DTensor upstream rules → decomp fallback → all-Replicate
- Each strategy is an `OpSpec` with output placements, required input placements, and redistribution costs

### Phase 3: ILP Optimization (`optimize_sharding.py`)

Formulate and solve the ILP:

- **Variables**: Binary `x[node, arg, output_placement, input_placement]`
- **Objective**: Minimize Σ(communication_cost + compute_cost + transition_cost)
- **Constraints**: Uniqueness, consistency, flow, validity, user constraints
- **Solver**: PuLP's CBC (COIN-OR Branch and Cut)

### Phase 4: Application (`apply_sharding.py` → `module_construction.py`)

Apply the solution to create a parallel model:

1. `ApplyShardingInterpreter` walks the graph, inserting redistribute collectives where placements change
2. Graph is traced again via `make_fx` and compiled (Inductor or eager)
3. Activation checkpointing pass tags ops for recompute vs. save
4. `make_parallel_module()` builds the final `nn.Module` with DTensor parameters

## Cost Models

AutoParallel has three cost models working together:

### Compute Cost (`compute_estimation.py`)
- Uses `FlopCounterMode` to count FLOPs per operation
- Models memory bandwidth (reads + writes × bandwidth)
- Runtime = max(compute_time, memory_bound_time)
- Hardcoded GPU specs: H100 (989 TFLOPS bf16), B200 (2200 TFLOPS), A100, etc.

### Communication Cost (`collective_runtime_estimation.py`)
- Models all-gather, reduce-scatter, all-reduce, all-to-all
- Default: PyTorch's `MeshTopoInfo`-based model
- Advanced: NCCL cost model simulating 6 algorithms (Ring, Tree, CollNet, NVLS) and 3 protocols (LL, LL128, Simple)

### Transition Cost
- Tie-breaker: penalty of 1 for each redistribution
- Favors fewer separate collectives → better fusion

## The Public API

```python
# Simple API
from autoparallel import auto_parallel
parallel_model = auto_parallel(model, mesh, sample_inputs)

# Full API with more control
from autoparallel import AutoParallel
with AutoParallel(model, mesh, ...) as ap:
    graph = ap.build_joint_graph(sample_inputs)
    optimizer = ap.build_sharding_optimizer(graph)
    solution = optimizer.solve()
    parallel_model = ap.apply_and_build(graph, solution)

# Pipeline parallelism
from autoparallel import AutoParallelPP
```
