# Examples Walkthrough

This chapter walks through the key examples, explaining what each does and why.

## Example 1: HuggingFace Integration (`example_hf.py`)

**The simplest way to use AutoParallel with any HuggingFace model.**

```bash
python examples/example_hf.py --model gpt2 --mesh 8
```

### What it does:
1. Loads a HuggingFace model config (GPT-2, LLaMA, T5, BERT)
2. Creates the model on `meta` device
3. Sets up a fake process group with the specified mesh size
4. Runs `auto_parallel()` to find optimal sharding
5. Initializes weights and runs a forward + backward pass

### Key code pattern:
```python
config = AutoConfig.from_pretrained(args.model)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

mesh = DeviceMesh("cuda", torch.arange(args.mesh))
parallel_model = auto_parallel(model, mesh, sample_inputs)
```

### Supports three model types:
- `--task causal-lm` (default): Decoder-only models (GPT-2, LLaMA)
- `--task seq2seq`: Encoder-decoder models (T5)
- `--task masked-lm`: Encoder-only models (BERT)

## Example 2: Core Demo (`example_autoparallel.py`)

**The most detailed example showing the full API.**

```bash
python examples/example_autoparallel.py
```

### What it does:
1. Defines a Transformer block with self-attention + FFN
2. Sets up a 2D mesh (dp=32, tp=8) or 1D mesh (256)
3. Applies mixed precision (bf16 params, fp32 reduce)
4. Uses selective activation checkpointing
5. Runs the full AutoParallel pipeline with the verbose API

### Key patterns demonstrated:

**Mixed precision:**
```python
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
```

**Activation checkpointing:**
```python
from torch.utils.checkpoint import checkpoint
# Applied selectively in the model's forward()
```

**The full API (not the convenience function):**
```python
with AutoParallel(model, mesh, mixed_precision=mp_policy) as ap:
    joint_graph = ap.build_joint_graph(sample_inputs)
    optimizer = ap.build_sharding_optimizer(joint_graph)
    solution = optimizer.solve()
    parallel_model = ap.apply_and_build(joint_graph, solution)
```

## Example 3: LLaMA-3 (`example_llama3.py`)

**Production-scale model with advanced features.**

```bash
python examples/example_llama3.py
```

### What it does:
1. Sets up LLaMA-3 8B (or 70B) on a 2D mesh
2. Applies vocabulary parallelism constraints
3. Optionally applies manual TP constraints
4. Uses autobucketing for collective optimization
5. Supports async TP

### Advanced features:

**User sharding constraints:**
```python
# Force vocabulary embedding to be sharded on TP dimension
from autoparallel import with_sharding_constraint
x = with_sharding_constraint(x, mesh, [Replicate(), Shard(1)])
```

**Autobucketing:**
```python
# Groups small collectives together for better bandwidth utilization
ap.enable_autobucketing(runtime_estimator=custom_estimator)
```

## Example 4: Local Map (`example_local_map.py`)

**Custom sharding for operations AutoParallel can't auto-shard.**

```bash
python examples/example_local_map.py
```

### What it does:
1. Uses a 3D mesh (dp, tp, cp — data, tensor, context parallelism)
2. Demonstrates `local_map` for custom communication patterns
3. Shows context-parallel attention via manual sharding

### The `local_map` pattern:
```python
from torch.distributed.tensor import local_map

@local_map(
    out_placements=[Shard(2)],  # output is sharded on seq dim
    in_placements=[Shard(2), Shard(2), Shard(2)],  # inputs too
    device_mesh=mesh["cp"],
)
def context_parallel_attention(q, k, v):
    # This function sees local tensors (already sharded)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

`local_map` lets you write custom distributed ops that the auto-optimizer can't discover. The function receives local (sharded) tensors and returns local tensors — DTensor handles redistribution.

## Example 5: Distributed Checkpointing (`example_dcp.py`)

**Save and resume training with sharded checkpoints.**

```bash
python examples/example_dcp.py  # needs 4 GPUs
```

### What it does:
1. Phase 1: Single process with fake PG generates sharding map
2. Phase 2: Multi-process (4 GPUs) trains, saves checkpoint at step 10
3. Phase 3: Loads checkpoint onto unsharded model, verifies loss curves match

### Key pattern:
```python
# Save sharded checkpoint
from torch.distributed.checkpoint import save, load
save(parallel_model.state_dict(), checkpoint_dir)

# Load on different mesh/sharding
load(new_model.state_dict(), checkpoint_dir)
```

## Example 6: Pipeline Parallelism (`example_pp_graph_passes.py`)

**Graph transformations for pipeline parallelism.**

```bash
python examples/example_pp_graph_passes.py
```

### What it does:
1. Uses DeepSeek V3 model
2. Tests 4 PP configurations:
   - Basic graph partition (split forward/backward)
   - Split FSDP collectives (separate prefetch/reduce-scatter)
   - Split dI/dW (separate input-grad and weight-grad)
   - Combined (all passes)

## Example 7: DeepSeek V3 with MoE (`example_ds3_local_map.py`)

**Mixture-of-Experts with expert parallelism.**

```bash
torchrun --standalone --nproc-per-node 4 examples/example_ds3_local_map.py
```

### What it does:
1. DeepSeek V3 model with MoE layers
2. 2D mesh (dp, ep — data, expert parallelism)
3. Uses `local_map` for expert routing and communication
4. Supports real execution and numerics validation

## Running Examples in Your Environment

```bash
# Activate your environment
conda activate autoparallel

# Start with the simplest example
python examples/example_hf.py --model gpt2 --mesh 8

# Then try the detailed example
python examples/example_autoparallel.py

# Then LLaMA-3
python examples/example_llama3.py
```

Most examples use fake process groups and work on a single GPU or even CPU (for the optimizer phase). The multi-GPU examples (`example_dcp.py`, `example_ds3_*.py`) need real GPUs and `torchrun`.
