# Real-World Usage

## Where AutoParallel Shines

### 1. Large Language Model Training
The primary use case. AutoParallel was built for and tested on LLaMA-3 (8B, 70B) and DeepSeek-V3. If you're training a decoder-only transformer at scale, this is the sweet spot.

**Why**: FSDP+TP hybrid parallelism is the standard for large LLMs, but the optimal FSDP/TP split depends on model size, hidden dimensions, number of heads, and mesh topology. AutoParallel finds this automatically.

### 2. Exploring Parallelism Strategies
Even if you don't use AutoParallel in production, it's a powerful **exploration tool**:
- "Should I use FSDP, TP, or both for my model at this scale?"
- "What's the optimal TP degree for my cluster topology?"
- "How much communication will my sharding strategy incur?"

Run the optimizer on meta device (no GPUs needed) and inspect the solution.

### 3. Rapid Prototyping
When iterating on model architectures, AutoParallel eliminates the need to manually re-derive sharding strategies every time you change the model structure.

### 4. Multi-Dimensional Meshes
For clusters with complex topologies (e.g., NVSwitch within nodes, InfiniBand across nodes), the multi-dimensional mesh + ILP naturally handles the asymmetric bandwidth.

## Where NOT to Use AutoParallel (Today)

### 1. Production Training at Scale
The project is experimental. APIs are unstable. Requires PyTorch nightly. Not battle-tested at Meta-scale production.

### 2. Models with Dynamic Control Flow
If your model has data-dependent branching, variable-length loops, or other dynamic behavior, FX tracing won't capture it correctly.

### 3. Small Models on Few GPUs
If your model fits on one GPU, or you only have 2-4 GPUs, simple DDP or FSDP is sufficient. AutoParallel's overhead (tracing, ILP solve) isn't worth it.

### 4. Non-GPU Hardware
TPUs, Intel Gaudi, or CPU-only setups are not supported. NVIDIA CUDA and AMD ROCm GPUs are supported (the cost model has specs for both).

### 5. Custom Kernels / Triton Ops
If your model relies heavily on custom CUDA/Triton kernels, they likely won't have sharding rules. The optimizer will fall back to all-Replicate for those ops.

## Who Benefits Most

| Role | Benefit |
|------|---------|
| **ML Engineers** training large models | Eliminates manual sharding — faster iteration |
| **Researchers** trying new architectures | No need to derive parallel strategies manually |
| **Infrastructure teams** | Consistent, optimal parallelism across models |
| **Students / learners** | Educational tool for understanding distributed training |

## Integration Points

### With TorchTitan
AutoParallel has a CI integration test with TorchTitan (Meta's distributed training framework). The test trains LLaMA-3 with AutoParallel on 4 GPUs with TP degree 4.

### With HuggingFace Transformers
The `example_hf.py` shows direct integration with any HuggingFace model. Just load the config, create on meta device, and pass to `auto_parallel()`.

### With Distributed Checkpointing
`example_dcp.py` shows save/load of sharded checkpoints. Compatible with PyTorch's `torch.distributed.checkpoint`.

### With Inductor
By default, AutoParallel compiles the parallelized graph with `torch.compile` / Inductor for optimized kernels.

## Estimating Gains

### Communication Reduction
AutoParallel's main win is reducing communication by finding the optimal placement. Compare:
- **DDP**: All-reduce all gradients (2× model_size per step)
- **FSDP**: All-gather params + reduce-scatter grads (2× model_size, but overlapped)
- **FSDP+TP**: Reduce both data-parallel and tensor-parallel communication by sharding across both dimensions

### How to Measure
1. Run AutoParallel with logging enabled
2. Compare estimated costs against your current manual strategy
3. Look at the sharding log: are there ops that your manual strategy replicates but AutoParallel shards?

### What to Expect
- For well-optimized manual sharding: AutoParallel should match or slightly improve
- For naive FSDP: AutoParallel may find TP opportunities you missed → 10-30% speedup
- For complex hybrid strategies: AutoParallel eliminates trial-and-error → development time savings

## Competitive Landscape

| Tool | Approach | Comparison |
|------|----------|------------|
| **Manual FSDP/TP** (PyTorch) | Hand-written parallel code | AutoParallel automates this |
| **Megatron-LM** (NVIDIA) | Library with fixed TP/PP/DP patterns | More mature but less flexible |
| **DeepSpeed** (Microsoft) | ZeRO optimizer + pipeline | Different approach (memory optimization) |
| **Alpa** (UC Berkeley) | ILP-based auto-parallelism for JAX | Similar idea, different framework |
| **GSPMD** (Google) | Compiler-based sharding for XLA | Similar idea, XLA-specific |
| **FlexFlow** (Stanford) | Simulation-based auto-parallelism | Uses simulation, not LP |

AutoParallel's unique position: **ILP-based, PyTorch-native, DTensor-integrated**. It's the closest thing to Alpa/GSPMD but built on PyTorch's own distributed primitives.
