# Prerequisites

Before diving into AutoParallel, you should be comfortable with these concepts. If you're missing any, the resources below will help.

## Must Know

### 1. PyTorch Basics
- `nn.Module`, `forward()`, parameters, buffers
- Autograd: how `.backward()` computes gradients
- The training loop: forward → loss → backward → optimizer step

### 2. Distributed Training Concepts
- **Data Parallelism (DP)**: Replicate model on every GPU, shard the batch. All-reduce gradients after backward.
- **Fully Sharded Data Parallel (FSDP)**: Shard model parameters across GPUs. All-gather before compute, reduce-scatter gradients after backward. Saves memory.
- **Tensor Parallelism (TP)**: Split individual weight matrices across GPUs. Column-parallel or row-parallel linear layers.
- **Pipeline Parallelism (PP)**: Split model layers across GPUs. Each GPU runs a subset of layers.

### 3. Device Mesh
A logical arrangement of GPUs into dimensions. For example, 32 GPUs arranged as a `(4, 8)` mesh:
- Dimension 0 (size 4): could be used for FSDP
- Dimension 1 (size 8): could be used for tensor parallelism

```python
from torch.distributed import DeviceMesh
mesh = DeviceMesh("cuda", torch.arange(32).reshape(4, 8), mesh_dim_names=("dp", "tp"))
```

### 4. Communication Collectives
- **All-Gather**: Every GPU collects the full tensor from all GPUs
- **Reduce-Scatter**: Reduce (sum) across GPUs and scatter the result
- **All-Reduce**: Reduce (sum) across all GPUs, every GPU gets the full result
- **All-to-All**: Each GPU sends a different shard to each other GPU

## Good to Know

### 5. FX Graphs
PyTorch FX traces a model into a graph of operations (nodes). Each node has:
- An `op` type: `placeholder`, `call_function`, `get_attr`, `output`
- `args` and `kwargs`
- Metadata like shape, dtype

AutoParallel works on FX graphs, not `nn.Module` directly. Understanding FX helps you read the codebase.

```python
import torch.fx
gm = torch.fx.symbolic_trace(model)
gm.graph.print_tabular()
```

### 6. DTensor (Distributed Tensor)
PyTorch's abstraction for distributed tensors. A DTensor wraps a local tensor with placement information:
- `Replicate()`: Full copy on every GPU
- `Shard(dim)`: Split along dimension `dim`
- `Partial()`: Each GPU holds a partial result (needs reduction)

```python
from torch.distributed.tensor import DTensor, Replicate, Shard
# A tensor sharded along dim 0 across the mesh
dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
```

### 7. Linear Programming (LP/ILP)
An optimization technique where you minimize a linear objective subject to linear constraints. AutoParallel uses **Integer Linear Programming** (variables are 0 or 1) to select the best sharding strategy for each operation.

You don't need to be an LP expert, but understanding the concept helps:
- **Variables**: Binary decisions (use this strategy or not)
- **Objective**: Minimize total cost
- **Constraints**: Rules that must be satisfied

## Resources

| Topic | Resource |
|-------|----------|
| PyTorch Distributed | [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) |
| FSDP | [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |
| Tensor Parallelism | [TP Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) |
| DTensor | [DTensor Tutorial](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) |
| FX | [FX Tutorial](https://pytorch.org/docs/stable/fx.html) |
| Linear Programming | [PuLP Documentation](https://coin-or.github.io/pulp/) |
