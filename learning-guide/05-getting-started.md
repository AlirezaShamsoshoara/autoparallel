# Getting Started

## Requirements

- Python >= 3.10
- PyTorch nightly (>= 2.10) — **not** a stable release
- CUDA GPU (for tests and real execution)
- PuLP (LP solver)
- `filecheck` (testing utility)

## Installation

### Option 1: Quick Install

```bash
pip install transformers
pip install -e .
```

### Option 2: Full Development Setup (with tests + linting)

```bash
# Install AutoParallel in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt

# Install pre-commit hooks for linting
pip install pre-commit
pre-commit install
```

### Existing Setup (your environment)

You already have a conda environment ready:

```bash
conda activate autoparallel
```

## Your First Run

The quickest way to see AutoParallel in action:

```bash
python examples/example_hf.py --model gpt2 --mesh 8
```

This runs GPT-2 on a **fake process group** with 8 virtual GPUs — no actual multi-GPU setup needed. It:
1. Creates a GPT-2 model on meta device
2. Sets up a 1D device mesh of size 8
3. Runs AutoParallel to find optimal sharding
4. Prints the sharding decisions

## Understanding the Output

When you run an example, you'll see output like:

```
[AutoParallel] Tracing model...
[AutoParallel] Building sharding metadata...
[AutoParallel] Solving ILP (N variables, M constraints)...
[AutoParallel] Solution found in X.XXs
[AutoParallel] Applying sharding...
```

The sharding log shows what the optimizer decided for each operation:
- Which operations are replicated
- Which are sharded (and on which dimension)
- What collectives were inserted (all-gather, reduce-scatter)
- The estimated costs

## Running Without GPUs (Fake Process Group)

AutoParallel supports a "fake process group" mode that simulates distributed training without actual GPUs. All examples use this by default:

```python
from torch.distributed import init_process_group
init_process_group(backend="fake", world_size=256)
mesh = DeviceMesh("cuda", torch.arange(256).reshape(32, 8))
```

This is how the optimizer runs: it only needs tensor shapes, not actual data. Real execution happens after optimization.

## Key Examples to Try

| Example | Command | What it demonstrates |
|---------|---------|---------------------|
| HuggingFace models | `python examples/example_hf.py --model gpt2 --mesh 8` | Simplest integration |
| Core demo | `python examples/example_autoparallel.py` | Full pipeline with mixed precision |
| LLaMA-3 | `python examples/example_llama3.py` | Production model, 2D mesh, vocab parallel |
| Local map | `python examples/example_local_map.py` | Custom sharding via `local_map` |
| Pipeline parallel | `python examples/example_pp_graph_passes.py` | PP graph transformations |

### Multi-GPU Examples (need real GPUs)

| Example | Command | GPUs |
|---------|---------|------|
| Checkpointing | `python examples/example_dcp.py` | 4 |
| DeepSeek V3 | `torchrun --standalone --nproc-per-node 4 examples/example_ds3_local_map.py` | 4 |
| DeepSeek V3 PP | `torchrun --standalone --nproc-per-node 4 examples/example_ds3_pp.py --use-loss-fn --fake-evaluate` | 4 |

## Minimal Code Example

Here's the simplest possible AutoParallel usage:

```python
import torch
from torch import nn
from torch.distributed import DeviceMesh, init_process_group
from autoparallel import auto_parallel

# Fake distributed setup (no real GPUs needed)
init_process_group(backend="fake", world_size=8)
mesh = DeviceMesh("cuda", torch.arange(8))

# Define a simple model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Create model on meta device (no memory allocated)
with torch.device("meta"):
    model = MLP()

# Sample input shape
sample_input = torch.randn(32, 1024, device="meta")

# AutoParallel finds optimal sharding
parallel_model = auto_parallel(model, mesh, sample_input)

# The result is a sharded nn.Module ready for distributed training
```

## What to Look For

When studying the examples, pay attention to:

1. **Meta device**: Models are created with `torch.device("meta")` — no GPU memory used during optimization
2. **Mesh shape**: How the GPUs are arranged affects what strategies are possible
3. **Sample inputs**: Only shapes matter, not values — used for tracing
4. **Sharding log**: The optimizer's decisions — which ops are sharded, which are replicated
5. **`init_weights()`**: How parameters are initialized after sharding (each GPU only initializes its shard)
