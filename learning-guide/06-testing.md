# Testing AutoParallel

## Running Tests

```bash
# Run all tests (requires CUDA GPU)
pytest tests/

# Run a specific test file
pytest tests/test_api.py

# Run a specific test
pytest tests/test_api.py::test_meta_model_init

# Verbose output
pytest tests/ -v

# Run with print output visible
pytest tests/ -s
```

## How Tests Work Without Multi-GPU

The test suite uses a **fake process group** (world_size=256) set up in `tests/conftest.py`. This means:

- No actual CUDA GPUs are needed for most tests (though a CUDA device must be available)
- The optimizer runs on fake tensors (shapes only, no data)
- Communication collectives are simulated, not executed

Three mesh fixtures are provided:
- `device_mesh_1d`: Shape `(256,)` — 1D mesh
- `device_mesh_2d`: Shape `(32, 8)` — 2D mesh (typical FSDP + TP)
- `device_mesh_3d`: Shape `(8, 8, 4)` — 3D mesh

## Test Coverage Map

### Core API Tests (`test_api.py` — 17 tests)
The most important test file. Covers:
- Meta model initialization
- FX graph annotation correctness
- Inference mode compilation
- `ModuleDict` preservation
- Cleanup on failure (FakeTensorMode)
- Unused parameter handling
- Aliased submodules (`isinstance` checks)
- User-defined methods, properties, classmethods on the parallel module
- EMA (Exponential Moving Average) updates
- Overlap scheduling (enabled/disabled)
- `compile=True` path

### Optimization Tests (`test_optimize_placement.py` — 7 tests)
- 1D FSDP/DDP discovery for FFN and Transformer models
- 2D FSDP+TP discovery
- In-graph tensor construction
- `local_map` placement constraints
- Parameter memory constraints
- Edge case: world_size larger than parameter size

### Sharding Application Tests (`test_apply_sharding.py`)
- Shard order computation
- Filter specs for `local_map`

### Cost Model Tests (`test_nccl_cost_model.py` — 50+ tests)
The most extensively tested component. Covers:
- Mesh topology derivation
- Algorithm bandwidth and latency (Ring, Tree, NVLS)
- Algorithm selection logic
- Protocol eligibility (LL, LL128, Simple)
- NVSwitch/CollNet feature flags
- Monotonicity (larger messages cost more)
- Multi-node empirical paths
- Blackwell GPU bandwidth scaling

### Propagation Rule Tests (`test_propagation_rules.py`)
- Permute + LayerNorm stride handling
- `torch.arange`/iota support with Embedding
- `index_put` with `List[Tensor]` args

### Module Construction Tests (`test_module_construction.py` — 11 tests)
- Parameters and buffers registered correctly
- `isinstance` preservation
- User attributes, methods copied
- Weight tying (parameter aliases)
- Buffer aliases
- Module aliases

### Weight Init Tests (`test_init_weights.py` — 12 tests)
- Basic initialization
- Inplace `.data[:]` patterns
- Aliased buffers (e.g., RoPE cache)
- Aliased parameters (weight tying)
- `load_state_dict`
- Submodule delegation
- Error detection for `.data = ...` assignment

### Activation Checkpointing Tests (`test_activation_checkpointing.py` — 10 tests)
- User-level `torch.utils.checkpoint` integration
- AutoParallel's `ac_joint_pass`
- `local_map` + AC interaction
- Recompute tag correctness

### Other Test Files
| File | Focus |
|------|-------|
| `test_auto_parallel_simple.py` | `auto_parallel()` convenience API |
| `test_dtensor.py` | DTensor sharding helpers |
| `test_estimate_graph_metrics.py` | Graph metrics estimation |
| `test_fsdp_all_gather_tagging.py` | FSDP AC tagging |
| `test_graph_utils.py` | View→einsum pattern matching |
| `test_input_validation.py` | Input constraint checking |
| `test_log_formatting.py` | Sharding log output |
| `test_ordered_sharding.py` | Ordered redistribution chains |
| `test_paired_output_constraint.py` | Forward-backward consistency |
| `test_placement_options_utils.py` | Placement option utilities |
| `test_prefetch_derived_sets.py` | Prefetch overlap sets |
| `test_tracing.py` | FakeTensor aliasing |
| `test_aot_eager.py` | AOT eager correctness (has xfail) |
| `test_api_pp.py` | Pipeline parallelism module |
| `test_ops.py` | Custom ops (sharding_constraint, permutation) |

## Linting

```bash
# Run all linters
pre-commit run --all-files

# Individual tools
black --check autoparallel/
isort --check autoparallel/
flake8 autoparallel/
mypy autoparallel/
```

## CI/CD

Three GitHub Actions workflows run automatically:

1. **`lint.yml`**: isort, black, mypy, flake8 on every PR
2. **`test_cuda.yml`**: Single-GPU tests + examples, multi-GPU examples (4 GPUs)
3. **`test_torchtitan.yml`**: Integration test with TorchTitan on 4 GPUs

## Writing New Tests

Follow the existing pattern:

```python
import pytest
import torch
from torch.distributed import DeviceMesh

def test_my_feature(device_mesh_2d):
    """Test with the 2D mesh fixture (32, 8)."""
    mesh = device_mesh_2d

    with torch.device("meta"):
        model = MyModel()

    sample_input = torch.randn(batch, seq_len, hidden, device="meta")

    # Use AutoParallel
    from autoparallel import auto_parallel
    parallel_model = auto_parallel(model, mesh, sample_input)

    # Assert properties of the result
    assert isinstance(parallel_model, MyModel)
    # Check sharding decisions, parameter placements, etc.
```
