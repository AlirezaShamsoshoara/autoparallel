# Examples: Deep Walkthroughs

---

## `example_hf.py` — HuggingFace Integration

This is the simplest way to use AutoParallel with a real model. It takes any HuggingFace model, creates it on the meta device, runs AutoParallel to find the optimal sharding, and does a forward + backward pass — all without needing multiple GPUs.

## How to Run It

```bash
# GPT-2 (decoder-only, simplest)
python examples/example_hf.py --model gpt2 --mesh 8

# LLaMA-style model
python examples/example_hf.py --model meta-llama/Llama-3-8B --mesh 8

# T5 (encoder-decoder)
python examples/example_hf.py --model google-t5/t5-base --mesh 8 --task seq2seq

# BERT (encoder-only)
python examples/example_hf.py --model google-bert/bert-base-uncased --mesh 8 --task masked-lm

# 2D mesh (FSDP + TP)
python examples/example_hf.py --model gpt2 --mesh 2,4

# Custom batch size and sequence length
python examples/example_hf.py --model gpt2 --mesh 8 --batch-size 32 --seq-len 256
```

## Line-by-Line Walkthrough

### Step 1: Parse Arguments (lines 44-73)

```python
parser.add_argument("--model", default="gpt2")          # any HuggingFace model name
parser.add_argument("--mesh", default="8")               # GPU mesh shape: "8" or "2,4"
parser.add_argument("--task", default="causal-lm",       # model type
    choices=["causal-lm", "seq2seq", "masked-lm"])
parser.add_argument("--batch-size", default=16)
parser.add_argument("--seq-len", default=128)
```

**What to understand**: The `--mesh` argument defines how GPUs are arranged. `"8"` means a 1D mesh of 8 GPUs (FSDP or DP). `"2,4"` means a 2D mesh with 2 groups of 4 GPUs (e.g., FSDP on dim 0, TP on dim 1).

### Step 2: Set Up Fake Distributed Environment (lines 79-99)

```python
mesh_shape = tuple(int(s) for s in args.mesh.split(","))   # e.g., (8,) or (2, 4)
world_size = math.prod(mesh_shape)                          # total GPUs: 8

# Create a fake process group — no real GPUs needed
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

# Create the device mesh
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", mesh_shape, mesh_dim_names=dim_names
)
```

**What to understand**: This creates a **fake** distributed setup that simulates `world_size` GPUs on your single machine. No real NCCL communication happens. AutoParallel only needs tensor shapes and the mesh structure to run the optimizer — not real GPUs.

### Step 3: Define Input Sharding (line 102)

```python
x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
```

**What to understand**: This tells AutoParallel how the input should be distributed across GPUs. For the input tensor (token IDs), it's sharded on dimension 0 (the batch dimension) across the first mesh dimension, and replicated on any remaining mesh dimensions.

For a 1D mesh `(8,)`:
```
x_sharding = (Shard(0),)
→ Batch of 16 split into 8 shards of 2 each
```

For a 2D mesh `(2, 4)`:
```
x_sharding = (Shard(0), Replicate())
→ Batch split on dim 0 (2 shards of 8), replicated across TP dim
```

### Step 4: Load Model on Meta Device (lines 112-124)

```python
config = AutoConfig.from_pretrained(args.model)
if hasattr(config, "use_cache"):
    config.use_cache = False          # KV cache is not compatible with tracing
vocab_size = config.vocab_size

auto_model_cls = {
    "causal-lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "masked-lm": AutoModelForMaskedLM,
}[task]

with torch.device("meta"):
    model = auto_model_cls.from_config(config)
```

**What to understand**: This is the most important pattern. Three things happen:

1. **`AutoConfig.from_pretrained`**: Downloads the model's configuration (layer count, hidden size, number of heads, etc.) from HuggingFace. Only the config — no weights.

2. **`use_cache = False`**: KV caching (storing attention keys/values from previous tokens) uses dynamic state that FX tracing can't capture. Must be disabled.

3. **`with torch.device("meta")`**: Creates the model with **zero memory**. All parameters exist as meta tensors — they have shapes and dtypes but no data. For GPT-2 this saves ~500 MB. For LLaMA-3 70B this saves 280 GB.

### Step 5: Define Sample Inputs (lines 131-153)

```python
# For causal-lm and masked-lm:
def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

# For seq2seq (like T5):
def input_fn():
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    return input_ids, attention_mask, decoder_input_ids
```

**What to understand**: `input_fn` returns the **global** input shape (before sharding). The values are random — they don't matter. AutoParallel uses these to trace the model and determine tensor shapes at each operation.

For seq2seq models like T5, the forward pass takes three inputs:
- `input_ids`: encoder input tokens
- `attention_mask`: which tokens to attend to (must be a tensor, not None, for tracing)
- `decoder_input_ids`: decoder input tokens

### Step 6: Run AutoParallel (lines 156-174)

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
)

with AutoParallel(
    model, input_fn, mesh, mp_policy, compile=True, repeated_subgraphs=True
) as autop:
    autop.add_input_constraints(input_constraints)
    autop.add_parameter_memory_constraint(low=None, high=None)

    sharding_placement = autop.optimize_placement(verbose=True)
    parallel_mod = autop.apply_placement(sharding_placement)
```

**What to understand**: This is the core of AutoParallel. Let's break down each piece:

**`MixedPrecisionPolicy`**: Parameters are stored in `bfloat16` (saves memory, faster compute), but gradient reductions use `float32` (more numerically stable). This is standard practice for large model training.

**`AutoParallel(...) as autop`**: Creates the AutoParallel context. Arguments:
- `model`: your nn.Module (on meta device)
- `input_fn`: function that returns sample inputs
- `mesh`: the device mesh
- `mp_policy`: mixed precision settings
- `compile=True`: compile the final graph with Inductor for faster execution
- `repeated_subgraphs=True`: detect repeated transformer layers and share ILP variables (faster solve)

**`add_input_constraints`**: Tells the optimizer that inputs must have the specified sharding. Without this, the optimizer might choose to replicate inputs (wasting memory).

**`add_parameter_memory_constraint(low=None, high=None)`**: Optional memory bounds. `None` means no constraint — the optimizer is free to replicate or shard parameters as it sees fit.

**`optimize_placement(verbose=True)`**: This is where the magic happens:
1. Traces the model into a joint FX graph (forward + backward)
2. Enumerates sharding options for every node
3. Formulates the ILP
4. Solves it with PuLP/CBC
5. Returns the optimal placement for each node

With `verbose=True`, it prints the sharding log showing what was decided.

**`apply_placement`**: Takes the solved placements and transforms the graph:
1. Inserts collectives (all-gather, reduce-scatter, etc.)
2. Compiles with Inductor (if `compile=True`)
3. Returns a new `nn.Module` with DTensor parameters

### Step 7: Forward + Backward (lines 177-200)

```python
# Allocate real GPU memory for the sharded parameters
parallel_mod.to_empty(device="cuda")

# Compute local batch size (global_batch / num_data_parallel_groups)
local_batch = batch_size // mesh_shape[0]

# Create local input (just this GPU's shard of the batch)
x = torch.randint(0, vocab_size, (local_batch, seq_len), device="cuda")

# Forward pass
out = parallel_mod(x)

# Compute loss and backward
if isinstance(out, torch.Tensor):
    loss = out.sum()
elif isinstance(out, (tuple, list)):
    loss = out[0].sum()
else:
    loss = next(v.sum() for v in out.__dict__.values() if isinstance(v, torch.Tensor))
loss.backward()
```

**What to understand**:

**`to_empty(device="cuda")`**: Until now, parameters were on the meta device (no memory). This allocates real GPU memory for each parameter — but only the **local shard** on this GPU. If FSDP shards a parameter 8-way, each GPU allocates only 1/8 of it.

Note: `init_weights()` is **not** called in this example. In a real training scenario, you'd call `parallel_mod.init_weights()` after `to_empty()` to fill the parameters with proper initial values. Here, the random uninitialized values are fine since we only care that the forward + backward runs without errors.

**`local_batch = batch_size // mesh_shape[0]`**: The global batch is split across GPUs on the first mesh dimension. With `batch_size=16` and `mesh_shape=(8,)`, each GPU gets `local_batch=2`.

**The loss handling**: Different HuggingFace models return different types:
- Some return a raw tensor (logits)
- Some return a tuple (logits, hidden_states, ...)
- Some return a dataclass-like object with named fields
The code handles all three cases.

**`loss.backward()`**: Runs the backward pass. Because AutoParallel already inserted all the necessary collectives (reduce-scatter for gradients, etc.) into the compiled graph, this single `.backward()` call handles all distributed communication automatically.

## The Full Flow in One Picture

```
┌─────────────────────────────────────────────────────┐
│ 1. Parse args: model=gpt2, mesh=8                   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ 2. Fake process group: simulate 8 GPUs              │
│    init_process_group("fake", world_size=8)         │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ 3. Create GPT-2 on meta device (0 bytes)            │
│    with torch.device("meta"):                       │
│        model = AutoModelForCausalLM.from_config(cfg)│
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ 4. AutoParallel optimizer                           │
│    ├── Trace → joint FX graph (forward + backward)  │
│    ├── Enumerate sharding options per node          │
│    ├── Solve ILP → optimal placements               │
│    └── Insert collectives → compiled graph          │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ 5. Allocate real memory                             │
│    parallel_mod.to_empty(device="cuda")             │
│    (each GPU only allocates its shard)              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│ 6. Forward + backward                               │
│    out = parallel_mod(x)  ← collectives run inside  │
│    loss.backward()        ← gradient sync automatic │
└─────────────────────────────────────────────────────┘
```

## What the Output Looks Like

When you run with `verbose=True`, you'll see something like:

```
Model:      gpt2
Task:       causal-lm
Mesh:       (8,), dim_names=('dim0',)
Batch size: 16 (global), Seq len: 128

Config:     GPT2Config, 124.4M params

[AutoParallel] Tracing model...
[AutoParallel] Building sharding metadata for 847 nodes...
[AutoParallel] Solving ILP (12,340 variables, 8,721 constraints)...
[AutoParallel] Solution found in 1.23s, cost: 0.00425
[AutoParallel] Applying sharding...

def forward(self, input_ids):
    embedding = ...         # placement=Shard(0), cost=(0.00, 0.00, 0.00)
    layer_norm = ...        # placement=Shard(0), cost=(0.00, 0.01, 0.00)
    mm = ...                # placement=Shard(0), cost=(0.00, 0.15, 0.00)
    ...

AutoParallel pipeline completed in 3.45s
Forward + backward OK
```

The sharding log shows the placement chosen for each operation and the (communication, compute, transition) cost tuple.

## Experimenting With This Example

### Try Different Mesh Shapes

```bash
# 1D mesh: FSDP only (or DDP for small models)
python examples/example_hf.py --model gpt2 --mesh 8

# 2D mesh: FSDP + TP
python examples/example_hf.py --model gpt2 --mesh 2,4

# Larger mesh
python examples/example_hf.py --model gpt2 --mesh 32
```

Compare the sharding logs — you'll see different strategies for different mesh shapes.

### Try Different Models

```bash
# Small model — optimizer may choose all-Replicate (DDP)
python examples/example_hf.py --model gpt2 --mesh 8

# Larger model — optimizer will shard more aggressively
python examples/example_hf.py --model gpt2-large --mesh 8

# Different architecture
python examples/example_hf.py --model google-bert/bert-base-uncased --mesh 8 --task masked-lm
```

### Try Different Sequence Lengths

```bash
# Short sequences — activations are small, less pressure to shard
python examples/example_hf.py --model gpt2 --mesh 8 --seq-len 32

# Long sequences — activations are large, more benefit from sharding
python examples/example_hf.py --model gpt2 --mesh 8 --seq-len 1024
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: transformers` | HuggingFace not installed | `pip install transformers` |
| `RuntimeError: use_cache` | KV cache incompatible with tracing | Already handled (line 114) |
| `NotImplementedError: Operator ... does not have a sharding strategy` | Op has no sharding rule | The op falls back to Replicate (may be suboptimal) |
| Very slow optimization | Large model + large mesh = many ILP variables | Use `repeated_subgraphs=True` (already set) |

## Key Takeaways

1. **Any HuggingFace model works** — as long as it can be traced by TorchDynamo (most standard architectures can)
2. **No model code changes** — you use `AutoModelForCausalLM.from_config` normally
3. **Meta device is essential** — without it, you'd need GPU memory just to create the model
4. **The optimizer runs on a single machine** — fake process group, no real multi-GPU needed
5. **Input shapes matter, values don't** — `input_fn` returns random data, only shapes are used
6. **Mixed precision is recommended** — `bfloat16` params with `float32` reduce is standard practice

---

## `example_autoparallel.py` — The Full-Feature Demo

While `example_hf.py` shows the simplest path, this example is the **most educational**. It builds a Transformer block from scratch (no HuggingFace) and demonstrates features that `example_hf.py` skips: selective activation checkpointing, custom `init_weights`, 2D meshes, output constraints, and the full `with AutoParallel(...) as autop` API.

### How to Run It

```bash
python examples/example_autoparallel.py
```

No arguments — everything is configured in the script. To switch between 1D and 2D mesh, change `use_1d_mesh` on line 90.

### What This Example Teaches You

| Concept | Where in the code | What to learn |
|---------|-------------------|---------------|
| Hand-written Transformer block | Lines 34-80 | How attention + FFN looks without HuggingFace |
| Selective activation checkpointing | Lines 22-31, 68-70 | Choose per-op what to save vs. recompute |
| Custom `init_weights` | Lines 46-50, 139 | DTensor-aware weight initialization |
| 2D mesh (FSDP + TP) | Lines 96-104 | How to set up hybrid parallelism |
| Input AND output constraints | Lines 128-131 | Control sharding at both ends |
| Mixed precision with reduce_dtype | Line 122 | bf16 params + fp32 gradient reduction |

### Line-by-Line Walkthrough

#### Part 1: Selective Activation Checkpointing (lines 22-31)

```python
def policy_fn(ctx, op, *args, **kwargs):
    if (
        op == torch.ops.aten._scaled_dot_product_flash_attention.default
        or op == torch.ops.aten._scaled_dot_product_efficient_attention.default
    ):
        return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
    return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE

context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
```

**What to understand**: This defines a **per-op** checkpointing policy. Instead of checkpointing entire layers (all-or-nothing), this function is called for every operation and decides individually:

- **SDPA (attention)**: `MUST_RECOMPUTE` — don't save the attention output during forward. It's large (scales with sequence length²) but relatively cheap to recompute. This saves a lot of memory.
- **Everything else**: `MUST_SAVE` — keep it in memory. Matmul outputs, projections, etc. are expensive to recompute.

This is **selective** activation checkpointing — the smartest strategy. Compare to the approaches from the concepts chapter:
```
Checkpoint nothing:     Save everything      → maximum memory, no extra compute
Checkpoint whole layer: Save almost nothing   → minimum memory, ~2× compute
Selective (this code):  Save cheap, drop big  → good memory, minimal extra compute  ✓
```

#### Part 2: The Transformer Block (lines 34-80)

```python
class Block(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)    # query projection
        self.wk = nn.Linear(dim1, dim1, bias=bias)    # key projection
        self.wv = nn.Linear(dim1, dim1, bias=bias)    # value projection
        self.wo = nn.Linear(dim1, dim1, bias=bias)    # output projection
        self.w1 = nn.Linear(dim1, dim2, bias=bias)    # FFN up-projection
        self.w2 = nn.Linear(dim2, dim1, bias=bias)    # FFN down-projection
```

**What to understand**: This is a single Transformer layer, written from scratch. No HuggingFace, no abstractions. It has two parts:

**Self-Attention** (`_compute_attention`, lines 52-65):
```
x → [Wq] → q ─┐
x → [Wk] → k ─┤→ SDPA(q,k,v) → [Wo] → attention output
x → [Wv] → v ─┘
```

The `unflatten` + `permute` reshapes from `(batch, seq, hidden)` to `(batch, heads, seq, head_dim)` — this is how multi-head attention splits the hidden dimension into separate heads.

**FFN** (lines 72-78):
```
attention_output + x  →  [W1]  →  ReLU  →  [W2]  →  + residual  →  output
         residual connection                    residual connection
```

**Key point**: This is a **standard** `nn.Module`. No parallelism annotations, no `ColumnParallelLinear`, no collective calls. AutoParallel handles all of that automatically.

#### Part 3: The Selective Checkpoint Wrapping (lines 67-70)

```python
def forward(self, x):
    o = torch.utils.checkpoint.checkpoint(
        self._compute_attention, x, use_reentrant=False, context_fn=context_fn
    )
```

**What to understand**: Only the attention computation is wrapped with `checkpoint()`. The FFN (lines 74-78) is NOT checkpointed. This means:

- Attention activations (especially SDPA outputs) are dropped and recomputed during backward
- FFN activations are kept in memory (no recomputation)

The `context_fn=context_fn` passes the selective policy — within the checkpointed region, the policy further decides per-op what to save vs. recompute.

#### Part 4: Custom `init_weights` (lines 46-50)

```python
def init_weights(self):
    for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
        torch.nn.init.normal_(lin.weight)
        if lin.bias is not None:
            torch.nn.init.normal_(lin.bias)
```

**What to understand**: When the model is created on meta device, parameters have no data. After AutoParallel shards the model, `init_weights()` is called to fill each GPU's shard with actual values. AutoParallel wraps this method to be **DTensor-aware** — each GPU only initializes its local shard, using the correct RNG seed so the global result is consistent.

This is called at line 139:
```python
parallel_mod.to_empty(device="cuda")   # allocate memory for shards
parallel_mod.init_weights()             # fill with values
```

The `example_hf.py` skipped `init_weights()` — this example shows the proper pattern.

#### Part 5: 2D Mesh Setup (lines 83-104)

```python
world_size = 256

use_1d_mesh = False

if use_1d_mesh:
    # 1D mesh: 256 GPUs in a line
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
else:
    # 2D mesh: 32 groups × 8 GPUs per group
    mesh = init_device_mesh("cuda", (world_size // 8, 8), mesh_dim_names=("dp", "tp"))
```

**What to understand**: This is where you choose the parallelism structure:

```
1D mesh (256,):           2D mesh (32, 8):

  All 256 GPUs in one       32 rows × 8 columns
  dimension. The ILP        "dp" dimension (32): likely FSDP
  chooses FSDP or DDP.      "tp" dimension (8):  likely tensor parallelism

  Simpler but limited.      The ILP discovers the optimal
                             FSDP+TP combination.
```

The 2D mesh is the default (`use_1d_mesh = False`) and the more interesting case. With 256 GPUs in a `(32, 8)` arrangement, the optimizer will likely choose FSDP on the dp dimension (shard parameters 32-way) and tensor parallelism on the tp dimension (split weight matrices 8-way).

#### Part 6: Model Dimensions (lines 106-110)

```python
bs = 8 * mesh.shape[0]     # batch size scales with dp dimension
seq_len = 256
nheads = 48                 # number of attention heads
dim1 = 6144                 # hidden dimension
dim2 = dim1 * 4             # FFN intermediate dimension (4× hidden, standard ratio)
```

**What to understand**: These dimensions are **realistic** — similar to a large model (6144 hidden is close to LLaMA-3 8B's 4096, and the 4× FFN ratio is standard). The batch size scales with the dp mesh dimension so each GPU gets a constant local batch of 8.

The number of heads (48) affects tensor parallelism: the TP dimension (8) must divide the number of heads evenly (48 / 8 = 6 heads per GPU). AutoParallel checks this via divisibility constraints.

#### Part 7: The AutoParallel Pipeline (lines 125-136)

```python
with AutoParallel(model, input_fn, mesh, mp_policy, compile=True) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    sharding_placement = autop.optimize_placement()
    parallel_mod = autop.apply_placement(sharding_placement)
```

**What to understand**: This uses the **full API** (not the simple `auto_parallel()` convenience function). The differences from `example_hf.py`:

**`add_output_constraints`**: Forces the output to have the same sharding as the input — `Shard(0)` on the batch dimension. This ensures the output is data-parallel (each GPU has its batch slice), which is needed for loss computation. `example_hf.py` didn't set output constraints.

**No `repeated_subgraphs=True`**: This model has only one Transformer block, so graph clustering wouldn't help. `example_hf.py` used it because GPT-2 has 12 identical layers.

**No `verbose=True`**: The sharding log isn't printed by default. Add `verbose=True` to `optimize_placement()` to see the optimizer's decisions.

#### Part 8: Forward + Backward (lines 138-146)

```python
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

x = (torch.rand(bs // mesh.shape[0], seq_len, dim1, device="cuda"),)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))
```

**What to understand**: Unlike `example_hf.py`, this example calls `init_weights()` — the proper pattern for real training. Also note:

- **Local batch**: `bs // mesh.shape[0]` divides by the dp dimension (32), so each GPU gets a batch of 8
- **`torch.randn_like(out)` as grad**: Since there's no loss function, a random gradient is used to drive the backward pass. This is fine for testing — we just need to verify that the backward runs without errors and all collectives work.

### What the Optimizer Likely Decides

For a 2D mesh `(32, 8)` with dim names `("dp", "tp")`:

```
Weight matrices (wq, wk, wv, wo, w1, w2):
  → [Shard(0), Shard(1)]  or  [Shard(0), Replicate()]
     FSDP          TP            FSDP        no TP

  The optimizer balances:
  - More TP sharding = less memory per GPU, but more communication
  - Less TP sharding = more memory per GPU, but less communication

Activations (q, k, v, attention output):
  → [Shard(0), Replicate()]  or  [Shard(0), Shard(1)]
     batch-parallel              batch + head-parallel

  The optimizer may shard attention heads across the TP dimension
  since nheads=48 divides evenly by tp=8.

SDPA output:
  → The optimizer sees this as an expensive op and may choose
     head-sharding to reduce per-GPU memory.

FFN weights (w1: 6144→24576, w2: 24576→6144):
  → Classic column-parallel / row-parallel split on TP dimension.
     w1: Shard(1) on TP (column-parallel)
     w2: Shard(0) on TP (row-parallel)
```

### How This Example Differs from `example_hf.py`

| Aspect | `example_hf.py` | `example_autoparallel.py` |
|--------|-----------------|---------------------------|
| Model | HuggingFace (GPT-2, T5, etc.) | Hand-written Transformer block |
| API | `with AutoParallel(...)` | Same, but exercises more features |
| Activation checkpointing | None | Selective per-op policy |
| `init_weights` | Skipped | Properly called |
| Output constraints | Not set | Set (matches input sharding) |
| Mesh | 1D (configurable) | 2D default, 1D option |
| `repeated_subgraphs` | True (multi-layer) | Not used (single block) |
| Mixed precision | bf16 params, fp32 reduce | Same |
| Model scale | Small (GPT-2 124M) | Large dims (6144 hidden, ~900M params) |

### What You Should Learn From This Example

1. **Selective AC is the practical approach** — don't checkpoint everything or nothing; choose per-op based on size vs. recompute cost
2. **The full API gives more control** — input constraints, output constraints, memory constraints are all available
3. **2D meshes enable hybrid parallelism** — the optimizer discovers FSDP+TP combinations automatically
4. **`init_weights()` is the right pattern** — create on meta, shard, allocate, then initialize
5. **No model code changes for parallelism** — the `Block` class is pure single-GPU PyTorch; AutoParallel adds all the distributed logic
