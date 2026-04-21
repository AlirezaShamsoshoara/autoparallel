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

---

## `example_llama3.py` — Production-Scale LLaMA-3

This is the most advanced example. It uses a real LLaMA-3 architecture (8B or 70B) with production features: vocabulary parallelism, optional manual TP constraints, autobucketing for collective optimization, and async tensor parallelism. If you want to understand how AutoParallel handles a real model, this is the one to study.

### How to Run It

```bash
python examples/example_llama3.py
```

No arguments — configuration is controlled by variables in the script. Key settings to change:
- `model_type = "8b"` (line 56): Change to `"70b"` for the larger model
- `use_1d_mesh = False` (line 34): Change to `True` for 1D FSDP-only
- `enable_asynctp = False` (line 57): Change to `True` for async tensor parallelism
- `enable_manual_constraint = False` (line 234): Change to `True` to force specific TP placement

### What This Example Teaches You

| Concept | Where in the code | What to learn |
|---------|-------------------|---------------|
| Real LLaMA-3 architecture | Lines 60-88 | GQA, RoPE, SwiGLU — production Transformer features |
| Vocabulary parallelism | Lines 226-229 | Shard the output logits across GPUs |
| Manual TP constraints | Lines 135-216, 234-236 | Override the optimizer for specific ops |
| Autobucketing | Lines 101-125 | Group small collectives for better bandwidth |
| Graph metrics estimation | Lines 117-118 | Measure expected compute/communication |
| Async TP | Lines 238-255 | Overlap TP communication with compute |
| `repeated_subgraphs=True` | Line 220 | 32 identical layers → share ILP variables |

### Part 1: The LLaMA-3 Architecture (lines 60-88)

```python
if model_type == "8b":
    model_args = TransformerModelArgs(
        dim=4096,           # hidden dimension
        n_layers=32,        # 32 transformer blocks
        n_heads=32,         # 32 query heads
        n_kv_heads=8,       # 8 key/value heads (GQA)
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,  # RoPE positional encoding
        vocab_size=128256,  # LLaMA-3 vocabulary
        max_seq_len=8192,   # 8K context length
    )
```

**What to understand**: This is the actual LLaMA-3 8B configuration. Three features are different from the simple `example_autoparallel.py`:

**Grouped-Query Attention (GQA)**: `n_heads=32` but `n_kv_heads=8`. Instead of 32 independent sets of Q, K, V, there are 32 query heads but only 8 key/value heads. Every 4 query heads share one K/V head. This saves memory and compute for K/V without losing much quality.

```
Standard Multi-Head (32 Q, 32 K, 32 V):
  Q₀→K₀,V₀  Q₁→K₁,V₁  Q₂→K₂,V₂  ...  Q₃₁→K₃₁,V₃₁

Grouped-Query (32 Q, 8 K, 8 V — LLaMA-3):
  Q₀,Q₁,Q₂,Q₃ → K₀,V₀     (4 queries share 1 K/V)
  Q₄,Q₅,Q₆,Q₇ → K₁,V₁
  ...
  Q₂₈,Q₂₉,Q₃₀,Q₃₁ → K₇,V₇
```

The `repeat_kv()` function in the model code handles this by repeating each K/V head 4 times to match the query head count before feeding into SDPA.

**RoPE (Rotary Position Embedding)**: Instead of adding positional embeddings to the input, RoPE applies a rotation to Q and K vectors based on their position. `rope_theta=500000` controls the frequency base. The `precompute_freqs_cis` function creates the rotation frequencies as complex numbers, and `apply_rotary_emb` applies them.

**SwiGLU FFN**: The feedforward network uses three linear layers instead of two:
```
Standard FFN:    x → W1 → ReLU → W2 → output
SwiGLU FFN:      x → W1 → SiLU ──┐
                 x → W3 ──────────┤× (element-wise multiply)
                                   └→ W2 → output
```
This is the `forward` in the `FeedForward` class: `self.w2(F.silu(self.w1(x)) * self.w3(x))`. It's more expressive than standard FFN.

### Part 2: The Mesh and Input Setup (lines 27-54)

```python
world_size = 64

mesh = init_device_mesh("cuda", (world_size // 8, 8), mesh_dim_names=("dp", "tp"))
# → (8, 8) mesh: 8 FSDP groups × 8 TP groups

batch_size = 2 * mesh.shape[0]    # 2 × 8 = 16 global batch
seqlen = 2048 * 4                 # 8192 sequence length
vocab_size = 128256               # LLaMA-3 tokenizer vocabulary
```

**What to understand**: 64 GPUs in an `(8, 8)` mesh. The batch size scales with the dp dimension (each GPU gets `batch_size // 8 = 2` sequences). The sequence length is 8192 — a long-context workload where activation memory is significant.

### Part 3: Vocabulary Parallelism (lines 226-229)

```python
if use_vocab_parallel:
    assert mesh.ndim == 2
    out_sharding = (Shard(0), Shard(2))
```

**What to understand**: The output of a language model has shape `(batch, seq_len, vocab_size)`. With `vocab_size=128256`, this is a **huge** tensor. Vocabulary parallelism shards it on dimension 2 (the vocab dimension) across the TP mesh dimension.

```
Without vocab parallel:
  Each GPU: output shape (2, 8192, 128256)  ← 8 GB per GPU in bf16!

With vocab parallel (tp=8):
  Each GPU: output shape (2, 8192, 16032)   ← 1 GB per GPU ✓
```

This is set as an **output constraint** — the optimizer is told "the output must be `Shard(0)` on dp and `Shard(2)` on tp." The optimizer then works backward from this constraint to figure out how to shard the final linear layer (`self.output`) to produce this placement naturally.

### Part 4: Autobucketing (lines 101-125)

```python
autobucketing_level = "aten"

custom_runtime_estimation = make_custom_runtime_estimation(mesh)
aten_autobucketing_config.custom_runtime_estimation = custom_runtime_estimation

torch._inductor.config.reorder_for_peak_memory = False
torch._inductor.config.reorder_for_compute_comm_overlap = False

def post_grad_pass(graph):
    new_gm = _aten_autobucketing_pass(graph)
    metrics = estimate_graph_metrics(new_gm, custom_runtime_estimation)
    print(metrics)
    return new_gm

torch._inductor.config.post_grad_custom_post_pass = post_grad_pass
```

**What to understand**: After AutoParallel inserts collectives into the graph, there may be many small all-gathers and reduce-scatters scattered throughout. Each collective has a **launch overhead** (~7μs). Sending many small messages is inefficient — better to **bucket** (group) them together into fewer, larger messages.

Autobucketing does this as a post-compilation pass:

```
Before bucketing:                     After bucketing:
  all-gather(64 KB)                     all-gather(256 KB)  ← fused!
  compute...                            compute...
  all-gather(64 KB)
  compute...
  all-gather(64 KB)
  compute...
  all-gather(64 KB)
```

The `estimate_graph_metrics` call prints the expected compute time, communication time, and overlap ratio — useful for understanding whether communication is the bottleneck.

Inductor's default reordering passes are disabled (`reorder_for_peak_memory = False`, `reorder_for_compute_comm_overlap = False`) because AutoParallel's own bucketing pass handles this better for its specific use case.

### Part 5: Manual TP Constraints (lines 135-216)

```python
enable_manual_constraint = False
if enable_manual_constraint and not use_1d_mesh:
    add_tp_constraints(autop)
```

**What to understand**: This is an **optional override** of the optimizer's decisions. When enabled, it forces specific TP patterns on the matmul operations:

```python
# The forced TP pattern for each transformer block's 7 matmuls:

# Q, K, V projections (3 matmuls) + FFN up-projections (2 matmuls):
#   Column-parallel: input Replicate on TP, weight Shard(1) on TP → output Shard(1) on TP

# Output projection (1 matmul) + FFN down-projection (1 matmul):
#   Row-parallel: input Shard on TP, weight Shard(0) on TP → output Partial on TP
```

This is exactly the Megatron-LM column/row parallel pattern, but expressed as ILP constraints instead of replacement layers. The comment on lines 151-153 explains the math:
```
out = x @ w     →  S(0)R × RS(1) → S(0)S(1)     (column-parallel)
g_w = g.T @ x   →  S(1)S(0) × S(0)R → PS(0)     (weight gradient)
g_x = g @ w.T   →  S(0)S(1) × RS(0) → S(0)P     (input gradient, needs reduce)
```

**Why would you use this?** Two reasons:
1. **Benchmarking**: Compare AutoParallel's auto-discovered strategy against the known-optimal Megatron-style TP
2. **Guardrails**: If you know the optimal TP pattern, you can force it and let the optimizer only decide the FSDP strategy

By default this is disabled (`enable_manual_constraint = False`), letting the ILP discover the strategy on its own.

### Part 6: Async Tensor Parallelism (lines 238-255)

```python
enable_asynctp = False
if enable_asynctp:
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group
    enable_symm_mem_for_group(mesh["dp"].get_group().group_name)
    enable_symm_mem_for_group(mesh["tp"].get_group().group_name)

    from autoparallel.asynctp import micro_pipeline_tp_pass
    torch._inductor.config.post_grad_custom_post_pass = _pass
```

**What to understand**: Normal TP does: compute matmul → wait → all-reduce → next matmul. Async TP **overlaps** the communication with the next computation:

```
Normal TP:     [matmul]─[wait]─[all-reduce]─[wait]─[matmul]─[wait]─[all-reduce]
Async TP:      [matmul]─[all-reduce──────]─[matmul]─[all-reduce──────]
                              ↑                           ↑
                         overlapped with next matmul start
```

This uses **symmetric memory** — a special memory region that all GPUs in a group can directly access (via NVLink), enabling zero-copy communication. The `micro_pipeline_tp_pass` splits operations into micro-pipelines to maximize overlap.

This is an advanced optimization and disabled by default.

### Part 7: The AutoParallel Pipeline (lines 219-260)

```python
with AutoParallel(
    model, input_fn, mesh, mp_policy, compile=True, repeated_subgraphs=True
) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    out_sharding = x_sharding
    if use_vocab_parallel:
        out_sharding = (Shard(0), Shard(2))

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])

    sharding_placement = autop.optimize_placement(verbose=True)
    parallel_mod = autop.apply_placement(sharding_placement)
```

**What to understand**: Key difference from previous examples:

**`repeated_subgraphs=True`**: LLaMA-3 8B has 32 identical transformer blocks. Without this flag, the ILP would create separate variables for each block — 32× more variables and constraints. With this flag, AutoParallel detects the repeated structure and shares ILP variables across all 32 blocks, assuming they all get the same sharding. This dramatically speeds up the solver.

**Different input vs. output sharding**: Input is `(Shard(0), Replicate())` (batch-sharded, replicated on TP). Output is `(Shard(0), Shard(2))` (batch-sharded, vocab-sharded on TP). This means the model must transition from replicated activations to vocab-parallel output somewhere in the final layers — the optimizer figures out where.

### Part 8: Forward + Backward (lines 262-277)

```python
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

x = (torch.randint(0, vocab_size, (batch_size // mesh.shape[0], seqlen), device="cuda"),)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))
```

Same pattern as previous examples, but note the local input: `batch_size // mesh.shape[0]` = `16 // 8` = 2 sequences per GPU, each of length 8192.

### What the Optimizer Discovers

For LLaMA-3 8B on an `(8, 8)` mesh, the optimizer typically finds:

```
Embedding (tok_embeddings):
  → [Shard(0), Replicate()]  on dp, replicated on TP
  The embedding table is huge (128256 × 4096) but only looked up, not matmul'd.

Attention projections (wq, wk, wv):
  → FSDP on dp, column-parallel on TP
  wq: [4096, 4096] → each TP GPU gets [4096, 512]
  wk: [4096, 1024] → each TP GPU gets [4096, 128]  (smaller due to GQA)
  wv: [4096, 1024] → each TP GPU gets [4096, 128]

Output projection (wo):
  → FSDP on dp, row-parallel on TP
  Produces Partial output → reduce-scatter or all-reduce

SwiGLU FFN (w1, w2, w3):
  → w1, w3: column-parallel on TP (up-projections)
  → w2: row-parallel on TP (down-projection, produces Partial → reduce)

Final output linear:
  → [Shard on dp, Shard(1) on TP] → produces vocab-parallel output
  Matches the output constraint (Shard(0), Shard(2))

RoPE freqs_cis buffer:
  → Replicate() everywhere (small, shared across all GPUs)
```

### How This Example Differs from the Previous Two

| Aspect | `example_hf.py` | `example_autoparallel.py` | `example_llama3.py` |
|--------|-----------------|---------------------------|---------------------|
| Model | Any HF model | 1-block Transformer | Full LLaMA-3 (8B/70B) |
| Architecture | Generic | Basic attention + FFN | GQA + RoPE + SwiGLU |
| Scale | ~124M params | ~900M params | 8B-70B params |
| Mesh | 1D (configurable) | 2D default | 2D default (64 GPUs) |
| Vocab parallel | No | No | Yes |
| AC | None | Selective per-op | None (model-level) |
| Autobucketing | No | No | Yes (aten-level) |
| Manual TP option | No | No | Yes (optional) |
| Async TP option | No | No | Yes (optional) |
| `repeated_subgraphs` | Yes (multi-layer) | No (1 block) | Yes (32 layers) |
| Graph metrics | No | No | Yes (prints expected perf) |

### What You Should Learn From This Example

1. **Real architectures have nuances** — GQA changes the TP sharding of K/V heads; SwiGLU adds a third weight matrix; RoPE adds a buffer that must be replicated
2. **Vocabulary parallelism matters** — for large vocabs, the output logits tensor dominates memory; sharding it on the TP dimension is essential
3. **Autobucketing improves real performance** — many small collectives from 32 layers get fused into fewer, larger ones
4. **Manual constraints let you guide the optimizer** — useful for benchmarking or when you know the optimal pattern
5. **`repeated_subgraphs` is critical for multi-layer models** — 32 identical layers means 32× reduction in ILP size
6. **Async TP is the next frontier** — overlapping communication with compute for TP is where production systems are heading
