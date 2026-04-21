# Concepts Deep Dive: Placements, Graphs, Tracing, and Operations

This chapter explains the foundational concepts that the rest of the guide assumes you know. Start here if terms like "Shard", "Replicate", "FX graph", or "addmm" feel unfamiliar.

## What Are Shard, Replicate, and Partial?

These three words describe **where a tensor's data lives** across multiple GPUs. They are the core vocabulary of AutoParallel — every decision the optimizer makes boils down to choosing one of these for each tensor on each mesh dimension.

### The Setup

Say you have 4 GPUs and a weight matrix `W` with shape `[1024, 1024]`. You have three choices for how to distribute it.

---

### Replicate — Every GPU has a full copy

```
                     W = [1024, 1024]

GPU 0: [1024, 1024]  ← full copy
GPU 1: [1024, 1024]  ← full copy (same data)
GPU 2: [1024, 1024]  ← full copy (same data)
GPU 3: [1024, 1024]  ← full copy (same data)
```

**What it means**: Every GPU stores the **entire** tensor. All copies are identical.

**Memory cost**: 4× (one full copy per GPU). This is the most expensive in memory.

**Communication cost**: None for reading. But when you update this tensor (e.g., gradients during training), you need an **all-reduce** to keep all copies in sync.

**When it's useful**: Small tensors like biases, layer norm scales. The memory waste is tiny, and avoiding communication for reads is worth it.

**Analogy**: Like giving every student a full copy of the textbook. Everyone can read any page instantly, but you need 4 copies.

---

### Shard(dim) — Each GPU has a slice along one dimension

```
                     W = [1024, 1024]

Shard(0) — split along rows:

GPU 0: [256, 1024]   ← rows 0-255
GPU 1: [256, 1024]   ← rows 256-511
GPU 2: [256, 1024]   ← rows 512-767
GPU 3: [256, 1024]   ← rows 768-1023


Shard(1) — split along columns:

GPU 0: [1024, 256]   ← columns 0-255
GPU 1: [1024, 256]   ← columns 256-511
GPU 2: [1024, 256]   ← columns 512-767
GPU 3: [1024, 256]   ← columns 768-1023
```

**What it means**: The tensor is **sliced** along dimension `dim`. Each GPU holds only 1/N of the data.

**Memory cost**: 1/4 per GPU. This is the cheapest in memory — exactly what FSDP does with parameters.

**Communication cost**: When any GPU needs the **full** tensor (e.g., to do a matmul), it must **all-gather** the shards from all other GPUs. This costs communication bandwidth.

**When it's useful**: Large tensors like weight matrices. Saves memory at the cost of communication.

**Analogy**: Like tearing a textbook into 4 parts and giving each student one part. Saves paper, but if a student needs a page from someone else's section, they have to ask for it.

**The `dim` matters**: `Shard(0)` and `Shard(1)` split different dimensions. For a weight matrix in a linear layer:
- `Shard(0)` = row-parallel (split input features)
- `Shard(1)` = column-parallel (split output features)

These lead to completely different communication patterns, which is why the optimizer has to choose carefully.

---

### Partial — Each GPU has a partial result that needs to be combined

```
                     result = X @ W
                     (should be [32, 1024])

GPU 0: [32, 1024]   ← partial sum (not the real answer yet!)
GPU 1: [32, 1024]   ← partial sum (different values!)
GPU 2: [32, 1024]   ← partial sum (different values!)
GPU 3: [32, 1024]   ← partial sum (different values!)

               To get the real answer:
               real_result = GPU0 + GPU1 + GPU2 + GPU3
```

**What it means**: Each GPU has a tensor of the **same shape**, but with **different values**. The **real** result is the sum (or other reduction) of all GPUs' values. No single GPU has the correct answer yet.

**Memory cost**: 1× per GPU (same as Replicate in size, but the data is incomplete).

**Communication cost**: To turn a `Partial` into something usable, you need:
- **All-reduce** → every GPU gets the full summed result (`Partial` → `Replicate`)
- **Reduce-scatter** → each GPU gets a shard of the summed result (`Partial` → `Shard`)

**When it happens**: `Partial` appears naturally from math. When you split a matrix multiply:

```
Full matmul:     Y = X @ W           where W is [1024, 1024]

With Shard(0) on W (row-parallel):
  GPU 0 computes:  Y₀ = X[:, 0:256]   @ W[0:256, :]     ← partial product
  GPU 1 computes:  Y₁ = X[:, 256:512] @ W[256:512, :]    ← partial product
  GPU 2 computes:  Y₂ = X[:, 512:768] @ W[512:768, :]    ← partial product
  GPU 3 computes:  Y₃ = X[:, 768:]    @ W[768:, :]       ← partial product

  Real answer:     Y  = Y₀ + Y₁ + Y₂ + Y₃               ← need to sum!
```

Each GPU computed part of the dot product. The result is `Partial` — it has the right shape but wrong values until summed.

**Analogy**: Like 4 people each counting cars on a different road. Each person has a number, but the **total** traffic count requires adding all 4 numbers together.

---

### How They Work Together

In a real model, different tensors have different placements, and the optimizer inserts communication (collectives) when adjacent operations need different placements:

```
Example: FSDP Linear Layer

  weight: Shard(0)     ← each GPU has 1/4 of the weight (saves memory)
       │
       ▼
  all-gather            ← communication: reconstruct full weight
       │
       ▼
  weight: Replicate     ← now every GPU has the full weight
       │
       ▼
  mm(input, weight)     ← compute: each GPU does the full matmul
       │
       ▼
  output: Replicate     ← every GPU has the full output
```

```
Example: Tensor-Parallel Linear Layer

  input: Replicate      ← every GPU has the full input
  weight: Shard(1)      ← each GPU has 1/4 of weight columns
       │
       ▼
  mm(input, weight)      ← compute: each GPU does a partial matmul
       │
       ▼
  output: Shard(1)       ← each GPU has 1/4 of the output columns
```

```
Example: Row-Parallel with Partial

  input: Shard(1)        ← each GPU has 1/4 of input features
  weight: Shard(0)       ← each GPU has 1/4 of weight rows (matching)
       │
       ▼
  mm(input, weight)       ← compute: each GPU multiplies its slices
       │
       ▼
  output: Partial         ← each GPU has a partial sum
       │
       ▼
  all-reduce              ← communication: sum the partial results
       │
       ▼
  output: Replicate       ← now every GPU has the correct full output
```

### Multi-Dimensional Meshes

On a 2D mesh (e.g., 4×8 = 32 GPUs), each tensor has a placement **per mesh dimension**:

```
mesh = DeviceMesh("cuda", shape=(4, 8), names=("dp", "tp"))

A tensor with placement [Shard(0), Replicate()]:
  - Shard(0) on the "dp" dimension (4-way split along tensor dim 0)
  - Replicate() on the "tp" dimension (full copy across 8 GPUs)

  Result: 4 groups of 8 GPUs. Within each group of 8, the data is identical.
          Across groups, each group has a different 1/4 of the rows.
```

This is how AutoParallel combines FSDP and TP:
- `[Shard(0), Replicate()]` = FSDP on dim 0, no TP → parameter is sharded 4-way
- `[Replicate(), Shard(1)]` = no FSDP, TP on dim 1 → weight columns split 8-way
- `[Shard(0), Shard(1)]` = both FSDP and TP → weight is split 4×8 = 32-way

### Summary Table

| Placement | What each GPU has | Memory per GPU | To use the data | Common in |
|-----------|------------------|----------------|-----------------|-----------|
| `Replicate()` | Full identical copy | Full size | Just read it | Small tensors, biases |
| `Shard(dim)` | One slice along dim | 1/N of full size | All-gather to get full | FSDP params, TP weights |
| `Partial()` | Same shape, partial values | Full size | All-reduce or reduce-scatter to get correct values | After sharded matmuls |

### The Optimizer's Job in One Sentence

AutoParallel's ILP optimizer assigns `Replicate`, `Shard(dim)`, or `Partial` to every tensor in the graph, choosing the combination that minimizes total communication cost while respecting memory constraints.

---

## What Is a "Graph" in a PyTorch Model?

When you write a normal PyTorch model, you write **Python code**:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

PyTorch normally runs this in **eager mode** — each line executes immediately, one at a time. This is great for debugging, but it means PyTorch never sees the "big picture" of your model. It can't optimize across operations or reason about the whole computation at once.

A **graph** is a different representation. Instead of executing line by line, PyTorch records **what** operations happen and **how** data flows between them, producing a data structure like this:

```
 ┌───────────────┐
 │  placeholder  │  x: (32, 768)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  fc1 (linear) │  fc1(x): (32, 3072)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │     relu      │  relu(fc1): (32, 3072)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  fc2 (linear) │  fc2(relu): (32, 768)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │    output     │  return fc2
 └───────────────┘
```

Each box is a **node** — one operation in the graph. Each arrow (`│` / `▼`) is a **data dependency** — it shows which node's output feeds into the next node's input. The graph captures the full computation without executing it.

**Why does this matter for AutoParallel?** To decide how to shard each operation, AutoParallel needs to see all operations at once — their shapes, their connections, their costs. You can't do that in eager mode where operations run and disappear. The graph is the "blueprint" that the optimizer reasons about.

## What Is FX?

**FX** (short for "effects") is PyTorch's graph representation framework. It converts your `nn.Module` into a `GraphModule` — an `nn.Module` that holds an explicit graph of operations.

### FX Node Types

Every node in an FX graph has one of these types:

| Node Type | What It Is | Example |
|-----------|-----------|---------|
| `placeholder` | An input to the graph | `x` in `forward(self, x)` |
| `get_attr` | A parameter or buffer | `self.fc1.weight` |
| `call_function` | A function call | `torch.relu(x)`, `torch.ops.aten.mm(a, b)` |
| `call_method` | A method call on a tensor | `x.view(32, -1)` |
| `call_module` | A call to a submodule | `self.fc1(x)` |
| `output` | The return value | `return x` |

### A Concrete Example

If you trace the MLP above, the FX graph (simplified) looks like:

```python
# This is actual FX graph code you can print with gm.print_readable()

def forward(self, x):
    fc1_weight = self.fc1.weight                    # get_attr
    fc1_bias = self.fc1.bias                        # get_attr
    mm = torch.ops.aten.mm(x, fc1_weight.t())       # call_function
    add = torch.ops.aten.add(mm, fc1_bias)          # call_function
    relu = torch.ops.aten.relu(add)                 # call_function
    fc2_weight = self.fc2.weight                    # get_attr
    fc2_bias = self.fc2.bias                        # get_attr
    mm_1 = torch.ops.aten.mm(relu, fc2_weight.t())  # call_function
    add_1 = torch.ops.aten.add(mm_1, fc2_bias)      # call_function
    return (add_1,)                                 # output
```

Notice how `nn.Linear` was decomposed into lower-level operations (`mm` + `add`). This is important — keep reading.

## What Is Tracing?

**Tracing** is the process of converting your Python `forward()` method into an FX graph. PyTorch feeds fake inputs through your model and records every operation that happens.

### The Three Tracing Steps in AutoParallel

AutoParallel uses a sophisticated tracing pipeline:

**Step 1: TorchDynamo** captures the Python code into an FX graph. (See "What Is TorchDynamo?" below for the full explanation.)

**Step 2: AOTAutograd** (Ahead-of-Time Autograd) takes the forward graph and automatically generates the backward graph. It produces a **joint graph** containing both forward and backward operations. (See "What Is a Joint Graph?" below.)

**Step 3: Decomposition** breaks high-level operations (like `nn.Linear`, `addmm`) into lower-level primitive operations (like `mm`, `add`). This gives the optimizer more granular control over sharding. (See "What Is `addmm`?" below.)

### What Is TorchDynamo?

TorchDynamo is **PyTorch's Python-level tracer**. It's the technology behind `torch.compile()`. To understand why it exists, you need to understand the problem it solves.

#### The Problem: Python Is Hard to Trace

The simplest way to capture a graph is **symbolic tracing** (`torch.fx.symbolic_trace`). You feed a fake "proxy" tensor into the model, and every operation on it gets recorded. But this breaks on normal Python code:

```python
def forward(self, x):
    if x.shape[0] > 16:      # ❌ symbolic_trace can't handle this
        x = self.big_path(x)
    else:
        x = self.small_path(x)

    for layer in self.layers:  # ❌ loops over modules are tricky
        x = layer(x)

    x = x.to(self.dtype)      # ❌ accessing Python attributes during trace
    return x
```

`symbolic_trace` replaces `x` with a Proxy object. But `x.shape[0] > 16` tries to evaluate the Proxy as a real number — and crashes. Python `if`, `for`, list comprehensions, attribute access, data-dependent control flow: all of these break simple tracing.

#### The Solution: Bytecode Analysis

TorchDynamo works at a completely different level. Instead of running your Python code and hoping to record it, Dynamo **intercepts Python bytecode** before it runs.

Python compiles your `forward()` method into bytecode instructions (the low-level operations that the Python interpreter executes). Dynamo hooks into Python's frame evaluation mechanism and analyzes these bytecodes:

```
Your code:          x = torch.relu(self.fc1(x))

Python bytecode:    LOAD_FAST x
                    LOAD_ATTR fc1
                    CALL_FUNCTION 1
                    LOAD_GLOBAL torch
                    LOAD_ATTR relu
                    CALL_FUNCTION 1
                    STORE_FAST x

Dynamo sees:        "Load x, call fc1, call relu, store result"
                    → Records: relu(fc1(x)) into FX graph
```

#### How Dynamo Handles Control Flow

When Dynamo encounters an `if` statement, it evaluates the condition with the **actual values** it has (or shape information from fake tensors). If `x.shape[0]` is 32 and the condition is `> 16`, Dynamo knows this is `True` and traces only the true branch. It then creates a **guard**: "this graph is valid only when `x.shape[0] > 16`". If a future input violates the guard, Dynamo re-traces.

```python
def forward(self, x):
    if x.shape[0] > 16:      # Dynamo evaluates this → True for shape[0]=32
        x = self.big_path(x)  # Only this branch is traced
    return x

# Guard: graph valid when input.shape[0] > 16
# If called with shape[0]=8, Dynamo traces a new graph for the else branch
```

This means: the graph captures **one specific execution path**, not all possible paths. For AutoParallel, this is fine — the model is traced once with representative inputs, and the graph captures that path.

#### Dynamo vs. symbolic_trace: Summary

| Feature | `symbolic_trace` | TorchDynamo |
|---------|------------------|-------------|
| How it works | Runs code with Proxy tensors | Analyzes Python bytecode |
| Python `if/else` | Crashes | Evaluates condition, traces one branch |
| Python `for` loops | Sometimes works | Unrolls and traces |
| Attribute access | Limited | Full support |
| Data-dependent shapes | Crashes | Guards + re-tracing |
| Closures / free vars | Limited | Full support |
| Speed | Fast | Slower (bytecode analysis) |
| Used by | Simple models, tutorials | `torch.compile()`, AutoParallel |

#### Why AutoParallel Uses Dynamo

AutoParallel needs to trace **real-world models** (LLaMA-3, GPT-2, DeepSeek-V3) that use Python `if` statements, loops over layers, attribute access, and other features that `symbolic_trace` can't handle. Dynamo makes this possible.

The actual call in AutoParallel (`api.py`):
```python
# Dynamo captures the model into an FX graph
gm, guards = _dynamo_graph_capture_for_export(model, fake_inputs, decomp_table)
```

After Dynamo captures the forward FX graph, it's passed to AOTAutograd to generate the backward graph — producing the joint graph that the sharding optimizer works on.

### What Is AOTAutograd?

AOTAutograd stands for **Ahead-of-Time Autograd**. To understand it, you first need to understand how normal PyTorch autograd works — and why it's a problem for AutoParallel.

#### How Normal Autograd Works (Runtime)

In standard PyTorch, the backward pass is computed **at runtime**, on the fly:

```python
# Forward pass: PyTorch builds a "tape" as you go
x = torch.randn(32, 768, requires_grad=True)
y = model(x)          # Each op secretly records itself onto the tape
loss = y.sum()

# Backward pass: PyTorch replays the tape in reverse
loss.backward()        # Walks the tape backward, computing gradients
```

The "tape" (technically the autograd graph) is built dynamically during the forward pass and consumed during the backward pass. This is great for flexibility — any Python code works — but it means **PyTorch never has a complete picture of the backward pass ahead of time**.

```
Forward (runtime):   x → linear → relu → linear → loss
                     ↓ records tape on-the-fly ↓

Tape:  [linear_bwd, relu_bwd, linear_bwd, sum_bwd]

Backward (runtime):  Replays tape in reverse → gradients
```

#### The Problem for AutoParallel

AutoParallel's optimizer needs to see the **entire computation** — both forward and backward — as a graph **before** anything runs. It needs to ask questions like:

- "If I shard this forward `mm` with `Shard(1)`, what communication does the backward need?"
- "What's the total cost of this sharding strategy across both forward and backward?"
- "Can I overlap this backward reduce-scatter with that forward all-gather?"

But with normal autograd, the backward pass doesn't exist as a graph until runtime. There's nothing to analyze ahead of time.

#### The Solution: Generate Backward at Trace Time

AOTAutograd solves this by generating the backward graph **during tracing**, before any real computation happens:

```
Normal PyTorch:
  Trace time:   capture forward graph only
  Runtime:      run forward → build tape → run backward from tape

AOTAutograd:
  Trace time:   capture forward graph → generate backward graph → join them
  Runtime:      run the pre-compiled joint graph
```

Here's what AOTAutograd does step by step:

**Step 1: Take the forward graph** (from Dynamo)
```
forward graph:
  x → mm(x, w1) → relu → mm(relu, w2) → output
```

**Step 2: Symbolically differentiate it** — for each forward op, AOTAutograd knows the corresponding backward formula:
```
mm(A, B)        →  backward: grad_A = grad_out @ B.T,  grad_B = A.T @ grad_out
relu(X)         →  backward: grad_X = grad_out * (X > 0)
```

**Step 3: Chain them together** using the chain rule, producing backward ops:
```
backward graph:
  grad_out → mm(grad_out, w2.T) → relu_bwd → mm(relu_bwd, w1.T) → grad_x
                                             → mm(x.T, relu_bwd)  → grad_w1
             → mm(relu.T, grad_out)                                → grad_w2
```

**Step 4: Join forward + backward** into a single FX graph:
```
joint graph:
  # Forward
  x, w1, w2 = placeholders
  mm1 = mm(x, w1)
  r = relu(mm1)
  mm2 = mm(r, w2)

  # Backward
  grad_out = placeholder
  grad_mm2 = mm(grad_out, w2.T)
  grad_r = relu_backward(grad_mm2, mm1)
  grad_w1 = mm(x.T, grad_r)
  grad_x = mm(grad_r, w1.T)
  grad_w2 = mm(r.T, grad_out)
```

#### What "Saved for Backward" Means

Some backward formulas need values from the forward pass. For example:
- `relu_backward` needs to know **where** the input was positive → needs the forward input `mm1`
- `mm` backward needs the **other operand** → `grad_w1 = mm(x.T, grad_r)` needs `x` from forward

In normal autograd, these are "saved tensors" stored on the tape during forward. In AOTAutograd's joint graph, these are simply **edges** — the backward node references the forward node directly. This is why the joint graph must contain both: the backward nodes literally point to forward nodes for their inputs.

```
Forward:     x ──→ mm1 ──→ relu ──→ mm2 ──→ output
             │      │        │
             │      │        │  (saved for backward)
             ▼      ▼        ▼
Backward:  grad_w1  relu_bwd  grad_mm2 ──→ grad_x
```

#### Why This Matters for AutoParallel

With the joint graph, AutoParallel's ILP optimizer can reason about the **full training step**:

1. **Coupled decisions**: Sharding `w1` as `Shard(1)` in the forward `mm` means `grad_w1 = mm(x.T, grad_r)` produces a `Partial()` result that needs a reduce-scatter. The optimizer sees both costs.

2. **Overlap opportunities**: An FSDP all-gather of `w2` in the forward can overlap with computation on `w1`. The optimizer sees both operations and their timing.

3. **Memory vs. communication trade-offs**: Saving `mm1` for backward (instead of recomputing it) costs memory. Recomputing it costs FLOPs. The activation checkpointing pass sees the full picture.

#### The Actual Call in AutoParallel

```python
# From api.py — the key function call
joint_graph = aot_export_joint_with_descriptors(
    forward_graph,        # From Dynamo
    sample_inputs,
    decompositions=decomp_table,  # addmm → mm + add, etc.
)
# joint_graph now contains forward + backward as one FX GraphModule
```

#### AOTAutograd vs. Normal Autograd: Summary

| Aspect | Normal Autograd | AOTAutograd |
|--------|-----------------|-------------|
| When backward is created | At runtime (during `.backward()`) | At trace time (before any computation) |
| Backward representation | Dynamic tape (autograd graph) | Static FX graph |
| Can optimize backward? | No — it doesn't exist yet | Yes — it's a graph you can analyze |
| Can see forward + backward together? | No | Yes — joint graph |
| Used by | Normal PyTorch training | `torch.compile()`, AutoParallel |
| Flexibility | Any Python code | Static graph (one execution path) |

### What Is the Meta Device?

In PyTorch, every tensor lives on a **device**: `cpu`, `cuda` (GPU), or `meta`.

The **meta device** is a special "virtual" device where tensors have **shapes and dtypes but no actual data**. No memory is allocated — not on the CPU, not on the GPU, nowhere. The tensor is just a description of what it *would* be.

```python
# Normal tensor on CPU — allocates 4 MB of RAM
cpu_tensor = torch.randn(1024, 1024)           # 1024 × 1024 × 4 bytes = 4 MB

# Normal tensor on GPU — allocates 4 MB of VRAM
gpu_tensor = torch.randn(1024, 1024, device="cuda")  # 4 MB on GPU

# Meta tensor — allocates ZERO bytes anywhere
meta_tensor = torch.randn(1024, 1024, device="meta")  # 0 bytes, just metadata
```

You can check a meta tensor's properties:
```python
meta_tensor.shape       # torch.Size([1024, 1024])  ✓ shape works
meta_tensor.dtype       # torch.float32              ✓ dtype works
meta_tensor.device      # device(type='meta')        ✓ device works
meta_tensor[0, 0]       # ERROR! No data to read     ✗ can't access values
meta_tensor + 1         # ERROR! No data to compute  ✗ can't do math
```

#### Why Meta Device Matters for AutoParallel

A model like LLaMA-3 70B has 70 billion parameters. In float32, that's **280 GB** — more than any single GPU can hold. You can't even *create* this model on a normal device without a cluster.

But with the meta device:

```python
# This works on a laptop with 8 GB RAM!
with torch.device("meta"):
    model = LLaMA3_70B()    # 0 bytes allocated

# Every parameter exists as metadata only
for name, param in model.named_parameters():
    print(name, param.shape, param.device)
    # "layers.0.attn.q_proj.weight torch.Size([8192, 8192]) meta"
    # "layers.0.attn.k_proj.weight torch.Size([1024, 8192]) meta"
    # ... all on meta device, zero memory used
```

AutoParallel creates models on the meta device, traces them (using fake tensors — see below), runs the ILP optimizer to find optimal sharding, and only allocates real GPU memory **at the very end** when training is about to begin:

```python
# Step 1: Create on meta device (0 bytes)
with torch.device("meta"):
    model = MyModel()

# Step 2: AutoParallel optimizes sharding (still 0 bytes)
parallel_model = auto_parallel(model, mesh, sample_inputs)

# Step 3: NOW allocate real memory (only each GPU's shard)
parallel_model.to_empty(device="cuda")   # allocate storage
parallel_model.init_weights()             # fill with values
```

The key insight: at step 3, each GPU only allocates memory for **its shard** of the parameters (e.g., 1/8 of the model), not the full model. The meta device lets you plan the distribution before committing any memory.

#### Meta Device vs. Fake Tensors

These are related but different:

| | Meta Device | Fake Tensors |
|---|---|---|
| **What** | A device type (`device="meta"`) | A tensor wrapper (FakeTensorMode) |
| **Data storage** | No storage at all | No storage, but *pretends* to be on a real device |
| **Operations** | Crash if you try to compute | Succeed! Return fake results with correct shapes |
| **Purpose** | Create models without memory | Trace models without memory |

Meta tensors can't be used in operations — they crash. That's fine for creating a model, but not for *tracing* it (which requires running the forward pass). That's where fake tensors come in.

### What Are Fake Tensors?

**Fake tensors** extend the meta device idea to support **computation**. They have shapes and dtypes like meta tensors, but they can actually participate in operations — the operations don't compute real values, they just figure out **what shape the output would be**.

```python
# Meta tensor: can't compute
meta_a = torch.randn(32, 768, device="meta")
meta_a + 1    # ❌ ERROR — no data to add to

# Fake tensor: CAN compute (but only shapes, not values)
from torch._subclasses import FakeTensorMode
with FakeTensorMode():
    fake_a = torch.randn(32, 768, device="cuda")  # 0 bytes, but pretends to be on CUDA
    fake_b = fake_a + 1                            # ✓ works! fake_b.shape = (32, 768)
    fake_c = fake_a @ torch.randn(768, 3072, device="cuda")  # ✓ works! fake_c.shape = (32, 3072)
    # No actual computation happened. No GPU memory used.
    # PyTorch just propagated the shapes through each operation.
```

**Why AutoParallel needs fake tensors**: Tracing requires running the model's `forward()` method to record all operations. With fake tensors, the forward pass executes but only propagates shapes — no real computation, no GPU memory. This is how AutoParallel traces a 70B model on a single machine.

The flow:
```
1. Create model on meta device              (0 bytes — just parameter shapes)
2. Convert parameters to fake tensors       (0 bytes — but now ops work)
3. Run forward() with fake inputs           (0 bytes — records FX graph)
4. AOTAutograd generates backward            (0 bytes — more graph nodes)
5. ILP optimizer runs on the graph           (only CPU memory for the solver)
6. to_empty(device="cuda") + init_weights()  (NOW real GPU memory is used)
```

## What Is a Process Group?

When you train on multiple GPUs, those GPUs need a way to find each other and communicate. A **process group** is the object that makes this possible — it's the "phone network" that connects GPUs together.

### One GPU = One Process

In distributed training, each GPU runs its own copy of the Python program as a separate **process**. If you have 8 GPUs, you have 8 Python processes running simultaneously:

```
Process 0 (GPU 0): python train.py    ← rank 0
Process 1 (GPU 1): python train.py    ← rank 1
Process 2 (GPU 2): python train.py    ← rank 2
...
Process 7 (GPU 7): python train.py    ← rank 7
```

Each process has a **rank** (its ID number, starting from 0) and knows the **world size** (total number of processes). These processes need a way to talk to each other — that's the process group.

### What a Process Group Does

A process group handles three things:

1. **Discovery** — "Who else is in this training run? What's their address?"
2. **Communication** — "Send this tensor to GPU 3" or "All-reduce this gradient across all GPUs"
3. **Synchronization** — "Wait until everyone has finished this step before continuing"

```python
import torch.distributed as dist

# Initialize the process group — every process calls this
dist.init_process_group(
    backend="nccl",      # Communication library (NCCL for NVIDIA GPUs)
    rank=my_rank,        # This process's ID (0, 1, 2, ...)
    world_size=8,        # Total number of processes
)

# Now all 8 processes can communicate
dist.all_reduce(tensor)   # Sum tensor across all 8 GPUs
dist.barrier()             # Wait for all 8 processes to reach this point
```

### Backends: How GPUs Actually Talk

The `backend` parameter tells PyTorch which communication library to use:

| Backend | Hardware | Speed | When to use |
|---------|----------|-------|-------------|
| `"nccl"` | NVIDIA GPUs (NVLink, NVSwitch, InfiniBand) | Fastest for GPUs | Real GPU training |
| `"gloo"` | CPU, or GPU fallback | Moderate | CPU training, debugging |
| `"fake"` | Nothing — simulates communication | Instant (no-op) | Testing, optimization planning |

### Subgroups: Not Everyone Talks to Everyone

A process group can contain a **subset** of GPUs. This is how mesh dimensions work:

```
8 GPUs total, arranged as mesh (2, 4):

  Row groups (TP):           Column groups (FSDP):
  [GPU 0, 1, 2, 3]          [GPU 0, 4]
  [GPU 4, 5, 6, 7]          [GPU 1, 5]
                             [GPU 2, 6]
                             [GPU 3, 7]
```

When GPU 0 does an all-gather on the TP dimension, it only communicates with GPUs 1, 2, 3 (its row group) — not all 8. Each subgroup has its own process group:

```python
mesh = DeviceMesh("cuda", [[0,1,2,3], [4,5,6,7]], mesh_dim_names=("dp","tp"))

# All-gather on TP dimension: GPU 0 talks to GPUs 1,2,3 only
# All-reduce on DP dimension: GPU 0 talks to GPU 4 only
```

### Fake Process Groups: Why AutoParallel Doesn't Need Real GPUs

AutoParallel's optimizer only needs tensor **shapes** and **costs** — it never moves real data. So it uses a **fake process group** that pretends to be a distributed setup without any actual GPUs:

```python
from torch.distributed import init_process_group

# This creates a "pretend" 256-GPU cluster on your single machine
init_process_group(backend="fake", world_size=256)
mesh = DeviceMesh("cuda", torch.arange(256).reshape(32, 8))

# AutoParallel can now plan sharding for a 32×8 mesh
# without having 256 actual GPUs
```

No communication happens. No GPUs are used. The fake process group just provides the **metadata** (world size, ranks, group structure) that AutoParallel needs to enumerate sharding options and run the ILP solver.

This is how AutoParallel's test suite works — `conftest.py` creates a fake process group with world_size=256, so all tests run on a single machine:

```python
# From tests/conftest.py
fake_store = FakeStore()
torch.distributed.init_process_group("fake", store=fake_store, rank=0, world_size=256)
```

### Real Process Groups: When Training Actually Happens

When you're done planning and ready to train for real, you use a real process group with `torchrun`:

```bash
# This launches 4 processes (one per GPU), each with a real NCCL process group
torchrun --standalone --nproc-per-node 4 train.py
```

Inside `train.py`, `init_process_group(backend="nccl")` sets up real NCCL communication. Now the collectives that AutoParallel inserted into the graph (all-gather, reduce-scatter, etc.) actually move data between GPUs.

### The Full Picture

```
Planning phase (fake process group):
  init_process_group("fake", world_size=256)
  → Create mesh → AutoParallel optimizer → sharding plan
  → No GPUs needed, runs on laptop

Training phase (real process group):
  torchrun --nproc-per-node 8 train.py
  → init_process_group("nccl")
  → Create mesh → Apply sharding plan → Train
  → Real GPUs, real communication
```

### Quick Reference

| Term | Meaning |
|------|---------|
| **Process** | One Python program running on one GPU |
| **Rank** | A process's ID number (0, 1, 2, ...) |
| **World size** | Total number of processes |
| **Process group** | The communication channel connecting processes |
| **Backend** | The library that implements communication (NCCL, Gloo, Fake) |
| **Subgroup** | A process group containing a subset of GPUs (e.g., one mesh row) |
| **Fake process group** | A no-op process group for testing/planning without GPUs |

## What Is a Joint Graph?

A normal forward pass through a model produces outputs. The backward pass computes gradients. These are usually separate:

```
Forward:   input → [layer1] → [layer2] → [layer3] → output
Backward:  grad_output → [layer3_bwd] → [layer2_bwd] → [layer1_bwd] → grad_input
```

A **joint graph** combines both into a single FX graph:

```
Joint graph:
    # === Forward ===
    x = placeholder
    w1 = get_attr(layer1.weight)
    mm1 = aten.mm(x, w1)                                      # forward op
    relu1 = aten.relu(mm1)                                    # forward op
    w2 = get_attr(layer2.weight)
    mm2 = aten.mm(relu1, w2)                                  # forward op
    output = mm2

  # === Backward ===
    grad_out = placeholder                                    # gradient from loss
    grad_mm2 = aten.mm(grad_out, w2.t())                      # backward of mm2
    grad_relu = aten.threshold_backward(grad_mm2, mm1, 0)     # backward of relu
    grad_mm1 = aten.mm(grad_relu, w1.t())                     # backward of mm1
    grad_w1 = aten.mm(x.t(), grad_relu)                       # weight gradient for layer1
    grad_w2 = aten.mm(relu1.t(), grad_out)                    # weight gradient for layer2
```

### Why Does AutoParallel Need a Joint Graph?

Because **forward and backward sharding decisions are coupled**:

- If you shard a weight matrix with `Shard(1)` in the forward pass (column-parallel), the backward pass needs a `reduce-scatter` to collect the gradient
- If you choose `Replicate()` in the forward, the backward is also replicated — no communication needed but more memory used

The optimizer must see both forward and backward operations simultaneously to make globally optimal decisions. An FSDP all-gather in the forward has a corresponding reduce-scatter in the backward — the optimizer balances both costs together.

## What Is `addmm`? Why Decompose It?

### The Operation

`addmm` stands for **"add matrix-matrix"**. It's a fused operation:

```python
# addmm(bias, input, weight) = bias + (input @ weight)
result = torch.addmm(bias, input, weight)

# This is what nn.Linear does internally:
# output = input @ weight.T + bias
# which is: addmm(bias, input, weight.T)
```

It's a single CUDA kernel that computes the matrix multiply and bias addition together — faster than doing them separately.

### The Problem for Parallelism

When `addmm` is a single operation in the graph, the sharding optimizer sees this:

```
addmm(bias, input, weight) → output
```

It must choose **one** sharding strategy for the entire fused operation. But tensor parallelism needs to treat the matrix multiply and the bias addition differently:

**Column-parallel** (shard weight on columns):
```
input: [32, 768]  Replicate    (every GPU has full input)
weight: [768, 3072]  Shard(1)  (each GPU has columns: [768, 384])
output: [32, 3072]  Shard(1)   (each GPU has part of output: [32, 384])
bias: [3072]  Shard(0)         (each GPU has matching bias slice: [384])
```

**Row-parallel** (shard weight on rows):
```
input: [32, 768]  Shard(1)     (each GPU has part of input: [32, 96])
weight: [768, 3072]  Shard(0)  (each GPU has rows: [96, 3072])
output: [32, 3072]  Partial    (each GPU has a partial sum, needs reduce)
bias: [3072]  Replicate        (add after reduce)
```

The key insight: the `mm` and the `add` need **different** sharding. The `mm` output might be `Partial()` (needs reduction), but the `bias` add happens **after** reduction. As a fused `addmm`, the optimizer can't express this.

### The Decomposition

AutoParallel's `tracing.py` replaces `addmm` with separate `mm` + `add`:

```python
# Before decomposition (fused):
addmm(bias, input, weight)   # one node, one sharding decision

# After decomposition (separate):
mm_result = mm(input, weight)  # node 1: can shard the matmul
output = add(mm_result, bias)  # node 2: can shard the addition independently
```

Now the optimizer can:
1. Choose column-parallel for the `mm` → output is `Shard(1)`
2. Choose matching `Shard(0)` for the bias in the `add` → no communication needed

The actual code is simple:

```python
# From autoparallel/tracing.py
def addmm_decomp(self, mat1, mat2, beta=1, alpha=1):
    return self + mat1 @ mat2
```

### What About Performance?

Yes, fused `addmm` is faster than separate `mm` + `add` on a single GPU. But in distributed training, the ability to shard the matmul independently is worth far more than the fusion benefit. And when the graph is later compiled with Inductor (`torch.compile`), the compiler can re-fuse them if they end up on the same device.

## Other Important Decompositions

AutoParallel's decomposition table makes several other choices:

### Operations That Are PRESERVED (not decomposed)

These ops have **custom sharding rules** in AutoParallel, so decomposing them would lose important information:

| Operation | Why preserved |
|-----------|---------------|
| `native_layer_norm` | Has a custom rule that knows norm dims can't be sharded |
| `softmax` / `_softmax` | Has a custom rule for attention sharding |
| `embedding_dense_backward` | Has a custom rule for vocabulary parallelism |
| `stack` | Has a custom rule in DTensor |

If these were decomposed into simpler ops, AutoParallel would lose the semantic understanding of what they do and make worse sharding decisions.

### Operations That ARE Decomposed

Everything else uses Inductor's standard decomposition table, which breaks complex operations into ~2000 primitive "ATen" operations. This gives the optimizer maximum granularity.

## How Alias Nodes Help the Optimizer

After tracing, AutoParallel inserts `aten.alias` nodes at every point where a tensor is used by multiple downstream operations:

```
# Before aliases:
mm_result → relu (consumer 1)
mm_result → save_for_backward (consumer 2)

# After aliases:
mm_result → alias_1 → relu (consumer 1)
mm_result → alias_2 → save_for_backward (consumer 2)
```

**Why?** Without aliases, both consumers are forced to use the same sharding placement (because they read the same tensor). With aliases, the optimizer can choose different placements for each consumer and insert a redistribution between them if needed. This gives the ILP more freedom to find the global optimum.

An alias is a zero-cost identity operation — it doesn't copy data. It's purely a graph-level trick to give the optimizer more decision points.

## Putting It All Together

Here's the complete tracing pipeline for a simple 2-layer MLP:

```
Step 1: User writes nn.Module
    forward(x) → fc1(x) → relu → fc2 → output

Step 2: TorchDynamo traces Python → FX graph
    placeholder(x) → call_module(fc1) → call_function(relu) → call_module(fc2) → output

Step 3: Decomposition lowers to ATen ops
    placeholder(x) → mm(x, w1.T) → add(mm, b1) → relu(add) → mm(relu, w2.T) → add(mm, b2) → output
                      ^^^^^^^^^^^^^^^^^^^^^^^^
                      This was addmm, now split!

Step 4: AOTAutograd adds backward ops → joint graph
    [forward ops] + [backward ops for all mm, add, relu]

Step 5: Alias insertion at fan-out points
    mm → alias_1 → relu
    mm → alias_2 → backward_save

Step 6: Ready for the sharding optimizer!
    Each node now has independent sharding options.
    The ILP picks the globally optimal assignment.
```

## What Is Activation Checkpointing?

Activation checkpointing (also called **gradient checkpointing** or **rematerialization**) is a technique to **trade compute for memory**. It lets you train models that would otherwise not fit in GPU memory, at the cost of doing some extra computation.

> **The name is confusing.** "Activation checkpointing" sounds like it means "save all activations." It's actually the opposite — it means **drop most activations and save only a few as "checkpoints"** (recovery points). During backward, you restart from the nearest checkpoint and recompute what you dropped. Think of it like save points in a video game: you don't record every frame, you just save at key moments and replay from there when needed.

### The Problem: Activations Eat Memory

During the forward pass, every intermediate result (called an **activation**) must be kept in memory because the backward pass needs it to compute gradients:

```
Forward pass of a 4-layer model:

  input → [Layer 1] → act1 → [Layer 2] → act2 → [Layer 3] → act3 → [Layer 4] → output
                       save     save       save     save       save     save

  Memory: act1 + act2 + act3 + output = 4 activations stored
```

Why does backward need these? Remember from the AOTAutograd section: the backward formula for an operation usually needs the **inputs** from the forward pass. For example:
- `relu_backward(grad, input)` needs the forward input to know where values were positive
- `mm_backward(grad, A, B)` needs both operands from the forward `mm(A, B)`

For a large model like LLaMA-3 70B with long sequences, these saved activations can consume **more memory than the model parameters themselves**. On a 80 GB H100, you might have:
- Model parameters: 35 GB (after FSDP sharding)
- Optimizer states: 20 GB
- Activations: 50+ GB ← doesn't fit!

### The Solution: Forget and Recompute

The idea is simple: **don't save some activations during forward. Recompute them during backward when you need them.**

```
WITHOUT activation checkpointing:

  Forward:   input → L1 → act1 → L2 → act2 → L3 → act3 → L4 → output
                          [save]       [save]       [save]
  Memory:    4 activations saved

  Backward:  Uses saved act1, act2, act3 to compute gradients


WITH activation checkpointing (checkpoint every 2 layers):

  Forward:   input → L1 → act1 → L2 → act2 → L3 → act3 → L4 → output
                          [DROP]       [save]       [DROP]
  Memory:    Only act2 saved (checkpoint boundary)

  Backward:
    Phase 1: Recompute L3, L4 from act2    ← extra forward work!
             Now we have act3 again → compute grads for L3, L4
    Phase 2: Recompute L1, L2 from input   ← extra forward work!
             Now we have act1 again → compute grads for L1, L2
```

We cut memory from 4 activations to 1, but we ran some forward operations **twice**.

### The Trade-Off Visualized

Memory and compute move in **opposite directions** — saving one costs the other:

```
                          Memory usage              Extra compute
                          ──────────────            ──────────────
  No checkpointing        ████████████████  100%    ░               0%
  Checkpoint every 2 lyrs ████████          50%     ████           33%
  Checkpoint every layer  ████              25%     ████████      100%
```

- **Left column (memory)**: bars shrink as you checkpoint more — fewer activations stored
- **Right column (compute)**: bars grow as you checkpoint more — more recomputation during backward

The sweet spot depends on your GPU: if you have plenty of memory, don't checkpoint. If memory is tight, checkpoint more aggressively.

### How It Works in PyTorch

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SelfAttention(...)
        self.ffn = FeedForward(...)

    def forward(self, x):
        # Without checkpointing:
        # x = self.attn(x)
        # x = self.ffn(x)

        # With checkpointing: activations inside attn and ffn are NOT saved
        # They'll be recomputed during backward
        x = checkpoint(self.attn, x, use_reentrant=False)
        x = checkpoint(self.ffn, x, use_reentrant=False)
        return x
```

The `checkpoint()` wrapper tells PyTorch: "Run this function normally during forward, but **don't save** the intermediate activations. During backward, re-run the forward to recreate them."

### What Gets Saved vs. Recomputed?

The choice of what to save and what to recompute is called the **checkpointing policy**. Common strategies:

| Strategy | What's saved | Memory | Extra compute |
|----------|-------------|--------|---------------|
| No checkpointing | Everything | Maximum | None |
| Checkpoint per layer | Only layer inputs | ~√N | ~1× forward |
| Checkpoint per block | Block boundaries | Moderate | Moderate |
| Selective (per-op) | Cheap ops recomputed, expensive ops saved | Fine-tuned | Minimal |

**Selective checkpointing** is the most advanced: instead of checkpointing entire layers, you choose per-operation. For example:
- **Recompute** dropout, relu, add (cheap — microseconds)
- **Save** matmul outputs, attention scores (expensive to recompute)

### How AutoParallel Handles Activation Checkpointing

AutoParallel has its own activation checkpointing system in `graph_passes/activation_checkpointing.py`. Because it operates on the joint graph (forward + backward in one FX graph), it can make fine-grained decisions:

**1. User-level checkpointing**: If you wrapped layers with `torch.utils.checkpoint` in your model, AutoParallel preserves those decisions. The wrapped operations are tagged as `MUST_RECOMPUTE`.

**2. FSDP all-gather recomputation**: When using FSDP with `reshard_after_forward=True`, the all-gathered parameters are freed after forward to save memory. During backward, AutoParallel recomputes (re-all-gathers) them. This is tagged automatically.

**3. Staged recomputation (`ac_joint_pass`)**: AutoParallel's own AC pass uses a heuristic — it estimates peak memory as `√(total activations)` and stages recomputation to stay within that budget. It marks operations as `PREFER_RECOMPUTE` and lets the compiler decide:

```
Joint graph:

  Forward:   mm → relu → mm → relu → mm → output
              │           │           │
              ▼           ▼           ▼
  Tags:    PREFER     PREFER     MUST_SAVE
           RECOMPUTE  RECOMPUTE

  Backward:  ◄── recompute relu, mm from saved checkpoint ──►
```

**4. What makes this different from manual checkpointing**: In manual checkpointing, you decide at the Python level which layers to wrap. AutoParallel decides at the **graph operation level** — it can choose to recompute one specific relu but save the matmul in the same layer. This is more fine-grained and can find better memory/compute trade-offs.

### A Concrete Example

For a Transformer layer with attention + FFN:

```
Forward:
  q = mm(x, Wq)           ← SAVE (expensive to recompute, needed for attention backward)
  k = mm(x, Wk)           ← SAVE
  v = mm(x, Wv)           ← SAVE
  attn = sdpa(q, k, v)    ← SAVE (very expensive)
  proj = mm(attn, Wo)     ← PREFER_RECOMPUTE (can be recomputed from saved attn)
  add1 = x + proj         ← PREFER_RECOMPUTE (cheap)
  norm = layer_norm(add1) ← PREFER_RECOMPUTE (cheap)
  ff1 = mm(norm, W1)      ← PREFER_RECOMPUTE (can be recomputed from saved norm input)
  relu = relu(ff1)        ← PREFER_RECOMPUTE (very cheap)
  ff2 = mm(relu, W2)      ← PREFER_RECOMPUTE (can be recomputed from saved relu)
  add2 = add1 + ff2       ← SAVE (checkpoint boundary)
```

Memory saved: instead of storing all 10 intermediate tensors, only ~4-5 are kept. The rest are recomputed from those checkpoints during backward.

### Why It Matters for Distributed Training

Activation checkpointing is **especially important** when combined with parallelism:

- **FSDP already saves parameter memory** by sharding weights. But activations are **not sharded** (each GPU computes on the full batch slice). AC reduces activation memory.

- **Tensor parallelism** reduces activation memory somewhat (each GPU sees a slice). But the attention scores in SDPA are still full-sized. AC helps.

- **The combination**: FSDP (shard params) + TP (shard activations partially) + AC (recompute remaining activations) is often necessary to fit large models in memory. AutoParallel optimizes all three together.

```
Memory breakdown for a large model:

  Without AC:    [params: 35 GB] + [optimizer: 20 GB] + [activations: 50 GB] = 105 GB  ❌ doesn't fit 80 GB
  With AC:       [params: 35 GB] + [optimizer: 20 GB] + [activations: 15 GB] = 70 GB   ✓ fits!
```

## What Is `local_map`?

`local_map` is a PyTorch utility that lets you write a function that operates on **local (already-sharded) tensors** and tell PyTorch what the input and output placements are. It's an escape hatch for operations that AutoParallel can't auto-shard.

### The Problem: Some Ops Can't Be Auto-Sharded

AutoParallel's optimizer has sharding rules for ~30 common operations (matmul, relu, layer norm, etc.). But some operations don't have rules — or the correct sharding requires domain knowledge the optimizer doesn't have. Examples:

- **Context-parallel attention**: Sharding SDPA on the sequence dimension requires knowing that causal masks need special handling
- **Expert routing in MoE**: Tokens are sent to different experts on different GPUs — the routing logic is model-specific
- **Custom operations**: Any operation you wrote yourself

For these, you need a way to tell PyTorch: "I know how to shard this. Here's what the inputs and outputs look like. Just trust me."

### How `local_map` Works

`local_map` is a decorator. You annotate a function with the DTensor placements of its inputs and outputs:

```python
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Shard, Replicate

@local_map(
    out_placements=((Shard(0),),),            # output is sharded on dim 0
    in_placements=((Shard(0),), (Shard(0),)), # both inputs sharded on dim 0
    redistribute_inputs=True,                  # redistribute inputs if needed
    device_mesh=mesh,
)
def my_custom_op(x, y):
    # Inside here, x and y are LOCAL tensors (just this GPU's shard)
    # You write normal single-GPU code
    return x + y
```

The key idea: **inside the function, you work with regular tensors** (the local shard on this GPU). **Outside the function, PyTorch sees DTensors** with the placements you declared. `local_map` handles the translation.

### A Visual Walkthrough

Say you have 4 GPUs and a tensor `x` of shape `[1024, 768]` with placement `Shard(0)`:

```
OUTSIDE local_map (DTensor world):
  x is a DTensor: shape [1024, 768], placement Shard(0)
  GPU 0 has x[0:256],  GPU 1 has x[256:512],
  GPU 2 has x[512:768], GPU 3 has x[768:1024]

         │
         ▼  local_map unwraps the DTensor

INSIDE local_map (local tensor world):
  x is a plain tensor: shape [256, 768]  ← just this GPU's shard
  You write normal PyTorch code as if it's a single GPU

         │
         ▼  local_map re-wraps the result

OUTSIDE local_map (DTensor world):
  result is a DTensor with the placement you declared in out_placements
```

### Real Example: Context-Parallel Attention

From `examples/example_local_map.py` — this shards attention across GPUs on the sequence dimension, the batch dimension, and the head dimension:

```python
@local_map(
    out_placements=((Shard(0), Shard(1), Shard(2)),),   # output: shard batch, heads, sequence
    in_placements=(
        (Shard(0), Shard(1), Shard(2)),   # query:  shard batch, heads, sequence
        (Shard(0), Shard(1), Shard(2)),   # key:    shard batch, heads, sequence
        (Shard(0), Shard(1), Shard(2)),   # value:  shard batch, heads, sequence
    ),
    redistribute_inputs=True,
    device_mesh=mesh,
)
def context_parallel_attention(query, key, value):
    # Inside here: q, k, v are local tensors — just this GPU's slice
    # of the batch, heads, and sequence
    out = nn.functional.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return out
```

Each GPU computes attention on its own slice of the sequence. No all-gather, no all-reduce — each GPU independently computes its portion. This is context parallelism, and AutoParallel can't auto-discover it (SDPA sequence sharding is disabled due to upstream bugs), so `local_map` lets you do it manually.

### Real Example: Sharded Pointwise Op

```python
@local_map(
    out_placements=((Shard(0), Shard(0), Replicate()),),
    in_placements=((Shard(0), Shard(0), Replicate()),),
    redistribute_inputs=True,
    device_mesh=mesh,
)
def sharded_pointwise(x):
    return x + 10   # Each GPU adds 10 to its local shard
```

This tells PyTorch: "the input is sharded on dim 0 across the first two mesh dimensions and replicated on the third. The output has the same placement. The function just adds 10 element-wise." No communication needed.

### How `local_map` Interacts with AutoParallel

When AutoParallel traces a model that uses `local_map`, it treats the `local_map` function as an **opaque block** with known input/output placements:

```
AutoParallel sees:
  ... → [node A: Replicate] → [local_map: in=Shard(0), out=Shard(0)] → [node B: ???] → ...
                                     ▲                      ▲
                                     │                      │
                          Placement is fixed         Placement is fixed
                          (you declared it)          (you declared it)
```

The optimizer doesn't look inside the `local_map` — it trusts your placement declarations. It **does** optimize the surrounding operations and inserts collectives to redistribute tensors into the placements that your `local_map` expects.

For example, if the node before `local_map` is `Replicate()` but your `local_map` wants `Shard(0)`, AutoParallel will insert a slice (or redistribute) before the call.

### When to Use `local_map`

| Situation | Use `local_map`? |
|-----------|-----------------|
| Standard ops (matmul, relu, layer norm) | No — AutoParallel handles these |
| Context-parallel attention | Yes — auto-sharding is disabled for SDPA sequence dim |
| MoE expert routing | Yes — token routing is model-specific |
| Custom CUDA/Triton kernels | Yes — no auto-sharding rules exist |
| Complex resharding patterns | Yes — when you know the optimal placement |

### `local_map` vs `with_sharding_constraint`

These solve different problems:

| | `local_map` | `with_sharding_constraint` |
|---|---|---|
| **Purpose** | Write custom sharded ops | Force a specific placement on a tensor |
| **What you write** | A function with local tensor code | Nothing — just declares a placement |
| **Optimizer sees** | Opaque block with fixed placements | A constraint the ILP must respect |
| **When to use** | Ops the optimizer can't handle | Override optimizer's choice for a specific tensor |

```python
# with_sharding_constraint: "I want this tensor to be Shard(1) on TP dim"
x = with_sharding_constraint(x, mesh, (Replicate(), Shard(1)))

# local_map: "Here's a whole function that expects sharded inputs"
@local_map(out_placements=..., in_placements=..., device_mesh=mesh)
def my_op(x):
    return custom_kernel(x)
```

## What Is SDPA (Scaled Dot-Product Attention)?

SDPA stands for **Scaled Dot-Product Attention**. It's the core mathematical operation inside every Transformer model — the thing that lets the model "pay attention" to different parts of the input.

### The Intuition

Imagine reading a sentence: "The cat sat on the mat because **it** was tired."

What does "it" refer to? To answer that, your brain looks back at every previous word and decides which ones are most relevant. That's attention — each word "attends to" other words to understand context.

In a Transformer, this is done with three vectors for each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

### The Math

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

Step by step:

```
1. Q × Kᵀ              Dot product: how much does each query match each key?
                        Shape: (seq_len, seq_len) — every token scores against every other

2. / √d                 Scale down by √(head_dimension) to prevent huge values
                        that would make softmax saturate

3. softmax(...)         Convert scores to probabilities (0 to 1, sum to 1)
                        Each token now has a probability distribution over all other tokens

4. × V                  Weighted sum: combine values using the attention probabilities
                        Shape: (seq_len, head_dim) — the attention output
```

### A Visual Example

```
Input: "The cat sat"     (3 tokens, each has Q, K, V vectors)

Step 1: Q × Kᵀ = attention scores
                    Key:
                    The   cat   sat
  Query:  The  [  0.8   0.1   0.1 ]   "The" mostly attends to itself
          cat  [  0.3   0.5   0.2 ]   "cat" attends to itself and "The"
          sat  [  0.2   0.6   0.2 ]   "sat" attends mostly to "cat"

Step 2: Scale by √d (just divides all numbers)

Step 3: Softmax (makes each row sum to 1.0)

Step 4: Multiply by V → weighted combination of value vectors
        "sat" gets 60% of "cat"'s information + 20% of others
```

### Multi-Head Attention

Instead of one set of Q, K, V, Transformers use **multiple heads** — each head independently computes attention and then the results are concatenated:

```
Hidden dim = 768, Heads = 12 → Head dim = 64

Head 0: Q₀, K₀, V₀ → Attention₀ (64 dims)   ← maybe learns syntax
Head 1: Q₁, K₁, V₁ → Attention₁ (64 dims)   ← maybe learns semantics
...
Head 11: Q₁₁, K₁₁, V₁₁ → Attention₁₁ (64 dims)

Concatenate: [Attention₀ | Attention₁ | ... | Attention₁₁] = 768 dims
```

This is why the code in `example_autoparallel.py` reshapes tensors:

```python
# (batch, seq_len, hidden) → (batch, heads, seq_len, head_dim)
q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
```

It splits the hidden dimension into `nheads` groups of `head_dim` each, then rearranges so the heads dimension comes before the sequence dimension (which is what SDPA expects).

### PyTorch's `scaled_dot_product_attention`

PyTorch provides SDPA as a single function call:

```python
output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
```

Under the hood, PyTorch picks the fastest implementation for your hardware:

| Backend | When used | Speed |
|---------|-----------|-------|
| **FlashAttention** | NVIDIA GPUs (A100, H100) | Fastest — fused CUDA kernel, O(N) memory |
| **Efficient Attention** | Broader GPU support | Fast — memory-efficient algorithm |
| **Math fallback** | CPU or unsupported GPU | Slowest — standard PyTorch ops |

FlashAttention is a breakthrough: instead of materializing the full `(seq_len × seq_len)` attention matrix (which is huge for long sequences), it computes attention in tiles, keeping memory usage linear in sequence length instead of quadratic.

### Why SDPA Matters for AutoParallel

SDPA is special in AutoParallel for several reasons:

**1. It has a custom sharding rule** (`propagation_rules.py`): AutoParallel knows that Q, K, V can be sharded on the **heads dimension** — each GPU computes attention for a subset of heads. This is tensor parallelism for attention.

```
8 GPUs, 48 heads → 6 heads per GPU

GPU 0: SDPA(Q[:,:6,:], K[:,:6,:], V[:,:6,:])   → Attention for heads 0-5
GPU 1: SDPA(Q[:,6:12,:], K[:,6:12,:], V[:,6:12,:]) → Attention for heads 6-11
...
```

**2. Context parallelism is disabled for SDPA**: Sharding on the sequence dimension (dim 2) would require special handling of causal masks. AutoParallel currently filters out `Shard(2)` strategies for SDPA due to upstream PyTorch bugs.

**3. Activation checkpointing targets SDPA**: The attention output is large (`batch × heads × seq_len × head_dim`) and the selective checkpointing policy in `example_autoparallel.py` marks SDPA as `MUST_RECOMPUTE` to save this memory.

**4. SDPA is opaque to decomposition**: AutoParallel preserves SDPA as a single node in the FX graph (doesn't decompose it into Q×K, softmax, ×V). This is because FlashAttention is a fused kernel — decomposing it would lose the performance benefit and create a huge `(seq_len × seq_len)` intermediate tensor.

### Quick Reference

| Term | Meaning |
|------|---------|
| **SDPA** | Scaled Dot-Product Attention — the core attention operation |
| **Q, K, V** | Query, Key, Value — the three input matrices to attention |
| **Head** | One independent attention computation; models have multiple heads |
| **Head dim** | hidden_size ÷ num_heads (e.g., 768 ÷ 12 = 64) |
| **FlashAttention** | Memory-efficient fused CUDA kernel for SDPA |
| **Causal mask** | Prevents tokens from attending to future tokens (for autoregressive models) |

## What Is Megatron-LM?

**Megatron-LM** is NVIDIA's library for training large language models. It was one of the first frameworks to show that you could train models with hundreds of billions of parameters by combining multiple parallelism strategies. It's the most well-known alternative to what AutoParallel does — and understanding it helps you understand why AutoParallel exists.

### The Origin Story

In 2019, NVIDIA researchers published a paper showing how to train a Transformer model with **8.3 billion parameters** (huge at the time) by splitting the model across GPUs in a specific way. The key insight was **tensor parallelism for Transformers**: splitting the attention heads and FFN weight matrices across GPUs within a node, while using data parallelism across nodes.

They called the framework **Megatron-LM** (named after Megatron, the Transformers villain — because it trains very large *Transformers*).

Over time, Megatron-LM grew to support:
- Tensor Parallelism (TP) — the original innovation
- Pipeline Parallelism (PP) — splitting layers across GPUs
- Data Parallelism (DP) — replicating across groups
- Sequence Parallelism (SP) — splitting along sequence length
- Expert Parallelism (EP) — for Mixture-of-Experts models
- Context Parallelism (CP) — for very long sequences

### How Megatron-LM Works (The Manual Approach)

Megatron-LM provides **pre-built parallel layer implementations**. You replace standard PyTorch layers with Megatron's parallel versions:

```python
# Standard PyTorch (single GPU):
class TransformerLayer(nn.Module):
    def __init__(self, hidden, heads):
        self.attn_qkv = nn.Linear(hidden, 3 * hidden)
        self.attn_proj = nn.Linear(hidden, hidden)
        self.ffn_up = nn.Linear(hidden, 4 * hidden)
        self.ffn_down = nn.Linear(4 * hidden, hidden)

# Megatron-LM (manual parallelism):
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class TransformerLayer(nn.Module):
    def __init__(self, hidden, heads):
        # You must manually decide: which layers are column-parallel vs row-parallel
        self.attn_qkv = ColumnParallelLinear(hidden, 3 * hidden)   # ← manual choice
        self.attn_proj = RowParallelLinear(hidden, hidden)          # ← manual choice
        self.ffn_up = ColumnParallelLinear(hidden, 4 * hidden)     # ← manual choice
        self.ffn_down = RowParallelLinear(4 * hidden, hidden)      # ← manual choice
```

The pattern is always:
1. **Column-parallel** linear: splits weight columns across GPUs, each GPU computes partial output
2. **Row-parallel** linear: splits weight rows across GPUs, all-reduce at the end

For a Transformer FFN:
```
Column-parallel               Row-parallel
    Linear1                     Linear2
  ┌───────────┐                ┌─────────┐
  │ W[:, 0:N] │──► partial ──► │W[0:N, :]│──► all-reduce ──► output
  │ W[:, N:2N]│──► partial ──► │W[N:2N,:]│──►    ↑
  └───────────┘                └─────────┘       sum
```

### Megatron-LM vs AutoParallel: The Key Difference

The fundamental difference is **manual vs automatic**:

```
Megatron-LM:
  1. Human expert reads the model architecture
  2. Human decides: "this layer should be column-parallel, that one row-parallel"
  3. Human replaces nn.Linear with ColumnParallelLinear / RowParallelLinear
  4. Human configures TP degree, PP stages, DP groups
  5. Human inserts communication collectives in the right places
  6. Train

AutoParallel:
  1. Write a normal nn.Module (no parallel code)
  2. Call auto_parallel(model, mesh, inputs)
  3. ILP solver figures out all of the above automatically
  4. Train
```

| Aspect | Megatron-LM | AutoParallel |
|--------|-------------|--------------|
| **How parallelism is specified** | Replace layers with parallel versions manually | Automatic via ILP optimization |
| **Model code changes** | Significant — must use Megatron's layer classes | None — standard nn.Module |
| **Supported strategies** | TP, PP, DP, SP, EP, CP (all manually configured) | FSDP, TP, DP, hybrid (auto-discovered); PP and EP require manual help |
| **New model effort** | Must write parallel version of each layer | Just write the model, run optimizer |
| **Optimality** | Relies on human expertise to choose right strategy | ILP guarantees global optimum (given accurate cost model) |
| **Maturity** | Battle-tested at massive scale (530B+ params) | Experimental, early development |
| **Performance** | Highly optimized CUDA kernels, fused ops | Relies on PyTorch Inductor for kernel fusion |
| **Flexibility** | Fixed patterns (column/row parallel) | Can discover non-standard strategies |
| **Framework** | Custom training framework (own data loader, optimizer, etc.) | Just the model sharding; uses PyTorch or TorchTitan for training |

### What Megatron-LM Does Well

**1. Proven at scale**: Megatron-LM has trained models up to 530 billion parameters (Megatron-Turing NLG). It's been used in production at NVIDIA and many other organizations.

**2. Highly optimized kernels**: Megatron-LM includes custom CUDA kernels for fused attention, fused layer norm, fused bias+gelu, etc. These are hand-tuned for NVIDIA hardware.

**3. Complete training framework**: Unlike AutoParallel (which only handles model sharding), Megatron-LM is a full training framework with data loading, mixed precision, checkpointing, and logging built in.

**4. Pipeline parallelism**: Megatron-LM has mature PP support with schedules like 1F1B (one forward, one backward), interleaved schedules, and virtual pipeline stages. AutoParallel's PP support is still basic.

**5. Sequence parallelism**: Megatron-LM shards layer norm and dropout along the sequence dimension to reduce activation memory. This is tightly integrated with their TP implementation.

### What Megatron-LM Struggles With

**1. Model-specific effort**: Every new architecture requires writing a parallel version. If your model isn't a standard Transformer, you're doing significant engineering work.

**2. Fixed patterns**: The column-parallel → row-parallel pattern works great for Transformers, but it's a fixed recipe. If a different sharding strategy would be better for your specific model/mesh/hardware, Megatron won't find it.

**3. Locked to NVIDIA**: Megatron-LM is heavily optimized for NVIDIA GPUs with NCCL. Running on AMD or other hardware requires significant porting.

**4. Hard to experiment with**: Changing the parallelism strategy (e.g., "what if I use TP=4 instead of TP=8?") requires reconfiguring and potentially rewriting parallel layers.

**5. Tight coupling**: Model code and parallelism code are intertwined. You can't easily take a Megatron-LM model and run it on a single GPU for debugging — you need the full distributed setup.

### Why AutoParallel Exists

AutoParallel aims to get the **benefits** of Megatron-LM's parallelism strategies (TP, PP, DP) **without the manual effort**:

```
Megatron's approach:
  "Here are the building blocks (ColumnParallelLinear, etc.).
   You figure out how to assemble them for your model."

AutoParallel's approach:
  "Give me your model. I'll figure out the optimal assembly automatically."
```

The trade-off is clear: Megatron gives you more control and is battle-tested, but requires expert knowledge and per-model engineering. AutoParallel gives you automation and optimality, but is experimental and less mature.

### Can You Use Both?

Not directly — they approach the problem differently (Megatron replaces layers; AutoParallel traces and transforms the graph). But the **ideas** from Megatron-LM heavily influenced AutoParallel:

- AutoParallel's `addmm → mm + add` decomposition enables the same column/row parallel patterns that Megatron uses
- AutoParallel's cost model accounts for the same communication patterns (all-gather for TP, reduce-scatter for FSDP)
- The benchmark sweeps in AutoParallel's MAST scripts compare against manually configured FSDP (similar to what Megatron would do)

### Other Frameworks in This Space

| Framework | Who | Approach |
|-----------|-----|----------|
| **Megatron-LM** | NVIDIA | Manual parallel layers, custom CUDA kernels |
| **DeepSpeed** | Microsoft | ZeRO optimizer (auto-shard optimizer states/gradients/params), plus some TP/PP |
| **FSDP** | Meta/PyTorch | Auto-shard parameters (like ZeRO-3), built into PyTorch |
| **Alpa** | UC Berkeley | ILP-based auto-parallelism for JAX (closest to AutoParallel's approach) |
| **GSPMD** | Google | Compiler-based sharding propagation for XLA |
| **AutoParallel** | Meta | ILP-based auto-parallelism for PyTorch |

AutoParallel's unique niche: **ILP-based automatic optimization, built on PyTorch's native DTensor infrastructure**. It brings the automation of Alpa/GSPMD to the PyTorch ecosystem.

## Key Terminology Reference

| Term | Meaning |
|------|---------|
| **Eager mode** | PyTorch's default: execute ops one at a time |
| **FX** | PyTorch's graph capture and transformation framework |
| **GraphModule** | An `nn.Module` that holds an explicit FX graph |
| **Node** | One operation in the graph (placeholder, call_function, output, etc.) |
| **Tracing** | Converting Python code into a graph by recording operations |
| **TorchDynamo** | PyTorch's Python-level tracer (handles control flow) |
| **AOTAutograd** | Generates backward ops ahead-of-time, producing a joint graph |
| **Joint graph** | Single graph containing both forward and backward operations |
| **Decomposition** | Breaking high-level ops into primitive ops (addmm → mm + add) |
| **ATen** | PyTorch's C++ tensor operation library (~2000 primitives) |
| **Fake tensors** | Tensors with shapes but no data (for tracing without GPUs) |
| **Meta device** | A virtual device where tensors have no storage |
| **Alias node** | Zero-cost identity op inserted to give the optimizer more flexibility |
| **Process group** | Communication channel connecting GPU processes for collectives |
| **Rank** | A process's ID number in the process group (0, 1, 2, ...) |
| **World size** | Total number of processes in the process group |
| **Activation checkpointing** | Trading compute for memory: drop activations during forward, recompute during backward |
| **addmm** | Fused bias + matmul: `bias + input @ weight` |
| **mm** | Pure matrix multiply: `input @ weight` |
