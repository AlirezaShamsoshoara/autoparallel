# Communication Collectives: From Basics to Advanced

When you split a model across multiple GPUs, those GPUs need to **talk to each other**. Communication collectives are the vocabulary of that conversation — they are standardized patterns for moving and combining data across GPUs.

This chapter starts from zero and builds up to how AutoParallel uses collectives.

## Why GPUs Need to Communicate

Imagine you shard a weight matrix across 4 GPUs, each holding 1/4 of the columns:

```
GPU 0: W[:, 0:256]     GPU 1: W[:, 256:512]
GPU 2: W[:, 512:768]   GPU 3: W[:, 768:1024]
```

To compute `Y = X @ W`, each GPU can compute its slice: `Y_partial = X @ W_slice`. But now each GPU only has **part** of the output. To get the full output, GPUs must communicate. The **how** of this communication is a collective.

## The Basics: Point-to-Point vs. Collective

**Point-to-point**: One GPU sends data to one other GPU.
```
GPU 0 ──send──► GPU 2
```

**Collective**: A group of GPUs coordinate together in a pattern. Every GPU in the group participates simultaneously.
```
GPU 0 ◄──►  GPU 1
  ▲            ▲
  │            │
  ▼            ▼
GPU 2 ◄──►  GPU 3
```

Collectives are what AutoParallel uses. They're faster than point-to-point because the network hardware (NVLink, NVSwitch, InfiniBand) is optimized for them.

## The Four Core Collectives

AutoParallel uses four collectives. Let's understand each one visually.

---

### 1. All-Gather

**Purpose**: Every GPU has a **piece**. After all-gather, every GPU has the **whole thing**.

**When AutoParallel uses it**: FSDP all-gathers sharded parameters before computation. If a parameter is `Shard(0)` across 4 GPUs, each GPU holds 1/4. Before the layer can compute, every GPU needs the full parameter.

```
BEFORE (each GPU has a shard):            AFTER (every GPU has everything):

GPU 0: [A]                                GPU 0: [A][B][C][D]
GPU 1: [B]          ──all-gather──►       GPU 1: [A][B][C][D]
GPU 2: [C]                                GPU 2: [A][B][C][D]
GPU 3: [D]                                GPU 3: [A][B][C][D]
```

**Data movement**: Each GPU sends its shard to all others. Total data moved: `N × shard_size` (where N = number of GPUs).

**Memory impact**: Memory usage **increases** N×. Each GPU goes from holding 1/N to holding the full tensor.

**Code example**:
```python
import torch
import torch.distributed as dist

# Each GPU has a different shard
# GPU 0: [1, 2], GPU 1: [3, 4], GPU 2: [5, 6], GPU 3: [7, 8]
local_shard = torch.tensor([rank * 2 + 1, rank * 2 + 2])

# After all-gather, every GPU has [1, 2, 3, 4, 5, 6, 7, 8]
full_tensor = torch.empty(8)
dist.all_gather_into_tensor(full_tensor, local_shard)
# full_tensor = [1, 2, 3, 4, 5, 6, 7, 8]  on every GPU
```

---

### 2. Reduce-Scatter

**Purpose**: Every GPU has a **full-sized** tensor (often with partial results). After reduce-scatter, each GPU has a **reduced (summed) shard**.

**When AutoParallel uses it**: After the backward pass with FSDP, each GPU has full-sized gradients. Reduce-scatter sums them across GPUs and gives each GPU only its shard of the summed gradient — saving memory.

```
BEFORE (each GPU has a full tensor):      AFTER (each GPU has a summed shard):

GPU 0: [a₀ b₀ c₀ d₀]                    GPU 0: [Σa]         (a₀+a₁+a₂+a₃)
GPU 1: [a₁ b₁ c₁ d₁]  ─reduce-scatter─► GPU 1: [Σb]         (b₀+b₁+b₂+b₃)
GPU 2: [a₂ b₂ c₂ d₂]                    GPU 2: [Σc]         (c₀+c₁+c₂+c₃)
GPU 3: [a₃ b₃ c₃ d₃]                    GPU 3: [Σd]         (d₀+d₁+d₂+d₃)
```

**Data movement**: Similar to all-gather but in reverse — data is summed and scattered.

**Memory impact**: Memory usage **decreases** N×. Each GPU goes from holding the full tensor to holding 1/N.

**Code example**:
```python
# Each GPU has full-sized gradients (e.g., from backward pass)
# GPU 0: [1, 2, 3, 4], GPU 1: [5, 6, 7, 8], GPU 2: [9, 10, 11, 12], GPU 3: [13, 14, 15, 16]
full_gradient = torch.tensor([(rank * 4) + i + 1 for i in range(4)])

# After reduce-scatter, each GPU has one summed shard
# GPU 0: [1+5+9+13]=28, GPU 1: [2+6+10+14]=32, etc.
local_shard = torch.empty(1)
dist.reduce_scatter_tensor(local_shard, full_gradient, op=dist.ReduceOp.SUM)
```

**Key insight**: All-gather and reduce-scatter are **inverses**:
```
all-gather:     shards → full copies    (expand)
reduce-scatter: full copies → shards    (compress + sum)
```

This is why FSDP uses them as a pair: all-gather in forward, reduce-scatter in backward.

---

### 3. All-Reduce

**Purpose**: Every GPU has a tensor. After all-reduce, every GPU has the **sum** (or other reduction) of all tensors.

**When AutoParallel uses it**: When a `Partial()` placement needs to become `Replicate()`. For example, in row-parallel tensor parallelism, each GPU computes a partial matrix product. All-reduce sums them so every GPU has the full result.

```
BEFORE (each GPU has partial results):    AFTER (every GPU has the sum):

GPU 0: [a₀ b₀]                           GPU 0: [Σa  Σb]
GPU 1: [a₁ b₁]       ──all-reduce──►     GPU 1: [Σa  Σb]
GPU 2: [a₂ b₂]                           GPU 2: [Σa  Σb]
GPU 3: [a₃ b₃]                           GPU 3: [Σa  Σb]
```

**Relationship to the other two**: All-reduce = reduce-scatter + all-gather. In fact, NCCL often implements it that way internally:
```
all-reduce  =  reduce-scatter (sum into shards)  +  all-gather (broadcast shards)
```

**Data movement**: 2× the tensor size total (equivalent to one reduce-scatter + one all-gather).

**Memory impact**: No change — every GPU starts and ends with a full-sized tensor.

**Code example**:
```python
# Each GPU has partial results from computing part of a matrix multiply
# GPU 0: [10, 20], GPU 1: [30, 40], GPU 2: [50, 60], GPU 3: [70, 80]
partial = torch.tensor([rank * 20 + 10, rank * 20 + 20], dtype=torch.float)

# After all-reduce, every GPU has the sum: [160, 200]
dist.all_reduce(partial, op=dist.ReduceOp.SUM)
# partial = [160, 200] on every GPU
```

---

### 4. All-to-All

**Purpose**: Each GPU sends a **different piece** to each other GPU. It's like a transpose of data across GPUs.

**When AutoParallel uses it**: When changing from `Shard(dim_a)` to `Shard(dim_b)` — resharding from one dimension to another. Common in expert parallelism (MoE) where tokens are routed to different expert GPUs.

```
BEFORE (sharded on dim 0):                AFTER (sharded on dim 1):

GPU 0: [a₀ a₁ a₂ a₃]                    GPU 0: [a₀ b₀ c₀ d₀]
GPU 1: [b₀ b₁ b₂ b₃]   ──all-to-all──►  GPU 1: [a₁ b₁ c₁ d₁]
GPU 2: [c₀ c₁ c₂ c₃]                    GPU 2: [a₂ b₂ c₂ d₂]
GPU 3: [d₀ d₁ d₂ d₃]                    GPU 3: [a₃ b₃ c₃ d₃]
```

Think of it as a matrix transpose: rows become columns, columns become rows.

**Data movement**: Each GPU sends `(N-1)/N` of its data and receives the same amount. Total: `N × (N-1) × shard_size`.

**Memory impact**: No change in total size, but the data is **rearranged** across GPUs.

**Code example**:
```python
# Each GPU has 4 elements, will send one to each GPU
# GPU 0: [0, 1, 2, 3], GPU 1: [10, 11, 12, 13], etc.
input_tensor = torch.tensor([rank * 10 + i for i in range(4)], dtype=torch.float)
output_tensor = torch.empty(4, dtype=torch.float)

# After all-to-all:
# GPU 0 gets element 0 from each GPU: [0, 10, 20, 30]
# GPU 1 gets element 1 from each GPU: [1, 11, 21, 31]
# etc.
dist.all_to_all_single(output_tensor, input_tensor)
```

---

## How Collectives Map to DTensor Placements

AutoParallel's optimizer chooses placements (`Replicate`, `Shard`, `Partial`) for each tensor. When adjacent operations need different placements, a collective is inserted to **redistribute** the data:

| From | To | Collective Needed |
|------|-----|-------------------|
| `Shard(d)` | `Replicate()` | **All-gather** on dim d |
| `Replicate()` | `Shard(d)` | **Local slice** (no communication!) |
| `Partial()` | `Replicate()` | **All-reduce** |
| `Partial()` | `Shard(d)` | **Reduce-scatter** on dim d |
| `Shard(d1)` | `Shard(d2)` | **All-to-all** (reshard) |
| `Replicate()` | `Replicate()` | Nothing needed |
| `Shard(d)` | `Shard(d)` | Nothing needed |

**The optimizer's goal**: minimize the total cost of these redistributions across the entire model.

## Collectives in FSDP (Full Story)

FSDP uses a paired all-gather / reduce-scatter pattern. Here's the full cycle:

```
State: Parameters sharded across GPUs
       GPU 0 has param[:256], GPU 1 has param[256:512], ...

  ┌──────────── FORWARD PASS ────────────┐
  │                                       │
  │  1. All-gather parameters             │
  │     [shard] ──all-gather──► [full]    │
  │                                       │
  │  2. Compute forward with full params  │
  │     output = input @ full_weight      │
  │                                       │
  │  3. (Optional) Reshard parameters     │
  │     Free the all-gathered copy        │
  │     to save memory                    │
  │                                       │
  └───────────────────────────────────────┘

  ┌──────────── BACKWARD PASS ────────────┐
  │                                       │
  │  4. All-gather parameters again       │
  │     (if resharded after forward)      │
  │                                       │
  │  5. Compute gradients                 │
  │     grad_weight = input.T @ grad_out  │
  │                                       │
  │  6. Reduce-scatter gradients          │
  │     [full_grad] ──reduce-scatter──►   │
  │     [grad_shard]                      │
  │     Each GPU has its shard of summed  │
  │     gradients                         │
  │                                       │
  └───────────────────────────────────────┘

  7. Optimizer step on sharded gradients
     Each GPU updates only its param shard
```

## Collectives in Tensor Parallelism (Full Story)

Tensor parallelism splits weight matrices. Here's a column-parallel + row-parallel FFN:

```
  FFN: output = Linear2(ReLU(Linear1(x)))

  Linear1 (column-parallel):
  ┌─────────────────────────────────────────────┐
  │  Weight split on columns:                   │
  │  GPU 0: W1[:, 0:512]   GPU 1: W1[:, 512:]   │
  │                                             │
  │  Each GPU computes:  y_local = x @ W1_local │
  │  GPU 0: y[:, 0:512]   GPU 1: y[:, 512:]     │
  │                                             │
  │  No communication needed! Each GPU has      │
  │  a different slice of the output.           │
  └─────────────────────────────────────────────┘
                    │
                    ▼
                  ReLU (element-wise, no communication)
                    │
                    ▼
  Linear2 (row-parallel):
  ┌─────────────────────────────────────────────────────┐
  │  Weight split on rows:                              │
  │  GPU 0: W2[0:512, :]   GPU 1: W2[512:, :]           │
  │                                                     │
  │  Each GPU computes: z_partial = y_local @ W2_local  │
  │  Each GPU has a PARTIAL result (needs sum)          │
  │                                                     │
  │  ► All-reduce (or reduce-scatter):                  │
  │    z_full = z_partial_0 + z_partial_1               │
  │    Now every GPU has the full output.               │
  └─────────────────────────────────────────────────────┘
```

The communication happens only once (the all-reduce after Linear2), not between every layer. This is why column-parallel → row-parallel is efficient.

## Cost of Collectives

Not all collectives cost the same. AutoParallel's cost model estimates each one:

### Bandwidth Cost (simplified)

For N GPUs each sending/receiving data of size S:

| Collective | Data Moved Per GPU | Total Across All GPUs |
|------------|-------------------|----------------------|
| All-gather | S × (N-1)/N (receive) | S × (N-1) |
| Reduce-scatter | S × (N-1)/N (send) | S × (N-1) |
| All-reduce | 2 × S × (N-1)/N | 2 × S × (N-1) |
| All-to-all | S × (N-1)/N | S × (N-1) |

**Key insight**: All-reduce costs ~2× what all-gather or reduce-scatter costs individually. But FSDP's paired all-gather + reduce-scatter also costs 2× combined. The difference is **when** the cost is paid (forward vs. backward).

### What Makes a Collective Fast or Slow?

1. **Message size**: Larger messages use bandwidth more efficiently. Many small collectives are worse than one large one (this is why AutoParallel has collective **bucketing**).

2. **Network topology**: Communication within a node (NVLink/NVSwitch: ~900 GB/s) is much faster than across nodes (InfiniBand: ~50-100 GB/s per link).

3. **Algorithm**: NCCL chooses between Ring, Tree, NVLS, and other algorithms depending on message size and topology. AutoParallel's NCCL cost model simulates this selection.

4. **Overlap with compute**: If a collective runs while the GPU is computing something else, it's essentially "free". AutoParallel models this with prefetch discounts.

## NCCL: The Engine Behind Collectives

**NCCL** (NVIDIA Collective Communications Library) is the library that actually executes collectives on NVIDIA GPUs. When AutoParallel inserts an all-gather into the graph, NCCL is what runs it at training time.

NCCL implements multiple **algorithms** for each collective and picks the best one at runtime:

| Algorithm | Best For | How It Works |
|-----------|----------|--------------|
| **Ring** | Medium-large messages | GPUs form a ring, data flows around it |
| **Tree** | Small-medium messages | GPUs form a binary tree, data flows up and down |
| **NVLS** | Large messages (NVSwitch) | Direct NVSwitch multicast, lowest latency |
| **CollNet Direct** | Multi-node | Uses network switch reduction |
| **CollNet Chain** | Multi-node | Chained switch reduction |

AutoParallel's `nccl_cost_model.py` simulates this algorithm selection to estimate costs accurately, rather than using a simple bandwidth model.

## Overlapping Communication with Computation

The real magic in distributed training is **hiding communication behind computation**. While GPU cores are busy computing one layer's matmul, the network can be transferring data for the next layer:

```
Time ──────────────────────────────────────────►

GPU compute:  [  mm layer 1  ][  mm layer 2  ][  mm layer 3  ]
Network:         [AG layer 2]    [AG layer 3]    [AG layer 4]
                  ▲                ▲
                  └── overlapped! ─┘
```

**AG** = all-gather of the next layer's parameters (FSDP prefetch)

AutoParallel models this overlap. When it estimates the cost of an FSDP all-gather, it applies a **prefetch discount** — if the all-gather can run concurrently with computation, its effective cost is reduced (or zero if fully hidden).

## Forward/Backward Pairs

Each collective in AutoParallel has a natural backward counterpart. The `collectives.py` implements these as `torch.autograd.Function` pairs:

| Forward Collective | Backward Collective | Why |
|--------------------|--------------------|----|
| All-gather | Reduce-scatter | Gathering shards forward → scatter gradients backward |
| Reduce-scatter | All-gather | Scattering forward → gathering gradient shards backward |
| All-reduce | All-reduce | Sum forward → sum gradients backward |
| All-to-all | All-to-all (reversed) | Transpose forward → transpose backward |

This pairing is automatic — AOTAutograd generates the backward collectives from the forward ones in the joint graph. The optimizer sees both and accounts for the total cost.

## Quick Reference

```
ALL-GATHER:       shards  →  full copies       (for FSDP param reconstruct)
REDUCE-SCATTER:   full    →  summed shards      (for FSDP grad sync)
ALL-REDUCE:       partial →  summed full copies (for TP partial results)
ALL-TO-ALL:       shard(d1) → shard(d2)         (for MoE expert routing)
```

## How AutoParallel Decides Which Collectives to Insert

The optimizer doesn't explicitly "choose collectives." Instead, it:

1. Chooses a **placement** for each tensor (e.g., `Shard(0)`, `Replicate()`, `Partial()`)
2. Where two adjacent operations disagree on placement, a **redistribution** is needed
3. The redistribution maps to a specific collective (see the placement table above)
4. The **cost** of that redistribution was already factored into the ILP objective

So when you see the final sharded model, the collectives are a **consequence** of the placement decisions, not direct decisions themselves. The optimizer minimizes total redistribution cost, which indirectly minimizes communication.
