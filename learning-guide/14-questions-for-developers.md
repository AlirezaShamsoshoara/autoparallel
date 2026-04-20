# Questions to Ask the AutoParallel Developers

Organized by topic, from strategic to deeply technical. Pick the ones that match your conversation context — you won't have time for all of them.

---

## Vision & Roadmap

These are the highest-value questions. They give you insight you can't get from reading code.

**1. What's the end-state vision for AutoParallel?**
Is the goal to become the default parallelization path in PyTorch (replacing manual FSDP/TP), or to remain an optional tool for advanced users? Is there a plan to upstream it into PyTorch core?

**2. What's blocking production adoption today?**
The project is labeled experimental. What specifically needs to happen before a team could use it in production training? Is it correctness, performance, API stability, or something else?

**3. What's the relationship with `torch.compile`?**
Both AutoParallel and `torch.compile` use TorchDynamo + AOTAutograd + FX graphs. Is AutoParallel heading toward being a compiler pass inside `torch.compile`, or will it always be a separate step?

**4. Is there a plan for automatic pipeline parallelism stage partitioning?**
Today users must manually assign PP stages. Is auto-partitioning on the roadmap? What makes it hard — the graph cutting problem, memory estimation, or something else?

**5. Where does AutoParallel fit in Meta's internal training stack?**
Is it used for any real training runs internally, or purely a research project? What models has it been validated on beyond LLaMA-3 and DeepSeek-V3?

---

## Design Decisions

These questions reveal the *why* behind the architecture — the kind of knowledge you can't get from reading code.

**6. Why ILP instead of other search methods?**
You chose Integer Linear Programming over alternatives like dynamic programming, reinforcement learning, simulation-based search (FlexFlow), or heuristic-based approaches (Megatron). What made ILP the right choice? Where does it struggle?

**7. Why a joint graph instead of separate forward/backward optimization?**
Alpa (for JAX) optimizes intra-op and inter-op parallelism in separate stages. AutoParallel optimizes the joint graph as one ILP. What are the trade-offs? Does the joint approach scale to very large models?

**8. Why decompose `addmm` instead of teaching the optimizer about fused ops?**
The `addmm → mm + add` decomposition enables TP but loses kernel fusion. Was it considered to keep `addmm` as one node and give it TP-aware strategies instead? What was the trade-off?

**9. How do you decide which ops get custom propagation rules vs. relying on DTensor upstream?**
There are ~30 ops with custom rules. What's the criteria? Is the goal to eventually push all of them upstream into PyTorch's DTensor?

**10. Why PuLP/CBC and not a commercial solver like Gurobi or CPLEX?**
CBC is open-source but slower. Have you benchmarked against commercial solvers? At what model size does solver time become a bottleneck?

---

## Technical Deep Dives

These show you've read the code and understand the nuances.

**11. How does graph clustering handle non-identical transformer layers?**
Graph clustering shares ILP variables for repeated subgraphs (e.g., transformer layers). But what happens with models where layers differ slightly (different head counts, MoE layers mixed with dense layers, adapter layers)?

**12. How accurate is the cost model in practice?**
The compute model uses hardcoded TFLOPS with a 70% efficiency factor and 7μs kernel launch floor. The NCCL model simulates algorithm selection. How close are these estimates to real wall-clock time? Have you validated against profiling data?

**13. What's the story with `kwargs` not being processed?**
`build_sharding_metadata()` only processes `node.args`, not `node.kwargs` (there are TODOs at lines 203 and 251 of `optimize_sharding.py`). Are there real ops where this causes incorrect sharding? Is this a known correctness gap or just a missing optimization?

**14. Why is context parallelism disabled in the SDPA rule?**
The code filters out `Shard(2)` strategies in SDPA due to upstream PyTorch issues (PR #131351). What's the nature of the bug — is it a correctness issue or a performance issue? Is there an ETA for fixing it upstream?

**15. How does the prefetch discount work in practice?**
`apply_prefetch_discount()` scales down communication costs for overlappable collectives. How is the discount factor determined? Is it a fixed constant, or does it model the actual compute/communication overlap ratio?

**16. The all-reduce backward is currently another all-reduce (with a TODO about it being wrong). What's the correct behavior?**
The comment at `collectives.py:200` says it should be split into an all-reduce and an identity. In what scenarios does the current implementation produce wrong gradients?

---

## Gaps & Limitations

These questions show you understand the weak spots and want to learn where to contribute or what to avoid.

**17. What's the plan for dynamic shapes?**
The `make_fx` usage in `apply_sharding.py:295` is flagged as "suspicious in case of dynamic shapes." How fundamental is this limitation? Would supporting dynamic shapes require a different architecture?

**18. How far away is MoE auto-discovery?**
Today, expert parallelism requires manual `local_map` wrappers. Is there a path to the optimizer automatically discovering expert parallelism strategies? What makes MoE harder than TP/FSDP for the ILP?

**19. What happens with ops that fall back to all-Replicate?**
When an op has no sharding rule and `enable_implicit_replication` is on, it defaults to all-Replicate. Is there a way to know how much performance is lost? Is there tooling to identify which ops are bottlenecks?

**20. Are there models where AutoParallel makes *worse* decisions than manual sharding?**
If so, what characterizes them? Is it cost model inaccuracy, missing op rules, or fundamental ILP limitations?

---

## Integration & Ecosystem

**21. What's the long-term plan for TorchTitan integration?**
Today AutoParallel plugs in as a TorchTitan experiment module. Will it become a first-class option in TorchTitan (e.g., `--parallelism.strategy=auto`)?

**22. How does AutoParallel interact with `torch.distributed.checkpoint`?**
The `example_dcp.py` shows basic checkpoint save/load. Are there edge cases with resharding between saves (e.g., saving with one mesh, loading with another)?

**23. Is there a plan for inference optimization?**
AutoParallel supports inference mode, but is there specific work on inference-optimized sharding (different cost model, latency vs. throughput trade-offs)?

**24. How do you test correctness?**
The test suite uses fake process groups (no real communication). How do you verify that the sharded model produces the same numerical results as the unsharded model? Is there a bit-exact equivalence test?

---

## Contributing & Learning

**25. Where's the best place to start contributing?**
For someone who understands the codebase, what are the highest-impact areas to contribute to? More op rules? Better cost models? Documentation?

**26. What's the hardest bug you've debugged in AutoParallel?**
This often reveals architectural assumptions and failure modes that aren't documented anywhere.

**27. What would you design differently if starting from scratch?**
Every project has things the authors wish they'd done differently. This reveals hard-won lessons.

**28. Are there papers or references that influenced the design?**
Besides Alpa and GSPMD, are there other papers or systems that shaped the architecture?

---

## Quick-Fire Questions (for casual conversation)

- How long does the ILP typically take to solve for LLaMA-3 70B?
- What's the largest model (in parameters) that's been run through AutoParallel?
- Has anyone tried AutoParallel on vision models or multi-modal architectures?
- Is there interest in supporting quantization-aware sharding (INT4/INT8 weight sharding)?
- What's the debugging workflow when the optimizer makes a surprising decision?

---

## How to Prioritize

If you only have **5 minutes**, ask: 1, 2, 6, 20, 27

If you have **15 minutes**, add: 4, 12, 18, 25

If you have **30+ minutes**, go for the full technical deep dives (11-16) — that's where you'll learn the most that you can't learn from the code alone.

**Pro tip**: Start with the vision/roadmap questions. They set the context for everything else and show the developers you care about the *project*, not just the code. Then dive into the technical questions that interest you most.
