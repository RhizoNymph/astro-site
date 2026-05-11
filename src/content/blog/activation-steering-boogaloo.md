---
title: Activation Steering in vLLM
pubDate: 2026-05-10
---
# Activation Steering Part 2: Electric Boogaloo

In my first blog on activation steering in vLLM (which you can read [here](https://www.rhizonymph.com/blog/activation-steering/)) I walked through the work I've
been doing adding support for steering vectors to vLLM. In it, I showed that in the end there was still real overhead vs the steering disabled case and detailed
some things that I planned to do in the future to try and close the gap. Since then, I've tried these out and tried other things I hadn't thought of back then that
further profiling revealed I should tackle. In this post I'm going to go over what I tried, what worked, and what didn't. I'm going to stick with the microbenches from
[my benchmark suite](https://www.github.com/RhizoNymph/steering-bench) but at the end we're going to compare the latency/throughput benchmarks that I posted results
from in the last article.

Let us start with reviewing the optimization roadmap that I ended with last time:
- Optimize config registration to further reduce per request overhead, the dominant remaining overhead at larger batch sizes and short workloads
- Name to vector resolution should happen on the worker side
- Write a custom Triton kernel for indexed scatter to steering tables 

## Config registration

Config registration itself (SteeringManager.register_config) which gets called when a new (hash, phase) config is admitted I was able to pretty simply batch into a single non-blocking H2D transfer instead of the previous approach that was a different cudaMemcpy for every (layer, hook) combination.

```python
for hook_point, layer_vecs in vectors.items():
    items = [
        (layer_idx, vec)
        for layer_idx, vec in layer_vecs.items()
        if locally_owned_layers is None or layer_idx in locally_owned_layers
    ]
    if not items:
        stored[hook_point] = {}
        continue
    layer_idxs = [layer_idx for layer_idx, _ in items]
    raw_vecs = [vec for _, vec in items]

    arr = np.asarray(raw_vecs, dtype=np.float32)
    cpu_t = torch.from_numpy(arr).pin_memory()
    stacked = cpu_t.to(self.device, non_blocking=True)

    stored[hook_point] = {
        layer_idx: stacked[i : i + 1] for i, layer_idx in enumerate(layer_idxs)
    }

self.config_vectors[key] = stored
```

The `pin_memory()` + `non_blocking=True` pair is what's actually doing the work here. The previous code path went through `torch.tensor(vec, device="cuda")` for each `(layer, hook)` pair, which issues a synchronous `cudaMemcpy` per row. Stacking on the CPU first and shipping one pinned non-blocking copy collapses ~80 syncs down to one per hook. The microbench for the steering manager showed a 1.86–1.93× speedup:

| layers / configs / hooks | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 1 / 1_hook | 3.46 ms | 1.79 ms | 1.93× |
| 26 / 16 / 3_hooks | 162.2 ms | 87.1 ms | 1.86× |
| 34 / 16 / 3_hooks | 212.2 ms | 113.9 ms | 1.86× |
| 42 / 16 / 3_hooks | 263.0 ms | 140.9 ms | 1.87× |

Another change made to the registration process was something I feel a bit silly for needing to do in the first place, using numpy arrays instead of lists of floats in
the vector resolution and config hashing parts of this path. The prior code re-converted the resolved list back to an ndarray inside `hash_steering_config` via `np.asarray(vec, dtype=np.float32).tobytes()` — once per layer, on the hot path, twice per request (prefill + decode). Pushing ndarrays through end-to-end skips the per-layer conversion entirely while keeping the hash bit-identical. The benchmarks show:

`hash_steering_config` — uniform 7.7–8.3× across the sweep

| layers / hooks / overlap | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 3_hooks / base_only | 5.53 ms | 0.71 ms | 7.78× |
| 34 / 3_hooks / base_plus_phase | 7.54 ms | 0.93 ms | 8.13× |
| 42 / 1_hook / base_plus_phase | 3.00 ms | 0.38 ms | 7.84× |
| 42 / 3_hooks / base_only | 9.53 ms | 1.14 ms | 8.34× |

`resolve_effective_vectors` — 1.13–1.63× when there's a merge

| layers / hooks / overlap | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 34 / 3_hooks / base_plus_phase | 20.58 ms | 13.27 ms | 1.55× |
| 42 / 3_hooks / base_plus_phase | 26.74 ms | 16.45 ms | 1.63× |
| 34 / 1_hook / base_plus_phase | 6.37 ms | 4.39 ms | 1.45× |
| 26 / 1_hook / base_only | 1.61 ms | 1.64 ms | 0.98× |

Combined per-request savings (34-layer / 3-hooks / base+phase)

| stage | baseline | new | saved |
| --- | ---: | ---: | ---: |
| resolve (× 1) | 20.58 ms | 13.27 ms | 7.31 ms |
| hash (× 2, prefill + decode) | 15.08 ms | 1.86 ms | 13.22 ms |
| total | 35.66 ms | 15.13 ms | 20.53 ms saved per request |

## Steering table/index building

This isn't strictly config registration, but the same per-config bookkeeping shows up again on the per-step populate path, and the wins there compound with the registration ones.

One of the more significant changes was something that I mentioned in the previous post, which I erroneously thought that I would need a triton kernel for but ended up being able to do with regular pytorch code. This is the consolidation of the steering tables into a contiguous tensor that gets copied to the GPU in one go. The work done here is called on every decode step, so bringing this down is important.

The code I actually used to do this is a bit verbose for putting in a blog post, but the basic idea is: order the configs by layer and by hook point within the layers, build a list of tensors that respects the ordering, use torch.stack(flat_rows).reshape(len(active_tables), num_rows, hidden_size) to create the contiguous tensor, move it to the GPU in one go, and then build the steering index to target this new format. The benchmarks for this showed very nice gains on the populate_steering_tables path:

| layers / configs / hooks | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 1 / 3_hooks | 1.64 ms | 0.83 ms | 1.97× |
| 26 / 16 / 3_hooks | 3.40 ms | 2.72 ms | 1.25× |
| 34 / 8 / 3_hooks | 3.16 ms | 2.08 ms | 1.52× |
| 42 / 1 / 3_hooks | 2.56 ms | 1.31 ms | 1.96× |
| 42 / 16 / 3_hooks | 5.46 ms | 4.26 ms | 1.28× |

Building the steering index had a similar optimization path, replacing the per-request slice-assign loop with a single `np.repeat` that expands the per-request rows into the per-token row array, then copies it in a single non-blocking H2D transfer:

```python
expanded = np.repeat(
    rows_scratch[:active_count],
    n_tokens_scratch[:active_count],
)
n_expanded = int(expanded.shape[0])
n_expanded = min(n_expanded, index_pinned.shape[0], steering_index.shape[0])

index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
steering_index[:n_expanded].copy_(
    index_pinned[:n_expanded], non_blocking=True
)
```

The benchmarks here were even better than the steering table population, with the caveat that the `np.repeat` + pinned-copy path has fixed setup cost that doesn't amortize at batch size 1 — there, the old per-element write into a single GPU slot was actually cheaper. I'm generally chasing speedups on configurations that resemble production workloads, so this is an acceptable tradeoff when the mixed steered/unsteered batch-size-16 path has a 3× improvement:

| scenario | num_reqs × tokens | baseline | new | speedup |
| --- | --- | ---: | ---: | ---: |
| decode_128 | 128 × 1 | 0.910 ms | 0.110 ms | 8.36× |
| decode_64 | 64 × 1 | 0.467 ms | 0.070 ms | 6.70× |
| decode_32 | 32 × 1 | 0.237 ms | 0.052 ms | 4.63× |
| decode_8 | 8 × 1 | 0.070 ms | 0.037 ms | 1.90× |
| mixed_16 | 1×256 + 15×1 | 0.128 ms | 0.042 ms | 3.07× |
| decode_1 | 1 × 1 | 0.017 ms | 0.033 ms | 0.53× ↓ |
| prefill_1×1024 | 1 × 1024 | 0.017 ms | 0.033 ms | 0.53× ↓ |

## Named vector resolution

The named vector support was initially added as a UX feature to make constructing requests for vectors that you know ahead of time and intend to reuse regularly, as well
as to lower the amount of bytes over the wire when you're repeatedly sending requests that include it. This turned out to be an important feature for optimizing config registration though! By auto promoting vectors that are seen multiple times to a named module, the hot path for registration and hashing can be short circuited. 

To recap, the problem here was that originally I was resolving the vectors from the names passed by a request as part of the OpenAI serving handlers and storing the
registry table on the server side. This meant that ultimately the actual IPC traffic sent from the server to the workers was the same as if the named vectors weren't used
at all. Like I said earlier, this feature was originally conceived as a UX improvement not a performance optimization. By reframing it as an important way to sidestep
the config registration path and lower the traffic that a worker has to receive for vectors that are repeatedly reused, we can save a lot of overhead. 

The basic way of handling this is: make a worker side _steering_module_registry, populate it with engine_client.collective_rpc, and only pay the cost of registration once
instead of every time a vector is sent. The microbench shows that the not only does the named resolution cut almost the entire remaining per-request overhead, but it removes the fixed cost that previously scaled with num_layers and num_hook_points:

| layers / hooks | baseline RT (inline) | new RT (module_ref) | speedup | bytes (base → new) |
| --- | ---: | ---: | ---: | ---: |
| 26 / 1_hook | 7.02 ms | 0.010 ms | 676× | 599 583 → 214 |
| 26 / 3_hooks | 22.15 ms | 0.011 ms | 2 096× | 1 798 377 → 214 |
| 34 / 1_hook | 9.06 ms | 0.010 ms | 867× | 784 001 → 214 |
| 34 / 3_hooks | 28.89 ms | 0.011 ms | 2 740× | 2 351 649 → 214 |
| 42 / 3_hooks | 37.43 ms | 0.011 ms | 3 548× | 2 904 930 → 214 |

## Gather-add Fused Kernel

This is a tricky one. Previously, the indexed gather that pulls the correct steering configs for each token from the steering tables and the residual add that is the
actual steering operation were two separate operations. Typically, you wouldn't actually need to use a Triton kernel to get these to fuse when using torch.compile
because it does automatic kernel fusion for simple operations like this one. In our case though, it can't do this because the `apply_steering` custom op is registered
as opaque. It has to be opaque because everything starts off as zero vectors, so while writing this I found that Inductor optimizes it out because at compile time
this looks like a no-op. If we let this happen, steering never actually happens! This means that we need to fuse the operations manually with a custom Triton kernel.

The win is a little more than "one fewer kernel launch" though. The eager body does two passes over `hidden_states`: PyTorch's advanced indexing first materializes `table[index]` as a temporary, then the add reads `hidden_states` again to produce the output. The fused kernel streams `hidden_states` once, reads its row out of `table`, and writes the sum back in a single pass — half the memory bandwidth per call. Microbench shows that at prefill shapes (N=2048 tokens) this is roughly a 2× win on the op itself; at decode-N=1 it's closer to 10% (the launch-overhead share, which gets captured into the CUDA graph anyway and effectively disappears at replay time). Either way, the op fires for every hook point at every layer when steering is on, so it's worth doing.

The core operation is:

```python
pid_n = tl.program_id(axis=0)

row = tl.load(index_ptr + pid_n)

hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
table_row_ptr = table_ptr + row * t_stride_r
out_row_ptr = out_ptr + pid_n * o_stride_n

for h_off in range(0, H, BLOCK_H):
    h_idx = h_off + tl.arange(0, BLOCK_H)
    mask = h_idx < H
    h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
    t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)

    result = h_vals + t_vals.to(h_vals.dtype)
    tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)
```

There are some other pieces around choosing the `BLOCK_H` for your hardware and doing the kernel warmup ahead of graph capture so the JIT cost is frontloaded, but this is the meat of it.



## Things I tried that didn't pay off

For honesty's sake, a handful of things I tried alongside the wins above that ended up closed without merging. Each was a reasonable hypothesis on paper:

- zstd-compressing the packed inline vectors over IPC - Predicted ~30 ms/request on the assumption IPC bandwidth was the bottleneck. Bench showed a 15–64 ms regression on every shape — on a single-host loopback ZMQ deployment with PCIe 4.0, a 520 KB transfer takes ~17 μs, while zstd compress+decompress costs ~5–10 ms per request. The msgspec dict walk dominated the cost, not zstd, so a cheaper compressor wouldn't have saved it either.
- Shared-memory IPC - Client mmaps a region, ships only `(offset, length, dtype, shape)` — ~32 bytes per layer instead of the packed blob. Came out flat (±9 ms) on every benched shape, because once the inline-pack PR shrank the payload to model dtype, the residual per-request overhead was hash + msgspec walk + worker-side resolve, none of which mmap touches.
- In-place `apply_steering` - `mutates_args=["hidden_states"]` to halve per-call bandwidth (skip the fresh output allocation). Didn't land — the any-active short-circuit PR landed instead and partially subsumed the gain.
- Autotuning the steering Triton kernel over a few `BLOCK_H` / `num_warps` configs - Didn't beat the hand-picked heuristic enough to be worth the cache-key complexity.
- int16 `steering_index` to quarter the per-gather index bandwidth - Capped `max_steering_configs` at 32764, cast to int32 inside the kernel before stride math. Real but small win, and not net-positive enough to justify the new admission-time guard.
- Fusing PRE_ATTN + POST_ATTN into one kernel for gemma3 - Drops 28 launches per forward. Closed in favor of broader cross-model work like the any-active short-circuit, which gets most of the value without the per-model surgery.
- Fusing `apply_steering` with the following `GemmaRMSNorm` - Microbench showed 3.6–5.5× on the kernel itself, but kept the scope gemma3-only and didn't generalize cleanly.

# Other features

Beyond raw latency, a few capabilities landed alongside the perf work that are worth a mention:

- I made sure it supported tensor and pipeline parallelism, and because the post_mlp hook point comes after the down projection from the mlp block it ends up not needing any adaptation for expert parallelism. 
- I added an activation capture and consumer plugin system that allows defining workloads to run based on the output of activations. Originally, this was made to be 
just a way to write out the activations to a filesystem (usually an NFS volume) but I wanted to expand it to allow things like computing a reward signal based on 
activation similarity to a feature vector so that it can support training models like what Anthropic does in their recent paper on emotion vectors. One of the next things
I want to do is tie this together with the steering system to allow dynamic steering based on activations, which is something I mentioned wanting to do in my first
article but needed to lay this groundwork for.
- I'm expanding the steering methods by adding support for SAE feature clamping, both with clamping vectors and full encode->clamp->decode passes. After that comes
adding support for cross-layer transcoder based steering.

As always, if you want to talk to me about any of this or have ideas that would help your own research please reach out to me on X at @RhizoNymph
