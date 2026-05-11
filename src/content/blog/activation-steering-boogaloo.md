---
title: Activation Steering in vLLM
pubDate: 2026-05-10
---
# Activation Steering Part 2: Electric Boogaloo

In my first blog on activation steering in vLLM (which you can read (here)[https://www.rhizonymph.com/blog/activaton-steering/]) I walked through the work I've
been doing adding support for steering vectors to vLLM. In it, I showed that in the end there was still real overhead vs the steering disabled case and detailed
some things that I planned to do in the future to try and close the gap. Since then, I've tried these out and tried other things I hadn't thought of back then that
further profiling revealed I should tackle. In this post I'm going to go over what I tried, what worked, and what didn't. I'm going to stick with the microbenches from
(my benchmark suite)[https://www.github.com/RhizoNymph/steering-bench] but at the end we're going to compare the latency/throughput benchmarks that I posted results
from in the last article.

Let us start with reviewing the optimization roadmap that I ended with last time:
- Optimize config registration to further reduce per request overhead, the dominant remaining overhead at larger batch sizes and short workloads
- Name to vector resolution should happen on the worker side
- Write a custom Triton kernel for indexed scatter to steering tables 

## Config registration

Config registration itself (SteeringManager.register_config) which gets called when a new (hash, phase) config is admitted I was able to pretty simply batch into a single non-blocking H2D transfer instead of the previous approach that was a different cudaMemcpy for every (layer, hook) combination.

```
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
    cpu_t = torch.from_numpy(arr)
    stacked = cpu_t.to(self.device)
    
    stored[hook_point] = {
        layer_idx: stacked[i : i + 1] for i, layer_idx in enumerate(layer_idxs)
    }
    
self.config_vectors[key] = stored
```

The microbench for the steering manager showed that this gave 86-93% speedup with this change:

| layers / configs / hooks | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 1 / 1_hook | 3.46 ms | 1.79 ms | 1.93× |
| 26 / 16 / 3_hooks | 162.2 ms | 87.1 ms | 1.86× |
| 34 / 16 / 3_hooks | 212.2 ms | 113.9 ms | 1.86× |
| 42 / 16 / 3_hooks | 263.0 ms | 140.9 ms | 1.87× |

Another change made to the registration process was something I feel a bit silly for needing to do in the first place, using numpy arrays instead of lists of floats in
the vector resolution and config hashing parts of this path. This allowed me to eliminate a per layer np.asarray(list, dtype=fp32) call while maintaining the hashes exactly. The benchmarks show:

`hash_steering_config` — uniform 7.7–8.3× across the sweep

| layers / hooks / overlap | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 3_hooks / base_only | 5.53 ms | 0.71 ms | 7.78× |
| 34 / 3_hooks / base_plus_phase | 7.54 ms | 0.93 ms | **8.13×** |
| 42 / 1_hook / base_plus_phase | 3.00 ms | 0.38 ms | 7.84× |
| 42 / 3_hooks / base_only | 9.53 ms | 1.14 ms | **8.34×** |

`resolve_effective_vectors` — 1.13–1.63× when there's a merge

| layers / hooks / overlap | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 34 / 3_hooks / base_plus_phase | 20.58 ms | 13.27 ms | 1.55× |
| 42 / 3_hooks / base_plus_phase | 26.74 ms | 16.45 ms | **1.63×** |
| 34 / 1_hook / base_plus_phase | 6.37 ms | 4.39 ms | 1.45× |
| 26 / 1_hook / base_only | 1.61 ms | 1.64 ms | 0.98× |

Combined per-request savings (34-layer / 3-hooks / base+phase)

| stage | baseline | new | saved |
| --- | ---: | ---: | ---: |
| resolve (× 1) | 20.58 ms | 13.27 ms | 7.31 ms |
| hash (× 2, prefill + decode) | 15.08 ms | 1.86 ms | **13.22 ms** |
| **total** | 35.66 ms | 15.13 ms | **20.53 ms saved per request** |

## Steering table/index building

While not strictly config registration, the building of the steering tables and index had a lot of improvement to make and is related to the per-config paths
that showed to be a significant source of overhead.

One of the more significant changes was something that I mentioned in the previous post, which I erroneously thought that I would need a triton kernel for but ended up being able to do with regular pytorch code. This is the consolidation of the steering tables into a contiguous tensor that gets copied to the GPU in one go. The work done here is called on every decode step, so bringing this down is important.

The code I actually used to do this is a bit verbose for putting in a blog post, but the basic idea is: order the configs by layer and by hook point within the layers, build a list of tensors that respects the ordering, use torch.stack(flat_rows).reshape(len(active_tables), num_rows, hidden_size) to create the contiguous tensor, move it to the GPU in one go, and then build the steering index to target this new format. The benchmarks for this showed very nice gains on the populate_steering_tables path:

| layers / configs / hooks | baseline | new | speedup |
| --- | ---: | ---: | ---: |
| 26 / 1 / 3_hooks | 1.64 ms | 0.83 ms | **1.97×** |
| 26 / 16 / 3_hooks | 3.40 ms | 2.72 ms | 1.25× |
| 34 / 8 / 3_hooks | 3.16 ms | 2.08 ms | 1.52× |
| 42 / 1 / 3_hooks | 2.56 ms | 1.31 ms | 1.96× |
| 42 / 16 / 3_hooks | 5.46 ms | 4.26 ms | 1.28× |

Building the steering index had a similar optimization path, replacing the per request assignment loop that uses np.repeat to expand the per-request rows into the per-token row array and copy it in a single non blocking H2D transfer: 

```
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

The benchmarks here were even better than the steering table population, though with the caveat that the fixed cost of performing the operation means that the batch size 1 scenarios are slower. This is unfortunate, but I'm generally chasing speedups on configurations that more closely resemble production workloads so this is an acceptable tradeoff when the mixed steered/unsteered batch size 16 path has a 3x improvement:

| scenario | num_reqs × tokens | baseline | new | speedup |
| --- | --- | ---: | ---: | ---: |
| **decode_128** | 128 × 1 | 0.910 ms | 0.110 ms | **8.36×** |
| **decode_64** | 64 × 1 | 0.467 ms | 0.070 ms | **6.70×** |
| decode_32 | 32 × 1 | 0.237 ms | 0.052 ms | 4.63× |
| decode_8 | 8 × 1 | 0.070 ms | 0.037 ms | 1.90× |
| **mixed_16** | 1×256 + 15×1 | 0.128 ms | 0.042 ms | **3.07×** |
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
| 34 / 3_hooks | 28.89 ms | 0.011 ms | **2 740×** | 2 351 649 → 214 |
| 42 / 3_hooks | 37.43 ms | 0.011 ms | **3 548×** | 2 904 930 → 214 |

## Gather-add Fused Kernel

This is a tricky one. Previously, the indexed gather that pulls the correct steering configs for each token from the steering tables and the residual add that is the
actual steering operation were two separate operations. Typically, you wouldn't actually need to use a Triton kernel to get these to fuse when using torch.compile
because it does automatic kernel fusion for simple operations like this one. In our case though, it can't do this because the apply_steering custom op is registered 
as opaque. It has to be opaque because everything starts off as zero vectors, so while writing this I found that Inductor optimizes it out because at compile time
this looks like a no-op. If we let this happen, steering never actually happens! This means that we need to fuse the operations manually with a custom Triton kernel.
This is a very simple operation so it's reasonable to do this. It's a small win, only removing one kernel launch between each operation, but this happens for every hook
point at every layer when you're steering so it's still worth it.

The core operation is:

```
pid_n = tl.program_id(axis=0)

row = tl.load(index_ptr + pid_n)

hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
table_row_ptr = table_ptr + row * t_stride_r
out_row_ptr = out_ptr * pid_n * o_stride_n

for h_off in range(0, H, BLOCK_H):
  h_idx = h_off + tl.arange(0, BLOCK_H)
  mask = h_idx < H
  h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
  t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)

  result = h_vals + t_vals.to(h_vals)
  tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)
```

There are some other pieces around choosing the block_H for your hardware and doing the kernel warmup ahead of graph capture so the JIT cost is frontloaded,
but this is the meat of it. The benchmarks show that this gives about a 4% reduction end to end latency under graph capture, which is small but respectable.

<div class="table-row">

| label | bs | pre-merge | post-merge | Δ |
| :--- | :--- | :--- | :--- | :--- |
| graphs_with_steer | 1 | 1575.6 ms | 1519.0 ms | 1.04× |
| graphs_with_steer | 4 | 1613.8 ms | 1554.7 ms | 1.04× |
| graphs_with_steer | 8 | 1706.6 ms | 1644.3 ms | 1.04× |
| eager_with_steer | 1 | 5237.6 ms | 5240.6 ms | 1.00× |
| eager_with_steer | 4 | 5247.2 ms | 5300.1 ms | 0.99× |
| eager_with_steer | 8 | 5336.1 ms | 5421.0 ms | 0.98× |

</div>

# Other features

I thought I would shout out some other things I've been working on in this fork that aren't just the performance of the straight steering vector functionality. 

- I made sure it supported tensor and pipeline parallelism, and because the post_mlp hook point comes after the down projection from the mlp block it ends up not needing any adaptation for expert parallelism. 
- I added an activation capture and consumer plugin system that allows defining workloads to run based on the output of activations. Originally, this was made to be 
just a way to write out the activations to a filesystem (usually an NFS volume) but I wanted to expand it to allow things like computing a reward signal based on 
activation similarity to a feature vector so that it can support training models like what Anthropic does in their recent paper on emotion vectors. One of the next things
I want to do is tie this together with the steering system to allow dynamic steering based on activations, which is something I mentioned wanting to do in my first
article but needed to lay this groundwork for.
- I'm expanding the steering methods by adding support for SAE feature clamping, both with clamping vectors and full encode->clamp->decode passes. After that comes
adding support for cross-layer transcoder based steering.

As always, if you want to talk to me about any of this or have ideas that would help your own research please reach out to me on X at @RhizoNymph
