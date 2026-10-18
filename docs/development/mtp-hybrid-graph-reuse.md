# MTP hybrid graph reuse investigation

The aggregate `input.llm_graph_input_mem_hybrid` rejection does not identify which predicate failed. Without model execution, the reported 24 hybrid rejections cannot be assigned to individual predicates. Do not remove these checks based on the aggregate counter.

## Topology dependencies

- `llm_graph_input_mem_hybrid::can_reuse` checks token count, attention mask dimensions, recurrent state count, sequence count, recurrent head, zero-state row, and the three GDN identity choices.
- `llama_kv_cache::get_n_kv` rounds the highest occupied cell index plus one to `max(n_pad, 256)`, bounded by cache capacity. Two speculative tokens can cross a rounding boundary, then rollback can cross it in the other direction. This changes both the mask dimensions and the K/V views. Matching token count alone does not permit reuse.
- `build_rs` embeds the recurrent head and zero-state row in view offsets and dimensions. They cannot be replaced by updated `s_copy` input values.
- Snapshot rollback itself is already dynamic: `llama_memory_recurrent_context::s_copy` selects `rs_idx * size + src0` and consumes the pending rollback index during input upload. With the gather path and otherwise equal topology, changing the snapshot plane does not require rebuilding.
- `LLAMA_GDN_SNAPSHOT_INPLACE` selects a different graph when `is_s_copy_snapshot_identity()` changes. A pending rollback makes that predicate false. A graph that reads current-state cache views must not be reused to read a rollback plane. The convolution and state-view identity choices have the same requirement.

## The generic input label

`llm_graph_input_i` is the default diagnostic name, not a concrete input type. In particular, Qwen4Exp's `llm_graph_input_kpool` and `llm_graph_input_qwen4exp_ple` override `can_reuse` but inherit this name. K-pool reuse checks token count, pool count, padded new-pool count, and `n_kv`. PLE checks row count. A generic-name rejection is not evidence that the base implementation was called. Record concrete input names and individual predicate values in a separate measurement change to attribute the three reported rejections.

## Safe alternatives

`LLAMA_GRAPH_REUSE_HYBRID=1` disables the snapshot identity fast path at graph construction and in its matching reuse predicate. It uses the existing dynamic gather path even when `LLAMA_GDN_SNAPSHOT_INPLACE=1`. This avoids the topology transition caused by a pending snapshot rollback. It retains every shape, head, zero-row, and identity check; it does not force reuse. When unset, the existing behavior is unchanged.

For a snapshot-identity transition, compare runs with `LLAMA_GRAPH_REUSE_HYBRID=1` while keeping all other options identical. This gives up the in-place optimization to keep snapshot selection dynamic. It has no effect if the snapshot fast path was already disabled, and does not remove `n_kv` or pool-shape transitions.

For alternating `n_kv` shapes, retaining graph/allocation candidates keyed by full graph parameters is possible, but would retain a second target compute buffer. A larger padding quantum could reduce crossings, but increases attention work and must also be evaluated at mask boundaries. Neither alternative should force reuse of a differently shaped graph. Per-predicate measurements are needed before choosing either approach.

This finding is based on source inspection only. It does not establish the numerical distribution of the supplied rejection counts or a performance gain.
