# Host step timing

Set `LLAMA_STEP_TIMING` to a JSONL output path before starting the process. This measures host wall time with `ggml_time_us()`, independently of `GGML_METAL_DISPATCH_STATS`. It adds no backend synchronization. Graph submission time includes any blocking inside `graph_compute`; it is not GPU execution time. Getter synchronization is recorded separately.

The path is read once. Records are buffered in memory and written together at normal process exit, outside all measured regions. The output file is overwritten. Memory use grows with the number of records; abnormal termination does not flush the buffer. File errors are reported on stderr at exit.

With the variable unset or empty, decode and speculative-simple use the untimed template instantiations. The dispatch checks a cached bool once per decode (once at example entry). The compiler removes phase clocks, record construction, graph diagnostics, JSON formatting and buffer operations from that path. Synchronization checks the cached context bool once. No per-step `getenv` is used.

Example commands (these execute a model; they are not part of build verification):

```sh
LLAMA_STEP_TIMING=/tmp/mtp-step.jsonl build-decode-dev/bin/llama-speculative-simple \
    -m /path/to/model.gguf --spec-type draft-mtp --spec-draft-n-max 1 \
    -p 'Hello' -n 128
python3 scripts/step_timing_summary.py /tmp/mtp-step.jsonl --max-tokens 2 --skip-steps 5
python3 scripts/step_timing_summary.py /tmp/mtp-step.jsonl --max-tokens 2 --skip-steps 5 --json
```

## Records

All durations are integer microseconds. Common fields are `type`, `context`, `step`, `status`, `total_us`, and `us` (phase name to duration). Step numbering starts at 0; initialization and calls outside the example generation loop have step -1. `status` is 0 for completion, -1 for an incomplete/error path, or 1 for a step that restored a checkpoint and continued. It is not the decode return code.

### Decode

`type` is `decode`. `context` is `target` for a default context or `mtp` for an MTP context, selected from the existing context type.

Additional fields:

- `decode_id`: process-wide identifier linking synchronization records to the most recent decode on that context.
- `n_tokens`, `n_outputs`: counts for the logical decode batch.
- `n_ubatches`: number of graph reuse decisions made.
- `n_kv`: maximum padded attention KV extent across the batch, obtained from plain KV or hybrid attention contexts; -1 for other memory types.
- `reused`: true if every graph was reused; interpret together with `status` and `n_ubatches`.
- `reasons`: non-reuse reasons, one entry per observed failure. `graph_reuse_disable`, `active_mismatch:OLD->NEW`, `params.FIELD`, or `input.CLASS`. Parameter diagnostics report the first failed condition; input diagnostics name every top-level input that returned false. Short-circuited checks are not called again for diagnostics.
- `graphs`: one object per ubatch, with `slot`, `active_slot`, `n_tokens`, `n_outputs`, `n_kv`, and `reused`. Slot 0 means no outputs, slot 1 means outputs, and active slot -1 means no active graph.

`us` phases are `batch_prepare`, `memory_prepare`, `output_prepare`, `ubatch_prepare`, `memory_apply`, `reuse_check`, `reuse_sync` (reused graphs), `sched_reset`, `build_graph`, `sched_alloc_graph` (rebuilt graphs), `set_inputs`, `graph_compute`, `output_transfer`, and `output_mapping`. Repeated phases accumulate across ubatches. `sched_reset` includes graph-result reset; `output_transfer` includes asynchronous copy scheduling and associated output bookkeeping. Aborted paths can have missing phases.

### Step

`type` is `step` and `context` is `speculative-simple`. Additional fields are `drafted` (verified draft length) and `accepted` (number of draft tokens accepted by sampling, excluding the extra target token). A checkpoint-restoration step records the sampling result even though it restores state for the next iteration.

`us` phases are `checkpoint_prepare`, `draft`, `checkpoint_seq_rm`, `verify_batch`, `verify_decode`, `speculative_process`, `sampling_accept`, `output`, `seq_rm_target`, and `seq_rm_draft`. The checkpoint retry path uses `checkpoint_restore_seq_rm` and omits output/final removal. `draft` encloses the complete speculative draft call. `output` includes token conversion, display, and token bookkeeping.

### Synchronization

`type` is `synchronize`. Additional fields are `decode_id` and `caller`; `us.wait` measures the existing backend scheduler wait. The caller identifies getters, state APIs, scheduler reservation, context destruction, or pipeline-parallel input protection. A wait can cover work from more than one queued decode; `decode_id` identifies the most recent one, not exclusive ownership of all waited work.

## Summary

`--max-tokens N` excludes decode records with larger logical batches and their linked synchronization records. `--skip-steps N` excludes step numbers below N and all initialization records. To exclude prefill for draft-one MTP, use `--max-tokens 2` (a very short prefill may also have two tokens).

Phase times from all calls with the same step/context are summed before computing median and mean ms/step. Retained steps with no occurrence of a phase contribute zero. Initialization and other unstepped calls are reported separately as ms/record. Context statistics include graph reuse rates and counts of observed non-reuse reasons. Build/alloc distributions include only decodes where the phase executed, with count, mean, min, median, p90, p95 and max; multiple ubatches are summed per decode.

Step, decode, and synchronization timings overlap. Do not sum across record types. Opt-in timing itself adds host overhead, especially formatting and buffering nested records; `total_us` includes nested instrumentation while phase measurements exclude their own bookkeeping.

Verification without model/backend execution:

```sh
cmake --build build-decode-dev -j
python3 scripts/test_step_timing_summary.py
```

## First command buffer timing

Set `GGML_FIRST_CB_TIMING` to a separate JSONL path. Do not use the same path as `LLAMA_STEP_TIMING`: both overwrite their output at normal exit. The environment is cached once. When disabled, the backend paths check the cached flag without clocks, records, or extra synchronization. When enabled, decode IDs are allocated even if `LLAMA_STEP_TIMING` is absent. The additional records contain `decode_id`, `step`, `context`, logical decode `n_tokens`, process-wide `graph_id` (one per llama graph_compute call), and scheduler `split_id`. Calls outside decode use decode ID -1 and n_tokens 0.

`split` records include backend name, node count, absolute `start_s`, and host `us.input_copy` and `us.compute_async`. Input copy includes inter-backend dependency waits and input copy submission, including CPU PLE get_rows output copies to Metal. Compute time sums graph_compute_async calls, including callback subgraphs, but excludes callback evaluation and callback synchronization. These are host durations, not device copy durations.

`metal_graph` records include `graph_start_s` (llama graph_compute entry), `metal_start_s` (Metal graph_compute entry), and a `buffers` array in queue order (main buffer first, then worker buffers). Each buffer has `index`, `commit_s` (before commit), `commit_return_s`, `graph_to_commit_us`, `metal_to_commit_us` (both through commit return), `status`, and, for successful buffers with valid GPU timestamps, `gpu_start_s` and `gpu_end_s`. A GPU can start before commit returns; commit-to-GPU-start uses the timestamp before commit. Graph `gpu_start_s`/`gpu_end_s` are the earliest start and latest end. Graph `us` includes `graph_to_metal`, `graph_to_gpu_start`, and `metal_to_gpu_start`.

Absolute times are seconds on the mach_absolute_time clock on Apple platforms, matching Metal GPUStartTime/GPUEndTime. GPU results are read only after waiting for completion, at the next normal backend synchronize (or backend destruction). Pending graphs retain their command buffers across later submissions. No wait is inserted into graph_compute. Failed/uncommitted buffers have no GPU intervals; early aborts can omit a graph record. The memory buffer and retained command buffers grow until synchronization. Existing dispatch profiling can still introduce its own waits; leave `GGML_METAL_DISPATCH_STATS` unset when measuring normal scheduling.

The summary's `first_cb_ms_per_record` groups by context and logical n_tokens. It reports median and mean per split/Metal graph/CB occurrence, rather than per generation step. `gpu_idle` is the nonnegative interval from the preceding Metal graph's latest GPU end to this graph's earliest GPU start, across contexts and splits, ordered by GPU start. The predecessor is selected before warmup/prefill filtering. The first observed graph has no idle sample. This graph-to-graph gap can contain copy/event work and is not a proof that the entire GPU was idle. Multiple devices/independent queues must be measured in separate runs. JSONL line order follows flushing, not necessarily submission order.

Example commands below execute Metal and are provided for a separate measurement run; they are not build/CPU verification commands. Replace the model path and tensor placement expression for your model.

```sh
GGML_METAL_N_CB=4 GGML_FIRST_CB_TIMING=/tmp/mtp-cb.jsonl LLAMA_STEP_TIMING=/tmp/mtp-step.jsonl \
    build-decode-dev/bin/llama-speculative-simple -m /path/to/model.gguf \
    --spec-type draft-mtp --spec-draft-n-max 1 -ot '.*per_layer_token_embd.*=CPU' -p 'Hello' -n 128
GGML_METAL_N_CB=4 GGML_FIRST_CB_TIMING=/tmp/base-cb.jsonl LLAMA_STEP_TIMING=/tmp/base-step.jsonl \
    build-decode-dev/bin/llama-completion -m /path/to/model.gguf \
    -ot '.*per_layer_token_embd.*=CPU' -p 'Hello' -n 128
python3 scripts/step_timing_summary.py /tmp/mtp-step.jsonl /tmp/mtp-cb.jsonl --max-tokens 2 --skip-steps 5
python3 scripts/step_timing_summary.py /tmp/base-step.jsonl /tmp/base-cb.jsonl --max-tokens 2 --json
```
