# MTP prefill benchmark

This tool uses the server's `common_speculative_init_from_params`, `common_speculative_init`, and `common_speculative_process` path. It supports single-head MTP models with separate draft KV, including Qwen35 and Qwen35MoE models. Use an MTP-enabled GGUF.

```sh
build-mtpbench/bin/mtp-prefill-bench --help
LLAMA_SPEC_DRAFT_UBATCH=512 build-mtpbench/bin/mtp-prefill-bench \
    --model /path/to/model-with-mtp.gguf --mode all \
    --n-ctx 65536 --prompt-length 65536 --batch-length 2048 \
    --target-ubatch 2048 --draft-ubatch 1024 \
    --target-k q8_0 --target-v q8_0 --draft-k f16 --draft-v f16 \
    --repetitions 3
```

`LLAMA_SPEC_DRAFT_UBATCH` caps `--draft-ubatch` using the same initialization code as the server. Flash attention is enabled. Random tokens use seed 1234. Context initialization, KV clearing, batch construction in the tool, and capture copies are outside the timed sections. There is no warmup; inspect the repetitions separately to identify first-use effects. Each mode processes exactly N tokens, with one logits output at the end of each logical batch.

- `target-only` creates no draft context and disables nextn extraction.
- `target+draft` processes each target batch, then calls the speculative process hook. It includes target nextn extraction and draft catch-up.
- `draft-only` first runs and reports `target+draft` to capture rows, then clears draft KV and recreates the speculative driver before replay. It preserves cross-batch pending hidden rows and uses the same process hook with captured target rows. It includes host batch assembly and hidden-row copies, but runs no target computation during replay.
- `all` runs all three modes in that order per repetition.

Replay stores N x hidden-width float32 values on the host. For long prompts this can consume substantial RAM. Capture can also affect caches and memory pressure. The tool prints batch depth, token count, target/draft/total wall time, per-repetition totals, and final aggregate throughput. Compare equal-sized batches when inspecting the draft-time slope; the first batch can include graph allocation.

## Optional internal profile

Set `LLAMA_SPEC_MTP_PROFILE=1` for the tool or server. The value is read once and cached; other values disable profiling. Example format (illustrative values, not measured results):

```text
mtp_profile: tokens=2048 pos_first=4096 pos_last=6143 host_ms=3.250 draft_ms=42.500
mtp_profile_ubatch: tokens=512 pos_first=4096 pos_last=4607 graph_ms=0.350 inputs_ms=0.800 compute_ms=9.100
```

The process line's `host_ms` includes sequence bookkeeping, hidden-state retrieval, draft batch assembly, verification-row copies, and pending-row updates. `draft_ms` includes the draft `llama_process` call and synchronization. Each successful nonempty token-only single-head process call emits one line.

Ubatch timing uses existing `ggml_time_us()` graph-build/input timing locations in `llama_context::process_ubatch` and the backend scheduler's synchronization:

- `graph_ms`: memory-context application, graph parameters/reuse, build and allocation. Reused graphs skip build/allocation. Scheduler reset and pipeline waits in this phase are included.
- `inputs_ms`: `set_inputs`, including embedding uploads, followed by scheduler synchronization.
- `compute_ms`: graph submission and scheduler synchronization until completion.

These are wall-clock phases, not device kernel timers. Output extraction and decode bookkeeping outside `process_ubatch` remain included in `draft_ms`; the ubatch times need not sum to that value. Profiling serializes ubatches and input uploads and includes log overhead in the outer call, so use profiling-disabled runs for throughput comparisons. Disabled profiling adds cached flag checks and branches, with no additional clocks, synchronization, or logs. Existing perf context totals do not separate these phases, and backend kernel timing alone would omit host construction/upload/waits.
