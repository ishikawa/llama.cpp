#pragma once

struct common_params;

// MoE router statistics collection for expert pruning.
// Target stats are enabled with LLAMA_MOE_STATS=<path>. Draft stats are enabled with
// LLAMA_MOE_STATS_DRAFT=<path>. The two collectors are independent and draft layer
// numbers are local to the draft model. The two paths must differ; if they are equal
// (string compare) the draft collector refuses to start.
// Hooks the eval callback to accumulate per-(layer, expert) selection counts and gate weight
// sums, dumped as JSON atexit, on SIGUSR1, and every LLAMA_MOE_STATS_INTERVAL=<seconds>
// when set (crash insurance for long collection runs; SIGUSR1 and interval dumps are
// POSIX-only - on Windows only the atexit dump fires). Semantics and caveats:
//   - the captured gate weight is the last "ffn_moe_weights*" tensor in the graph
//     (post-normalization/scaling), i.e. the actual mixing weight of the expert output
//   - a forward pass is committed when "result_output" is evaluated; prompt-processing
//     ubatches without logits do not commit, so the stats lean towards decode tokens
//   - for draft-dspark, one draft block decode graph with "result_output" is one commit;
//     KV injection and encoder passes have no matching tensors and are skipped naturally
//   - concurrent slots are not separated (fine with --parallel 1); with -ngl > 0 the
//     per-layer host reads add a device sync each
//   - the collector and callback data are intentionally leaked so the SIGUSR1 thread and
//     atexit hook can never race static destruction
void common_moe_stats_maybe_init(common_params & params);
void common_moe_stats_maybe_init_draft(common_params & params);
