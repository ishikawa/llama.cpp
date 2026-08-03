#pragma once

struct common_params;

// MoE router statistics collection for expert pruning (enabled via LLAMA_MOE_STATS=<path>).
// Hooks the eval callback to accumulate per-(layer, expert) selection counts and gate weight
// sums, dumped as JSON atexit, on SIGUSR1, and every LLAMA_MOE_STATS_INTERVAL=<seconds>
// when set (crash insurance for long collection runs). Semantics and caveats:
//   - the captured gate weight is the last "ffn_moe_weights*" tensor in the graph
//     (post-normalization/scaling), i.e. the actual mixing weight of the expert output
//   - a forward pass is committed when "result_output" is evaluated; prompt-processing
//     ubatches without logits do not commit, so the stats lean towards decode tokens
//   - concurrent slots are not separated (fine with --parallel 1); with -ngl > 0 the
//     per-layer host reads add a device sync each
//   - the collector and callback data are intentionally leaked so the SIGUSR1 thread and
//     atexit hook can never race static destruction
void common_moe_stats_maybe_init(common_params & params);
