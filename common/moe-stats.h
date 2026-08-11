#pragma once

struct common_params;
struct llama_vocab;

// MoE router statistics collection for expert pruning (enabled via LLAMA_MOE_STATS=<path>).
// Hooks the eval callback to accumulate per-(layer, expert) and per-(class, layer, expert)
// selection counts and gate weight sums, dumped as JSON atexit, on SIGUSR1, and every LLAMA_MOE_STATS_INTERVAL=<seconds>
// when set (crash insurance for long collection runs; SIGUSR1 and interval dumps are
// POSIX-only - on Windows only the atexit dump fires). Semantics and caveats:
//   - the captured gate weight is the last "ffn_moe_weights*" tensor in the graph
//     (post-normalization/scaling), i.e. the actual mixing weight of the expert output
//   - top-1 margin stats use the actual selection score space when available
//     (masked, then biased, then raw probs) plus raw probs for the mixing-space margin
//   - LLAMA_MOE_STATS_RAW=<path> also appends per-token raw router scores for
//     knockout replay; dumps are large ((1 + has_sel) x layers x tokens x n_expert
//     x 4B, i.e. doubled when a separate selection tensor exists), so use them only
//     with small calibration text; records are native-endian and flushed per record,
//     so a crash loses at most the record being written
//   - LLAMA4 assigns raw logits as the selection scores without a cb() name, so its
//     margins fall back to the probs space (ranking is unaffected, absolute
//     sel_margin values are not logit-space differences)
//   - layers whose experts are selected without argsort over router scores (e.g. the
//     hash-routed leading layers of deepseek4, selected_experts_in) still get margin
//     fields, but they are meaningless there (often negative) - filter them downstream
//   - a forward pass is committed when "result_output" is evaluated; prompt-processing
//     ubatches without logits do not commit, so the stats lean towards decode tokens
//   - concurrent slots are not separated (fine with --parallel 1); with -ngl > 0 the
//     per-layer host reads add a device sync each, plus one input-token read per forward
//   - the collector and callback data are intentionally leaked so the SIGUSR1 thread and
//     atexit hook can never race static destruction
void common_moe_stats_maybe_init(common_params & params);
void common_moe_stats_maybe_init_vocab(const llama_vocab * vocab);
