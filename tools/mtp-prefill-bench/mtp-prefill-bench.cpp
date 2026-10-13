#include "common.h"
#include "speculative.h"
#include "ggml-backend.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::puts("Usage: mtp-prefill-bench --model MODEL [options]\n"
              "  --mode all|target-only|target+draft|draft-only (default all)\n"
              "  --n-ctx N              context size (default 65536)\n"
              "  --target-ubatch N      target physical batch (default 512)\n"
              "  --draft-ubatch N       draft physical batch (default target value)\n"
              "                        LLAMA_SPEC_DRAFT_UBATCH additionally caps this value\n"
              "  --target-k TYPE --target-v TYPE (default f16)\n"
              "  --draft-k TYPE --draft-v TYPE   (default f16)\n"
              "  --prompt-length N      deterministic random tokens (default 32768)\n"
              "  --batch-length N       logical batch (default 2048)\n"
              "  --repetitions N        repetitions (default 3)\n"
              "  --gpu-layers N         offloaded layers (default 99)\n"
              "  --help\n"
              "draft-only first captures target+draft rows, then clears draft KV and replays.\n"
              "Capture copies and context initialization are excluded from timings.\n"
              "LLAMA_SPEC_MTP_PROFILE=1 enables synchronized per-ubatch profiling.");
}

static int32_t positive(const std::string & value) {
    size_t end;
    const long long n = std::stoll(value, &end);
    if (end != value.size() || n <= 0 || n > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("expected a positive 32-bit integer: " + value);
    }
    return (int32_t) n;
}

static ggml_type kv_type(const std::string & value) {
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                       GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                       GGML_TYPE_IQ4_NL }) {
        if (value == ggml_type_name(type)) {
            return type;
        }
    }
    throw std::invalid_argument("unsupported KV type: " + value);
}

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    try {
        common_params params;
        params.n_ctx = 65536;
        params.n_batch = 2048;
        params.n_ubatch = 512;
        params.n_gpu_layers = 99;
        params.n_parallel = 1;
        params.warmup = false;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        int32_t n_prompt = 32768;
        int32_t n_repetitions = 3;
        int32_t n_ubatch_dft = 0;
        std::string mode = "all";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            }
            check(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if      (arg == "--model")         { params.model.path = value; }
            else if (arg == "--mode")          { mode = value; }
            else if (arg == "--n-ctx")         { params.n_ctx = positive(value); }
            else if (arg == "--target-ubatch") { params.n_ubatch = positive(value); }
            else if (arg == "--draft-ubatch")  { n_ubatch_dft = positive(value); }
            else if (arg == "--batch-length")  { params.n_batch = positive(value); }
            else if (arg == "--prompt-length") { n_prompt = positive(value); }
            else if (arg == "--repetitions")   { n_repetitions = positive(value); }
            else if (arg == "--gpu-layers")    {
                size_t end;
                const long long n = std::stoll(value, &end);
                check(end == value.size() && n >= 0 && n <= std::numeric_limits<int32_t>::max(), "invalid GPU layer count");
                params.n_gpu_layers = (int32_t) n;
            }
            else if (arg == "--target-k")      { params.cache_type_k = kv_type(value); }
            else if (arg == "--target-v")      { params.cache_type_v = kv_type(value); }
            else if (arg == "--draft-k")       { params.speculative.draft.cache_type_k = kv_type(value); }
            else if (arg == "--draft-v")       { params.speculative.draft.cache_type_v = kv_type(value); }
            else { throw std::invalid_argument("unknown option: " + arg); }
        }
        check(!params.model.path.empty(), "--model is required");
        check(n_prompt <= params.n_ctx, "prompt exceeds context size");
        check(mode == "all" || mode == "target-only" || mode == "target+draft" || mode == "draft-only", "invalid mode");
        common_init();
        llama_backend_init();
        ggml_backend_load_all();
        auto target = common_init_from_params(params);
        auto * ctx_tgt = target->context();
        auto * model = target->model();
        check(ctx_tgt && model, "target initialization failed");
        check(n_prompt <= (int64_t) llama_n_ctx(ctx_tgt), "prompt exceeds allocated context size");
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        check(n_vocab > 0, "empty vocabulary");
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int32_t> token_dist(0, n_vocab - 1);
        llama_tokens tokens(n_prompt);
        for (auto & token : tokens) {
            token = token_dist(rng);
        }
        const bool run_only = mode == "all" || mode == "target-only";
        const bool replay = mode == "all" || mode == "draft-only";
        std::vector<std::vector<float>> hidden;
        std::printf("%-14s %4s %8s %8s %12s %12s %12s\n", "mode", "rep", "depth", "tokens", "target_ms", "draft_ms", "total_ms");
        auto report = [&](const char * name, int rep, int depth, int count, int64_t tgt, int64_t dft) {
            std::printf("%-14s %4d %8d %8d %12.3f %12.3f %12.3f\n", name, rep, depth, count,
                        tgt / 1000.0, dft / 1000.0, (tgt + dft) / 1000.0);
        };
        auto summary = [&](const char * name, int rep, int64_t total) {
            std::printf("total %-14s rep=%d ms=%.3f tokens=%d t/s=%.2f\n", name, rep, total / 1000.0,
                        n_prompt, total > 0 ? n_prompt * 1e6 / total : 0.0);
        };
        int64_t totals[3] = {};
        for (int rep = 1; rep <= n_repetitions; ++rep) {
            if (run_only) {
                llama_set_embeddings_nextn(ctx_tgt, false, false);
                llama_memory_clear(llama_get_memory(ctx_tgt), true);
                int64_t total = 0;
                for (int pos = 0; pos < n_prompt;) {
                    const int count = std::min(params.n_batch, n_prompt - pos);
                    common_batch batch(ctx_tgt);
                    for (int i = 0; i < count; ++i) {
                        batch.add(tokens[pos + i], pos + i, 0, i == count - 1);
                    }
                    const auto start = ggml_time_us();
                    check(llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target process failed");
                    llama_synchronize(ctx_tgt);
                    const auto elapsed = ggml_time_us() - start;
                    total += elapsed;
                    report("target-only", rep, pos, count, elapsed, 0);
                    pos += count;
                }
                totals[0] += total;
                summary("target-only", rep, total);
            }
            if (mode == "target-only") {
                continue;
            }
            check(llama_model_n_layer_nextn(model) == 1, "benchmark requires a single MTP head");
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.ctx_tgt = ctx_tgt;
            auto params_dft = common_base_params_to_speculative(params);
            params_dft.n_ubatch = n_ubatch_dft > 0 ? n_ubatch_dft : params.n_ubatch;
            auto draft = common_speculative_init_from_params(params_dft, model, ctx_tgt);
            auto * ctx_dft = draft->context();
            check(ctx_dft, "draft initialization failed");
            check(llama_get_ctx_other(ctx_dft) != ctx_tgt, "shared-KV MTP is not supported by this benchmark");
            params.speculative.draft.ctx_dft = ctx_dft;
            common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
            check(spec != nullptr, "speculative initialization failed");
            llama_memory_clear(llama_get_memory(ctx_tgt), true);
            hidden.clear();
            int64_t total = 0;
            const int32_t n_embd = llama_model_n_embd_out(model);
            check(n_embd > 0, "invalid hidden width");
            for (int pos = 0; pos < n_prompt;) {
                const int count = std::min(params.n_batch, n_prompt - pos);
                common_batch batch(ctx_tgt);
                for (int i = 0; i < count; ++i) {
                    batch.add(tokens[pos + i], pos + i, 0, i == count - 1);
                }
                const auto start = ggml_time_us();
                check(llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "target process failed");
                llama_synchronize(ctx_tgt);
                const auto t_target = ggml_time_us();
                check(common_speculative_process(spec.get(), batch), "draft process failed");
                llama_synchronize(ctx_dft);
                const auto end = ggml_time_us();
                total += end - start;
                report("target+draft", rep, pos, count, t_target - start, end - t_target);
                if (replay) {
                    const float * h = llama_get_embeddings_nextn(ctx_tgt);
                    check(h, "target hidden rows unavailable");
                    hidden.emplace_back(h, h + (size_t) count * n_embd);
                }
                pos += count;
            }
            totals[1] += total;
            summary("target+draft", rep, total);
            if (replay) {
                spec.reset();
                llama_memory_clear(llama_get_memory(ctx_dft), true);
                spec.reset(common_speculative_init(params.speculative, 1));
                check(spec != nullptr, "replay initialization failed");
                total = 0;
                size_t index = 0;
                for (int pos = 0; pos < n_prompt;) {
                    const int count = std::min(params.n_batch, n_prompt - pos);
                    common_batch batch(ctx_tgt);
                    for (int i = 0; i < count; ++i) {
                        batch.add(tokens[pos + i], pos + i, 0, i == count - 1);
                    }
                    const auto start = ggml_time_us();
                    check(common_speculative_process_mtp_hidden(spec.get(), batch, hidden[index++]), "draft replay failed");
                    llama_synchronize(ctx_dft);
                    const auto elapsed = ggml_time_us() - start;
                    total += elapsed;
                    report("draft-only", rep, pos, count, 0, elapsed);
                    pos += count;
                }
                totals[2] += total;
                summary("draft-only", rep, total);
            }
        }
        const char * names[] = { "target-only", "target+draft", "draft-only" };
        for (int i = 0; i < 3; ++i) {
            if ((i == 0 && run_only) || (i == 1 && mode != "target-only") || (i == 2 && replay)) {
                std::printf("aggregate %-14s reps=%d ms=%.3f t/s=%.2f\n", names[i], n_repetitions,
                            totals[i] / 1000.0, totals[i] > 0 ? (double) n_prompt * n_repetitions * 1e6 / totals[i] : 0.0);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "mtp-prefill-bench: %s\n", e.what());
        return 1;
    }
}
