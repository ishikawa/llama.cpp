#pragma once

#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Experimental host timing. Records are buffered until normal process exit.
LLAMA_API bool llama_step_timing_enabled();
// Step association is thread-local; -2 reads without changing it.
LLAMA_API int64_t llama_step_timing_step(int64_t step = -2);
LLAMA_API int64_t llama_step_timing_next_decode();
LLAMA_API void llama_step_timing_submit(const std::string & json);
LLAMA_API std::string llama_step_timing_quote(const std::string & value);

struct llama_step_timing_record {
    std::string type;
    std::string context;
    std::string caller;
    int64_t step;
    int64_t decode_id = -1;
    int64_t n_tokens = 0;
    int64_t n_outputs = 0;
    int64_t n_kv = -1; // Maximum attention KV extent across ubatches; -1 if unavailable.
    int64_t n_ubatches = 0;
    int64_t accepted = 0;
    int64_t drafted = 0;
    int64_t status = -1;
    bool reused = true;
    std::vector<std::string> reasons;
    std::vector<std::string> graphs;
    std::map<std::string, int64_t> us;
    int64_t start;
    int64_t last;
    llama_step_timing_record ** owner;

    llama_step_timing_record(const char * type, const char * context, llama_step_timing_record ** owner = nullptr)
        : type(type), context(context), step(llama_step_timing_step()), start(ggml_time_us()), last(start), owner(owner) {
        if (owner) {
            *owner = this;
        }
    }

    void mark(const char * phase) {
        const int64_t end = ggml_time_us();
        us[phase] += end - last;
        last = ggml_time_us();
    }

    ~llama_step_timing_record() {
        const int64_t total = ggml_time_us() - start;
        if (owner) {
            *owner = nullptr;
        }
        std::string json = "{\"type\":" + llama_step_timing_quote(type) + ",\"context\":" + llama_step_timing_quote(context);
        json += ",\"step\":" + std::to_string(step) + ",\"status\":" + std::to_string(status);
        if (type == "decode") {
            json += ",\"decode_id\":" + std::to_string(decode_id);
            json += ",\"n_tokens\":" + std::to_string(n_tokens) + ",\"n_outputs\":" + std::to_string(n_outputs);
            json += ",\"n_kv\":" + std::to_string(n_kv) + ",\"n_ubatches\":" + std::to_string(n_ubatches);
            json += ",\"reused\":" + std::string(reused ? "true" : "false") + ",\"reasons\":[";
            for (size_t i = 0; i < reasons.size(); ++i) {
                json += (i ? "," : "") + llama_step_timing_quote(reasons[i]);
            }
            json += "],\"graphs\":[";
            for (size_t i = 0; i < graphs.size(); ++i) {
                json += (i ? "," : "") + graphs[i];
            }
            json += "]";
        } else if (type == "step") {
            json += ",\"accepted\":" + std::to_string(accepted) + ",\"drafted\":" + std::to_string(drafted);
        } else if (type == "synchronize") {
            json += ",\"decode_id\":" + std::to_string(decode_id) + ",\"caller\":" + llama_step_timing_quote(caller);
        }
        json += ",\"total_us\":" + std::to_string(total) + ",\"us\":{";
        bool first = true;
        for (const auto & entry : us) {
            json += (first ? "" : ",") + llama_step_timing_quote(entry.first) + ":" + std::to_string(entry.second);
            first = false;
        }
        llama_step_timing_submit(json + "}}");
    }
};
