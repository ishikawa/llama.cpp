#include "llama-step-timing.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {
struct step_timing_sink {
    std::string path;
    std::string buffer;
    std::mutex mutex;

    step_timing_sink() {
        const char * value = std::getenv("LLAMA_STEP_TIMING");
        if (value && *value) {
            path = value;
        }
    }

    ~step_timing_sink() {
        if (path.empty()) {
            return;
        }
        FILE * file = std::fopen(path.c_str(), "w");
        if (!file) {
            std::fprintf(stderr, "LLAMA_STEP_TIMING: cannot open %s\n", path.c_str());
            return;
        }
        if (std::fwrite(buffer.data(), 1, buffer.size(), file) != buffer.size()) {
            std::fprintf(stderr, "LLAMA_STEP_TIMING: write failed for %s\n", path.c_str());
        }
        if (std::fclose(file) != 0) {
            std::fprintf(stderr, "LLAMA_STEP_TIMING: close failed for %s\n", path.c_str());
        }
    }
};

step_timing_sink & timing_sink() {
    static step_timing_sink sink;
    return sink;
}
}

bool llama_step_timing_enabled() {
    return !timing_sink().path.empty();
}

int64_t llama_step_timing_step(int64_t step) {
    static thread_local int64_t current = -1;
    if (step != -2) {
        current = step;
    }
    return current;
}

int64_t llama_step_timing_next_decode() {
    static int64_t next = 0;
    auto & sink = timing_sink();
    std::lock_guard<std::mutex> lock(sink.mutex);
    return ++next;
}

void llama_step_timing_submit(const std::string & json) {
    auto & sink = timing_sink();
    std::lock_guard<std::mutex> lock(sink.mutex);
    sink.buffer += json;
    sink.buffer += '\n';
}

std::string llama_step_timing_quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += c;
        } else if (c < 0x20) {
            const char * hex = "0123456789abcdef";
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += c;
        }
    }
    return result + "\"";
}
