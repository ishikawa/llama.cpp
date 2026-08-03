#include "moe-stats.h"

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

struct common_moe_stats_tensor {
    ggml_type type = GGML_TYPE_COUNT;
    int64_t   ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    size_t    nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    std::vector<uint8_t> data;
};

struct common_moe_stats_layer_pending {
    bool topk_ready    = false;
    bool weights_ready = false;

    common_moe_stats_tensor topk;
    common_moe_stats_tensor weights;
};

struct common_moe_stats_expert {
    uint64_t count      = 0;
    double   weight_sum = 0.0;
};

static std::string common_moe_stats_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);

    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0x0f];
                } else {
                    out += (char) c;
                }
                break;
        }
    }

    return out;
}

class common_moe_stats_collector {
public:
    void configure(const std::string & output_path, const std::string & model_path) {
        std::lock_guard<std::mutex> lock(mutex);

        if (this->output_path.empty()) {
            this->output_path = output_path;
        } else if (this->output_path != output_path) {
            LOG_WRN("%s: LLAMA_MOE_STATS changed from '%s' to '%s', keeping first path\n", __func__, this->output_path.c_str(), output_path.c_str());
        }

        if (this->model_path.empty()) {
            this->model_path = model_path;
        } else if (this->model_path != model_path && !warned_multiple_models) {
            warned_multiple_models = true;
            LOG_WRN("%s: multiple model paths observed, stats file will use first model path '%s'\n", __func__, this->model_path.c_str());
        }
    }

    void add_forward(std::map<int, std::map<int, common_moe_stats_expert>> && forward_stats, uint64_t n_tokens) {
        std::lock_guard<std::mutex> lock(mutex);

        total_tokens += n_tokens;

        for (const auto & layer_it : forward_stats) {
            auto & layer = stats[layer_it.first];
            for (const auto & expert_it : layer_it.second) {
                auto & dst = layer[expert_it.first];
                dst.count      += expert_it.second.count;
                dst.weight_sum += expert_it.second.weight_sum;
            }
        }
    }

    void dump() {
        std::lock_guard<std::mutex> lock(mutex);

        if (output_path.empty()) {
            return;
        }

        // write to a temp file and rename so a crash mid-dump cannot destroy the
        // previously accumulated stats file
        const std::string tmp_path = output_path + ".tmp";

        std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
        if (!out) {
            LOG_ERR("%s: failed to open '%s'\n", __func__, tmp_path.c_str());
            return;
        }

        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "{\n";
        out << "  \"model\": \"" << common_moe_stats_json_escape(model_path) << "\",\n";
        out << "  \"total_tokens\": " << total_tokens << ",\n";
        out << "  \"layers\": {";

        bool first_layer = true;
        for (const auto & layer_it : stats) {
            out << (first_layer ? "\n" : ",\n");
            first_layer = false;
            out << "    \"" << layer_it.first << "\": {";

            bool first_expert = true;
            for (const auto & expert_it : layer_it.second) {
                out << (first_expert ? "\n" : ",\n");
                first_expert = false;
                out << "      \"" << expert_it.first << "\": {\"count\": " << expert_it.second.count << ", \"weight_sum\": " << expert_it.second.weight_sum << "}";
            }

            if (!layer_it.second.empty()) {
                out << "\n";
            }
            out << "    }";
        }

        if (!stats.empty()) {
            out << "\n";
        }
        out << "  }\n";
        out << "}\n";

        out.close();
        if (!out) {
            LOG_ERR("%s: failed to write '%s'\n", __func__, tmp_path.c_str());
            return;
        }

        if (std::rename(tmp_path.c_str(), output_path.c_str()) != 0) {
            LOG_ERR("%s: failed to rename '%s' to '%s'\n", __func__, tmp_path.c_str(), output_path.c_str());
        }
    }

private:
    std::mutex mutex;
    std::string output_path;
    std::string model_path;
    std::map<int, std::map<int, common_moe_stats_expert>> stats;
    uint64_t total_tokens = 0;
    bool warned_multiple_models = false;
};

struct common_moe_stats_cb_data {
    explicit common_moe_stats_cb_data(common_moe_stats_collector & collector) : collector(collector) {}

    common_moe_stats_collector & collector;
    std::mutex mutex;
    std::map<int, common_moe_stats_layer_pending> pending;
    bool warned_bad_topk    = false;
    bool warned_bad_weights = false;
};

static common_moe_stats_collector & common_moe_stats_get_collector() {
    static common_moe_stats_collector * collector = new common_moe_stats_collector();
    return *collector;
}

static bool common_moe_stats_name_layer(const char * name, const char * prefix, int & layer) {
    const size_t prefix_len = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_len) != 0 || name[prefix_len] != '-') {
        return false;
    }

    const char * p = name + prefix_len + 1;
    if (*p < '0' || *p > '9') {
        return false;
    }

    int value = 0;
    while (*p >= '0' && *p <= '9') {
        value = value*10 + (*p - '0');
        ++p;
    }

    if (*p != '\0') {
        return false;
    }

    layer = value;
    return true;
}

static bool common_moe_stats_topk_name(const char * name, int & layer) {
    return common_moe_stats_name_layer(name, "ffn_moe_topk", layer);
}

static bool common_moe_stats_weights_name(const char * name, int & layer) {
    return common_moe_stats_name_layer(name, "ffn_moe_weights",         layer) ||
           common_moe_stats_name_layer(name, "ffn_moe_weights_softmax", layer) ||
           common_moe_stats_name_layer(name, "ffn_moe_weights_norm",    layer) ||
           common_moe_stats_name_layer(name, "ffn_moe_weights_scaled",  layer);
}

static bool common_moe_stats_wants_tensor(const char * name) {
    int layer;
    return std::strcmp(name, "result_output") == 0 ||
           common_moe_stats_topk_name(name, layer) ||
           common_moe_stats_weights_name(name, layer);
}

static common_moe_stats_tensor common_moe_stats_copy_tensor(const ggml_tensor * t) {
    common_moe_stats_tensor result;
    result.type = t->type;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }

    const size_t nbytes = ggml_nbytes(t);
    result.data.resize(nbytes);
    if (nbytes == 0) {
        return result;
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr || ggml_backend_buffer_is_host(buffer)) {
        std::memcpy(result.data.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, result.data.data(), 0, nbytes);
    }

    return result;
}

static int32_t common_moe_stats_get_i32(const common_moe_stats_tensor & t, int64_t i0, int64_t i1) {
    const size_t off = i1*t.nb[1] + i0*t.nb[0];
    return *(const int32_t *) (t.data.data() + off);
}

static double common_moe_stats_get_float(const common_moe_stats_tensor & t, int64_t i0, int64_t i1, int64_t i2) {
    const size_t off = i2*t.nb[2] + i1*t.nb[1] + i0*t.nb[0];

    switch (t.type) {
        case GGML_TYPE_F32:
            return *(const float *) (t.data.data() + off);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (t.data.data() + off));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (t.data.data() + off));
        default:
            return 0.0;
    }
}

static bool common_moe_stats_commit_layer(
        const common_moe_stats_layer_pending & pending,
        std::map<int, common_moe_stats_expert> & layer_stats,
        bool & warned_bad_topk,
        bool & warned_bad_weights) {
    const common_moe_stats_tensor & topk    = pending.topk;
    const common_moe_stats_tensor & weights = pending.weights;

    if (topk.type != GGML_TYPE_I32 || topk.ne[0] <= 0 || topk.ne[1] <= 0) {
        if (!warned_bad_topk) {
            warned_bad_topk = true;
            LOG_WRN("%s: unexpected ffn_moe_topk tensor type or shape\n", __func__);
        }
        return false;
    }

    if (weights.type != GGML_TYPE_F32 && weights.type != GGML_TYPE_F16 && weights.type != GGML_TYPE_BF16) {
        if (!warned_bad_weights) {
            warned_bad_weights = true;
            LOG_WRN("%s: unexpected ffn_moe_weights tensor type\n", __func__);
        }
        return false;
    }

    const int64_t n_expert_used = topk.ne[0];
    const int64_t n_tokens      = topk.ne[1];
    const bool weights_3d = weights.ne[0] == 1 && weights.ne[1] == n_expert_used && weights.ne[2] == n_tokens;
    const bool weights_2d = weights.ne[0] == n_expert_used && weights.ne[1] == n_tokens;

    if (!weights_3d && !weights_2d) {
        if (!warned_bad_weights) {
            warned_bad_weights = true;
            LOG_WRN("%s: ffn_moe_topk and ffn_moe_weights shapes do not match\n", __func__);
        }
        return false;
    }

    for (int64_t it = 0; it < n_tokens; ++it) {
        for (int64_t ie = 0; ie < n_expert_used; ++ie) {
            const int32_t expert = common_moe_stats_get_i32(topk, ie, it);
            if (expert < 0) {
                continue;
            }

            const double weight = weights_3d ?
                common_moe_stats_get_float(weights, 0, ie, it) :
                common_moe_stats_get_float(weights, ie, it, 0);

            auto & stat = layer_stats[expert];
            stat.count += 1;
            stat.weight_sum += weight;
        }
    }

    return true;
}

static void common_moe_stats_commit_forward(common_moe_stats_cb_data & cb_data) {
    std::map<int, std::map<int, common_moe_stats_expert>> forward_stats;
    uint64_t n_tokens = 0;

    {
        std::lock_guard<std::mutex> lock(cb_data.mutex);

        for (const auto & layer_it : cb_data.pending) {
            const auto & pending = layer_it.second;
            if (!pending.topk_ready) {
                continue;
            }

            n_tokens = std::max<uint64_t>(n_tokens, pending.topk.ne[1] > 0 ? (uint64_t) pending.topk.ne[1] : 0);

            if (!pending.weights_ready) {
                continue;
            }

            std::map<int, common_moe_stats_expert> layer_stats;
            if (common_moe_stats_commit_layer(
                pending,
                layer_stats,
                cb_data.warned_bad_topk,
                cb_data.warned_bad_weights) && !layer_stats.empty()) {
                forward_stats[layer_it.first] = std::move(layer_stats);
            }
        }

        cb_data.pending.clear();
    }

    if (!forward_stats.empty() || n_tokens > 0) {
        cb_data.collector.add_forward(std::move(forward_stats), n_tokens);
    }
}

static bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return common_moe_stats_wants_tensor(t->name);
    }

    auto & cb_data = *(common_moe_stats_cb_data *) user_data;

    if (std::strcmp(t->name, "result_output") == 0) {
        common_moe_stats_commit_forward(cb_data);
        return true;
    }

    int layer;
    if (common_moe_stats_topk_name(t->name, layer)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        auto & pending = cb_data.pending[layer];
        pending.topk = std::move(snapshot);
        pending.topk_ready = true;
        return true;
    }

    if (common_moe_stats_weights_name(t->name, layer)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        auto & pending = cb_data.pending[layer];
        pending.weights = std::move(snapshot);
        pending.weights_ready = true;
        return true;
    }

    return true;
}

static void common_moe_stats_dump_atexit() {
    common_moe_stats_get_collector().dump();
}

#if defined(SIGUSR1) && !defined(_WIN32)
static int common_moe_stats_signal_pipe[2] = { -1, -1 };

static void common_moe_stats_sigusr1_handler(int) {
    const uint8_t byte = 1;
    if (common_moe_stats_signal_pipe[1] != -1) {
        const ssize_t ret = write(common_moe_stats_signal_pipe[1], &byte, sizeof(byte));
        (void) ret;
    }
}

static void common_moe_stats_install_sigusr1() {
    if (pipe(common_moe_stats_signal_pipe) != 0) {
        LOG_WRN("%s: failed to create signal pipe: %s\n", __func__, std::strerror(errno));
        return;
    }

    const int flags = fcntl(common_moe_stats_signal_pipe[1], F_GETFL, 0);
    if (flags < 0 || fcntl(common_moe_stats_signal_pipe[1], F_SETFL, flags | O_NONBLOCK) != 0) {
        LOG_WRN("%s: failed to set signal pipe non-blocking: %s\n", __func__, std::strerror(errno));
        close(common_moe_stats_signal_pipe[0]);
        close(common_moe_stats_signal_pipe[1]);
        common_moe_stats_signal_pipe[0] = -1;
        common_moe_stats_signal_pipe[1] = -1;
        return;
    }

    std::thread([] {
        uint8_t buf[64];
        for (;;) {
            const ssize_t n = read(common_moe_stats_signal_pipe[0], buf, sizeof(buf));
            if (n > 0) {
                common_moe_stats_get_collector().dump();
            } else if (n < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
    }).detach();

    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = common_moe_stats_sigusr1_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1, &action, nullptr) != 0) {
        LOG_WRN("%s: failed to install SIGUSR1 handler: %s\n", __func__, std::strerror(errno));
    }
}
#endif

static void common_moe_stats_install_dump_handlers() {
    std::atexit(common_moe_stats_dump_atexit);

#if defined(SIGUSR1) && !defined(_WIN32)
    common_moe_stats_install_sigusr1();
#endif
}

void common_moe_stats_maybe_init(common_params & params) {
    const char * output_path = std::getenv("LLAMA_MOE_STATS");
    if (output_path == nullptr || output_path[0] == '\0') {
        return;
    }

    auto & collector = common_moe_stats_get_collector();
    collector.configure(output_path, params.model.path);

    static std::once_flag once;
    std::call_once(once, common_moe_stats_install_dump_handlers);

    if (params.cb_eval == common_moe_stats_cb_eval && params.cb_eval_user_data != nullptr) {
        return;
    }

    if (params.cb_eval != nullptr || params.cb_eval_user_data != nullptr) {
        LOG_WRN("%s: cb_eval is already set, MoE router stats disabled\n", __func__);
        return;
    }

    params.cb_eval           = common_moe_stats_cb_eval;
    params.cb_eval_user_data = new common_moe_stats_cb_data(collector);
}
