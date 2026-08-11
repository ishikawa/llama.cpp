#include "moe-stats.h"

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"
#include "unicode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
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
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#endif

struct common_moe_stats_tensor {
    ggml_type type = GGML_TYPE_COUNT;
    int64_t   ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    size_t    nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    std::vector<uint8_t> data;
};

static int32_t common_moe_stats_get_i32(const common_moe_stats_tensor & t, int64_t i0, int64_t i1);

struct common_moe_stats_layer_pending {
    bool topk_ready         = false;
    bool weights_ready      = false;
    bool probs_ready        = false;
    bool probs_biased_ready = false;
    bool probs_masked_ready = false;

    common_moe_stats_tensor topk;
    common_moe_stats_tensor weights;
    common_moe_stats_tensor probs;
    common_moe_stats_tensor probs_biased;
    common_moe_stats_tensor probs_masked;
};

static constexpr int COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS = 14;

struct common_moe_stats_expert {
    uint64_t count                 = 0;
    uint64_t top1_count            = 0;
    double   weight_sum            = 0.0;
    double   sel_margin_sum        = 0.0;
    double   sel_margin_sumsq      = 0.0;
    double   mix_margin_sum        = 0.0;
    double   mix_margin_sumsq      = 0.0;
    std::array<uint64_t, COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS> sel_margin_hist = {};
};

using common_moe_stats_layer_stats   = std::map<int, common_moe_stats_expert>;
using common_moe_stats_forward_stats = std::map<int, common_moe_stats_layer_stats>;

enum common_moe_stats_token_class : uint8_t {
    COMMON_MOE_STATS_CLASS_BYTE = 0,
    COMMON_MOE_STATS_CLASS_DIGIT,
    COMMON_MOE_STATS_CLASS_CJK,
    COMMON_MOE_STATS_CLASS_CODE,
    COMMON_MOE_STATS_CLASS_SPACE,
    COMMON_MOE_STATS_CLASS_LATIN,
    COMMON_MOE_STATS_CLASS_OTHER,
    COMMON_MOE_STATS_CLASS_COUNT,
};

static const std::array<const char *, COMMON_MOE_STATS_CLASS_COUNT> common_moe_stats_class_names = {
    "byte",
    "digit",
    "cjk",
    "code",
    "space",
    "latin",
    "other",
};

static const std::array<double, COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS> common_moe_stats_sel_margin_hist_edges = {
    0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
};

static void common_moe_stats_add_expert(common_moe_stats_expert & dst, const common_moe_stats_expert & src) {
    dst.count            += src.count;
    dst.top1_count       += src.top1_count;
    dst.weight_sum       += src.weight_sum;
    dst.sel_margin_sum   += src.sel_margin_sum;
    dst.sel_margin_sumsq += src.sel_margin_sumsq;
    dst.mix_margin_sum   += src.mix_margin_sum;
    dst.mix_margin_sumsq += src.mix_margin_sumsq;

    for (int i = 0; i < COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS; ++i) {
        dst.sel_margin_hist[i] += src.sel_margin_hist[i];
    }
}

static int common_moe_stats_sel_margin_hist_bin(double margin) {
    int bin = 0;
    for (int i = 1; i < COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS; ++i) {
        if (margin < common_moe_stats_sel_margin_hist_edges[i]) {
            break;
        }
        bin = i;
    }
    return bin;
}

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

static bool common_moe_stats_parse_utf8(const std::string & s, std::vector<uint32_t> & codepoints) {
    if (!common_utf8_is_complete(s)) {
        return false;
    }

    for (size_t offset = 0; offset < s.size();) {
        const utf8_parse_result parsed = common_parse_utf8_codepoint(s, offset);
        if (parsed.status != utf8_parse_result::SUCCESS || parsed.bytes_consumed == 0) {
            return false;
        }
        codepoints.push_back(parsed.codepoint);
        offset += parsed.bytes_consumed;
    }

    return true;
}

static bool common_moe_stats_is_ascii_digit(uint32_t cpt) {
    return cpt >= '0' && cpt <= '9';
}

static bool common_moe_stats_is_ascii_alpha(uint32_t cpt) {
    return (cpt >= 'a' && cpt <= 'z') || (cpt >= 'A' && cpt <= 'Z');
}

static bool common_moe_stats_is_space(uint32_t cpt) {
    return cpt == 0x09 || cpt == 0x0a || cpt == 0x0b || cpt == 0x0c || cpt == 0x0d ||
           cpt == 0x20 || cpt == 0x85 || cpt == 0xa0 || cpt == 0x1680 ||
           (cpt >= 0x2000 && cpt <= 0x200a) || cpt == 0x2028 || cpt == 0x2029 ||
           cpt == 0x202f || cpt == 0x205f || cpt == 0x3000;
}

static bool common_moe_stats_is_cjk(uint32_t cpt) {
    return (cpt >= 0x3000  && cpt <= 0x303f)  ||
           (cpt >= 0x3040  && cpt <= 0x30ff)  ||
           (cpt >= 0x31f0  && cpt <= 0x31ff)  ||
           (cpt >= 0x3400  && cpt <= 0x4dbf)  ||
           (cpt >= 0x4e00  && cpt <= 0x9fff)  ||
           (cpt >= 0xac00  && cpt <= 0xd7af)  ||
           (cpt >= 0xf900  && cpt <= 0xfaff)  ||
           (cpt >= 0x20000 && cpt <= 0x2ebef) ||
           (cpt >= 0x2f800 && cpt <= 0x2fa1f);
}

static bool common_moe_stats_is_code_symbol(uint32_t cpt) {
    switch (cpt) {
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
        case '<':
        case '>':
        case '+':
        case '-':
        case '*':
        case '/':
        case '%':
        case '=':
        case '!':
        case '&':
        case '|':
        case '^':
        case '~':
        case '?':
        case ':':
        case ';':
        case '.':
        case ',':
        case '#':
        case '@':
        case '\\':
        case '$':
        case '"':
        case '\'':
        case '`':
            return true;
        default:
            return false;
    }
}

static bool common_moe_stats_is_digit_piece(const std::vector<uint32_t> & codepoints) {
    bool has_digit = false;

    for (uint32_t cpt : codepoints) {
        if (common_moe_stats_is_ascii_digit(cpt)) {
            has_digit = true;
            continue;
        }
        if (cpt == ',' || cpt == '.') {
            continue;
        }
        return false;
    }

    return has_digit;
}

static bool common_moe_stats_is_space_piece(const std::vector<uint32_t> & codepoints) {
    if (codepoints.empty()) {
        return false;
    }

    for (uint32_t cpt : codepoints) {
        if (!common_moe_stats_is_space(cpt)) {
            return false;
        }
    }

    return true;
}

static bool common_moe_stats_is_code_piece(const std::vector<uint32_t> & codepoints) {
    int n_code = 0;
    int n_text = 0;

    for (uint32_t cpt : codepoints) {
        if (common_moe_stats_is_space(cpt)) {
            continue;
        }
        if (cpt == '_') {
            return true;
        }
        if (common_moe_stats_is_code_symbol(cpt)) {
            ++n_code;
        } else {
            ++n_text;
        }
    }

    return n_code > 0 && n_code >= n_text;
}

static bool common_moe_stats_is_latin_piece(const std::vector<uint32_t> & codepoints) {
    int n_alpha = 0;
    int n_other = 0;

    for (uint32_t cpt : codepoints) {
        if (common_moe_stats_is_space(cpt)) {
            continue;
        }
        if (common_moe_stats_is_ascii_alpha(cpt)) {
            ++n_alpha;
        } else {
            ++n_other;
        }
    }

    return n_alpha > 0 && n_alpha >= n_other;
}

static common_moe_stats_token_class common_moe_stats_classify_token(const llama_vocab * vocab, llama_token id) {
    const llama_token_attr attr = llama_vocab_get_attr(vocab, id);
    if (attr & LLAMA_TOKEN_ATTR_BYTE) {
        return COMMON_MOE_STATS_CLASS_BYTE;
    }

    const std::string piece = common_token_to_piece(vocab, id, true);
    std::vector<uint32_t> codepoints;
    if (!common_moe_stats_parse_utf8(piece, codepoints)) {
        return COMMON_MOE_STATS_CLASS_BYTE;
    }

    if (common_moe_stats_is_digit_piece(codepoints)) {
        return COMMON_MOE_STATS_CLASS_DIGIT;
    }

    for (uint32_t cpt : codepoints) {
        if (common_moe_stats_is_cjk(cpt)) {
            return COMMON_MOE_STATS_CLASS_CJK;
        }
    }

    if (common_moe_stats_is_code_piece(codepoints)) {
        return COMMON_MOE_STATS_CLASS_CODE;
    }

    if (common_moe_stats_is_space_piece(codepoints)) {
        return COMMON_MOE_STATS_CLASS_SPACE;
    }

    if (common_moe_stats_is_latin_piece(codepoints)) {
        return COMMON_MOE_STATS_CLASS_LATIN;
    }

    return COMMON_MOE_STATS_CLASS_OTHER;
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

    void enable() {
        std::lock_guard<std::mutex> lock(mutex);
        enabled = true;
    }

    bool is_enabled() {
        std::lock_guard<std::mutex> lock(mutex);
        return enabled;
    }

    void configure_vocab(const llama_vocab * vocab) {
        std::lock_guard<std::mutex> lock(mutex);

        if (!enabled || vocab == nullptr || vocab_ready) {
            return;
        }

        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        token_classes.resize(n_vocab);
        for (llama_token id = 0; id < n_vocab; ++id) {
            token_classes[id] = (uint8_t) common_moe_stats_classify_token(vocab, id);
        }

        vocab_ready = true;
    }

    bool make_forward_token_classes(
            const common_moe_stats_tensor & tokens,
            int64_t n_tokens,
            std::vector<uint8_t> & forward_token_classes,
            std::array<uint64_t, COMMON_MOE_STATS_CLASS_COUNT> & forward_class_tokens,
            bool & warned_bad_tokens) {
        std::lock_guard<std::mutex> lock(mutex);

        if (!vocab_ready) {
            return false;
        }

        // Embedding-only ubatches can leave the token GET_ROWS branch unused. If the
        // captured leaf does not match the router token axis, keep plain stats only.
        if (tokens.type != GGML_TYPE_I32 || tokens.ne[0] != n_tokens) {
            if (!warned_bad_tokens) {
                warned_bad_tokens = true;
                LOG_WRN("%s: unexpected inp_tokens tensor type or shape, class stats disabled for this forward\n", __func__);
            }
            return false;
        }

        forward_token_classes.clear();
        forward_token_classes.reserve(n_tokens);
        forward_class_tokens.fill(0);

        for (int64_t it = 0; it < n_tokens; ++it) {
            const int32_t token_id = common_moe_stats_get_i32(tokens, it, 0);
            if (token_id < 0 || token_id >= (int32_t) token_classes.size()) {
                forward_token_classes.clear();
                forward_class_tokens.fill(0);
                if (!warned_bad_tokens) {
                    warned_bad_tokens = true;
                    LOG_WRN("%s: inp_tokens contains token id outside the vocab, class stats disabled for this forward\n", __func__);
                }
                return false;
            }

            const uint8_t token_class = token_classes[token_id];
            forward_token_classes.push_back(token_class);
            if (token_class < COMMON_MOE_STATS_CLASS_COUNT) {
                forward_class_tokens[token_class] += 1;
            }
        }

        return true;
    }

    void add_forward(
            common_moe_stats_forward_stats && forward_stats,
            std::array<common_moe_stats_forward_stats, COMMON_MOE_STATS_CLASS_COUNT> && forward_class_stats,
            const std::array<uint64_t, COMMON_MOE_STATS_CLASS_COUNT> & forward_class_tokens,
            uint64_t n_tokens) {
        std::lock_guard<std::mutex> lock(mutex);

        total_tokens += n_tokens;

        for (const auto & layer_it : forward_stats) {
            auto & layer = stats[layer_it.first];
            for (const auto & expert_it : layer_it.second) {
                auto & dst = layer[expert_it.first];
                common_moe_stats_add_expert(dst, expert_it.second);
            }
        }

        for (int ic = 0; ic < COMMON_MOE_STATS_CLASS_COUNT; ++ic) {
            class_total_tokens[ic] += forward_class_tokens[ic];

            for (const auto & layer_it : forward_class_stats[ic]) {
                auto & layer = class_stats[ic][layer_it.first];
                for (const auto & expert_it : layer_it.second) {
                    auto & dst = layer[expert_it.first];
                    common_moe_stats_add_expert(dst, expert_it.second);
                }
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
        // finite lower edges for sel_margin_hist; the last bin has implicit +inf upper edge
        out << "  \"sel_margin_hist_edges\": [";
        for (int i = 0; i < COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS; ++i) {
            out << (i == 0 ? "" : ", ") << common_moe_stats_sel_margin_hist_edges[i];
        }
        out << "],\n";
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
                out << "      \"" << expert_it.first << "\": {\"count\": " << expert_it.second.count << ", \"weight_sum\": " << expert_it.second.weight_sum;
                if (expert_it.second.top1_count > 0) {
                    out << ", \"top1\": {\"count\": " << expert_it.second.top1_count
                        << ", \"sel_margin_sum\": " << expert_it.second.sel_margin_sum
                        << ", \"sel_margin_sumsq\": " << expert_it.second.sel_margin_sumsq
                        << ", \"mix_margin_sum\": " << expert_it.second.mix_margin_sum
                        << ", \"mix_margin_sumsq\": " << expert_it.second.mix_margin_sumsq
                        << ", \"sel_margin_hist\": [";
                    for (int i = 0; i < COMMON_MOE_STATS_SEL_MARGIN_HIST_BINS; ++i) {
                        out << (i == 0 ? "" : ", ") << expert_it.second.sel_margin_hist[i];
                    }
                    out << "]}";
                }
                out << "}";
            }

            if (!layer_it.second.empty()) {
                out << "\n";
            }
            out << "    }";
        }

        if (!stats.empty()) {
            out << "\n";
        }
        out << "  },\n";
        out << "  \"classes\": {";

        bool first_class = true;
        for (int ic = 0; ic < COMMON_MOE_STATS_CLASS_COUNT; ++ic) {
            if (class_total_tokens[ic] == 0 && class_stats[ic].empty()) {
                continue;
            }

            out << (first_class ? "\n" : ",\n");
            first_class = false;
            out << "    \"" << common_moe_stats_class_names[ic] << "\": {\n";
            out << "      \"total_tokens\": " << class_total_tokens[ic] << ",\n";
            out << "      \"layers\": {";

            bool first_layer = true;
            for (const auto & layer_it : class_stats[ic]) {
                out << (first_layer ? "\n" : ",\n");
                first_layer = false;
                out << "        \"" << layer_it.first << "\": {";

                bool first_expert = true;
                for (const auto & expert_it : layer_it.second) {
                    out << (first_expert ? "\n" : ",\n");
                    first_expert = false;
                    out << "          \"" << expert_it.first << "\": {\"count\": " << expert_it.second.count << ", \"weight_sum\": " << expert_it.second.weight_sum;
                    if (expert_it.second.top1_count > 0) {
                        out << ", \"top1_count\": " << expert_it.second.top1_count;
                        out << ", \"top1_sel_margin_sum\": " << expert_it.second.sel_margin_sum;
                    }
                    out << "}";
                }

                if (!layer_it.second.empty()) {
                    out << "\n";
                }
                out << "        }";
            }

            if (!class_stats[ic].empty()) {
                out << "\n";
            }
            out << "      }\n";
            out << "    }";
        }

        if (!first_class) {
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
    common_moe_stats_forward_stats stats;
    std::array<common_moe_stats_forward_stats, COMMON_MOE_STATS_CLASS_COUNT> class_stats;
    std::array<uint64_t, COMMON_MOE_STATS_CLASS_COUNT> class_total_tokens = {};
    std::vector<uint8_t> token_classes;
    uint64_t total_tokens = 0;
    bool warned_multiple_models = false;
    bool enabled = false;
    bool vocab_ready = false;
};

struct common_moe_stats_cb_data {
    explicit common_moe_stats_cb_data(common_moe_stats_collector & collector) : collector(collector) {}

    common_moe_stats_collector & collector;
    std::mutex mutex;
    std::map<int, common_moe_stats_layer_pending> pending;
    bool tokens_ready = false;
    common_moe_stats_tensor tokens;
    bool warned_bad_topk    = false;
    bool warned_bad_weights = false;
    bool warned_bad_tokens  = false;
    bool warned_bad_probs   = false;
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

static bool common_moe_stats_probs_name(const char * name, int & layer) {
    return common_moe_stats_name_layer(name, "ffn_moe_probs",        layer) ||
           common_moe_stats_name_layer(name, "ffn_moe_probs_biased", layer) ||
           common_moe_stats_name_layer(name, "ffn_moe_probs_masked", layer);
}

static bool common_moe_stats_tokens_probe_name(const char * name) {
    return std::strcmp(name, "inp_tokens_probe") == 0;
}

static bool common_moe_stats_wants_tensor(const ggml_tensor * t) {
    const char * name = t->name;
    int layer;
    return std::strcmp(name, "result_output") == 0 ||
           (common_moe_stats_tokens_probe_name(name) && (t->flags & GGML_TENSOR_FLAG_COMPUTE)) ||
           common_moe_stats_topk_name(name, layer) ||
           common_moe_stats_weights_name(name, layer) ||
           common_moe_stats_probs_name(name, layer);
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

static bool common_moe_stats_is_float_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static bool common_moe_stats_probs_ok(const common_moe_stats_tensor & t, int64_t n_tokens) {
    return common_moe_stats_is_float_type(t.type) && t.ne[0] > 1 && t.ne[1] == n_tokens;
}

static bool common_moe_stats_topk_ids_ok(
        const common_moe_stats_tensor & topk,
        int64_t n_expert_probs,
        int64_t n_expert_sel) {
    for (int64_t it = 0; it < topk.ne[1]; ++it) {
        for (int64_t ie = 0; ie < topk.ne[0]; ++ie) {
            const int32_t expert = common_moe_stats_get_i32(topk, ie, it);
            if (expert >= n_expert_probs || expert >= n_expert_sel) {
                return false;
            }
        }
    }

    return true;
}

static double common_moe_stats_max_other(
        const common_moe_stats_tensor & scores,
        int64_t n_expert,
        int64_t expert,
        int64_t token) {
    double max_score = -std::numeric_limits<double>::infinity();
    for (int64_t other = 0; other < n_expert; ++other) {
        if (other == expert) {
            continue;
        }
        max_score = std::max(max_score, common_moe_stats_get_float(scores, other, token, 0));
    }
    return max_score;
}

static bool common_moe_stats_commit_layer(
        const common_moe_stats_layer_pending & pending,
        common_moe_stats_layer_stats & layer_stats,
        std::array<common_moe_stats_layer_stats, COMMON_MOE_STATS_CLASS_COUNT> * class_layer_stats,
        const std::vector<uint8_t> * forward_token_classes,
        bool & warned_bad_topk,
        bool & warned_bad_weights,
        bool & warned_bad_probs) {
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

    const common_moe_stats_tensor * sel = nullptr;
    if (pending.probs_masked_ready) {
        sel = &pending.probs_masked;
    } else if (pending.probs_biased_ready) {
        sel = &pending.probs_biased;
    } else if (pending.probs_ready) {
        sel = &pending.probs;
    }

    bool margins_ready = false;
    if (!pending.probs_ready) {
        if (!warned_bad_probs) {
            warned_bad_probs = true;
            LOG_WRN("%s: ffn_moe_probs tensor not captured, top1 margin stats disabled for this layer\n", __func__);
        }
    } else {
        bool probs_ok = common_moe_stats_probs_ok(pending.probs, n_tokens);
        if (pending.probs_biased_ready) {
            probs_ok = probs_ok && common_moe_stats_probs_ok(pending.probs_biased, n_tokens) && pending.probs_biased.ne[0] == pending.probs.ne[0];
        }
        if (pending.probs_masked_ready) {
            probs_ok = probs_ok && common_moe_stats_probs_ok(pending.probs_masked, n_tokens) && pending.probs_masked.ne[0] == pending.probs.ne[0];
        }

        if (sel == nullptr || !probs_ok) {
            if (!warned_bad_probs) {
                warned_bad_probs = true;
                LOG_WRN("%s: unexpected ffn_moe_probs tensor type or shape, top1 margin stats disabled for this layer\n", __func__);
            }
        } else if (!common_moe_stats_topk_ids_ok(topk, pending.probs.ne[0], sel->ne[0])) {
            if (!warned_bad_probs) {
                warned_bad_probs = true;
                LOG_WRN("%s: ffn_moe_topk contains expert id outside ffn_moe_probs, top1 margin stats disabled for this layer\n", __func__);
            }
        } else {
            margins_ready = true;
        }
    }

    for (int64_t it = 0; it < n_tokens; ++it) {
        if (margins_ready) {
            const int32_t top1_expert = common_moe_stats_get_i32(topk, 0, it);
            if (top1_expert >= 0) {
                const double sel_score  = common_moe_stats_get_float(*sel, top1_expert, it, 0);
                const double mix_score  = common_moe_stats_get_float(pending.probs, top1_expert, it, 0);
                const double sel_margin = sel_score - common_moe_stats_max_other(*sel, sel->ne[0], top1_expert, it);
                const double mix_margin = mix_score - common_moe_stats_max_other(pending.probs, pending.probs.ne[0], top1_expert, it);

                if (std::isfinite(sel_margin) && std::isfinite(mix_margin)) {
                    auto & top1_stat = layer_stats[top1_expert];
                    top1_stat.top1_count += 1;
                    top1_stat.sel_margin_sum += sel_margin;
                    top1_stat.sel_margin_sumsq += sel_margin*sel_margin;
                    top1_stat.mix_margin_sum += mix_margin;
                    top1_stat.mix_margin_sumsq += mix_margin*mix_margin;
                    top1_stat.sel_margin_hist[common_moe_stats_sel_margin_hist_bin(sel_margin)] += 1;

                    if (class_layer_stats != nullptr && forward_token_classes != nullptr) {
                        const uint8_t token_class = (*forward_token_classes)[it];
                        if (token_class < COMMON_MOE_STATS_CLASS_COUNT) {
                            auto & class_stat = (*class_layer_stats)[token_class][top1_expert];
                            class_stat.top1_count += 1;
                            class_stat.sel_margin_sum += sel_margin;
                        }
                    }
                }
            }
        }

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

            if (class_layer_stats != nullptr && forward_token_classes != nullptr) {
                const uint8_t token_class = (*forward_token_classes)[it];
                if (token_class < COMMON_MOE_STATS_CLASS_COUNT) {
                    auto & class_stat = (*class_layer_stats)[token_class][expert];
                    class_stat.count += 1;
                    class_stat.weight_sum += weight;
                }
            }
        }
    }

    return true;
}

static void common_moe_stats_commit_forward(common_moe_stats_cb_data & cb_data) {
    common_moe_stats_forward_stats forward_stats;
    std::array<common_moe_stats_forward_stats, COMMON_MOE_STATS_CLASS_COUNT> forward_class_stats;
    std::array<uint64_t, COMMON_MOE_STATS_CLASS_COUNT> forward_class_tokens = {};
    std::vector<uint8_t> forward_token_classes;
    uint64_t n_tokens = 0;

    {
        std::lock_guard<std::mutex> lock(cb_data.mutex);

        for (const auto & layer_it : cb_data.pending) {
            const auto & pending = layer_it.second;
            if (!pending.topk_ready) {
                continue;
            }

            n_tokens = std::max<uint64_t>(n_tokens, pending.topk.ne[1] > 0 ? (uint64_t) pending.topk.ne[1] : 0);
        }

        if (cb_data.tokens_ready && n_tokens > 0) {
            cb_data.collector.make_forward_token_classes(
                cb_data.tokens,
                n_tokens,
                forward_token_classes,
                forward_class_tokens,
                cb_data.warned_bad_tokens);
        }

        for (const auto & layer_it : cb_data.pending) {
            const auto & pending = layer_it.second;
            if (!pending.topk_ready) {
                continue;
            }

            if (!pending.weights_ready) {
                continue;
            }

            common_moe_stats_layer_stats layer_stats;
            std::array<common_moe_stats_layer_stats, COMMON_MOE_STATS_CLASS_COUNT> class_layer_stats;
            const bool has_classes = !forward_token_classes.empty() && (int64_t) forward_token_classes.size() == pending.topk.ne[1];
            if (common_moe_stats_commit_layer(
                pending,
                layer_stats,
                has_classes ? &class_layer_stats : nullptr,
                has_classes ? &forward_token_classes : nullptr,
                cb_data.warned_bad_topk,
                cb_data.warned_bad_weights,
                cb_data.warned_bad_probs) && !layer_stats.empty()) {
                forward_stats[layer_it.first] = std::move(layer_stats);
                if (has_classes) {
                    for (int ic = 0; ic < COMMON_MOE_STATS_CLASS_COUNT; ++ic) {
                        if (!class_layer_stats[ic].empty()) {
                            forward_class_stats[ic][layer_it.first] = std::move(class_layer_stats[ic]);
                        }
                    }
                }
            }
        }

        cb_data.pending.clear();
        cb_data.tokens_ready = false;
        cb_data.tokens = common_moe_stats_tensor();
    }

    if (!forward_stats.empty() || n_tokens > 0) {
        cb_data.collector.add_forward(std::move(forward_stats), std::move(forward_class_stats), forward_class_tokens, n_tokens);
    }
}

static bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return common_moe_stats_wants_tensor(t);
    }

    auto & cb_data = *(common_moe_stats_cb_data *) user_data;

    if (std::strcmp(t->name, "result_output") == 0) {
        common_moe_stats_commit_forward(cb_data);
        return true;
    }

    if (common_moe_stats_tokens_probe_name(t->name)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        cb_data.tokens = std::move(snapshot);
        cb_data.tokens_ready = true;
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

    if (common_moe_stats_name_layer(t->name, "ffn_moe_probs_masked", layer)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        auto & pending = cb_data.pending[layer];
        pending.probs_masked = std::move(snapshot);
        pending.probs_masked_ready = true;
        return true;
    }

    if (common_moe_stats_name_layer(t->name, "ffn_moe_probs_biased", layer)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        auto & pending = cb_data.pending[layer];
        pending.probs_biased = std::move(snapshot);
        pending.probs_biased_ready = true;
        return true;
    }

    if (common_moe_stats_name_layer(t->name, "ffn_moe_probs", layer)) {
        auto snapshot = common_moe_stats_copy_tensor(t);
        std::lock_guard<std::mutex> lock(cb_data.mutex);
        auto & pending = cb_data.pending[layer];
        pending.probs = std::move(snapshot);
        pending.probs_ready = true;
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

static void common_moe_stats_install_sigusr1(int interval_s) {
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

    std::thread([interval_s] {
        uint8_t buf[64];
        const int timeout_ms = interval_s > 0 ? interval_s * 1000 : -1;
        for (;;) {
            struct pollfd pfd = { common_moe_stats_signal_pipe[0], POLLIN, 0 };
            const int pr = poll(&pfd, 1, timeout_ms);
            if (pr < 0) {
                if (errno == EINTR) {
                    continue;
                }
                LOG_WRN("common_moe_stats: dump thread exiting on poll error: %s (SIGUSR1/interval dumps disabled)\n", std::strerror(errno));
                break;
            }
            if (pr == 0) {
                // periodic dump (LLAMA_MOE_STATS_INTERVAL elapsed with no signal)
                common_moe_stats_get_collector().dump();
                continue;
            }
            const ssize_t n = read(common_moe_stats_signal_pipe[0], buf, sizeof(buf));
            if (n > 0) {
                common_moe_stats_get_collector().dump();
            } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
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
    // LLAMA_MOE_STATS_INTERVAL=<seconds> enables periodic dumps so a force-killed
    // process loses at most one interval of accumulated stats
    int interval_s = 0;
    if (const char * env = std::getenv("LLAMA_MOE_STATS_INTERVAL")) {
        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(env, &end, 10);
        // reject trailing garbage, non-positive values, and anything that would
        // overflow the poll() millisecond timeout (int)
        if (errno != 0 || end == env || *end != '\0' || parsed <= 0 || parsed > INT_MAX / 1000) {
            LOG_WRN("%s: ignoring invalid LLAMA_MOE_STATS_INTERVAL '%s' (want 1..%d seconds)\n", __func__, env, INT_MAX / 1000);
        } else {
            interval_s = (int) parsed;
        }
    }
    common_moe_stats_install_sigusr1(interval_s);
#endif
}

void common_moe_stats_maybe_init(common_params & params) {
    const char * output_path = std::getenv("LLAMA_MOE_STATS");
    if (output_path == nullptr || output_path[0] == '\0') {
        return;
    }

    auto & collector = common_moe_stats_get_collector();
    collector.configure(output_path, params.model.path);

    if (params.cb_eval == common_moe_stats_cb_eval && params.cb_eval_user_data != nullptr) {
        return;
    }

    if (params.cb_eval != nullptr || params.cb_eval_user_data != nullptr) {
        LOG_WRN("%s: cb_eval is already set, MoE router stats disabled\n", __func__);
        return;
    }

    collector.enable();

    params.cb_eval           = common_moe_stats_cb_eval;
    params.cb_eval_user_data = new common_moe_stats_cb_data(collector);

    // install the dump handlers only once collection is actually wired up, so a
    // disabled run (cb_eval conflict) does not keep dumping empty stats
    static std::once_flag once;
    std::call_once(once, common_moe_stats_install_dump_handlers);
}

void common_moe_stats_maybe_init_vocab(const llama_vocab * vocab) {
    const char * output_path = std::getenv("LLAMA_MOE_STATS");
    if (output_path == nullptr || output_path[0] == '\0') {
        return;
    }

    auto & collector = common_moe_stats_get_collector();
    if (!collector.is_enabled()) {
        return;
    }

    collector.configure_vocab(vocab);
}
