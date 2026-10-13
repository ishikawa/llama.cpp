#pragma once

#include "ggml.h"

#include <stdint.h>
#include <string.h>

static inline bool ggml_metal_profile_input_name(const char * name, const char * expected) {
    if (strcmp(name, expected) == 0) {
        return true;
    }
    const size_t len = strlen(expected);
    for (const char * p = strchr(name, '#'); p; p = strchr(p + 1, '#')) {
        if (strncmp(p + 1, expected, len) == 0 && p[1 + len] == '#') {
            return true;
        }
    }
    return false;
}

static inline int64_t ggml_metal_profile_tensor_tokens(const struct ggml_tensor * t) {
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 0 || t->ne[3] <= 0) {
        return -1;
    }
    if (t->type == GGML_TYPE_I32 && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 && ggml_metal_profile_input_name(t->name, "inp_tokens")) {
        return t->ne[0];
    }
    if (t->type == GGML_TYPE_F32 && t->ne[2] == 1 && t->ne[3] == 1 && ggml_metal_profile_input_name(t->name, "inp_embd")) {
        return t->ne[1];
    }
    if (t->op == GGML_OP_ROPE && t->src[0] && t->src[1]) {
        const struct ggml_tensor * pos = t->src[1];
        int32_t mode;
        memcpy(&mode, t->op_params + 2, sizeof(mode));
        const int n_pos = mode & GGML_ROPE_TYPE_MROPE ? GGML_MROPE_SECTIONS : 1;
        if (pos->type == GGML_TYPE_I32 && pos->ne[0] > 0 && pos->ne[1] == 1 && pos->ne[2] == 1 && pos->ne[3] == 1 && pos->ne[0] % n_pos == 0 && pos->ne[0] / n_pos == t->src[0]->ne[2]) {
            return pos->ne[0] / n_pos;
        }
    }
    // qwen4exp names this reshape [channels, sequence tokens, sequences].
    const char * prefix = "linear_attn_qkv_mixed-";
    if (t->type == GGML_TYPE_F32 && t->op == GGML_OP_RESHAPE && strncmp(t->name, prefix, strlen(prefix)) == 0 && t->ne[3] == 1 && t->ne[1] <= INT64_MAX / t->ne[2]) {
        const char * suffix = t->name + strlen(prefix);
        const char * p = suffix;
        while (*p >= '0' && *p <= '9') {
            ++p;
        }
        if (p != suffix && *p == '\0') {
            return t->ne[1] * t->ne[2];
        }
    }
    return -1;
}
