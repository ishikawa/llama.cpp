#include "reasoning-budget.h"
#include "common.h"
#include "unicode.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_token> tokens;
    size_t pos = 0;

    bool advance(llama_token token) {
        if (tokens.empty()) {
            return false;
        }

        if (token == tokens[pos]) {
            pos++;
            if (pos >= tokens.size()) {
                pos = 0;
                return true;
            }
        } else {
            pos = 0;
            if (token == tokens[0]) {
                pos = 1;
            }
        }
        return false;
    }

    void reset() { pos = 0; }

    llama_token next() const {
        if (tokens.empty()) {
            return LLAMA_TOKEN_NULL;
        }
        return tokens[pos];
    }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;
    std::vector<llama_token> min_forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget
    int32_t min_tokens;       // minimum tokens in reasoning block
    int32_t min_remaining;    // tokens remaining before natural end is allowed
    int32_t max_min_forces;   // maximum continuation injections
    int32_t min_forces;       // continuation injections used

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static bool common_reasoning_budget_min_active(const common_reasoning_budget_ctx * ctx) {
    return ctx->min_tokens >= 0 && ctx->min_remaining > 0 && ctx->min_forces < ctx->max_min_forces && !ctx->min_forced_tokens.empty();
}

static void common_reasoning_budget_start_counting(common_reasoning_budget_ctx * ctx) {
    ctx->state = REASONING_BUDGET_COUNTING;
    ctx->remaining = ctx->budget;
    ctx->min_remaining = ctx->min_tokens;
    ctx->min_forces = 0;
    COM_TRC("activated, budget=%d tokens, min=%d tokens\n", ctx->budget, ctx->min_tokens);

    if (ctx->remaining <= 0) {
        ctx->state = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
        COM_TRC("%s", "budget=0, forcing immediately\n");
    }
}

static void common_reasoning_budget_count_token(common_reasoning_budget_ctx * ctx) {
    ctx->remaining--;
    if (ctx->min_remaining > 0) {
        ctx->min_remaining--;
    }
}

static void common_reasoning_budget_start_min_forcing(common_reasoning_budget_ctx * ctx) {
    ctx->state = REASONING_BUDGET_MIN_FORCING;
    ctx->force_pos = 0;
    ctx->min_forces++;
    ctx->end_matcher.reset();
    COM_TRC("min not reached, forcing continuation (%d/%d)\n", ctx->min_forces, ctx->max_min_forces);
}

static void common_reasoning_budget_start_budget_forcing(common_reasoning_budget_ctx * ctx) {
    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token)) {
                common_reasoning_budget_start_counting(ctx);
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            if (ctx->end_matcher.advance(token)) {
                if (ctx->state == REASONING_BUDGET_COUNTING && common_reasoning_budget_min_active(ctx)) {
                    common_reasoning_budget_start_min_forcing(ctx);
                } else {
                    ctx->state = REASONING_BUDGET_DONE;
                    COM_TRC("%s", "deactivated (natural end)\n");
                }
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                common_reasoning_budget_count_token(ctx);
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        common_reasoning_budget_start_budget_forcing(ctx);
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        case REASONING_BUDGET_MIN_FORCING:
            common_reasoning_budget_count_token(ctx);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->min_forced_tokens.size()) {
                if (ctx->remaining <= 0) {
                    common_reasoning_budget_start_budget_forcing(ctx);
                    COM_TRC("%s", "budget exhausted after continuation\n");
                } else {
                    ctx->state = REASONING_BUDGET_COUNTING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "continuation complete, counting\n");
                }
            }
            break;
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token)) {
                common_reasoning_budget_start_counting(ctx);
                ctx->end_matcher.reset();
                COM_TRC("re-activated on new start tag, budget=%d tokens, min=%d tokens\n", ctx->budget, ctx->min_tokens);
            }
            break;
    }
}

static bool common_reasoning_budget_force_token(llama_token token, llama_token_data_array * cur_p) {
    bool found = false;
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != token) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            found = true;
        }
    }
    return found;
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_COUNTING && common_reasoning_budget_min_active(ctx)) {
        const llama_token end_next = ctx->end_matcher.next();
        size_t end_i = cur_p->size;
        size_t max_i = cur_p->size;
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id == end_next) {
                end_i = i;
            }
            if (std::isfinite(cur_p->data[i].logit) && (max_i == cur_p->size || cur_p->data[i].logit > cur_p->data[max_i].logit)) {
                max_i = i;
            }
        }

        if (end_i != cur_p->size) {
            if (ctx->end_matcher.pos > 0 || end_i == max_i) {
                common_reasoning_budget_start_min_forcing(ctx);
            } else {
                cur_p->data[end_i].logit = -INFINITY;
                return;
            }
        }
    }

    if (ctx->state != REASONING_BUDGET_FORCING && ctx->state != REASONING_BUDGET_MIN_FORCING) {
        return;
    }

    const std::vector<llama_token> & tokens = ctx->state == REASONING_BUDGET_MIN_FORCING ? ctx->min_forced_tokens : ctx->forced_tokens;
    if (ctx->force_pos >= tokens.size()) {
        return;
    }

    common_reasoning_budget_force_token(tokens[ctx->force_pos], cur_p);
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->min_remaining = ctx->min_tokens;
    ctx->min_forces = 0;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens, const std::vector<llama_token> & forced_tokens,
        int32_t budget, int32_t min_tokens, const std::vector<llama_token> & min_forced_tokens,
        int32_t max_min_forces, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab             * vocab,
        const std::vector<llama_token>       & start_tokens,
        const std::vector<llama_token>       & end_tokens,
        const std::vector<llama_token>       & forced_tokens,
        int32_t                                budget,
        int32_t                                min_tokens,
        const std::vector<llama_token>       & min_forced_tokens,
        int32_t                                max_min_forces,
        common_reasoning_budget_state          initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ { start_tokens, 0 },
            /* .end_matcher   = */ { end_tokens, 0 },
            /* .forced_tokens = */ forced_tokens,
            /* .min_forced_tokens = */ min_forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .min_tokens     = */ min_tokens,
            /* .min_remaining  = */ min_tokens,
            /* .max_min_forces = */ max_min_forces,
            /* .min_forces     = */ 0,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, -1, {}, 0, initial_state);
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        int32_t                          min_tokens,
        const std::vector<llama_token> & min_forced,
        int32_t                          max_min_forces,
        common_reasoning_budget_state    initial_state) {
    return common_reasoning_budget_init_state(vocab, start_tokens, end_tokens, forced_tokens, budget, min_tokens, min_forced, max_min_forces, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
