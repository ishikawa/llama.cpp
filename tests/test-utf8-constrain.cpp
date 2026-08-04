#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece;
    piece.resize(piece.capacity());
    int32_t n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, true);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, true);
    }
    piece.resize(n_chars);
    return piece;
}

static llama_token find_piece(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        if (token_to_piece(vocab, id) == piece) {
            return id;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static llama_token find_eog(const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token id = 0; id < n_vocab; ++id) {
        if (llama_vocab_is_eog(vocab, id)) {
            return id;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void require(bool ok, const char * msg) {
    if (!ok) {
        fprintf(stderr, "%s\n", msg);
        exit(1);
    }
}

static bool is_neg_inf(float v) {
    return std::isinf(v) && v < 0.0f;
}

static void apply(llama_sampler * smpl, std::vector<llama_token_data> & data) {
    llama_token_data_array cur_p = {
        /* .data       = */ data.data(),
        /* .size       = */ data.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };
    llama_sampler_apply(smpl, &cur_p);
}

int main(int argc, char ** argv) {
    require(argc >= 2, "usage: test-utf8-constrain <vocab.gguf>");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    require(model != nullptr, "failed to load vocab");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const llama_token tok_prefix = find_piece(vocab, std::string("\xe9\x81", 2));
    const llama_token tok_finish = find_piece(vocab, std::string("\x85", 1));
    const llama_token tok_ext    = find_piece(vocab, std::string("\xe5\xbb\xb6", 3));
    const llama_token tok_ascii  = find_piece(vocab, "a");
    const llama_token tok_eog    = find_eog(vocab);

    require(tok_prefix != LLAMA_TOKEN_NULL, "missing token for E9 81");
    require(tok_finish != LLAMA_TOKEN_NULL, "missing token for 85");
    require(tok_ascii  != LLAMA_TOKEN_NULL, "missing token for ascii 'a'");
    require(tok_eog    != LLAMA_TOKEN_NULL, "missing EOG token");

    const llama_token tok_normal = tok_ext != LLAMA_TOKEN_NULL ? tok_ext : tok_ascii;

    llama_sampler * smpl = llama_sampler_init_utf8_constrain(vocab);
    require(smpl != nullptr, "failed to init utf8 constrain sampler");

    {
        std::vector<llama_token_data> data = {
            { tok_finish, 10.0f, 0.0f },
            { tok_ascii,   1.0f, 0.0f },
            { tok_eog,     0.0f, 0.0f },
        };
        apply(smpl, data);
        require(is_neg_inf(data[0].logit), "clean state did not mask continuation-start token");
        require(data[1].logit == 1.0f, "clean state changed ascii token");
        require(data[2].logit == 0.0f, "clean state changed EOG token");
        require(llama_sampler_utf8_constrain_n_interventions(smpl) == 1, "clean state intervention was not counted");
    }

    llama_sampler_reset(smpl);

    {
        llama_sampler_accept(smpl, tok_prefix);
        std::vector<llama_token_data> data = {
            { tok_finish,  1.0f, 0.0f },
            { tok_normal, 10.0f, 0.0f },
            { tok_eog,     0.0f, 0.0f },
        };
        apply(smpl, data);
        require(data[0].logit == 1.0f, "pending state changed valid continuation token");
        require(is_neg_inf(data[1].logit), "pending state did not mask non-continuation token");
        require(data[2].logit == 0.0f, "pending state masked EOG token");
        require(llama_sampler_utf8_constrain_n_interventions(smpl) == 1, "pending state intervention was not counted");
    }

    llama_sampler_reset(smpl);

    {
        std::vector<llama_token_data> step0 = {
            { tok_prefix, 5.0f, 0.0f },
            { tok_ascii,  4.0f, 0.0f },
        };
        apply(smpl, step0);
        require(step0[0].logit == 5.0f, "valid prefix token was changed");
        require(step0[1].logit == 4.0f, "valid ascii token was changed");
        llama_sampler_accept(smpl, tok_prefix);

        std::vector<llama_token_data> step1 = {
            { tok_finish, 5.0f, 0.0f },
            { tok_eog,    4.0f, 0.0f },
        };
        apply(smpl, step1);
        require(step1[0].logit == 5.0f, "valid finish token was changed");
        require(step1[1].logit == 4.0f, "valid EOG token was changed");
        llama_sampler_accept(smpl, tok_finish);

        std::vector<llama_token_data> step2 = {
            { tok_normal, 5.0f, 0.0f },
            { tok_eog,    4.0f, 0.0f },
        };
        apply(smpl, step2);
        require(step2[0].logit == 5.0f, "valid normal token was changed");
        require(step2[1].logit == 4.0f, "valid EOG token was changed after complete codepoint");
        require(llama_sampler_utf8_constrain_n_interventions(smpl) == 0, "valid replay caused an intervention");
    }

    llama_sampler_free(smpl);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
