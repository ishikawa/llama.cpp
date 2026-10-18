#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

static uint32_t remove_reference(llama_kv_cells & cells, llama_seq_id seq, llama_pos p0, llama_pos p1) {
    p0 = p0 < 0 ? 0 : p0;
    p1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    uint32_t first = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }
        if (seq == -1) {
            cells.rm(i);
            first = std::min(first, i);
        } else if (cells.seq_has(i, seq) && cells.seq_rm(i, seq)) {
            first = std::min(first, i);
        }
    }
    return first;
}

static void compare(const llama_kv_cells & a, const llama_kv_cells & b) {
    GGML_ASSERT(a.size() == b.size());
    GGML_ASSERT(a.get_used() == b.get_used());
    GGML_ASSERT(a.used_min() == b.used_min());
    GGML_ASSERT(a.used_max_p1() == b.used_max_p1());
    GGML_ASSERT(a.get_has_shift() == b.get_has_shift());
    for (uint32_t i = 0; i < a.size(); ++i) {
        GGML_ASSERT(a.is_empty(i) == b.is_empty(i));
        GGML_ASSERT(a.seq_get_all(i) == b.seq_get_all(i));
        if (!a.is_empty(i)) {
            GGML_ASSERT(a.pos_get(i) == b.pos_get(i));
            GGML_ASSERT(a.get_shift(i) == b.get_shift(i));
            GGML_ASSERT(a.ext_get(i).x == b.ext_get(i).x);
            GGML_ASSERT(a.ext_get(i).y == b.ext_get(i).y);
            GGML_ASSERT(a.ext_get(i).tok == b.ext_get(i).tok);
        }
    }
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        GGML_ASSERT(a.seq_pos_get(s) == b.seq_pos_get(s));
        GGML_ASSERT(a.seq_pos_min(s) == b.seq_pos_min(s));
        GGML_ASSERT(a.seq_pos_max(s) == b.seq_pos_max(s));
    }
}

static void test(bool fast) {
    llama_kv_cells expected;
    expected.resize(4096);
    for (uint32_t i = 0; i < expected.size(); i += 3) {
        expected.pos_set(i, (i * 17) % 257);
        expected.ext_set(i, {int32_t(i), int32_t(i + 1), int32_t(i + 2)});
        expected.seq_add(i, i % 4);
        if (i % 2 == 0) {
            expected.seq_add(i, (i + 1) % 4);
        }
    }
    // A used cell need not belong to a sequence.
    expected.pos_set(1, 19);
    expected.pos_set(2, std::numeric_limits<llama_pos>::max());
    auto actual = expected;
    uint32_t head_expected = 4000;
    uint32_t head_actual = head_expected;
    const auto remove = [&](llama_seq_id seq, llama_pos p0, llama_pos p1) {
        const auto old_first = remove_reference(expected, seq, p0, p1);
        const auto new_first = actual.seq_rm_range(seq, p0, p1, fast);
        GGML_ASSERT(old_first == new_first);
        head_expected = std::min(head_expected, old_first);
        head_actual = std::min(head_actual, new_first);
        GGML_ASSERT(head_expected == head_actual);
        compare(expected, actual);
    };

    remove(0, 17, 25);
    remove(1, -7, 12);
    remove(2, 240, -3);
    remove(-1, 19, 20);
    remove(-1, 30, 30);
    remove(3, 100, 20);

    uint32_t rng = 12345;
    const auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 200; ++step) {
        const uint32_t i = next() % expected.size();
        if (expected.is_empty(i)) {
            const llama_pos pos = next() % 300;
            const llama_seq_id seq = next() % 4;
            expected.pos_set(i, pos);
            actual.pos_set(i, pos);
            expected.seq_add(i, seq);
            actual.seq_add(i, seq);
        } else {
            expected.pos_add(i, 3);
            actual.pos_add(i, 3);
            expected.pos_div(i, 2);
            actual.pos_div(i, 2);
        }
        if (step % 11 == 0) {
            expected.reset_shift();
            actual.reset_shift();
            const auto saved = expected.cp(0, expected.size());
            expected.set(0, saved);
            actual.set(0, saved);
        }
        const llama_seq_id seq = int(next() % 5) - 1;
        const llama_pos p0 = int(next() % 310) - 5;
        const llama_pos p1 = int(next() % 310) - 5;
        remove(seq, p0, p1);
    }
    remove(0, -1, -1);
    remove(-1, -1, -1);
    remove(-1, -1, -1);
}

int main() {
    test(false);
    test(true);
    std::puts("KV cell removal matches reference");
    return 0;
}
