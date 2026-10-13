import io
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import unittest

from metal_dispatch_stats import component, render, summarize


class DispatchStatsTest(unittest.TestCase):
    def test_decode_filter_and_mode_separation(self):
        record = dict(mode="dispatch_timing", n_tokens=1, barriers=3, gpu_sum_ms=2.0, gpu_span_ms=4.0,
                      dispatches=[dict(op="MUL_MAT", node="ffn_moe_down-17", pipeline="expert", count=2, gpu_ms=1.5),
                                  dict(op="ADD", node="", component_node="hc_combine", pipeline="add", count=1, gpu_ms=0.5)])
        records = [record, record, dict(record, n_tokens=32), dict(record, n_tokens=4),
                   dict(record, n_tokens=-1), dict(record, failed=True),
                   dict(record, mode="counts", dispatches=[dict(record["dispatches"][0], gpu_ms=None)])]
        groups = summarize(json.loads(json.dumps(records)))
        timing = groups["dispatch_timing"]
        self.assertEqual(timing["graphs"], 2)
        self.assertEqual(timing["kernels"]["expert"], [4, 3.0, 0])
        self.assertEqual(timing["components"]["MoE/total"], [4, 3.0, 0])
        self.assertEqual(timing["components"]["hyper-connections"], [2, 1.0, 0])
        output = io.StringIO()
        render(groups, output)
        self.assertIn("barrier/token=3.000", output.getvalue())
        self.assertIn("| expert | 2.000 | 0.000 | 1.500000 |", output.getvalue())
        self.assertIn("N/A", output.getvalue())

    def test_split_graph_denominator(self):
        record = dict(mode="counts", n_tokens=1, barriers=2, gpu_sum_ms=1.0, gpu_span_ms=1.0,
                      dispatches=[dict(node="hc_gate", pipeline="mv", count=5)])
        output = io.StringIO()
        render(summarize([record, record]), output, decode_tokens=1)
        self.assertIn("dispatch/token=10.000", output.getvalue())
        self.assertIn("barrier/token=4.000", output.getvalue())

    def test_components(self):
        for name, expected in [("ple_conv_out-1", "PLE"), ("ple_gate-1 (cont)", "PLE"), ("indexer_top_k-7", "attention/QSA"),
                               ("ffn_moe_weights_norm-2", "MoE/router"), ("ffn_swiglu", "MoE/shared expert"),
                               ("q_conv_predelta-3", "GDN"), ("result_norm", "output"), ("", "unclassified")]:
            self.assertEqual(component(name), expected)

    def test_jsonl_cli(self):
        record = dict(mode="counts", n_tokens=1, barriers=2, gpu_sum_ms=1.0, gpu_span_ms=1.5,
                      dispatches=[dict(node="", op="MUL", pipeline="mul", count=3, gpu_ms=None, barriers=2)],
                      component_dispatches=[dict(node="", component_node="ple_gate", count=1, barriers=1),
                                            dict(node="", component_node="hc_gate", count=2, barriers=1)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            path.write_text(json.dumps(record) + "\n\n" + json.dumps(dict(record, n_tokens=512)) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("metal_dispatch_stats.py")),
                                     str(path), "--decode-tokens", "1"], capture_output=True, text=True, check=True)
        self.assertIn("dispatch/token=3.000", result.stdout)
        self.assertIn("| PLE | 1.000 | 1.000 | N/A |", result.stdout)
        self.assertIn("| hyper-connections | 2.000 | 1.000 | N/A |", result.stdout)

    def test_token_tensor_shapes(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[1]
        source = r'''#include "ggml-metal-profile.h"
#include <assert.h>
int main(void) {
    struct ggml_tensor t = {.type = GGML_TYPE_F32, .ne = {2560, 4, 1, 1}};
    strcpy(t.name, "hc_gate-3");
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    strcpy(t.name, "model.input_embed (reshaped)");
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    struct ggml_tensor pos = {.type = GGML_TYPE_I32, .ne = {4, 1, 1, 1}};
    strcpy(pos.name, "Metal#inp_pos#0");
    struct ggml_tensor q = {.type = GGML_TYPE_F32, .ne = {256, 24, 1, 1}};
    t.op = GGML_OP_ROPE; t.src[0] = &q; t.src[1] = &pos;
    t.op_params[2] = GGML_ROPE_TYPE_IMROPE;
    assert(ggml_metal_profile_tensor_tokens(&t) == 1);
    pos.ne[0] = 128; q.ne[2] = 32;
    assert(ggml_metal_profile_tensor_tokens(&t) == 32);
    q.ne[2] = 1;
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    pos.ne[0] = 32; q.ne[2] = 32; t.op_params[2] = GGML_ROPE_TYPE_NORMAL;
    assert(ggml_metal_profile_tensor_tokens(&t) == 32);
    t.op = GGML_OP_NONE; t.type = GGML_TYPE_I32;
    t.ne[0] = 32; t.ne[1] = t.ne[2] = t.ne[3] = 1;
    strcpy(t.name, "Metal#inp_tokens#0");
    assert(ggml_metal_profile_tensor_tokens(&t) == 32);
    strcpy(t.name, "out_ids"); t.ne[0] = 1;
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    strcpy(t.name, "inp_pos"); t.ne[0] = 4;
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    strcpy(t.name, "linear_attn_qkv_mixed-0"); t.type = GGML_TYPE_F32;
    t.op = GGML_OP_RESHAPE; t.ne[0] = 10240; t.ne[1] = 1; t.ne[2] = 1;
    assert(ggml_metal_profile_tensor_tokens(&t) == 1);
    t.ne[1] = 16; t.ne[2] = 2;
    assert(ggml_metal_profile_tensor_tokens(&t) == 32);
    strcpy(t.name, "linear_attn_qkv_mixed-0 (reshaped)");
    assert(ggml_metal_profile_tensor_tokens(&t) == -1);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "tokens.c").write_text(source, encoding="utf-8")
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-I" + str(root / "ggml/include"), "-I" + str(root / "ggml/src/ggml-metal"),
                            str(path / "tokens.c"), "-o", str(path / "tokens")], check=True, capture_output=True)
            subprocess.run([str(path / "tokens")], check=True, capture_output=True)

    def test_empty(self):
        output = io.StringIO()
        render(summarize([]), output)
        self.assertIn("No successful decode", output.getvalue())


if __name__ == "__main__":
    unittest.main()
