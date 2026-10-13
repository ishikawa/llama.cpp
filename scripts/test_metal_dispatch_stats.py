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

    def test_bandwidth_and_weight_aggregation(self):
        members = [dict(weight="blk.N.attn_q.weight", type="q6_K", shape=[2560, 12288], ne=[2560, 12288, 1, 1]),
                   dict(weight="blk.N.attn_k.weight", type="q6_K", shape=[2560, 512], ne=[2560, 512, 1, 1])]
        record = dict(mode="dispatch_timing", n_tokens=1, barriers=0, gpu_sum_ms=3.0, gpu_span_ms=4.0,
                      dispatches=[dict(op="MUL_MAT", node="Qcur", pipeline="mv", count=2, gpu_ms=3.0)],
                      traffic=[dict(op="MUL_MAT", pipeline="mv", weight="blk.N.attn_q.weight", type="q6_K", shape=[2560, 12288],
                                    count=1, gpu_ms=1.0, read_bytes=2000000, write_bytes=1000000),
                               dict(op="MUL_MAT", pipeline="mv", weight="blk.N.attn_q.weight", type="q6_K", shape=[2560, 12288],
                                    count=1, gpu_ms=2.0, read_bytes=4000000, write_bytes=2000000)],
                      shared_inputs=[dict(members=members, groups=1, count=2, gpu_ms=3.0, read_bytes=6000000, write_bytes=3000000,
                                          row_concat_compatible=True, combined_shape=[2560, 12800], combined_rows=12800, src1_ids=["0x100"])])
        groups = summarize([record, record, dict(record, n_tokens=32)])
        group = groups["dispatch_timing"]
        self.assertEqual(group["weights"][("blk.N.attn_q.weight", "q6_K", (2560, 12288))], [4, 6.0, 12000000.0, 6000000.0])
        self.assertEqual(next(iter(group["shared_inputs"].values()))["groups"], 2)
        output = io.StringIO()
        render(groups, output, decode_tokens=2, shared_inputs=True)
        self.assertIn("| mv | 2.000 | 0.000 | 3.000000 | 3.000 |", output.getvalue())
        self.assertIn("| 2.000 | 3.000000 | 1500.000 | 6.000000 | 3.000000 | 3.000 |", output.getvalue())
        self.assertIn("[2560, 12800] | yes | 1.000 | 2.000 | 3.000000 |", output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("metal_dispatch_stats.py")),
                                     str(path), "--shared-inputs"], check=True, text=True, capture_output=True)
            self.assertIn("shared inputs", result.stdout)
            self.assertIn("blk.N.attn_q.weight", result.stdout)

    def test_byte_estimates_native(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        root = Path(__file__).resolve().parents[1]
        lib = root / "build-dispatch-profile/bin"
        if not compiler or not (lib / "libggml-base.dylib").exists():
            self.skipTest("C compiler or built ggml-base unavailable")
        source = r'''#include "ggml-metal-profile.h"
#include <assert.h>
static struct ggml_tensor tensor(enum ggml_type type, int64_t cols, int64_t rows, int64_t batches) {
    struct ggml_tensor t = {.type=type, .ne={cols, rows, batches, 1}};
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = ggml_row_size(type, cols);
    t.nb[2] = t.nb[1] * rows;
    t.nb[3] = t.nb[2] * batches;
    return t;
}
int main(void) {
    struct ggml_tensor weight = tensor(GGML_TYPE_Q6_K, 256, 8, 1);
    struct ggml_tensor input = tensor(GGML_TYPE_F32, 256, 1, 1);
    struct ggml_tensor out = tensor(GGML_TYPE_F32, 8, 1, 1);
    out.op = GGML_OP_MUL_MAT; out.src[0] = &weight; out.src[1] = &input;
    struct ggml_metal_profile_bytes b = ggml_metal_profile_tensor_bytes(&out);
    assert(b.src0 == 210 * 8 && b.src1 == 1024 && b.dst == 32 && b.other_src == 0);
    weight = tensor(GGML_TYPE_Q6_K, 256, 8, 512);
    input = tensor(GGML_TYPE_F32, 256, 1, 3);
    struct ggml_tensor ids = tensor(GGML_TYPE_I32, 2, 3, 1);
    out = tensor(GGML_TYPE_F32, 8, 2, 3);
    out.op = GGML_OP_MUL_MAT_ID; out.src[0] = &weight; out.src[1] = &input; out.src[2] = &ids;
    b = ggml_metal_profile_tensor_bytes(&out);
    assert(b.src0 == 210 * 8 * 2 * 3 && b.src1 == 3072 && b.dst == 192 && b.other_src == 24);
    assert(b.src0 < ggml_nbytes(&weight));
    weight = tensor(GGML_TYPE_F32, 8, 1, 1);
    input = tensor(GGML_TYPE_F32, 8, 1, 1);
    out = tensor(GGML_TYPE_F32, 8, 1, 1);
    out.op = GGML_OP_ADD; out.src[0] = &weight; out.src[1] = &input;
    b = ggml_metal_profile_tensor_bytes(&out);
    assert(b.src0 + b.src1 + b.other_src == 64 && b.dst == 32);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "bytes.c").write_text(source, encoding="utf-8")
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-I" + str(root / "ggml/include"), "-I" + str(root / "ggml/src/ggml-metal"),
                            str(path / "bytes.c"), "-L" + str(lib), "-Wl,-rpath," + str(lib), "-lggml-base",
                            "-o", str(path / "bytes")], check=True, capture_output=True)
            subprocess.run([str(path / "bytes")], check=True, capture_output=True)

    def test_shared_input_native(self):
        compiler = shutil.which("clang")
        if not compiler or sys.platform != "darwin":
            self.skipTest("macOS clang unavailable")
        root = Path(__file__).resolve().parents[1]
        implementation = (root / "ggml/src/ggml-metal/ggml-metal-context.m").read_text()
        start = implementation.index("static NSArray * ggml_metal_profile_shared_inputs(")
        end = implementation.index("static void ggml_metal_profile_write(", start)
        source = '#import <Foundation/Foundation.h>\n#include <assert.h>\n' + implementation[start:end] + r'''
static NSDictionary * member(NSString * weight, NSString * type, int cols, int rows, int count) {
    return @{ @"weight":weight, @"type":type, @"shape":@[@(cols), @(rows)], @"ne":@[@(cols), @(rows), @1, @1],
              @"contiguous":@YES, @"count":@(count), @"gpu_ms":@1, @"read_bytes":@100, @"write_bytes":@10 };
}
int main(void) {
    @autoreleasepool {
        NSDictionary * a = member(@"blk.N.q.weight", @"q6_K", 2560, 100, 1);
        NSDictionary * b = member(@"blk.N.k.weight", @"q6_K", 2560, 20, 1);
        NSDictionary * c = member(@"blk.N.k.weight", @"f32", 2560, 20, 1);
        NSDictionary * single = member(@"blk.N.q.weight", @"q6_K", 2560, 100, 5);
        NSArray * groups = ggml_metal_profile_shared_inputs(@{@"shared_inputs":@{
            @"input0":@{@"node0":a, @"node1":b}, @"input1":@{@"node2":a, @"node3":b},
            @"input2":@{@"node4":single}, @"input3":@{@"node5":a, @"node6":c}}});
        assert(groups.count == 2);
        for (NSDictionary * group in groups) {
            assert([group[@"combined_rows"] intValue] == 120);
            if ([group[@"same_type"] boolValue]) {
                assert([group[@"groups"] intValue] == 2 && [group[@"count"] intValue] == 4);
                assert([group[@"row_concat_compatible"] boolValue]);
                assert([group[@"src1_ids"] count] == 2);
            } else {
                assert([group[@"groups"] intValue] == 1 && [group[@"count"] intValue] == 2);
                assert(![group[@"row_concat_compatible"] boolValue]);
            }
        }
        // Different input pointers with singleton consumers must not merge by normalized weight name.
        groups = ggml_metal_profile_shared_inputs(@{@"shared_inputs":@{@"input0":@{@"node0":a}, @"input1":@{@"node1":b}}});
        assert(groups.count == 0);
        return 0;
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "shared.m").write_text(source, encoding="utf-8")
            subprocess.run([compiler, "-fblocks", "-Wall", "-Wextra", "-Werror", "-framework", "Foundation",
                            str(path / "shared.m"), "-o", str(path / "shared")], check=True, capture_output=True)
            subprocess.run([str(path / "shared")], check=True, capture_output=True)

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
