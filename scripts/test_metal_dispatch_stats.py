import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from metal_dispatch_stats import component, render, summarize


class DispatchStatsTest(unittest.TestCase):
    def test_decode_filter_and_mode_separation(self):
        record = dict(mode="dispatch_timing", n_tokens=1, barriers=3, gpu_sum_ms=2.0, gpu_span_ms=4.0,
                      dispatches=[dict(op="MUL_MAT", node="ffn_moe_down-17", pipeline="expert", count=2, gpu_ms=1.5),
                                  dict(op="ADD", node="", component_node="hc_combine", pipeline="add", count=1, gpu_ms=0.5)])
        records = [record, record, dict(record, n_tokens=32), dict(record, failed=True),
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

    def test_empty(self):
        output = io.StringIO()
        render(summarize([]), output)
        self.assertIn("No successful decode", output.getvalue())


if __name__ == "__main__":
    unittest.main()
