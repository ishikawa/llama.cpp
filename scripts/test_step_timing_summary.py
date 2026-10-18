#!/usr/bin/env python3
"""Synthetic tests; no model or backend execution."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from step_timing_summary import distribution, summarize


def decode(step, decode_id, context="mtp", tokens=1, reused=False, build=1000):
    return {
        "type": "decode", "context": context, "step": step, "decode_id": decode_id,
        "n_tokens": tokens, "status": 0, "reused": reused,
        "graphs": [{"reused": reused}],
        "reasons": [] if reused else ["active_mismatch:0->1"],
        "us": {"graph_compute": 100, **({} if reused else {"build_graph": build, "sched_alloc_graph": build * 2})},
        "total_us": build + 100,
    }


class StepTimingSummaryTest(unittest.TestCase):
    def test_per_step_aggregation_and_zero_steps(self):
        records = [decode(0, 1), decode(0, 2), decode(1, 3, reused=True)]
        result = summarize(records)
        phase = result["phases_ms_per_step"]["decode.mtp.build_graph"]
        self.assertEqual(phase["count"], 2)
        self.assertEqual(phase["mean_ms"], 1)
        self.assertEqual(phase["median_ms"], 1)
        stats = result["contexts"]["mtp"]
        self.assertAlmostEqual(stats["reuse_rate"], 1 / 3)
        self.assertEqual(stats["non_reuse_reasons"], {"active_mismatch:0->1": 2})
        self.assertEqual(stats["build_graph"]["count"], 2)

    def test_filters_and_sync_link(self):
        records = [decode(-1, 1, tokens=100), decode(0, 2), decode(1, 3), decode(2, 4, context="target")]
        records += [{"type": "synchronize", "step": 2, "decode_id": 1, "us": {"wait": 9999}}]
        result = summarize(records, skip_steps=1, max_tokens=2)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["steps"], 2)
        self.assertEqual(set(result["contexts"]), {"mtp", "target"})
        self.assertEqual(result["phases_ms_per_step"]["decode.mtp.build_graph"]["mean_ms"], 0.5)
        self.assertFalse(any(name.startswith("synchronize") for name in result["phases_ms_per_step"]))
        self.assertEqual(summarize(records, max_tokens=2)["records"], 3)

    def test_distributions_and_empty_input(self):
        self.assertEqual(distribution([]), {"count": 0})
        stats = distribution([1000, 3000])
        self.assertEqual(stats["median_ms"], 2)
        self.assertEqual(stats["p90_ms"], 2.8)
        self.assertEqual(summarize([])["contexts"], {})
        self.assertEqual(summarize([decode(-1, 1)])["unstepped_phases_ms_per_record"]["decode.mtp.build_graph"]["mean_ms"], 1)

    def test_mixed_records_and_sync_callers(self):
        records = [decode(-1, 1, tokens=100), decode(0, 2), decode(1, 3, reused=True)]
        records += [
            {"type": "step", "context": "speculative-simple", "step": step,
             "accepted": step, "drafted": 1, "us": {"draft": 2000}, "total_us": 4000}
            for step in (0, 1)
        ]
        records += [{"type": "synchronize", "context": "mtp", "step": 0, "decode_id": 2,
                     "caller": "llama_get_logits_ith", "us": {"wait": 3000}, "total_us": 3000}]
        result = summarize(records)
        self.assertEqual(result["phases_ms_per_step"]["step.speculative-simple.draft"]["mean_ms"], 2)
        self.assertEqual(result["phases_ms_per_step"]["synchronize.mtp.llama_get_logits_ith.wait"]["mean_ms"], 1.5)
        self.assertEqual(result["unstepped_phases_ms_per_record"]["decode.mtp.build_graph"]["mean_ms"], 1)
        records[1]["graphs"] = [{"reused": True}, {"reused": False}]
        stats = summarize(records, max_tokens=2)["contexts"]["mtp"]
        self.assertEqual(stats["graphs"], 3)
        self.assertAlmostEqual(stats["reuse_rate"], 2 / 3)

    def test_cli(self):
        script = Path(__file__).with_name("step_timing_summary.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.write_text(json.dumps(decode(0, 1)) + "\n", encoding="utf-8")
            output = subprocess.check_output([sys.executable, str(script), str(path), "--json"], text=True)
            self.assertEqual(json.loads(output)["records"], 1)
            path.write_text("invalid\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("line 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
