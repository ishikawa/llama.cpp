#!/usr/bin/env python3
"""Summarize Metal JSONL records. GPU times are milliseconds per decode graph."""
import argparse
import json
import re
import sys
from collections import defaultdict


def component(name):
    name = re.sub(r" \(.*$", "", name)
    name = re.sub(r"-\d+$", "", name)
    if name.startswith("ple_"):
        return "PLE"
    if name.startswith("hc_") or name == "l_last":
        return "hyper-connections"
    if name.startswith("result_"):
        return "output"
    if name.startswith(("ffn_shexp", "shared_expert", "ffn_up", "ffn_gate", "ffn_down", "ffn_swiglu", "ffn_silu")):
        return "MoE/shared expert"
    if name.startswith(("ffn_moe_logits", "ffn_moe_probs", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_weights")):
        return "MoE/router"
    if name in ("ffn_out", "ffn_moe_out", "ffn_moe_weighted"):
        return "MoE/reduction"
    if name.startswith("ffn_moe_"):
        return "MoE/expert"
    if name.startswith(("linear_attn", "conv_", "state", "q_conv", "k_conv", "v_conv", "gdn", "delta")) or name in (
        "new_state", "q_in", "k_in", "v_in", "g_in", "b_in", "z", "beta", "beta_sigmoid", "alpha", "a_softplus", "gate", "final_output", "recurrent_attn",
    ):
        return "GDN"
    if name.startswith(("indexer", "attn_", "kq", "Qcur", "Kcur", "Vcur", "gate_")):
        return "attention/QSA"
    if "norm" in name:
        return "norm"
    if name.startswith(("inp_", "model.input", "embd")):
        return "input"
    return "unclassified"


def summarize(records):
    groups = {}
    for record in records:
        if record.get("n_tokens") != 1 or record.get("failed", False):
            continue
        mode = record["mode"]
        group = groups.setdefault(mode, {
            "graphs": 0, "barriers": 0, "gpu_sum_ms": 0.0, "gpu_span_ms": 0.0,
            "kernels": defaultdict(lambda: [0, 0.0, 0]), "components": defaultdict(lambda: [0, 0.0, 0]),
        })
        group["graphs"] += 1
        for field in ("barriers", "gpu_sum_ms", "gpu_span_ms"):
            group[field] += record[field]
        for entry in record["dispatches"]:
            count, gpu = entry["count"], entry.get("gpu_ms") or 0.0
            group["kernels"][entry["pipeline"]][0] += count
            group["kernels"][entry["pipeline"]][1] += gpu
            group["kernels"][entry["pipeline"]][2] += entry.get("barriers", 0)
        for entry in record.get("component_dispatches", record["dispatches"]):
            count, gpu = entry["count"], entry.get("gpu_ms") or 0.0
            part = component(entry.get("component_node", entry["node"]))
            group["components"][part][0] += count
            group["components"][part][1] += gpu
            group["components"][part][2] += entry.get("barriers", 0)
            if part.startswith("MoE/"):
                group["components"]["MoE/total"][0] += count
                group["components"]["MoE/total"][1] += gpu
                group["components"]["MoE/total"][2] += entry.get("barriers", 0)
    return groups


def render(groups, output=sys.stdout, decode_tokens=None):
    if not groups:
        print("No successful decode graphs (n_tokens == 1).", file=output)
    for mode, group in sorted(groups.items()):
        n = decode_tokens if decode_tokens is not None else group["graphs"]
        dispatches = sum(row[0] for row in group["kernels"].values()) / n
        print(f"\n{mode}: {group['graphs']} decode Metal graphs, denominator={n} "
              f"({'tokens supplied by caller' if decode_tokens is not None else 'Metal graph invocations; assumes one graph per token'})", file=output)
        print(f"dispatch/token={dispatches:.3f}, barrier/token={group['barriers']/n:.3f}, "
              f"GPU sum/token={group['gpu_sum_ms']/n:.6f} ms, GPU span/token={group['gpu_span_ms']/n:.6f} ms", file=output)
        for table in ("kernels", "components"):
            print(f"\n{table}\n| name | dispatch/token | barrier/token | GPU ms/token |\n| --- | ---: | ---: | ---: |", file=output)
            for key, (count, gpu, barriers) in sorted(group[table].items(), key=lambda item: (-item[1][1], -item[1][0], item[0])):
                value = f"{gpu/n:.6f}" if mode == "dispatch_timing" else "N/A"
                print(f"| {key} | {count/n:.3f} | {barriers/n:.3f} | {value} |", file=output)
        print("Barriers before an op go to its first dispatch; barriers within an op go to its preceding dispatch. No per-kernel GPU time is available in counts mode. "
              "MoE/total overlaps its four subcomponents. Graph splits and warmup invocations are included.", file=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--decode-tokens", type=int, help="Actual measured decode tokens per mode; use when the scheduler splits each token into several Metal graphs")
    args = parser.parse_args()
    if args.decode_tokens is not None and args.decode_tokens <= 0:
        parser.error("--decode-tokens must be positive")
    def records():
        for path in args.paths:
            with open(path, encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f"{path}:{line_number}: {error}") from error
    render(summarize(records()), decode_tokens=args.decode_tokens)


if __name__ == "__main__":
    main()
