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
            "kernel_traffic": defaultdict(lambda: [0.0, 0.0, 0.0]),
            "weights": defaultdict(lambda: [0, 0.0, 0.0, 0.0]), "shared_inputs": {},
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
        for entry in record.get("traffic", []):
            count, gpu = entry["count"], entry.get("gpu_ms") or 0.0
            read, write = entry["read_bytes"], entry["write_bytes"]
            traffic = group["kernel_traffic"][entry["pipeline"]]
            traffic[0] += read
            traffic[1] += write
            traffic[2] += gpu
            if entry["op"] in ("MUL_MAT", "MUL_MAT_ID"):
                key = (entry["weight"], entry["type"], tuple(entry["shape"]))
                weight = group["weights"][key]
                weight[0] += count
                weight[1] += gpu
                weight[2] += read
                weight[3] += write
        for entry in record.get("shared_inputs", []):
            key = tuple((member["weight"], member["type"], tuple(member["ne"])) for member in entry["members"])
            key = (key, entry["row_concat_compatible"])
            shared = group["shared_inputs"].setdefault(key, dict(entry, groups=0, count=0, gpu_ms=0.0, read_bytes=0.0, write_bytes=0.0))
            for field in ("groups", "count", "gpu_ms", "read_bytes", "write_bytes"):
                shared[field] += entry[field]
    return groups


def bandwidth(read, write, gpu_ms):
    return f"{(read + write) / gpu_ms / 1e6:.3f}" if gpu_ms > 0 else "N/A"


def render(groups, output=sys.stdout, decode_tokens=None, shared_inputs=False):
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
            extra = " GB/s (R+W) |" if table == "kernels" else ""
            rule = " ---: |" if table == "kernels" else ""
            print(f"\n{table}\n| name | dispatch/token | barrier/token | GPU ms/token |{extra}\n| --- | ---: | ---: | ---: |{rule}", file=output)
            for key, (count, gpu, barriers) in sorted(group[table].items(), key=lambda item: (-item[1][1], -item[1][0], item[0])):
                value = f"{gpu/n:.6f}" if mode == "dispatch_timing" else "N/A"
                suffix = ""
                if table == "kernels":
                    traffic = group["kernel_traffic"].get(key)
                    rate = bandwidth(*traffic) if traffic and mode == "dispatch_timing" else "N/A"
                    suffix = f" {rate} |"
                print(f"| {key} | {count/n:.3f} | {barriers/n:.3f} | {value} |{suffix}", file=output)
        if group["weights"]:
            print("\nweights\n| weight | type | shape (ne0, ne1) | dispatch/token | GPU ms/token | us/dispatch | read MB/token | write MB/token | GB/s (R+W) |\n| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |", file=output)
            for (name, dtype, shape), (count, gpu, read, write) in sorted(group["weights"].items(), key=lambda item: (-item[1][1], item[0])):
                average = f"{1000 * gpu / count:.3f}" if count else "N/A"
                print(f"| {name} | {dtype} | {shape} | {count/n:.3f} | {gpu/n:.6f} | {average} | {read/n/1e6:.6f} | {write/n/1e6:.6f} | {bandwidth(read, write, gpu)} |", file=output)
        if shared_inputs:
            print("\nshared inputs (distinct MUL_MAT nodes within each graph)\n| weights | types | shapes (ne0, ne1) | combined (columns, rows) | row concat | groups/token | dispatch/token | GPU ms/token |\n| --- | --- | --- | --- | --- | ---: | ---: | ---: |", file=output)
            for entry in sorted(group["shared_inputs"].values(), key=lambda item: -item["gpu_ms"]):
                names = "; ".join(member["weight"] for member in entry["members"])
                types = "; ".join(member["type"] for member in entry["members"])
                shapes = "; ".join(str(tuple(member["shape"])) for member in entry["members"])
                combined = [int(value) for value in entry["combined_shape"]] if entry["combined_shape"] is not None else None
                compatible = "yes" if entry["row_concat_compatible"] else "no"
                print(f"| {names} | {types} | {shapes} | {combined} | {compatible} | {entry['groups']/n:.3f} | {entry['count']/n:.3f} | {entry['gpu_ms']/n:.6f} |", file=output)
            if not group["shared_inputs"]:
                print("No recorded shared MUL_MAT inputs (requires new timing-mode JSONL).", file=output)
        if group["kernel_traffic"]:
            print("Bytes are logical estimates, not measured DRAM traffic. MB=1e6 bytes; GB/s=(read+write bytes)/(GPU ms*1e6). "
                  "Per-dispatch estimates can count reused data and multi-dispatch op buffers more than once.", file=output)
        print("Barriers before an op go to its first dispatch; barriers within an op go to its preceding dispatch. No per-kernel GPU time is available in counts mode. "
              "MoE/total overlaps its four subcomponents. Graph splits and warmup invocations are included.", file=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--decode-tokens", type=int, help="Actual measured decode tokens per mode; use when the scheduler splits each token into several Metal graphs")
    parser.add_argument("--shared-inputs", action="store_true", help="Show MUL_MAT groups sharing a tensor input within a graph, identified before weight-name aggregation")
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
    render(summarize(records()), decode_tokens=args.decode_tokens, shared_inputs=args.shared_inputs)


if __name__ == "__main__":
    main()
