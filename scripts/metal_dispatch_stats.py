#!/usr/bin/env python3
"""Summarize Metal JSONL records in milliseconds per graph, grouped by mode, token count and kind.

Qwen4exp (Qwen3.8-Flash-Next), qwen3next and qwen35moe graph_mtp builders name the token embedding, hidden norm and projection mtp_tok_embd, mtp_hnorm and mtp_eh_proj.
The trunk has h_nextn too, so h_nextn is not a draft marker.
Use the backend graph_kind marker (any mtp_ node before fusion), or fall back to mtp_ dispatch/component names for older JSONL.
Do not use a node-count threshold: fusion and scheduler splits change it.
Splits without a marker cannot be identified reliably; do not treat split graph averages as complete model steps.
One- and two-token graphs are kept separately; larger prefill graphs and failed graphs are excluded.
Timing mode serializes dispatches; compare its component times with counts mode gpu_span_ms for the normal graph GPU interval.
Warmup is included.
Traffic is logical bytes per dispatch, not measured DRAM traffic; fused ops inherit the leading node's estimate.
MUL_MAT_ID counts expert selections per token, including duplicates across tokens.
The expert-union table instead uses ids read from shared memory after a completed timing dispatch, once per leading MUL_MAT_ID node.
It excludes other weights absorbed by fusion.
Unsupported ids shapes/storage and failed dispatches have no union sample; missing is not zero.
"""
import argparse
import json
import re
import sys
from collections import defaultdict


def component(name):
    name = re.sub(r" \(.*$", "", name)
    name = re.sub(r"-\d+$", "", name)
    if name.startswith("mtp_"):
        name = name[4:]
        if name in ("tok_embd", "h_input", "concat", "eh_proj"):
            return "input"
        if name in ("hnorm", "enorm", "shared_head_norm"):
            return "norm"
        if name == "post_ffn":
            return "MoE/reduction"
    if name == "h_nextn":
        return "hyper-connections"
    if name.startswith("ple_"):
        return "PLE"
    if name.startswith("hc_") or name == "l_last":
        return "hyper-connections"
    if name.startswith("result_"):
        return "output"
    if name.startswith(("ffn_shexp", "shared_expert", "ffn_up", "ffn_gate", "ffn_down", "ffn_swiglu", "ffn_silu")):
        return "MoE/shared expert"
    if name.startswith(("ffn_moe_logits", "ffn_moe_probs", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_group_topk", "ffn_moe_weights")):
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


def graph_kind(record):
    if record.get("graph_kind") in ("main", "mtp"):
        return record["graph_kind"]
    for field in ("dispatches", "component_dispatches"):
        for entry in record.get(field, []):
            if any(entry.get(key, "").startswith("mtp_") for key in ("node", "component_node")):
                return "mtp"
    return "main"


def summarize(records):
    groups = {}
    for record in records:
        if record.get("n_tokens") not in (1, 2) or record.get("failed", False):
            continue
        mode = record["mode"]
        group = groups.setdefault((mode, record["n_tokens"], graph_kind(record)), {
            "graphs": 0, "barriers": 0, "gpu_sum_ms": 0.0, "gpu_span_ms": 0.0,
            "kernels": defaultdict(lambda: [0, 0.0, 0]), "components": defaultdict(lambda: [0, 0.0, 0]),
            "expert_unions": defaultdict(lambda: [0, 0, 0.0, 0.0]),
            "component_traffic": defaultdict(float),
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
            part = component(entry.get("component_node", entry.get("node", "")))
            group["component_traffic"][part] += read
            if part.startswith("MoE/"):
                group["component_traffic"]["MoE/total"] += read
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
        for entry in record.get("expert_unions", []):
            row = group["expert_unions"][entry["weight"]]
            row[0] += 1
            row[1] += entry["union_experts"]
            row[2] += entry["selected_weight_bytes"]
            row[3] += entry["union_weight_bytes"]
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
        print("No successful decode graphs (n_tokens in (1, 2)).", file=output)
    for (mode, n_tokens, kind), group in sorted(groups.items()):
        n = decode_tokens if decode_tokens is not None else group["graphs"]
        dispatches = sum(row[0] for row in group["kernels"].values()) / n
        print(f"\n{mode}, {kind}, n_tokens={n_tokens}: {group['graphs']} decode Metal graphs, denominator={n} "
              f"({'denominator override supplied by caller' if decode_tokens is not None else 'Metal graph invocations'})", file=output)
        print(f"dispatch/graph={dispatches:.3f}, barrier/graph={group['barriers']/n:.3f}, "
              f"GPU sum/graph={group['gpu_sum_ms']/n:.6f} ms, gpu_span_ms/graph={group['gpu_span_ms']/n:.6f} ms", file=output)
        for table in ("kernels", "components"):
            extra = " read MB/graph | GB/s (R+W) |" if table == "kernels" else " read MB/graph |"
            rule = " ---: | ---: |" if table == "kernels" else " ---: |"
            print(f"\n{table}\n| name | dispatch/graph | barrier/graph | GPU ms/graph |{extra}\n| --- | ---: | ---: | ---: |{rule}", file=output)
            for key, (count, gpu, barriers) in sorted(group[table].items(), key=lambda item: (-item[1][1], -item[1][0], item[0])):
                value = f"{gpu/n:.6f}" if mode == "dispatch_timing" else "N/A"
                read = group["component_traffic"].get(key)
                suffix = f" {read/n/1e6:.6f} |" if read is not None else " N/A |"
                if table == "kernels":
                    traffic = group["kernel_traffic"].get(key)
                    rate = bandwidth(*traffic) if traffic and mode == "dispatch_timing" else "N/A"
                    suffix = f" {traffic[0]/n/1e6:.6f} | {rate} |" if traffic else " N/A | N/A |"
                print(f"| {key} | {count/n:.3f} | {barriers/n:.3f} | {value} |{suffix}", file=output)
        if group["weights"]:
            print("\nweights\n| weight | type | shape (ne0, ne1) | dispatch/graph | GPU ms/graph | us/dispatch | read MB/graph | write MB/graph | GB/s (R+W) |\n| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |", file=output)
            for (name, dtype, shape), (count, gpu, read, write) in sorted(group["weights"].items(), key=lambda item: (-item[1][1], item[0])):
                average = f"{1000 * gpu / count:.3f}" if count else "N/A"
                print(f"| {name} | {dtype} | {shape} | {count/n:.3f} | {gpu/n:.6f} | {average} | {read/n/1e6:.6f} | {write/n/1e6:.6f} | {bandwidth(read, write, gpu)} |", file=output)
        if group["expert_unions"]:
            print("\nexpert unions (leading MUL_MAT_ID weights, once per op)\n| weight | samples/graph | experts/sample | selected weight MB/graph | union weight MB/graph |\n| --- | ---: | ---: | ---: | ---: |", file=output)
            for name, (samples, experts, selected, union) in sorted(group["expert_unions"].items()):
                print(f"| {name} | {samples/n:.3f} | {experts/samples:.3f} | {selected/n/1e6:.6f} | {union/n/1e6:.6f} |", file=output)
        if shared_inputs:
            print("\nshared inputs (distinct MUL_MAT nodes within each graph)\n| weights | types | shapes (ne0, ne1) | combined (columns, rows) | row concat | groups/graph | dispatch/graph | GPU ms/graph |\n| --- | --- | --- | --- | --- | ---: | ---: | ---: |", file=output)
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

    timing = {key[1:]: value for key, value in groups.items() if key[0] == "dispatch_timing"}
    if timing:
        print("\ncomponent comparison (GPU ms/graph)\n| component | main 1 token | main 2 tokens | MTP draft (1 token) | 2 / 1 |\n| --- | ---: | ---: | ---: | ---: |", file=output)
        keys = ((1, "main"), (2, "main"), (1, "mtp"))
        parts = sorted({part for key in keys if key in timing for part in timing[key]["components"]})
        for part in parts:
            values = [timing[key]["components"].get(part, [0, 0.0, 0])[1] / timing[key]["graphs"] if key in timing else None for key in keys]
            ratio = values[1] / values[0] if values[0] and values[1] is not None else None
            cells = [f"{value:.6f}" if value is not None else "N/A" for value in [*values, ratio]]
            print(f"| {part} | " + " | ".join(cells) + " |", file=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--decode-tokens", type=int, help="Override the denominator per group for detailed tables; comparison always uses graph invocations")
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
