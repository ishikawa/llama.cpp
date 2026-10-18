#!/usr/bin/env python3
"""Summarize LLAMA_STEP_TIMING JSONL host timings (standard library only)."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


def distribution(values):
    """Return milliseconds, with linearly interpolated percentiles."""
    ordered = sorted(value / 1000 for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def first_cb_summary(records, skip_steps=0, max_tokens=None):
    """Use GPU order across contexts; flush order can differ from submission order."""
    values = defaultdict(lambda: defaultdict(list))
    graphs = sorted((record for record in records if record["type"] == "metal_graph"
                     and "gpu_start_s" in record), key=lambda record: record["gpu_start_s"])
    previous_end = None
    gaps = {}
    for record in graphs:
        key = id(record)
        if previous_end is not None:
            gaps[key] = max(0, record["gpu_start_s"] - previous_end) * 1e6
        previous_end = max(previous_end or 0, record["gpu_end_s"])
    for record in records:
        if record["type"] not in ("split", "metal_graph"):
            continue
        if skip_steps and record.get("step", -1) < skip_steps:
            continue
        if max_tokens is not None and record.get("n_tokens", 0) > max_tokens:
            continue
        group = f"{record.get('context', 'unknown')}.n_tokens={record.get('n_tokens', 0)}"
        metrics = values[group]
        if record["type"] == "split":
            for phase, duration in record.get("us", {}).items():
                metrics[f"split.{record['backend']}.{phase}"].append(duration)
            continue
        for phase, duration in record.get("us", {}).items():
            metrics[phase].append(duration)
        key = id(record)
        if key in gaps:
            metrics["gpu_idle"].append(gaps[key])
        for buffer in record.get("buffers", []):
            prefix = f"cb{buffer['index']}"
            for phase in ("graph_to_commit_us", "metal_to_commit_us"):
                metrics[f"{prefix}.{phase[:-3]}"].append(buffer[phase])
            if "gpu_start_s" in buffer:
                metrics[f"{prefix}.commit_to_gpu_start"].append((buffer["gpu_start_s"] - buffer["commit_s"]) * 1e6)
                metrics[f"{prefix}.gpu"].append((buffer["gpu_end_s"] - buffer["gpu_start_s"]) * 1e6)
    return {group: {phase: distribution(samples) for phase, samples in sorted(metrics.items())}
            for group, metrics in sorted(values.items())}


def summarize(records: list[dict[str, Any]], skip_steps: int = 0, max_tokens: int | None = None) -> dict[str, Any]:
    excluded_decodes = {
        record["decode_id"]
        for record in records
        if record["type"] == "decode"
        and max_tokens is not None
        and record["n_tokens"] > max_tokens
    }
    kept = []
    for record in records:
        step = record.get("step", -1)
        if skip_steps and step < skip_steps:
            continue
        if record.get("decode_id") in excluded_decodes:
            continue
        kept.append(record)

    # Include zero-time steps when a phase/context is absent in a retained step.
    steps = sorted({record["step"] for record in kept if record.get("step", -1) >= 0})
    phases = defaultdict(lambda: defaultdict(int))
    contexts: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(kept):
        kind = record["type"]
        if kind in ("split", "metal_graph"):
            continue
        context = record.get("context", "unknown")
        step = record.get("step", -1)
        key = ("step", step) if step >= 0 else ("record", index)
        prefix = f"{kind}.{context}"
        if kind == "synchronize":
            prefix += "." + record.get("caller", "unknown")
        for phase, duration in record.get("us", {}).items():
            phases[f"{prefix}.{phase}"][key] += duration
        phases[f"{prefix}.total"][key] += record.get("total_us", 0)
        if kind != "decode":
            continue
        stats = contexts.setdefault(context, {
            "decodes": 0, "failed_decodes": 0, "graphs": 0, "reused_graphs": 0,
            "non_reuse_reasons": Counter(), "build_graph": [], "sched_alloc_graph": [],
        })
        stats["decodes"] += 1
        stats["failed_decodes"] += record.get("status", 0) != 0
        graphs = record.get("graphs", [])
        stats["graphs"] += len(graphs)
        stats["reused_graphs"] += sum(graph.get("reused", False) for graph in graphs)
        stats["non_reuse_reasons"].update(record.get("reasons", []))
        for phase in ("build_graph", "sched_alloc_graph"):
            if phase in record.get("us", {}):
                stats[phase].append(record["us"][phase])

    phase_stats = {}
    initialization_stats = {}
    for name, per_step in sorted(phases.items()):
        initialization = [value for (kind, _), value in per_step.items() if kind == "record"]
        if initialization:
            initialization_stats[name] = distribution(initialization)
        if steps:
            values = [per_step.get(("step", step), 0) for step in steps]
            if any(kind == "step" for kind, _ in per_step):
                phase_stats[name] = distribution(values)
    for stats in contexts.values():
        stats["reuse_rate"] = stats["reused_graphs"] / stats["graphs"] if stats["graphs"] else 0.0
        stats["non_reuse_reasons"] = dict(sorted(stats["non_reuse_reasons"].items()))
        for phase in ("build_graph", "sched_alloc_graph"):
            stats[phase] = distribution(stats[phase])
    return {"records": len(kept), "steps": len(steps), "phases_ms_per_step": phase_stats,
            "unstepped_phases_ms_per_record": initialization_stats, "contexts": contexts,
            "first_cb_ms_per_record": first_cb_summary(records, skip_steps, max_tokens)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="+", help="step and/or first-command-buffer JSONL files")
    parser.add_argument("--skip-steps", type=int, default=0, help="exclude step numbers below N and initialization records")
    parser.add_argument("--max-tokens", type=int, help="exclude decode batches larger than N and their synchronize records")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args()
    if args.skip_steps < 0 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("--skip-steps must be nonnegative; --max-tokens must be positive")
    try:
        records = []
        for path in args.path:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as error:
                            raise ValueError(f"line {line_number}: {error}") from error
        result = summarize(records, args.skip_steps, args.max_tokens)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{args.path}: {error}\n")
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return
    print(f"{result['records']} records, {result['steps']} steps")
    print("Host durations overlap across record types; do not add decode, step and synchronize totals.")
    print("Phase                                             median ms/step  mean ms/step")
    for phase, stats in result["phases_ms_per_step"].items():
        print(f"{phase:55} {stats['median_ms']:12.4f} {stats['mean_ms']:13.4f}")
    if result["unstepped_phases_ms_per_record"]:
        print("\nInitialization/unstepped phases (ms per record):")
        for phase, stats in result["unstepped_phases_ms_per_record"].items():
            print(f"{phase:55} {stats['median_ms']:12.4f} {stats['mean_ms']:13.4f}")
    for group, metrics in result["first_cb_ms_per_record"].items():
        print(f"\nFirst command buffer: {group} (ms per split/Metal graph/CB occurrence)")
        for phase, stats in metrics.items():
            print(f"  {phase:53} {stats['median_ms']:12.4f} {stats['mean_ms']:13.4f}")
    for context, stats in sorted(result["contexts"].items()):
        print(f"\n{context}: reuse {stats['reused_graphs']}/{stats['graphs']} ({stats['reuse_rate']:.1%}), {stats['failed_decodes']} failed decodes")
        for reason, count in stats["non_reuse_reasons"].items():
            print(f"  {reason}: {count}")
        for phase in ("build_graph", "sched_alloc_graph"):
            print(f"  {phase} per decode (ms, executed only): {json.dumps(stats[phase], sort_keys=True)}")


if __name__ == "__main__":
    main()
