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
        if record["type"] in ("decode", "synchronize") and record.get("decode_id") in excluded_decodes:
            continue
        kept.append(record)

    # Include zero-time steps when a phase/context is absent in a retained step.
    steps = sorted({record["step"] for record in kept if record.get("step", -1) >= 0})
    phases = defaultdict(lambda: defaultdict(int))
    contexts: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(kept):
        kind = record["type"]
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
            "unstepped_phases_ms_per_record": initialization_stats, "contexts": contexts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--skip-steps", type=int, default=0, help="exclude step numbers below N and initialization records")
    parser.add_argument("--max-tokens", type=int, help="exclude decode batches larger than N and their synchronize records")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args()
    if args.skip_steps < 0 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("--skip-steps must be nonnegative; --max-tokens must be positive")
    try:
        records = []
        with args.path.open(encoding="utf-8") as source:
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
    for context, stats in sorted(result["contexts"].items()):
        print(f"\n{context}: reuse {stats['reused_graphs']}/{stats['graphs']} ({stats['reuse_rate']:.1%}), {stats['failed_decodes']} failed decodes")
        for reason, count in stats["non_reuse_reasons"].items():
            print(f"  {reason}: {count}")
        for phase in ("build_graph", "sched_alloc_graph"):
            print(f"  {phase} per decode (ms, executed only): {json.dumps(stats[phase], sort_keys=True)}")


if __name__ == "__main__":
    main()
