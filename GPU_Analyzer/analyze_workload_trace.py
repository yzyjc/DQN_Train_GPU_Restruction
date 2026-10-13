"""GPT6-Astra: summarize bounded HIP event envelopes; never infer GPU idle/duty."""
import argparse
import csv
import json
import math
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), mean_us=sum(values)/len(values),
                median_us=(values[(len(values)-1)//2]+values[len(values)//2])/2,
                p95_us=values[math.ceil(.95*len(values))-1], peak_us=values[-1])


def analyze(path):
    with Path(path).open(encoding="utf-8") as f:
        meta = dict(item.split("=", 1) for item in next(f).removeprefix("# ").strip().split(","))
        rows = list(csv.DictReader(f))
    if meta.get("version") != "1" or any(meta[k] != v for k, v in
            (("overflow", "0"), ("drained", "1"), ("committed", "1"))):
        raise ValueError("Incomplete, failed or unsupported profile; refusing performance conclusion")
    if not rows or not 1 <= int(meta["batch"]) <= 32:
        raise ValueError("Missing operations or invalid batch size")
    for row in rows:
        for key in ("host_us", "start_us", "end_us", "duration_us"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite timing")
        row["bytes"] = int(row["bytes"])
        if row["api_ok"] != "1" or row["host_us"] < 0 or row["bytes"] < 0:
            raise ValueError("Invalid operation")
        row["event_valid"] = row["event_ok"] == "1" and row["duration_us"] >= 0
        row["timeline_valid"] = row["timeline_ok"] == "1" and row["start_us"] >= 0 and row["end_us"] >= row["start_us"]
    device_rows = [r for r in rows if r["kind"] != "sync"]
    device_clock_valid = all(r["event_valid"] for r in device_rows)
    groups = {}
    for row in rows:
        key = row["stage"]+"/"+row["kind"]+"/"+(row["name"] if row["kind"] == "copy" else "all")
        groups.setdefault(key, []).append(row)
    stats = {}
    for key, group in groups.items():
        event_rows = [r for r in group if r["kind"] != "sync"]
        valid_rows = [r for r in event_rows if r["event_valid"]]
        durations = [r["duration_us"] for r in valid_rows]
        complete = len(valid_rows) == len(event_rows)
        size = sum(r["bytes"] for r in group)
        valid_elapsed = sum(durations)
        valid_bytes = sum(r["bytes"] for r in valid_rows)
        host_us = sum(r["host_us"] for r in group)
        stats[key] = dict(
            calls=len(group), bytes=size, host_us=host_us,
            event_complete=complete,
            event_coverage=len(valid_rows) / len(event_rows) if event_rows else None,
            event_envelope_us=valid_elapsed if complete else None,
            valid_event_count=len(valid_rows),
            valid_event_bytes=valid_bytes,
            valid_event_envelope_us=valid_elapsed,
            valid_event_duration=distribution(durations),
            valid_event_byte_rate_GBs=(valid_bytes / valid_elapsed / 1000
                                       if valid_elapsed > 0 and valid_bytes else None),
            host_api_byte_rate_GBs=size / host_us / 1000 if size and host_us > 0 else None,
            event_envelope_byte_rate_GBs=(size / valid_elapsed / 1000
                                          if complete and valid_elapsed > 0 and size else None),
            duration=distribution(durations) if complete else None)
    kernels = [r for r in rows if r["kind"] == "kernel"]
    device = [r for r in rows if r["kind"] != "sync"]
    adjacent_pairs = list(zip(device, device[1:]))
    valid_pairs = [(a, b) for a, b in adjacent_pairs
                   if a["timeline_valid"] and b["timeline_valid"] and b["start_us"] >= a["end_us"]]
    timeline_good = device_clock_valid and len(valid_pairs) == len(adjacent_pairs)
    gaps = [b["start_us"] - a["end_us"] for a, b in valid_pairs]
    kernel_gaps = [b["start_us"] - a["end_us"] for a, b in valid_pairs
                   if a["kind"] == b["kind"] == "kernel"]
    valid_kernels = [r for r in kernels if r["event_valid"]]
    wall = float(meta["wall_us"])
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("Invalid batch wall time")
    return dict(batch_samples=int(meta["batch"]), wall_us=wall, groups=stats,
                kernel_launches=len(kernels),
                kernel_event_count=len(valid_kernels),
                kernel_event_coverage=len(valid_kernels) / len(kernels) if kernels else None,
                kernel_envelopes=distribution([r["duration_us"] for r in valid_kernels]),
                invalid_device_events=sum(not r["event_valid"] for r in device), timeline_valid=timeline_good,
                valid_adjacent_operation_pairs=len(valid_pairs),
                adjacent_operation_pair_coverage=(len(valid_pairs) / len(adjacent_pairs)
                                                   if adjacent_pairs else None),
                adjacent_kernel_gap_envelopes=distribution(kernel_gaps),
                between_operation_gaps=distribution(gaps),
                host_sync_us=sum(r["host_us"] for r in rows if r["kind"] == "sync"),
                gpu_executing_us=None, gpu_idle_us=None,
                note="Instrumented batch, not game step. Valid-event subsets are partial when coverage<100%; event envelopes/gaps include dispatch and instrumentation effects, not hardware execution/idle. Host wait overlaps GPU activity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(analyze(args.trace), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text+"\n", encoding="utf-8")
    else:
        print(text)
