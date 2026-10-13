"""GPT6-Astra: compare HOST module shares; never infer GPU busy or corrected work."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_host(path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = {r["module"]: float(r["host_us"]) for r in rows}
    if len(values) != len(rows) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Invalid or duplicate host interval")
    if set(values) != {"setup", "input_clear", "forward", "huber", "backward", "validation", "wall"}:
        raise ValueError("Incomplete module coverage")
    wall = values.pop("wall")
    total = sum(values.values())
    if total <= 0 or wall < total - .01:
        raise ValueError("Invalid host denominator")
    return {k: v / total * 100 for k, v in values.items()}, wall


def compare(directory):
    captures = []
    for path in sorted(directory.glob("gpu_modules_*.csv")):
        with path.open(encoding="utf-8") as f:
            meta = dict(x.split("=", 1) for x in next(f).removeprefix("# ").strip().split(","))
            rows = list(csv.DictReader(f))
        if meta.get("version") not in ("1", "2") or meta.get("committed") != "1" or meta.get("overflow") != "0":
            raise ValueError("Incomplete background capture")
        times = [float(r["host_us"]) for r in rows]
        wall = float(meta["wall_us"])
        if not times or not math.isfinite(wall) or wall <= 0 or any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("Invalid background timing")
        if sum(times) > wall + .5:
            raise ValueError("Overlapping host intervals")
        grouped = {}
        samples = {}
        for row, time in zip(rows, times):
            name = row["module"]
            grouped[name] = grouped.get(name, 0) + time
            samples.setdefault(name, []).append(time)
        counters = None
        if meta["version"] == "2":
            names = ("kernel_launches", "memset_calls", "memset_bytes", "sync_calls", "h2d_calls", "h2d_bytes", "d2h_calls", "d2h_bytes", "d2d_calls", "d2d_bytes")
            counters = {k: int(meta[k]) for k in names}
            counters["sync_us"] = float(meta["sync_us"])
            if any(not math.isfinite(v) or v < 0 for v in counters.values()) or counters["sync_us"] > wall:
                raise ValueError("Invalid operation counters")
        captures.append(dict(file=path.name, intervals=len(rows), host_us=grouped,
                             operation_counters=counters,
                             module_distributions={k: dict(count=len(v),mean_us=statistics.mean(v),median_us=statistics.median(v),p95_us=sorted(v)[math.ceil(.95*len(v))-1],peak_us=max(v)) for k,v in samples.items()},
                             boundary_gap_us=wall-sum(times)))
    modes = {}
    for mode in ("off", "modules", "deep"):
        paths = sorted(directory.glob(mode+"_*.bin.host.csv"))
        if not paths:
            raise ValueError("Missing mode: "+mode)
        runs = [read_host(p) for p in paths]
        modes[mode] = dict(runs=len(runs), median_wall_us=statistics.median(w for _, w in runs),
            shares={k: dict(mean_pct=statistics.mean(s[k] for s, _ in runs),
                           min_pct=min(s[k] for s, _ in runs), max_pct=max(s[k] for s, _ in runs))
                    for k in runs[0][0]})
    if len({m["runs"] for m in modes.values()}) != 1:
        raise ValueError("Unbalanced mode counts")
    if len(captures) != modes["modules"]["runs"]:
        raise ValueError("Missing background captures")
    for mode in ("modules", "deep"):
        modes[mode]["share_shift_pp_vs_off"] = {k: v["mean_pct"]-modes["off"]["shares"][k]["mean_pct"]
                                                for k, v in modes[mode]["shares"].items()}
    return dict(modes=modes, module_captures=captures, denominator="Sum of six disjoint legacy HOST module intervals, not GPU execution",
                caveat="Independent cold-process synthetic batches with concurrent workload; wall includes profiler setup/flush. No causal or equivalence claim; inspect share shifts and within-mode ranges.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.directory), indent=2))
