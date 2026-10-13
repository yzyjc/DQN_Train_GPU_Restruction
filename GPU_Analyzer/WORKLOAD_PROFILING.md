# Bounded training workload diagnostics

Collaboration signature: GPT6-Astra.

## Primary design rule: preserve module proportions

Measurement plane and Timing plane are separate. The goal is trustworthy module
time proportions even when profiling slows overall training. Removing bookkeeping
from intervals does NOT remove changes to scheduling, contention, caches or clocks.
Never subtract a constant overhead and claim the original execution was recovered.

Set `DQN_GPU_PROFILE_MODE=modules` for HOST module-boundary sampling. This mode
does not allocate GPU events, query GPU timing or introduce any synchronization.
It uses a bounded, preallocated set of eight capture slots (512 intervals each);
the same `DQN_GPU_PROFILE_BATCHES=1..8` and explicit output directory apply.
Hot boundaries read QPC and store raw records; a background worker formats and
writes `gpu_modules_<pid>_<index>.csv`. No producer waits for the consumer, and
overflow is marked rather than silently accepted. Slots are never reused in this
bounded session. Worker startup is before the first measured interval; shutdown
joins the worker outside training. First-use overhead still affects wall time.

Module transitions stop the previous interval before recording metadata and start
the next afterward. The small gap is excluded from module sums; timestamp cost and
preemption cannot be completely separated. Raw host intervals include existing
business synchronization and legacy timer bookkeeping. They are NOT pure GPU work.
`backward_*` children are disjoint; sum them plus `backward` transition work to
recover a host backward grouping, never add parent and children twice.

`DQN_GPU_PROFILE_MODE=deep` (also the legacy unset default) uses the earlier
per-operation GPU event diagnostics. Its end-of-capture drain and synchronous file
write remain, so it is for short detailed investigation, not authoritative module
shares. Modules mode bypasses that entire collector.

Run `tests/run_module_timing.ps1` after building the isolated smoke executable.
It rotates OFF/modules/deep order over three rounds, checks parameter equality,
and compares mean/min/max host shares using the same six legacy intervals in all
three modes. This permits a direct comparison despite finer modules-mode records.
Wall includes startup and deep flush; do not use it as steady-state slowdown.
The synthetic cold-process experiment can reveal large share distortion but cannot
prove equivalence on real training. CPU and GPU percentages must have separate
denominators; a GPU-event failure remains a GPU measurement failure regardless of
whether the host proportions look stable.

Validation (2026-09-27): three rotated rounds, nine isolated batches, all trained
parameter bytes identical. Background output contained 387 disjoint intervals per
batch, no overflow; excluded boundary gaps were 11.6–23.8 us (not total profiler
overhead). Host mean shares in percent:

| Module | OFF | Modules | Deep |
|---|---:|---:|---:|
| Input + clear | 24.87 | 23.45 | 27.32 |
| Forward | 5.57 | 5.86 | 4.94 |
| Backward | 67.41 | 68.43 | 65.32 |

Backward per-run ranges: OFF 66.86–68.18%, modules 66.90–69.28%, deep
64.70–66.11%. This is evidence for checking instrumentation-dependent ratios,
not proof of steady-state equivalence or attribution of every shift to profiling.
Local raw evidence: `build_verify/workload_profile/module_compare_20260927_133047_9792`.

## Expanded lightweight coverage

Module CSV V2 adds actual launch count, clear calls/bytes, explicit device-sync
calls/host wait time, and H2D/D2H/D2D calls/bytes. These use counters plus two QPC
reads around existing device synchronization, without additional GPU events.
Sync wait is INCLUDED in host module time: never add it to the module total.
Requested copy/clear bytes are not physical DRAM traffic. Copy completion time
and GPU-effective bandwidth are still unavailable in this lightweight mode.
Per-module host distributions now include count, mean, median, P95 and peak;
their sample unit is one module occurrence inside the captured batch, not episodes.

Nine isolated comparison runs verified identical parameters. All three lightweight
captures asserted: 1,025 launches; 193 clears / 464,004 bytes (includes validation);
225 device-sync calls; H2D 1 / 3,328 bytes; D2H 2 / 132 bytes;
D2D 34 / 6,792,008 bytes. Evidence directory:
`build_verify/workload_profile/module_compare_20260927_133721_57724`.

`tests/gpu_event_probe.cpp` further isolates clock validity. On this installed
runtime, 64-byte memset produced invalid durations in 4/32 default-event and
10/32 blocking-event trials. For 64 MiB memset, both 32-trial sets had no invalid
durations (mean event span 409.739 / 385.806 us). This narrows the observed issue
to include small-operation timing; it does not establish a root cause or validate
production kernel timings. The probe synchronizes deliberately for validation only;
this synchronization is not added to Trainer. No production bandwidth or WGP value
is inferred from this synthetic large-clear experiment.

## Scope and units

`gpu_nn_train_step_batch` is one optimization batch, NOT one board move or episode.
Default operation is unchanged: no profile events or trace files unless
`DQN_GPU_PROFILE_BATCHES` is set to 1–8 before process startup. Set
`DQN_GPU_PROFILE_DIR` to an existing explicit directory. Each selected batch writes
`gpu_profile_<pid>_<index>.csv`. Invalid/failed/truncated batches are not performance evidence.
Do not enable profiling in a formal long run: it adds events and a diagnostic drain.

Captured per operation: stage, kind/kernel name, requested bytes, host call time,
device event duration and timeline availability. Copy directions are separate.
Kernel timing uses HIP's extended launch start/stop events in diagnostic mode.
No extra per-operation synchronization is inserted. Existing seven backward
device synchronizations per sample remain in place; optimization is a separate change.

Analyze with:

```
python GPU_Analyzer/analyze_workload_trace.py <trace.csv> --output <summary.json>
```

The report gives launch counts, clear/copy bytes, requested-byte/host-API-time ratios,
explicit sync wait, stage summaries, and per-operation HIP event-duration distributions.
It reports valid-event counts and coverage for each group: one bad event no longer
erases measurements from unrelated groups. Complete groups get a full event-envelope
byte rate; incomplete groups expose a separately named valid-subset rate using only
the bytes and events that passed validation. Bytes use decimal GB/s (`bytes / us / 1000`).
Neither rate is physical DRAM bandwidth: HIP event intervals include runtime/dispatch
effects, and the valid-subset rate omits failed samples. Host API byte rate includes
runtime/dispatch/wait overhead and is NOT completed-transfer or DRAM bandwidth;
asynchronous copy returns can produce rates above physical bandwidth. Host wait
overlaps GPU execution; do not add the two together.
GPU executing/idle are deliberately null without a trustworthy complete device
timeline. Incomplete event groups report coverage but not full-group timing; incomplete
or inconsistent cross-operation intervals do not qualify for complete gap statistics.
Missing data is never replaced with zero or clamped into a plausible result.

## Source audit findings

- Six gradient clears per sample: `4*(25 + 4*25*32 + 25*16) = 14,500` bytes.
- Batch 32: 192 clears / 464,000 bytes, plus a separate 4-byte validation clear.
- Old `Memset(6x)` host interval also includes the sample input D2D copy. TUI now
  calls this `Input+clear` under `HOST BATCH BREAKDOWN`.
- The existing overall HIP Event span is an interval, not pure GPU busy time.
- CPU MaxCore mean/P95/peak uses the last 60 valid TUI sampler readings (~60s),
  not a Trainer-thread utilization measurement.

## Validation on 8060S / Windows (2026-09-27)

`tests/gpu_workload_profile_smoke.cpp` calls HIP directly and never initializes the
production named kernel-timer mapping. It creates its own network and runs one
deterministic batch=32; no Trainer, replay, checkpoint or production logs are used.
With profiling off/on, all 3,394,340 parameter bytes and loss matched exactly.
SHA256: `77C826CC93B1BFE0E1F32FAF9B1E60BA3E859AC9F15B0D171DEEB358682BD7D1`.

One instrumented batch counted 1,025 launches, 192 gradient clears, and 224 backward
device sync calls (225 including validation). Host wall was 136.411 ms and explicit
sync call time 82.865 ms. Gradient clear host time was 38.854 ms for 464,000 bytes:
0.01194 GB/s requested-byte/host-API-time ratio. This is synthetic, instrumented, and ran
alongside the user's workload: it is NOT a production speed benchmark.

**Full device timeline is not closed:** ordinary events returned negative durations.
Extended kernel launch events still produced 21 invalid device durations in the
captured run. Reported API success did not imply a valid time. The analyzer now keeps
independent valid groups instead of discarding all GPU timing when one event fails.
Re-reading the saved trace yields valid durations for 1,009 / 1,025 kernel launches
(98.44% coverage), and all 192 / 192 gradient-clear events (464,000 requested bytes).
Those are partial HIP event-envelope observations from the isolated instrumented batch;
their requested-byte rate is not physical memset bandwidth. The cross-operation timeline
still fails validation, so complete gap, GPU-executing/idle, and physical bandwidth
measurements remain unavailable. Do not extrapolate this trace to a full training step.

An idle-GPU repeat on 2026-09-28 ran the same opt-in smoke profile four more times;
each baseline/profiled parameter snapshot matched and each trace contained all 1,025
launches. The four traces had 8, 10, 7, and 12 negative-duration events, leaving
99.22%, 99.02%, 99.32%, and 98.83% kernel-event coverage. Across 4,100 launch
events, 37 durations were invalid: 36 kernel events and one input D2D copy; no
memset event was invalid. The failures mostly occurred in backward kernels (26/37),
but spanned several kernel symbols. Removing concurrent training activity therefore
did not eliminate the issue. The complete HIP event envelope remains invalid by the
strict validator; valid event subsets and RGP's separate full-batch timeline remain
usable for their explicitly bounded purposes. This repeat does not identify whether
the occasional reversed timestamps come from the HIP event API, driver scheduling,
or this instrumentation pattern. A follow-up mixed-operation probe compared
extended-launch event pairs with ordinary event pairs bracketing the same kernel,
with periodic copy, memset, and device-sync operations. Across four 256-kernel
repeats, extended timing had 2, 3, 3, and 1 invalid intervals; ordinary bracketing
had 1, 5, 1, and 0. Neither method was consistently valid. Ordinary event spans
averaged 47–100 us per repeat versus 23–56 us for extended events, so they also
include queue/dispatch gaps and cannot be substituted as kernel execution time.
Keep strict validity checks and partial coverage; the independent full RGP event
timeline remains the complete dispatch source for this synthetic batch. The probe
source and executable remain under ignored `build_verify/rdts/` and are not
production changes.

## Deployment and remaining work

The production Trainer is not replaced or restarted. Rebuild its HIP object and
relink before requesting bounded captures during a future isolated run. The smoke
object is in `build_verify/workload_profile`; never confuse it with deployed code.
TUI formatting/statistics can be deployed independently.

Magpie and rocprof/rocprofv3 are absent from the current PATH; the installed
`D:/AMD_ROCm/bin` contains hipcc but no rocprof tool. Native Windows RGP has
captured an isolated HIP batch-32 smoke workload with hardware counter
collection enabled. A 256-dispatch prefix reports a 22.12 ms window and 60.52%
GPU idle, and exposes cache-hit, memory-unit busy/stalled, Fetch/Write,
local-video-memory and PCIe counter tracks. Counter collection is confirmed,
but numeric samples still require reading RGP graph tooltips or selected-region
summaries; they have not been extracted. Per-event timing distributions and
shader resource signatures are now exported. Actual VALU/DRAM/stall values, an
actual Trainer capture, and per-WGP measurements remain open. Do not treat a
Linux or experimental WSL support statement as support for this running Windows
Trainer. No driver/toolchain change is made by this task.

An alternate direct counter-reader was checked before attempting a custom
integration: the public GPUPerfAPI usage API lists Vulkan, DirectX 12/11 and
OpenGL backends/context types, but no HIP backend. The presence of
`GPUPerfAPICounters-x64.dll` in the Radeon tool bundle therefore does not by
itself provide a supported HIP-context API for this Trainer; RGP's existing
HIP capture path remains the available route in this environment. This is a
conclusion from the documented public API surface, not a claim about AMD's
internal RGP implementation. References: [GPUPerfAPI usage](https://gpuopen.com/manuals/gpu_performance_api_manual/usage/),
[GPUPerfAPI overview](https://gpuopen.com/gpuperfapi/).

Remaining non-WGP GPU-internal measurements are numeric cache-hit rates,
Memory Unit busy/stalled percentages, Fetch/Write and local-memory byte
samples, numeric per-shader Hardware Utilization/VALU values, and a valid
full-step GPU-execute/idle timeline. Shader ISA instruction timing is now
partially available: the isolated batch-32 profile exposed opcode hit counts,
instruction-cost percentages and latency clocks for 20 of 109 checked events,
across 9 of 20 kernel symbols. This is not an aggregate VALU utilization or a
physical-memory-stall measurement. Counter graph values and the complete device
timeline remain unavailable through the tested background automation path. To
close those, capture instruction timing where supported and read counter graphs
through RGP's foreground hover/region-summary UI or a future supported export
interface. Treat that as a separate bounded profiling task and record counter
pass count and perturbation. Do not fill missing hardware counters or byte rates
with estimates. WGP remains separately deferred. See
[`RGP_CAPTURE.md`](RGP_CAPTURE.md).

Official export requests checked on 2026-09-27:
- https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/112
- https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/124

Official references:
- https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/component-support.html
- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3-on-wsl.html
- https://rocm.docs.amd.com/projects/HIP/en/docs-5.0.2/user_guide/programming_manual.html
