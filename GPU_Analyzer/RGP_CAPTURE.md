# Native Windows HIP capture with Radeon GPU Profiler

Collaboration signature: GPT6-Astra.

## What this path adds

The lightweight workload trace in `WORKLOAD_PROFILING.md` measures host-side
module proportions and requested operation sizes. RGP adds a separate, short
hardware trace: HIP dispatch events, a GPU-side event timeline, wavefront
occupancy views, and supported hardware counters. These views can help separate
device execution from dispatch gaps and inspect memory/compute behavior at event
or shader level. They do not turn host API time or requested copy bytes into
physical DRAM bandwidth.

RGP's compute documentation lists HIP on Windows 11 and Ryzen processors with
Radeon graphics. The installed tool successfully captured this machine's
8060S, but that does not guarantee every counter is exposed on every driver.
Available counter views must be checked in the captured profile itself.

## Reproduce the bounded test

Build `tests/gpu_workload_profile_smoke.cpp` as the isolated `smoke.exe` used by
the workload-profile tests. It runs one deterministic batch of 32 and does not
load Trainer, replay, checkpoints, or production logs. Then run from the project
root while no formal training is running:

```powershell
.\tests\run_rgp_capture.ps1 `
  -RdpCli 'C:\path\to\RadeonDeveloperPanelCLI.exe' `
  -IConfirmNoTraining
```

The script creates a timestamped directory under `build_verify\rdts`, copies
the smoke binary and matching `amdhip64_7.dll` into an isolated app directory,
checks an unprofiled/profiled parameter snapshot pair, and requests at most 32
dispatches with counter collection. The smoke process remains alive for the
bounded capture window. It writes `capture_manifest.txt`, the `.rgp` profile,
and RDP stdout/stderr for diagnosis. Hardware trace collection can raise GPU
clocks and perturb the device; it is intentionally restricted to the isolated
test executable. Do not attach this workflow to a formal run as a background
monitor.

## Read the profile

Open the `.rgp` file in Radeon GPU Profiler. Inspect the compute queue's events
and wavefront-occupancy/counter views. For a representative set of dispatches,
record the event name, GPU duration, time gaps between events, occupancy view,
and whatever memory/VALU/cache counters the driver exposes. Record the tool and
driver versions, counter collection state, dispatch range, and whether capture
required multiple counter passes.

There is no supported batch CLI export of RGP's analysis tables in this tool
path. The profile is a binary RDF capture; this project does not parse its
private payload. The Windows UI exposes the selected event table and its
accessible rows, so the complete event table can be transcribed through the
public UI without reading the private capture format. Use the repository
script from an open RGP profile to export the event timing and a summary:

```powershell
.\GPU_Analyzer\tools\export_rgp_timing.ps1 `
  -OutCsv 'build_verify\rdts\event_timing.csv'
```

It selects the Event timing tab, reads the event rows through Windows UI
Automation, and writes event timings plus a `.summary.csv`. The first
successful 8060S profile was a synthetic batch-32 smoke workload, not an
actual Trainer interval; raw captures and exports are kept under ignored
`build_verify\rdts\`.

## Current evidence and limits (2026-09-27)

- RDP/RGP capture succeeded with hardware counter collection enabled for the
  isolated HIP batch-32 workload. A repeatable 256-dispatch profile was
  23,764,120 bytes with SHA256
  `CF5D0244C0E484B5D47D709431B4CDBF5D6AD38D748DFF5F01EBCE8540EADECB`.
  It is stored under the ignored
  `build_verify\rdts\capture_20260927_143942_27380\` directory.
- Parameter snapshots from the no-profile and profile paths matched in the
  independent smoke check. This checks workload determinism, not zero profiler
  overhead.
- The 256-dispatch overview reports a 22,121.287 us capture window, 60.52% GPU
  idle (39.48% active, derived), and classifies this isolated segment as CPU
  bound. RGP reports 19.00 MB of profiling overhead, or 0.88 GB/s of video
  memory. These are values for this captured prefix only; they are not
  full-batch or Trainer-wide rates. The smoke app is intentionally held open
  for trace collection, so idle percentages must be interpreted only inside
  the profile's bounded capture window.
- The exported all-events table contains 512 rows: 256 dispatches and 256
  `CmdBarrier` rows. Dispatches comprise 93 fill-buffer, 6 copy-buffer, and
  157 user-kernel calls (including 25 `kernel_bias_update`, 20 `kernel_fc`,
  20 `kernel_fc_update_w`, and 15 `kernel_fc_delta`). The separate host-side
  full-batch counter reported 1,025 launches, but the RGP event cap/window is
  not a proportional sample: do not describe these 256 events as a quarter of
  the batch or extrapolate their counts to a full training step.
- The event table's `Duration` column gives this captured set's per-dispatch
  GPU durations. Aggregation of all 512 rows (nearest-rank P95) produced:

  | Dispatch | Calls | Mean (us) | Median (us) | P95 (us) | Sum (us) |
  |---|---:|---:|---:|---:|---:|
  | `kernel_attn_backward` | 5 | 472.978 | 472.929 | 493.495 | 2,364.890 |
  | `kernel_pool_backward` | 5 | 215.110 | 214.862 | 233.622 | 1,075.550 |
  | `kernel_fc_delta` | 15 | 59.419 | 55.970 | 156.968 | 891.280 |
  | `kernel_fc` | 20 | 32.996 | 24.300 | 70.980 | 659.910 |
  | `kernel_attn_out_backward` | 5 | 36.706 | 36.240 | 39.260 | 183.530 |
  | `kernel_flatten_backward` | 5 | 26.378 | 26.100 | 28.470 | 131.890 |
  | `kernel_fc_update_w` | 20 | 4.633 | 3.460 | 9.870 | 92.650 |
  | fill-buffer (`__amd_rocclr_fillBufferAligned`) | 93 | 1.431 | 1.300 | 2.080 | 133.046 |

  Across the 256 dispatch rows, summed `Duration` is 5,871.405 us: user
  kernels 5,720.213 us, fill-buffer 133.046 us, and copy-buffer 18.146 us.
  These sums describe only the bounded capture; they are not the 22.121 ms
  capture-window duration and should not be used as a full-step timing total.
- The event timing view supplied start/end timestamps for all 512 events. The
  repository exporter reproduced the independent manual export exactly
  (512/512 event IDs, names, starts, ends, and durations matched). Across the
  255 intervals between consecutive non-barrier dispatches, the signed gap
  mean was 62.945 us, median 0.340 us, P95 230.093 us, minimum -0.381 us, and
  maximum 3,972.285 us. There were 15 small overlaps (each within 0.381 us),
  52 positive gaps over 100 us, and 2 over 1 ms. Positive-only mean was
  66.901 us. These are dispatch-to-dispatch idle intervals inside this bounded
  synthetic capture, not a direct measurement of CPU-side launch latency.
- A second UI Automation exporter reads one representative dispatch per
  distinct `kernel_*` symbol from RGP's Wavefront occupancy view:

  ```powershell
  .\GPU_Analyzer\tools\export_rgp_shader_resources.ps1 `
    -EventTimingCsv 'build_verify\rdts\event_timing.csv' `
    -OutCsv 'build_verify\rdts\shader_resources.csv'
  ```

  It collected 20 kernel-symbol rows. All showed theoretical occupancy
  `16 / 16` in RGP's resource model; this is not measured achieved occupancy,
  nor CU/WGP activity. The largest captured launch was `kernel_pool_update`
  (1,600 wavefronts / 51,200 threads). Its resource row used 7 VGPR
  (24 allocated) and 14 SGPR (128 allocated). Among the smaller launches,
  `kernel_attn_wqkv_update` used 84 VGPR (96 allocated), while
  `kernel_attention` used 75 (96 allocated). The selected representative
  events and full per-stage resource fields are in the generated CSV under
  ignored `build_verify\rdts\`; this groups by symbol and does not represent
  every shape or every dispatch. No LDS allocation was reported for these
  rows. These shader resource values describe launch geometry and compiler
  resource use; they cannot establish actual wave residency over time.
- RGP's `CmdBarrier` timing row spans the barrier's start/end and includes
  overlapping dependency work. Its mean span was 20.445 us (median 1.237 us,
  P95 108.439 us, max 493.502 us), while the event table's barrier `Duration`
  mean was 0.313 us; its `WorkDuration` matched the timeline span for all 256
  barriers. Barrier spans must not be added to dispatch time or called pure
  synchronization wait.
- In the earlier 32-dispatch capture, the nine named forward kernels had these
  individual durations: token embedding 2.409 us, QKV projection 6.592 us,
  attention 11.027 us, concat/projection 3.877 us, flatten 1.350 us, pooling
  5.352 us, and the three FC kernels 8.699, 53.754 and 27.924 us. This tiny
  nine-event sample is useful as a sanity check, not a stable P50/P95 estimate.
- The profile's Wavefront occupancy view visibly exposes cache-hit tracks
  (instruction/scalar/L0/L1/L2), Memory Unit busy/stalled, and Fetch size,
  Write size, local-video-memory bytes, and PCIe bytes. RDP's capture log
  confirms counter collection completed. Their numeric samples remain
  unavailable in the exported evidence: RGP presents values in graph-hover
  tooltips or for a selected graph region. Although the RGP window was brought
  forward and maximized for inspection, `SetCursorPos` and `GetCursorPos`
  returned failure in this execution session, and UI Automation exposes no
  numeric graph samples. Thus cache hit rate, memory busy/stall percentages,
  and physical-memory byte totals are still **not measured**. Counter-track
  visibility is availability evidence only.
- Per-event memory traffic and derived DRAM GB/s remain unavailable. The RGP
  memory-byte tracks are documented for RDNA 3+ and account for cache/memory
  effects, but the profile's displayed time-series values have not been
  extracted. Keep requested HIP copy/clear bytes and their host-call ratios
  separate from these hardware counters.
- Automation capability recheck (2026-09-28): UI Automation exposes the RGP
  counter-track names and graph bounds, but not the graph's numeric samples or
  tooltip values. The current profile also shows `No instruction timing data`
  for the selected kernel, so it cannot supply shader-level ALU utilization.
  A background `WM_MOUSEMOVE` still does not expose the tooltip; this session's
  `SetCursorPos` and `GetCursorPos` calls both failed, so foreground pointer
  inspection is unavailable here. This closes the extraction routes tested in
  this session, not the underlying hardware measurements. A supported
  machine-readable RGP export could reopen these items. AMD's public RGP
  repository still has open requests for counter/timing export and
  command-line batch analysis; this confirms the absence of a supported export
  route in the checked tool path, not a guarantee about all future RGP
  releases.
- A second bounded synthetic capture on 2026-09-27 enabled RGP instruction
  tracing through the RDP CLI. It requested 256 dispatches, produced a
  24,156,891-byte profile
  (SHA256
  `EBD94D25C56200127288E93320BA5C72F33BC54C3D59258DBCB63CB5A11E7701`), and
  retained exact parameter parity with the unprofiled smoke run. The ignored
  evidence is under
  `build_verify/rdts/capture_20260927_235423_49092/` (profile, timing CSV,
  resource CSV, and instruction CSV). The profile title reports instruction
  tracing limited to Shader Engine 0. The background exporter checked 109
  early events (up to eight per kernel symbol); 20 events
  across 9 of 20 kernel symbols had instruction data, yielding 1,988 ISA rows.
  These rows include opcode, hit count, instruction-cost percentage and latency
  clocks. They are a sampled per-shader instruction view, not aggregate GPU
  unit utilization or a DRAM-stall counter. In the traced subset, the `s_wait*`
  opcodes accounted for a mean 47.88% of instruction cost across 8
  `kernel_fc_update_w` events and 57.20% across 5 `kernel_pool_update` events;
  single-event samples ranged from 53% (`kernel_attn_wqkv_update`) to 92%
  (`kernel_token_embedding`). Treat these as explicit shader dependency-wait
  issue cost in this profile only, not as proof that physical memory is the
  bottleneck. RGP's numeric Hardware Utilization chart values remain unavailable
  to the tested background automation path.
- No full Trainer profile was taken; no production training process was
  attached, stopped, or modified.
- Per-WGP metrics are explicitly deferred. This capture does not establish a
  physical WGP mapping or WGP utilization.

## Full synthetic batch capture (2026-09-28)

The first 1,025-dispatch request completed in RDP, but exporting its RGP event
table showed only 256 dispatches. The helper had set `--rgp-auto-capture` but
left the separate capture mode and render-operation count at their defaults.
The capture helper now explicitly sets `--rgp-capture-mode dispatch` and
`--rgp-render-op-count <DispatchCount>` as well. A repeat using both explicit
options captured and exported all 1,025 requested dispatches (plus 1,025
`CmdBarrier` timeline rows), matching the isolated smoke's 1,025 host launch
count. Baseline and profiled parameter snapshots matched exactly
(`77C826CC93B1BFE0E1F32FAF9B1E60BA3E859AC9F15B0D171DEEB358682BD7D1`). The
verified profile is 58,028,715 bytes and is under
`build_verify/rdts/capture_explicit_dispatch_20260928_015140/`.
After updating the capture helper, a separate 32-dispatch run also produced
the expected 64 event rows (32 dispatches and 32 barrier rows).

For this one instrumented synthetic batch, the dispatch envelope spans
71,077.686 us and the sum of non-barrier event durations is 21,542.981 us
(30.31% of that envelope; 69.69% is inter-dispatch time by subtraction).
This is an RGP event-timeline duty estimate for the isolated capture, not a
Trainer-wide GPU utilization figure. The 1,024 signed gaps between consecutive
dispatches have mean 48.37 us, median 0.34 us, P95 200.85 us and maximum
4,870.84 us; 154 positive gaps exceed 100 us and 2 exceed 1 ms. Small negative
gaps (61, minimum -1.06 us) reflect timestamp overlap/precision and are not
negative idle time.

There are 640 user-kernel dispatches, 363 runtime fill-buffer dispatches and
22 runtime copy-buffer dispatches. User kernels sum to 21,062.351 us. The four
largest user-kernel groups account for 86.75% of user-kernel time:

| Kernel | Calls | Mean (us) | P95 (us) | Sum (us) |
|---|---:|---:|---:|---:|
| `kernel_attn_backward` | 20 | 433.00 | 482.75 | 8,659.94 |
| `kernel_pool_backward` | 20 | 199.05 | 223.06 | 3,981.03 |
| `kernel_fc_delta` | 60 | 53.02 | 111.07 | 3,181.36 |
| `kernel_fc` | 80 | 30.62 | 69.95 | 2,449.26 |

The dominant kernels support the earlier host-module finding that backward is
the largest training cost. The much longer dispatch envelope than summed
kernel durations also indicates that this batch is not continuously executing
user kernels; the trace alone cannot separate host launch starvation from
runtime scheduling or counter-capture effects. This is evidence of substantial
non-dispatch intervals in the smoke workload; it does not assign the remaining
time in a Trainer step. This remains a synthetic batch, and counter graphs are
present but their numeric memory/cache/VALU samples were not extracted. RGP
profile SHA256:
`8FE23FF3A365EE51C98FFF45A2F7C7E4039BDDA1982534997ADAC6572F382E5D`.
The earlier 116 MB profile was captured without explicit dispatch mode and
render-op count; its event table contained only 256 dispatches and is not the
full-batch timing evidence.

Official references:

- [Radeon GPU Profiler](https://gpuopen.com/rgp/)
- [RGP events and wavefront occupancy](https://gpuopen.com/manuals/rgp_manual/events_windows/)
- [Radeon Developer Panel CLI](https://gpuopen.com/manuals/rdp_manual/radeon_developer_panel_cli/)
- [RDP profiling features and instruction tracing](https://gpuopen.com/manuals/rdp_manual/features/)
- [RDP initial setup and process restrictions](https://gpuopen.com/manuals/rdp_manual/initial_setup/)
- [RGP timing/counter export request](https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/112)
- [RGP command-line batch-analysis request](https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/124)
