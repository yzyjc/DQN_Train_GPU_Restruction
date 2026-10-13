# Analyzer telemetry and layout

Author: GPT6-Astra (user-authorized collaboration signature)

## Implemented

- CPU maximum and per-thread readings use Processor Time, not frequency-scaled Processor Utility. Invalid/nonfinite samples are unavailable, not fabricated zeroes.
- Core/SMT pairing comes from Windows processor topology, not assumed adjacent thread numbers.
- Wide windows show metrics on either side of a logical CPU/GPU schematic. This is not a physical die map.
- Narrow windows omit the schematic; wheel, touch-drag wheel routing and Page Up/Down scroll all metric sections. Each column stops at its own last row.
- Corrected system CPU and memory capacity labels, Phase parsing and stale trailing text.

## Remaining work

- **Non-WGP, sampled system/device sensors:** GPU temperature, GPU power (not
  total APU/system power), core clock, and ADLX-reported memory clock are
  integrated into Analyzer, TUI, binary/CSV fields, and JSON statistics.
  Hotspot is not supported by this adapter's ADLX metrics; throttling has no
  validated source. Multiple
  GPUs deliberately return unavailable until exact adapter matching is
  implemented.
- **Non-WGP, GPU-internal profiling:** host module proportions, launch/copy/clear
  counts and requested bytes, explicit synchronization host-wait time, partial
  valid HIP event envelopes, event distributions, dispatch gaps, shader
  resource signatures, and a partial RGP ISA instruction trace are captured. A
  complete full-step device timeline, numeric RGP cache/memory counters,
  hardware byte totals, and numeric per-shader Hardware Utilization/VALU values
  are still unavailable. The instruction trace covers only a subset of sampled
  events on one shader engine; it does not replace aggregate counters. RGP
  counter graph values are not exposed by the tested background UI Automation
  path. See
  [`WORKLOAD_PROFILING.md`](WORKLOAD_PROFILING.md) and [`RGP_CAPTURE.md`](RGP_CAPTURE.md)
  for the measured subsets and the exact limits.
- **Per-WGP utilization:** intentionally paused. First prove an actual per-WGP
  counter source exists for this Windows driver; aggregate GPU engine usage
  cannot be split into WGP usage. Display remains unavailable, and sampling
  overhead must be measured separately.
- Physical chip artwork is presentation only and independent of telemetry.
  The current schematic deliberately makes no claim about physical CPU core
  placement.

Quota MCP did not return during this work; no remaining quota percentage or time estimate was inferred from it.

Validation: analyzer_telemetry_regression tests actual PDH/topology, invalid values and clipping; analyzer_layout_regression renders the production page at 85/160/240 columns and verifies the last metric remains reachable. These tests do not launch Trainer.

## Sensor validation (2026-09-27)

Official SDK probe on AMD Radeon 8060S returned success for capability and value queries for all three sensors (82 C, 45 W, 1094 MHz at that instant).
The old TUI ADL initialization used an incorrect no-argument declaration; this unsafe legacy path was removed. Sensor access now lives in Analyzer's sampling thread, using the official SDK declarations and releasing interfaces before terminating ADLX.

- Real backend regression: absent backend and mismatched name return unavailable; five consecutive full process lifecycles passed, with three valid readings each.
- An earlier probe run exited with access violation before buffered output was visible. Subsequent instrumented and five-repeat runs passed; its cause was not established. Do not describe this as proof of long-run driver stability.
- Analyzer window lifecycle, binary record completeness and CSV smoke passed.
- JSON parsing passed; seven new valid sensor records reported 65–66 C, 42–54 W and 1191–1333 MHz. Earlier unavailable records were excluded from sensor statistics.
- TUI production-page regression covers sensor values and disconnected fallback in addition to three layout sizes.

References: https://gpuopen.com/manuals/adlx/adlx-sdk-references/adlx-interfaces/performance-monitoring/
and https://gpuopen-librariesandsdks.github.io/adl/adl_8h.html .

## Additional ADLX sensor check (2026-09-28)

The official ADLX metrics support query on the single AMD Radeon 8060S adapter
reported VRAM clock supported and returned 1000 MHz. Hotspot support returned
false; a direct read returned ADLX unsupported (12). The shared-memory sample
remains version 1 and 64 bytes: the new readings reuse reserved padding marked
with a sensor-extension sentinel, so old TUI builds retain the same layout and
new TUI builds treat old publishers' sensor fields as unavailable. New disk
logs go to `metric_gpu_telemetry_v2.bin` so existing 9-field logs are preserved.

The isolated real-device regression then sampled the new path three times:
memory clock was 1000 MHz in all three readings and hotspot stayed unavailable.
The updated Analyzer, report generator and TUI translation units compile. A
full MSBuild rebuild was blocked before compilation because the sandbox denied
MSBuild access to `C:\Users\ROG\AppData\Local\Microsoft SDKs`; the changed
translation units were compiled directly with the installed compiler instead.
## Die-shot presentation (2026-09-27)

The user's root `Die shot.txt` is preserved unchanged. On Ryzen AI Max+ 395 with 16 OS-reported cores, sufficiently wide/tall windows render a compact derivative: two CCD-shaped groups, 16 core boxes with both logical-thread busy percentages, and 20 WGP/CU labels. OS core numbering is not verified against physical die placement or CCD identity; the diagram states this limitation. Per-WGP utilization is intentionally absent. NPU/IO artwork is omitted to fit. Other CPUs retain the generic topology view. The live data columns retain their previous width; smaller windows hide the diagram instead of squeezing metrics.

## Non-WGP closeout status (2026-09-28)

The data that can be obtained reliably from the current Windows Trainer/8060S
setup is implemented. This includes host-side module proportions, requested
launch/copy/clear counts and byte totals, host time inside existing device-sync
calls, valid HIP event subsets and distributions, RGP dispatch timings/gaps and
shader resource signatures, CPU topology/utilization, GPU utilization/memory,
and ADLX edge temperature, power, core clock and memory clock. ADLX returned
live readings for all four supported sensors on the single 8060S adapter;
hotspot is unsupported. Throttle state has no validated source. The isolated
Analyzer lifecycle/layout, telemetry compatibility and CSV checks passed, as
did the workload-trace parser (4/4). These checks did not launch Trainer.

The full synthetic-batch RGP timeline is now measured. Remaining non-WGP gaps
are numeric hardware-counter samples, a Trainer-specific device timeline, and
the cross-operation HIP event envelope that still fails consistency checks:

| Measurement | Current evidence | Needed to close it |
|---|---|---|
| Full synthetic GPU execute/idle and inter-kernel timeline | The explicit-mode RGP capture exported all 1,025/1,025 synthetic batch dispatches. Non-barrier dispatch durations sum to 21.543 ms inside a 71.078 ms dispatch envelope (30.31% derived activity); complete gap and kernel distributions are recorded in `RGP_CAPTURE.md`. This is not a Trainer capture. | Use as synthetic workload evidence; a representative Trainer batch is needed before making training-specific claims. |
| HIP cross-operation event envelope | Four idle-GPU instrumented smoke repeats had 7–12 reversed timestamps per 1,025 launches (99.32%–98.83% valid coverage), mostly backward kernel events. A mixed-operation probe also found invalid intervals with both extended launch events (1–3/256) and ordinary event brackets (0–5/256); ordinary brackets include queue/dispatch gaps and are not pure kernel time. Full envelope validation remains false, so no HIP-derived full-batch GPU idle/duty is reported. | Keep strict validity checks and partial coverage; don't substitute ordinary event spans for execution time. A validated timing source is required. |
| Physical DRAM read/write bytes and bandwidth | RGP memory tracks are visible, but numeric graph samples were not extracted. HIP requested bytes and host-call ratios are not physical traffic. | Numeric RGP counter samples from a supported export or selected-region UI readout. |
| Memory-controller busy/stall and cache-hit rates | RGP exposes counter tracks; background UI Automation returns labels/bounds, not samples. | Numeric samples from a supported export or selected-region UI readout. |
| Numeric VALU / hardware utilization | Shader instruction tracing gives a bounded ISA subset, not aggregate VALU utilization. | Numeric hardware-counter samples from a supported export or selected-region UI readout. |
| Achieved occupancy | RGP resource rows report theoretical occupancy only. | A time-varying achieved-occupancy counter with numeric samples. |
| FLOP/s and arithmetic intensity | No validated operation count plus physical-byte count exists for the complete workload. | A validated FLOP count and corresponding physical-memory byte count over the same interval. |

The current RDP CLI documents capture options, including hardware-counter
collection, but not exporting RGP counter graph samples. RGP's public issue
tracker still lists timing/counter export and batch analysis as open requests.
These items are not claimed complete, and no estimated values are substituted.
A supported machine-readable RGP export or a usable interactive desktop session
is still required for numeric counter samples. During the 2026-09-28 review the
RGP window was foregrounded and maximized, but cursor APIs failed in the
execution session and no numeric counter values were extracted.

Official references checked on 2026-09-28:
[RDP CLI capture options](https://gpuopen.com/manuals/rdp_manual/radeon_developer_panel_cli/),
[RGP counter tracks](https://gpuopen.com/manuals/rgp_manual/events_windows/),
[RGP export request #112](https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/112),
[RGP batch-analysis request #124](https://github.com/GPUOpen-Tools/radeon_gpu_profiler/issues/124).
