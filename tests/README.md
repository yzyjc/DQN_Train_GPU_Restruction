# Training regression suite

## Current full suite

Run `tests/run_regressions.ps1` for all 19 groups and the five active projects
(Core, NoTerminal Trainer/TUI/Watchdog, Log_Vindicator). It delegates to the dated
runner rather than maintaining a second copy of compiler/linker orchestration.
`-OutputDirectory` selects a separate build output; `-SkipBuild` reuses project
artifacts but still compiles and runs the tests. Do not use SkipBuild after a
production-source change. Default output: `build_verify/architecture_20260831`.

Each invocation creates a fresh `run_...` directory. All fixture logs, models,
quarantine files and fault injections stay there. Tests do not invoke the actual
Trainer/TUI/Watchdog entry points or attach to production training. GPU tests do
perform bounded updates of synthetic networks on the local GPU.

The latest additions cover shared metric format/reader/writer compatibility,
unsupported-version preservation, partial headers and records, denied access,
Dashboard tail alignment and missing/empty distinction, NaN rejection,
Vindicator directory loss/recovery, and deferred signal shutdown.
Earlier groups cover atomic CPU/GPU updates, checkpoint restore/rollback,
skills, save retry, replay boundaries, Guard reset, and UI timeline/history load.
Freeze-critical groups additionally cover batch/TQ gates, Sentinel rollback,
400-column checkpoint roundtrip, trajectory analysis and Probe 3D rendering.
See `../ARCHITECTURE.md` for ownership and on-disk coordinate contracts.

## Original focused training suite

Run `tests/run_training_regressions.ps1` from PowerShell. It builds the actual Core
project (including HIP kernels), then runs in a newly created directory under
`build_verify/training_regressions`. It does not stop any process, load the live
model, or write the live training logs/checkpoints.

Coverage:

- Side-to-move encoding, colour-swap equivalence, CPU/GPU forward agreement.
- Poisoned attention cache overwritten; every softmax row sums to one.
- Eight CPU/GPU training updates: finite parameters, learned Q/K/V projections,
  parameter agreement within 2e-6.
- Fixed winning/blocking questions: nonterminal, no duplicate boards, legal
  winning answers or moves removing every immediate enemy threat.
- Pure solitaire empty-board test, forced-first-move robustness, invalid output
  not counted as wins, signed correct-versus-incorrect margin.
- Graduation baseline stays fixed; slower completion triggers regression alert.
- V7 checkpoint roundtrip of LR, consolidation target/passes, skills and replay;
  V5 body compatibility; legacy log replay stops at the resumed episode.
- Skill metric absent initially, subsequently created, and partial final record.

## Compatibility

### Checkpoint memory pairing (GPT6-Astra, 2026-09-08)

`checkpoint_memory_regression` is included in the full suite. It generates its
own fixtures and uses real save/load functions without IPC or a live Trainer.
It covers current V11 checkpoint/memory pairs, historical V8/V10, V9/V10 and V9/V11 pairs,
missing and mismatched memory, an occupied V10 sidecar during upgrade, retry,
preservation of live state after rejection, and strict matching of a zero CRC.
Historical fixture conversion includes nonempty replay and opponent-pool data.

Focused run after building the projects into a chosen output directory:
`tests/run_recheck2_regressions.ps1 -SkipBuild -Tests checkpoint_memory_regression -OutputDirectory <build-output>`.
The new format requires V11 memory even when its expected CRC is zero. Legacy
untagged memory remains readable; a present tagged legacy sidecar must match.

The current checkpoint writer emits V11. The following describes the retained
V6 layout boundary; current restore also handles the later V7 training state.

V6 checkpoints retain the V5 metadata layout and append a versioned training-state
block before replay; CRC covers both. Pre-V6 models still load with a warning:
their absolute-colour training semantics differ from the corrected canonical
side-to-move network input. Their old scores are not comparable to v2 scores.
No model, replay or live log is deleted by these fixes.

Trainer and TUI share `run_skill_tests` and the live greedy near-tie policy.
In suite v3, `SoloEmpty` uses 64 fixed-seed trials; the 25 forced opening
locations are reported separately. No training/evaluation RNG is consumed.
New skill curves use `logs/metric_skill_v3.bin`; old v1/v2 curves remain untouched.
This suite originally used V7; the existing baseline marker distinguishes legacy
(1) from v3 (3) scores. Legacy baselines are measured again before comparison.
TUI evaluates a frozen copy of loaded weights and identifies it by fingerprint,
without borrowing the current trainer episode as snapshot metadata.

The ROCm path and architecture are overridable with `DqnRocmRoot` and
`DqnGpuArch` MSBuild properties. Core project references replace library-name
search paths, so consumers link the library produced in the same build.

## Training fix regression (2026-09-01)

`train_fix_regression` is included in `run_regressions.ps1`. It exercises actual
Core/Trainer functions without starting their entry points. The full minimax
evaluator is tested for completion, mid-run cancellation and invalid actions;
only the separate metadata test stubs expensive evaluation to a known score.
Other cases cover read-only candidate assessment, rejected candidate state
preservation, graduation checkpoint/restart replay isolation, legacy wrapped
replay filtering, identical live/test near-tie actions, reproducible zero-model
scores, correct evaluated-vs-next depth, skill torn-tail append, denied sharing,
old/future version preservation, invalid metrics and v3 log rollback.

The fixed Win1/Block1 banks remain local pattern tests, not a comprehensive
held-out collection of dense, legally reached game positions. Passing these
regressions is not a claim of policy convergence or resistance to forgetting.
