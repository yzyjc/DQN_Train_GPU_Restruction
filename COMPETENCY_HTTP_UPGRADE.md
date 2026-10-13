# Competency course and local TUI control — implementation checkpoint

Collaboration signature: GPT6-Astra. Date: 2026-10-13.

GPU Analyzer work remains paused. No production executable has been replaced and no formal training has been started for this upgrade.

## Runtime settings

The competency course and rehearsal default to disabled. Existing configurations retain legacy startup behavior. To test the new course in an isolated runtime, set `curriculum_enabled=1` and optionally `rehearsal_enabled=1` in `dqn_config.txt` before starting Trainer. All 30 keys and limits are centralized in `DQN_Core/include/competency_state.h`, persisted by the existing configuration system, and exposed through `GET_CONFIG` / HTTP.

Stages 0–2 use finite single-color line-completion teacher batches with progressively mixed depth-1 Minimax games (default stage 1: 25%, stage 2: 50%). Stage 3 uses per-color frozen snapshots, refreshed every 100 course games; stage 4 uses older snapshots, refreshed every 500 games; stage 5 uses independent online self-play. Opponent snapshots are separate from Bellman target networks. Promotion requires both colors to pass EmptyWin and (after Stage 0) Win1/Block1 gates, a minimum 5000 stage games, three consecutive evaluations and a 1000-game cooldown. These are engineering defaults, not evidence of learning effectiveness.

Rehearsal requires a mastered skill to decline across consecutive evaluations. It updates only the selected color online network, with a 200-committed-batch / 400-attempt budget and 4000-game cooldown. Target weights are not changed by rehearsal. Tactical training and evaluation puzzles use disjoint board-hash partitions. EmptyWin is explicitly a single-player diagnostic, not a legal alternating game evaluation.

The reduced LR is scoped to rehearsal. The side-specific LR is an actual override in the batch entry, so changing only the caller's LR is insufficient. The controller applies the reduction once and restores normal side LR on all exits; an actual numerical safety backoff is preserved proportionally. Normal scheduling is not permanently replaced by rehearsal LR. A real three-batch controller regression verifies normal training can commit afterward with its original LR.

V14 checkpoints append course state and four frozen/older snapshot networks to the existing layout under CRC. Legacy loading remains supported; absent course state begins Stage 0 when explicitly enabling the course. Failed staged loads use separate temporary independent-network storage before committing live state.

Status and history: `logs/competency_status.json`, `logs/competency_events.jsonl`; HTTP `/api/competency`. TUI System panel and `course status` show the saved per-color skill snapshot, gate progress and rehearsal/cooldown summary. They explicitly label saved skill metrics; they do not imply independent Minimax win rates. Dedicated course sliders are not yet implemented. Startup flags must be set before startup; runtime enabling is rejected while independent networks are inactive. Invalid/incomplete skill evaluation never promotes or triggers rehearsal from stale scores.

## Local HTTP module

The service is embedded in TUI, bound only to `127.0.0.1`. Preferred port is 17865; if occupied, it tries the previous fallback then ports 17866–17874. Read `tui_http_endpoint.json` beside the executable for actual URL, PID and random bearer token. Put `enabled=0` or `port=17865` in `tui_http_config.txt` to configure it. This is local HTTP; future remote HTTPS is outside this implementation.

TUI commands `http status`, `http on`, `http off` query or manage the service without stopping Trainer.

Public: `GET /api/help`, `GET /api/health`. Other routes require `Authorization: Bearer <token>`:

- `GET /api/status`, `/api/config`, `/api/competency`, `/api/action`.
- `PATCH /api/config`: `{"key":"learning_rate","value":-1}`. One finite validated key per request; -1 restores AUTO for learning rate and epsilon.
- `POST /api/trainer/action`: `{"action":"start"}`, `save_and_stop`, `save_snapshot`, or `resume`.

Errors include a tutorial. HTTP 202 means queued, not applied; read actual config/status afterward. Start/stop are dispatched by the UI thread and `/api/action` reports dispatch progress, not proof of saved data. Queries use their own pipe connection. Browser Origin requests are rejected. The API contract, Win32 socket adapter and Trainer pipe bridge are separate modules.

For isolated TUI checks only, `DQN_TUI_AUTOSTART=0` suppresses automatic Trainer startup. Normal startup behavior is unchanged.

## Probe

Q/K/V receive merged token+position inputs, rather than forming an adjacent-layer chain. Computed EmbeddedTokens/AttentionMix nodes and gray dependency edges explain the real tensor topology; gray edges are not measured gradients. EffGrad is the snapshot parameter displacement divided by update count and effective learning rate, with separate history per color; it is not raw backpropagation gradient.

## Verification and remaining acceptance

Isolated library, Trainer and TUI builds passed with 64-bit MSBuild/compiler and separate intermediate directories. `tests/run_competency_regressions.ps1` gives every test a new runtime directory and places the matching SDK HIP DLL beside test executables. The old latest-checkpoint assertion failed because an EP7000 artifact from a different test shared the directory; it passes in a fresh directory. A sandbox GPU crash occurred with both runtime variants; the matching-runtime unrestricted GPU test passes. These observations do not establish a hardware fault.

Passing checks: outcome formulas and three CPU/GPU update comparisons, Probe history/topology helpers, course gates, per-color regression/cooldown, holdout partition separation and legal color-to-move counts, V14 exact frozen-snapshot/cooldown restoration, failed-memory load preserving live state, frozen/lagged cadence, CPU rehearsal isolation, actual GPU teacher batches (two consecutive batches per color, trained parameters change and opposite resident parameters do not), course presentation, HTTP contract and real sockets, real native Trainer IPC queued/apply/readback, legacy V12/V13 persistence and reward propagation. Hidden real TUI smoke (`tests/tui_http_smoke.ps1`) passes with automatic Trainer startup disabled.

Hot configuration now accepts both `key=value` and `key = value`, validates course settings, and rejects runtime enabling while independent networks are inactive. Regression coverage explicitly tests this guard. Use `tests/run_competency_regressions.ps1 -IncludeGpu` from the repository for isolated builds and tests; it refuses to run beside a live Trainer.

Manual LR control is also dispatched to both side overrides, rather than only updating the global display/config value. When leaving a live manual override, AUTO restores the previously captured per-side LR, including an existing asymmetric numerical safety backoff. Configuration and effective parameter changes must still be distinguished from evidence of useful learning.

Remaining: real TUI→HTTP→running Trainer session lifecycle (especially graceful save/stop), full production curriculum progression and interrupted save/recovery, broader corruption/migration cases, dedicated TUI course controls and independent Minimax evaluation integration, long learning-effect comparisons. The tests exercise bounded batches and real adapters, not an entire curriculum or learning effectiveness. No production binary was installed; no formal long training was run.

## Final verification record

- Build: core library, Trainer, TUI, all passed with the 64-bit toolchain.
- Main suite: 15/15 passed; evidence `build_verify/competency_final_20261013/run_20261013_164914_797/`.
- Additional: outcome TUI renderer/readback regression passed; latest hidden TUI HTTP smoke passed on port 17865 with automatic Trainer startup disabled.
- Source implementation commits: `95503f2`, `a022ed7`, `fa35231`, `e49ae9e`, `ee6907d`; earlier outcome and Probe commits remain `d5c5249`, `248fcb6`.
- No production deployment, no formal long training, GPU Analyzer development remains paused.

## Changed files

- `COMPETENCY_HTTP_UPGRADE.md`
- `DQN_Core/include/competency_state.h`
- `DQN_Core/include/competency_training.h`
- `DQN_Core/include/core.h`
- `DQN_Core/include/probe_diagnostics.h`
- `DQN_Core/include/skill_test.h`
- `DQN_Core/include/step_outcome.h`
- `DQN_Core/include/terminal_outcome_credit.h`
- `DQN_Core/src/checkpoint.cpp`
- `DQN_Core/src/dqn_agent.cpp`
- `DQN_Core/src/ipc_server.cpp`
- `DQN_Core/src/neural_probe_publisher.cpp`
- `DQN_Core/src/skill_test.cpp`
- `DQN_Core/src/utils.cpp`
- `DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp`
- `DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.vcxproj`
- `DQN_Main_TUI_No_Terminal/competency_view.h`
- `DQN_Main_TUI_No_Terminal/local_http_api.h`
- `DQN_Main_TUI_No_Terminal/local_http_server.cpp`
- `DQN_Main_TUI_No_Terminal/local_http_server.h`
- `DQN_Main_TUI_No_Terminal/neural_viewer_3d.cpp`
- `DQN_Main_TUI_No_Terminal/neural_viewer_layout.cpp`
- `DQN_Main_TUI_No_Terminal/trainer_http_bridge.cpp`
- `DQN_Main_TUI_No_Terminal/trainer_http_bridge.h`
- `DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp`
- `tests/competency_checkpoint_regression.cpp`
- `tests/competency_gpu_regression.cpp`
- `tests/competency_rehearsal_lr_regression.cpp`
- `tests/competency_state_regression.cpp`
- `tests/competency_view_regression.cpp`
- `tests/hot_config_course_regression.cpp`
- `tests/local_http_api_regression.cpp`
- `tests/local_http_transport_regression.cpp`
- `tests/outcome_revision_regression.cpp`
- `tests/probe_diagnostics_regression.cpp`
- `tests/run_competency_regressions.ps1`
- `tests/step_outcome_regression.cpp`
- `tests/step_outcome_ui_regression.cpp`
- `tests/trainer_http_bridge_regression.cpp`
- `tests/tui_http_smoke.ps1`
