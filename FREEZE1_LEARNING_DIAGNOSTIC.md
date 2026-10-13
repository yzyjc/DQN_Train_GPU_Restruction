# Freeze-1 bounded learning diagnostic

Author: GPT6-Astra, 2026-09-19.

Baseline: formal-static-training-v1, EP50000 archive copied into an isolated
directory. Checkpoint SHA256:
`95CC4105793F044DFA76468E8A36F95C1CD225AAF62C0F04A9635C2EBFD071FC`.
Probe built against source a302f3e (checkpoint recovery fix only), including
fresh HIP kernels. AMD Radeon 8060S, ROCm runtime from D:\AMD_ROCm\bin.
No production files, processes, or archived evidence were modified.

`tests/freeze1_learning_probe.cpp <copied-checkpoint> <batches>` requires an
isolated working directory and the corresponding `.mem`. It validates every
constructed transition and requires GPU activation/download and no training
fault. This is a diagnostic, not an automatic claim that training improved.

## Results

Loaded phase 1, EP50000, LR 0.000159999996; KB2048/cache1076. Each physical replay
buffer has 20000 valid entries, 1166 terminal entries and zero negative terminal
entries. 5116 entries offer an immediate win; 1110 select a winning action
(21.70%). Buffers include mirrored observations; these are not independent
games, action-policy frequencies, or a reconstruction of earlier training.

Eight legal black-first Win1 positions: both physical sides and four rotations.
Repeat to batch32 and feed the real replay/DDQN/guard/GPU path; configured
anchors remain mixed in. Existing target network stays fixed in this bounded
terminal-learning diagnostic. It is not a complete training-loop comparison.

| Batches | Mean correct Q | Mean correct-minus-best-wrong | Top choice |
|---:|---:|---:|---:|
| 0 | 0.02479108 | -0.01401765 | 0/8 |
| 100 | 0.02831650 | -0.01080626 | 1/8 |
| 1000 | 0.05933506 | 0.00193558 | 4/8 |

1000 attempts, 1000 commits, zero rejects, no training fault. Most sampled
targets are near 1.0. Learning direction improves but remains weak on half the
small set; this excludes complete update starvation in this experiment, not
all historical gate problems. It does not establish a root cause or warrant
changing LR, representation, or replay sampling without further controls.

Evidence: build_verify/astra_20260919/freeze_probe/result_1000.log (ignored).
Next discriminating tests: per-side/per-layer gradients, exact same batch CPU
control, initial-model control, and Block1 negative bootstrap. New outcome
shaping is a separate feature, not a proven repair for Freeze-1.

## Layer update follow-up

The same copied baseline and seed were rerun for 100 batches with layer delta
reporting. The feature worktree was compiled with outcome shaping disabled;
its batch100 Q/margin match the original 100-batch run above. No production
parameter was changed. Evidence: freeze_probe/layer_result.log in the same
isolated output directory. This probe is reproducible from the accompanying
source; unlike the first run, its library includes the disabled feature code.

| Parameter block | Changed floats | Delta L2 |
|---|---:|---:|
| token_embed | 45/48 | 1.969e-4 |
| pos_embed | 400/400 | 1.640e-4 |
| attn_wq | 1/512 | 4.229e-8 |
| attn_wk | 1/512 | 5.530e-10 |
| attn_wv | 255/512 | 1.214e-4 |
| attn_wo | 196/512 | 1.082e-4 |
| pool_w | 47442/51200 | 8.223e-4 |
| w1 | 94700/131072 | 1.124e-3 |
| w2 | 302287/524288 | 1.343e-3 |
| w3 | 69186/131072 | 1.229e-3 |
| w4 | 4064/6400 | 1.192e-3 |
| b4 | 25/25 | 3.615e-3 |

This rules out all deep parameters being frozen for this input set, while
identifying nearly stationary attention Q/K projections. It does not distinguish
small mathematical gradients, float32 update rounding, weak attention signal,
or a shared CPU/GPU derivative defect. Finite-difference gradients and an
initial-model control are needed before calling this a bug or changing the
optimizer. The observed weakness is not evidence that new shaping cures it.
