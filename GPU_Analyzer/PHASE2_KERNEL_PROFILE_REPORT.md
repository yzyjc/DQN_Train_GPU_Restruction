# PHASE 2 — Kernel Timing 与 Backward 分解报告

## 测试环境

- GPU: AMD Radeon 8060S (gfx1151 / RDNA 3.5)
- 20 CU, 2900 MHz, 800 MHz memory clock, 512-bit bus
- 理论带宽: 102.4 GB/s
- WDDM Dedicated: 31.8 GB, Shared: 76.5 GB

## 实测数据 (训练中, Phase 1, Episode ~33000-34500)

### 吞吐量

| 指标 | 值 |
|------|-----|
| Episodes/s | 0.22-0.62 |
| s/ep | 2.72-4.60 |
| Steps/s | 5.4-7.4 |
| ms/step | 159-223 |
| Forward (avg/step) | 5-8 ms (3-4%) |
| Backward (avg/step) | 152-216 ms (96-97%) |
| Target Sync | 0.00 ms |

### 按方拆分 (Black=0, White=1)

| 方 | Forward | Backward | Steps |
|----|---------|----------|-------|
| Black | 3-5 ms | 82-110 ms | 12-24 |
| White | 3-5 ms | 80-110 ms | 12-24 |

两方耗时基本对称。

### Replay 细分 (per batch)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Sample (replay 采样) | 0.05 ms | 0.1% |
| Forward (GPU batch forward) | 17-19 ms | 20% |
| Target (CPU 目标计算) | 0.01 ms | 0% |
| Train(GPU) (train_agent_batch) | 65-89 ms | 78% |

### Backward 分层 (per call, 1536 calls)

| 层 | 耗时 | 占比 |
|----|------|------|
| FC4 (输出层) | 0.24-0.26 ms | 16% |
| FC3 | 0.13-0.14 ms | 9% |
| FC2 | 0.18-0.19 ms | 12% |
| FC1 | 0.25-0.28 ms | 17% |
| Pool | 0.10-0.11 ms | 7% |
| Attention | 0.47-0.54 ms | 32% |
| Embedding | 0.09-0.10 ms | 6% |
| **合计** | **~1.46 ms/call** | |

### Train Step Overhead (per batch)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Setup(copy) | 0.44-0.45 ms | 0.6% |
| **Memset(6x)** | **17-18 ms** | **24%** |
| Forward | 0.66-0.73 ms | 1% |
| Huber | 0.06-0.07 ms | 0.1% |
| Backward | 47-52 ms | 71% |
| Validate | 0.29-0.31 ms | 0.4% |
| **Total** | **66-71 ms** | |

## 关键发现

### 1. Backward 占 96.4% 的训练步时间

这是最显著的发现。Forward 只占 3-4%。

### 2. Memset(6x) 是最大单项开销

每个样本 6 次 `hipMemset`（清零梯度缓冲区），32 样本 = 192 次 memset 调用，共 17-18 ms，占 train step 的 24%。

### 3. Attention backward 是最贵的层

0.47-0.54 ms/call，占 backward_update 的 32%。这是因为 attention 需要保存 scores 并做 softmax 反向。

### 4. 实际 GPU 工作 vs 开销

```
Train step total: ~68 ms
  backward_update: 32 × 1.46 = 46.7 ms (实际 GPU kernel 工作)
  memset:          17.5 ms (梯度清零)
  setup/forward/validate: ~3 ms
```

### 5. GPU Compute Engine ~44%

GPU 利用率 44% 与训练数据吻合：Forward 3% + Backward 96% 中的 GPU 部分 ≈ 44% 平均利用率（因为有 CPU 等待和 memset 间隙）。

## 可用指标总结

| 指标 | 状态 | 来源 |
|------|------|------|
| GPU Compute Engine | ✅ 可用 | PDH |
| Copy Engine | ✅ 可用 | PDH |
| GPU Memory (HIP/WDDM) | ✅ 可用 | HIP + PDH |
| CPU/RAM | ✅ 可用 | GetSystemTimes |
| Temperature | ✅ 可用 | ADL (gfx1151) |
| Forward/Backward timing | ✅ 可用 | QPC instrumentation |
| Per-side breakdown | ✅ 可用 | QPC instrumentation |
| Replay breakdown | ✅ 可用 | QPC instrumentation |
| Backward per-layer | ✅ 可用 | QPC instrumentation |
| Train step overhead | ✅ 可用 | QPC instrumentation |
| Kernel timing | ✅ 可用 | hipEvent |
| GFX/Memory Clock | ❌ 不可用 | ADL 无此接口 |
| GPU Power | ❌ 不可用 | ADL 不支持 |
| VALU/Occupancy | ❌ 不可用 | 需要 rocprof (Linux) |
| DRAM bandwidth | ❌ 不可用 | 需要 rocprof (Linux) |

## 下一步建议

1. **优化 Memset(6x)**: 合并 6 次 `hipMemset` 为 1 次（连续内存区间），或用一个清零 kernel
2. **减少 hipDeviceSynchronize**: `backward_update` 里每个阶段都 sync，可改为批量末尾一次 sync
3. **Attention 优化**: 0.5ms/call 是最贵层，考虑融合 kernel 减少 launch 开销
4. **MI50 对比**: 用相同 benchmark session 格式在 MI50 上运行，对比 DRAM 带宽和 kernel 时间
