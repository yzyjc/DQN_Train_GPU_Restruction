# 黑白独立网络：生产接入与隔离验证交付报告

作者：deepseek-flash，2026-09-20。分支 `feature/independent-black-white-networks`。

HANDOFF：`INDEPENDENT_NETWORKS_HANDOFF.md`（作者 GPT6-Astra，基础提交 `ac42b50`）。

## 1. 已接入的生产路径

| 路径 | 文件 | 说明 |
|---|---|---|
| 按方权重所有权 | `DQN_Core/include/independent_networks.h`、`DQN_Core/src/independent_networks.cpp` | 物理棋色 `Side{Black,White}` 是唯一权威定义；每方独占 online/target/按方状态；堆分配、禁拷贝 |
| 按方 Q 路由 | `DQN_Core/src/dqn_agent.cpp` | `replay_from_buffer` 由回放缓冲区显式解析物理棋色，online/target/学习率全部按方取用 |
| 按方提交与按方状态 | `DQN_Core/src/neural_net.cpp` | `train_agent_batch_side()`：批次属于单一棋色；DELTA 门 EMA/预热、提交/拒绝计数、哨兵连续拒绝、学习率回退逐方归属 |
| 按方目标同步 | `DQN_Core/src/dqn_agent.cpp` | `update_target_side()`：只把该方 online 混入该方 target，另一方零改动 |
| 按方 GPU 镜像 | `DQN_Core/src/neural_net.cpp` | `gpu_net_bind_side` / `gpu_net_download_side` / `gpu_net_upload_side` / `gpu_net_snapshot_side` |
| Trainer 接线 | `DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp` | 预训练毕业后播种并启用；每局显式绑定在线方；hard/soft target 按方；快照按方；首步反思按方；LR 衰减下发到两方 |
| 存档 V13 | `DQN_Core/src/checkpoint.cpp` | pool 之后追加双方 online/target/按方状态，同一 CRC；截断/非法第二方整体拒绝；自动发现拒绝坏档 |

**默认行为不变**：未开启按方路由（`agent->indep == nullptr` 或 `active()==0`）时，
所有路径与历史单一 `q_net`/`target_net` 语义逐字节一致。旧测试与未迁移调用方不受影响。

## 2. 折扣与经验语义

见 `EXPERIENCE_SEMANTICS_AND_DISCOUNT.md` 与 `DQN_Core/include/experience_semantics.h`。

结论：当前生产中**不存在需要 `gamma^2` 的位置**——两种经验的 `next_state` 都只比
`state` 远一手。发现一个**未修复的语义缺口**（桥接缺少对手那一手的奖励，且折扣端点
差一手），已在文档中给出闭式差额、实测值与成对修复方案，等批准后再改行为。

## 3. 明确未接入 / 未部署

* **未部署**：未替换任何正式 EXE，未覆盖正式 checkpoint、replay、config。
* **未启动正式训练**，未启动长时间跑。
* 第 5 步（冻结对手与历史池按方来源、评测分色报告、Probe 显式棋色）**未做**：
  `opponent_pool` / `opponent_net` 仍是单一网络，白方仍会用到黑方快照，已在
  `DQN_Train_Trainer_NoTerminal` 的 `select_action_with_net(...,&agent->opponent_net)` 处保持原样。
* 第 6 步（V13 存档）**已做代码**，但**旧档迁移策略只是「显式播种 + 显式记录」**，
  没有做「清空不兼容回放并记录原因」——因为当前 V11/V12 的一步回放语义与 V13 相同，
  没有需要清空的样本。真正需要清空的是上文的语义缺口修复（那会改变回放含义）。

## 4. 测试证据

证据根目录：`build_verify/indep_verify_20260920/run_<时间戳>/<测试名>/output.txt`
（`build_verify/` 在 `.gitignore` 中，未提交）。最终一次全绿运行的摘要如下。

| 测试 | 覆盖 | 结果 |
|---|---|---|
| `independent_networks_regression` | 基础容器隔离（GPT6-Astra） | PASS |
| `independent_side_isolation_regression` | **真实 `replay_from_buffer` + `train_agent_batch_side`，黑/白双向逐字节隔离** | PASS |
| `experience_semantics_regression` | 折扣指数、己方/对手终局、白和/黑和、已知缺口手算 | PASS |
| `independent_checkpoint_v13_regression` | V13 精确恢复、坏档整体拒绝、自动发现、V12 迁移 | PASS |
| `anchor_regression` | 记忆锚点 | PASS (0 failures) |
| `sentinel_regression` | 哨兵保护 | PASS (0 failures) |
| `training_gate_regression` | DELTA 提交门 | PASS (0 failures) |
| `checkpoint_memory_regression` | checkpoint + `.mem` 同代与中断恢复 | PASS |

隔离测试实测读数（`independent_side_isolation_regression/output.txt`）：

```
[train-black] trained-side online changed=1; parameter L2 delta=2.4439506e-06;
              changed params=70286; max|db4|=1.70810381e-06
[train-black] other side (white): online_same=1 target_same=1 state_same=1
              parameter L2 delta=0/0 target L2 delta=0/0
              (committed=0 delta_ema=0 warmup=2 rejected=0 target_sync=0 lr=0)
[train-white] trained-side online changed=1; parameter L2 delta=2.72480532e-06;
              changed params=93329; max|db4|=1.62635115e-06
[train-white] other side (black): online_same=1 target_same=1 state_same=1
              parameter L2 delta=0/0 target L2 delta=0/0
              (committed=1 delta_ema=2.4439506e-06 warmup=1 rejected=0 target_sync=0 lr=9.99999975e-05)
```

即：被训练方有 7 万+ 个可学习参数变化、L2 增量非零；另一方的 online、target 与
全部按方状态字段逐字节零变化。**没有比较 loss、缓存或指针地址。**

V13 负向用例实测（`independent_checkpoint_v13_regression/output.txt`）：

```
[load]      V13 exact restore: black/white online+target+state byte-identical
[corrupt]   V13 独立网络权重非法（截断/非有限）；整体拒绝加载 -> load_checkpoint=0
[truncated] V13 checkpoint 文件长度异常 -> load_checkpoint=0; indep_active=0
[discovery] find_latest_checkpoint 选中 5555，跳过被截断的 6666
[migration] V12 loaded -> 显式 migrate-seed；激活后再播种被拒绝
```

**GPU 验证状态：③ 已补齐 HANDOFF:84 两项，记为通过**（deepseek-flash 提交 `383701f`）

原「部分通过」是因两项未做，现已由 `tests/gpu_side_multibatch_parity_regression.cpp`
补齐，原始证据 `build_verify/indep_flash_run/gpu_side_multibatch_parity_regression/output.txt`
（exit 0）：

1. **按方路径 CPU/GPU 一致性**（同一批 32 条真实两 ply 样本、同一初始权重，走
   `train_agent_batch_side` 入口）：
   `[A cpu] rc=1 loss=0.0164441559`、`[A gpu] rc=1 loss=0.0164441559`、
   `|dCPU-dGPU| L2=2.32830671e-10 relative=1.01363795e-11 changed=2 |loss gap|=0`
   （判据 `relative<0.005 && gap<1e-5`，实测远优于阈值；未训练的一方在两套后端下
   逐字节相同）。
2. **每方向连续 3 个批次**（设备每批都前进，另一方全程冻结 online/target/按方状态，
   且每批都断言「设备 online == 绑定方影子」）：
   * 黑方三批 DEVICE delta `2.67077276e-06/64842`、`3.41443983e-06/91365`、`2.48146795e-06/70500`
   * 白方三批 DEVICE delta `4.74582809e-06/115190`、`3.3683716e-06/112054`、`2.12849769e-06/60821`
   * 全程 `training_fault=0`

注意实现顺序：CPU 那一半必须在 `gpu_init()` **之前**跑，因为 `gpu_net_active()`
是进程级标志，否则 CPU agent 也会被路由到 GPU。

**其余 GPU 按方隔离证据**（`independent_gpu_side_isolation_regression/output.txt`，exit 0）：
L1 设备==绑定方影子（online 与 target 都 0/0）、L2 另一方影子/target/按方状态零变化、
L3 切回后设备回读等于对方训练前的值。真实设备数字：黑方训练使设备黑 online
变化 `L2=2.4439506e-06/70286` 参数；白方训练 `L2=2.73110523e-06/93658`。

环境问题**已解决**，不再是阻断项：

* 根因是 Windows SafeDllSearchMode 顺序（exe 目录 → SYSTEM32 → **CWD**），
  不是 DLL 版本或代码问题。修法是让测试 EXE 与正确的 `amdhip64_7.dll` 同目录
  （`tests/run_isolated_test.ps1`），并在日志里落盘指纹。
* 已用探针**回读模块路径**自证，正负两侧都有日志：
  * 负面对照 `build_verify/indep_gpu_control/gpu_runtime_load_probe/output.txt`：
    `[MODULE] amdhip64_7.dll path=C:\Windows\SYSTEM32\amdhip64_7.dll` → `0xC0000005`；
  * 正例 `build_verify/indep_gpu_load_20260920/A_path_only/output2.txt`：
    `path=...\A_path_only\amdhip64_7.dll`（20,687,360 B，sha256 `2ADFD520…`）→
    `probe finished without crash`。
* 已解锁的既有 GPU 测试（`build_verify/indep_gpu_run_20260920/<测试>/output.txt`，
  exit 0）：`training_regression`、`recheck_regression`、`step_outcome_regression`、
  `recheck2_regression`、`audit_regression`、`train_fix_regression`。

**已取得的 GPU 按方隔离证据**（`independent_gpu_side_isolation_regression/output.txt`，exit 0）：
L1 设备==绑定方影子（online 与 target 都 0/0）、L2 另一方影子/target/按方状态零变化、
L3 切回后设备回读等于对方训练前的值。真实设备数字：黑方训练使设备黑 online
变化 `L2=2.4439506e-06/70286` 参数；白方训练 `L2=2.73110523e-06/93658`。

**原「仍缺的两项」已闭合**（`383701f`）：

1. HANDOFF:84「每方向 **≥2 个连续批次**」→ 现每方向 3 个连续批次，见上文 B 组数字。
2. HANDOFF:84「**按方路径** CPU/GPU 一致性」→ 现走 `train_agent_batch_side`
   真实入口测得 `relative=1.01e-11`、`loss gap=0`，见上文 A 组数字。

构建产物（均在本机通过）：`DQN_Core.lib`、`DQN_Train_Trainer_NoTerminal.exe`、
`DQN_Main_TUI_No_Terminal.exe`、`Watchdog_NoTerminal.exe`、`Log_Vindicator.exe`
→ `build_verify/indep_base_20260920/`。正式 `x64\Release/` 未被覆盖。

## 4b. 真实 Trainer 隔离短训与重启恢复（第 ⑤ 步，腿 B）

方案 (b) 的批次循环修复（`8c2659d`）之前，每步同时回放黑白两缓冲而 GPU 只绑一方，
两边都攒够 `BATCH_SIZE` 时必然为未绑定方提交一批并闩死
（原始证据 `build_verify/indep_trainer_short/run3/`：`metric_loss.bin` 只有头、
`training_fault.log = component=indep_side_not_bound`）。修复后：

**运行段**（`build_verify/indep_trainer_short/run5/`，垫档 = 新布局 V13 `checkpoint_phase1_4242.bin`，
720 秒后强杀）：

* `logs/training_fault.log` = **空**（不再闩死）
* `logs/metric_loss.bin` = 142,544 B（真的在训练；`replay_type` 0/1 分方）
* `logs/active_timeline.txt` = `timeline_v1 1`（已在正式阶段）
* **切换开销与吞吐**（`logs/indep_switch_cost.log`，区间速率而非生命周期均值）：
  ```
  ep=4400 binds=1450 avg_ms=2.275810 interval_episodes=100 interval_s=98.7 eps_per_s=1.013
  ep=4700 binds=4212 avg_ms=2.239704 interval_episodes=100 interval_s=92.7 eps_per_s=1.079
  ep=5000 binds=7012 avg_ms=2.252224 interval_episodes=100 interval_s=92.5 eps_per_s=1.082
  ```
  即 **≈2.24–2.33 ms/bind**（一次 2×3.4MB 设备↔影子往返），稳态 **≈1.01–1.12 局/秒**。
* **按方提交/拒绝数**（`logs/indep_side_state.log`，两方分别给出）：
  ```
  ep=5000 black_committed=3517 black_rejected=1 black_rejected_delta=0 black_rejected_loss=0
          black_delta_ema=1.16606344e-05 black_lr=0.0003
          white_committed=3528 white_rejected=2 white_rejected_delta=0 white_rejected_loss=0
          white_delta_ema=1.71011962e-05 white_lr=0.0004
  ```
* Trainer 自己写出 `checkpoints/checkpoint_phase1_5000.bin`（**36,010,292 B = 新布局 V13**）
  + 同代 `.mem` + `dqn_model_best_phase1.bin`；`logs/checkpoint_recovery.log` 为空。

**重启段**（同目录再启动，150 秒）：载入 V13 后按方计数**接着涨而不是归零**：
```
ep=5000(上一段) black_committed=3517 white_committed=3528
ep=5100(重启后) black_committed=3992 white_committed=4003
```
`logs/training_fault.log` 仍为空，`logs/replay_semantics_migration.log` 为空
（V13 精确恢复路径没有误清回放）。切换开销 2.283 ms/bind。
注意：绑定计数与耗时是**进程内累计**，重启后从 0 重新计数，这是预期行为。

## 4c. 存档覆写风险排查（三个不同问题）

| # | 问题 | 判定 | 证据类型 |
|---|---|---|---|
| 1 | 正常同名保存覆盖健康档 | **不会留半档**：写 `<file>.tmp` → `_commit` → `MoveFileExA(REPLACE_EXISTING\|WRITE_THROUGH)`；`.mem` 先备份为 `.mem.bak`，任一步失败即回滚并保留旧恢复点 | **运行实验**：`tests/checkpoint_save_safety_regression.cpp` case1（覆盖后可重新加载、无 `.tmp`/`.mem.bak` 残留） |
| 2 | 故障退出时的保存行为 | **不会把污染状态写成新档**：训练循环条件含 `!g_training_fault`，闩死后不再进入训练；存档前 `gpu_net_download_agent` 失败即拒绝写档 | **运行实验**：case2a/2b/2c（目录不存在 / 回放缓冲缺失 / 回放计数非法 ⇒ `save_checkpoint` 返回 0，已有健康档逐字节不变，无残留） |
| 3 | 未同步或部分更新状态覆盖健康档 | **判定不可达**（见下），因此不再挂为缺口 | **代码级论证**，不是注入实验 |

第 3 条的论证链（每一步都可核验）：

1. `gpu_net_bind_side` 只要 `result != 0` 就置 `g_gpu_untrusted = 1`（`neural_net.cpp:406`）；
2. `g_gpu_untrusted` 全仓库**只在 `gpu_net_activate` 里被清零**（`neural_net.cpp:224`）；
3. 因此任何一次切换/下载/上传失败之后，`gpu_net_download_agent` → `download_side`
   （要求 `!g_gpu_untrusted`）必然返回非 0；
4. 而 `save_checkpoint` 的第一道门就是它（`checkpoint.cpp:487`），失败即打印
   `[SAVE] GPU state is unavailable/untrusted; checkpoint not committed.` 并返回 0
   —— **旧的健康档根本不会被替换**；
5. 非失败情形下也不存在代次不一致：一次**成功**切换的顺序是
   「下载旧方影子 → `set_bound` → 上传新方 online+target」，所以每次成功切换后
   两方影子都是当代；绑定方在绑定期间产生的更新由存档前的 `gpu_net_download_agent` 补齐。
   两边都当代 ⇒ 没有混合代次。

**证据类型声明（不混淆）**：第 1、2 条是**运行实验**；第 3 条是**代码级论证**，
**没有做注入实验**。要注入需要给 `neural_net.cpp` 加一个 test-only 的
`g_gpu_untrusted` setter；本仓库刻意不在生产接口上加这种可被误用的开关。
另注：`gpu_net_bind_side(agent, 7)` 会在 `side_from_physical` 抛异常时提前返回，
**不会**置位，所以它不能当注入手段。

## 5. 隔离短训启动方法

1. 用 `build_verify/indep_base_20260920/` 里的 `DQN_Train_Trainer_NoTerminal.exe`
   （源码 hash 见报告第 2 节，或直接 `git rev-parse HEAD`）。
2. **必须**在全新空目录里运行，只放 EXE + `amdhip64_7.dll`（取自 `x64\Release\`，
   20687360 B），不要指向正式训练目录。
3. 启动后 Trainer 会在预训练结束后打印：
   `[INDEP] 黑白独立网络已接入：由共享预训练权重播种两方；...`
   看到这一行才说明按方路由真的生效。
4. 本轮**没有实际执行**这条短训，因此不提供「已跑通」的结论。

## 6. 诚实边界

* 不声称训练效果提升。
* 不声称 Freeze-1 历史根因已闭合。
* 未执行的测试一律标注为未完成，不写成通过。

## 7. 工作区事故与恢复记录（必须知悉）

本次过程中我犯了两个工作区操作错误，均已完全恢复，但必须记录：

1. **`git stash push -u` 移除了受保护的三项未跟踪内容。**
   我用它临时对照基线，`-u` 把 `PROJECT_OVERVIEW.md`、`ROCm_DEV_GUIDE.md`、
   `正式训练记录/` 一并收进 stash 并从工作区删除（当时只看到 `git status`
   仍列出它们，那是 stash 部分失败回滚造成的假象）。
   **恢复方式**：stash 对象 `ffd14aeb9e6eeb95b964146459387b8827b813ba`
   仍然完整存在，用 `git restore --source=<该对象> --worktree -- <路径>`
   按原始 blob 恢复；`PROJECT_OVERVIEW.md` 与 `ROCm_DEV_GUIDE.md` 的
   `git hash-object` 与 stash 内 blob 逐字节一致。
   `正式训练记录/` 恢复 16 个文件、30,293,225 字节。

2. **一次误执行的 `git restore --source=<stash> --worktree -- .` 删除了全部已跟踪文件。**
   该 stash 的 index 树停在旧的 `ac42b50` 快照，因此 restore 把 125 个已跟踪文件
   从工作区删掉。**内容从未丢失**（我的两个提交与所有 blob 都在对象库中）。
   **恢复方式**：`git restore --source=HEAD --worktree -- .`（从我的
   `efd4555` 恢复全部已跟踪文件），再用 `git restore --staged` 把三项未跟踪内容
   放回未跟踪状态。恢复后 `git status` 只剩那三项 `??`，已跟踪文件零差异，
   三个新测试重新执行全部 PASS。

**未执行**任何 `reset --hard`、`git clean`、历史改写或强推；`ac42b50` 及其之前
的所有提交签名与内容未被我改动。上述两次操作都是**工作区**层面的，
`git log` 只有我新增的两个 `deepseek-flash` 提交。
