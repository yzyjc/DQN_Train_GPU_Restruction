# 训练轨迹指标清单（指标口径与数据来源）

> 本清单只记录**代码中已核实**的字段语义。每一条都给出写入点或格式定义出处。
> 标注「未记录」的项目表示当前日志中**确实不存在**该历史信息，分析器必须输出 unavailable，
> 不得由相关变化反推。

## 0. 读取入口与时间线

| 项 | 事实 | 出处 |
|---|---|---|
| 日志根目录 | `logs/`（相对 Trainer 工作目录；TUI 为 `x64\Release\logs\`） | `training_log_timeline.h:32` |
| 时间线指针 | `active_timeline.txt`，内容 `timeline_v1 <0\|1>`；0 → `logs/pretrain/`，1 → `logs/` | `training_log_timeline.h:14-27` |
| 时间线数量 | 只有两条：pretrain(0) 与正式(1)。二者都可能从 ep1 开始，**记录不可混算** | `training_log_timeline.h:7-8` |
| 读取器 | 必须复用 `metric_log_reader.h` 的 `br_*` API 与 `metric_log_format.h` 的 schema 表 | `bin_reader.h` → `metric_log_reader.h` |
| 校验器 | 复用 `metric_eval_reader.h` 的 `metric_win_row_valid` / `metric_eval_row_valid`，以及 `Log_Vindicator.cpp` 的 `invalid_value()` 口径 | 同左 |

## 1. 二进制指标日志通用格式

```
Header(24B) = { uint32 magic=0x44514E4C, uint32 field_count, uint32 version(<=1), uint32 reserved[3] }
Record     = { int32 episode, int32 step, float fields[field_count] }
stride     = 8 + 4*field_count
```

- **缺失行占位**：`episode==-1 && step==-1 && 所有 field 为 NaN` → 占位行，跳过。
- **截断**：`(size - 24) % stride != 0` → 尾部半条记录，**忽略并报告**，不修改源文件。
- **字段非法**：整数字段出现 NaN/非整/越界 → 按非法剔除，不参与统计。

## 2. 指标明细

| 文件 | fields | 字段含义（顺序严格） | `episode` 语义 | `step` 语义 | 每局条数 | zero_based | 写入点 |
|---|---|---|---|---|---|---|---|
| `metric_loss.bin` | 3 | `loss`, `q_value`, `target_q` | `chart_ep = 已完成局数+1` | **replay_type**（0=黑方回放缓冲，1=白方回放缓冲） | **多条/局** | 否 | `dqn_agent.cpp:1266` |
| `metric_epsilon.bin` | 1 | `epsilon` | 同上 | **恒为 0**（注意：本文件的 `step` **不是** replay_type） | 多条/局 | 否 | `dqn_agent.cpp:1272-1278` |
| `metric_epsilon_episode.bin` | 1 | `epsilon`（每局一个） | episode | 0 | 1 | 否 | `Trainer:322,336` |
| `metric_rewards.bin` | 8 | `actual[0..6]` 7 项实际奖励分量，`fields[7]` 总分 | episode | 0 | 1 | 否 | `Trainer:335-338` |
| `metric_episode_time.bin` | 1 | 该局墙钟耗时（秒，有限且 ≥0） | episode | 0 | 1 | 否 | `Trainer:361-366` |
| `metric_game.bin` | 1 | **局长（决策步数）** = `step+1` | episode | 0 | 1 | **是** | `dqn_agent.cpp:1213` |
| `metric_steps.bin` | 4 | `player`, `action`, `reward`, `done` | episode | 步序号 | 多步/局 | **是** | `dqn_agent.cpp:1198-1204` |
| `metric_win.bin` | 4 | `mode`, `win%`, `draw%`, `lose%` | 评估 episode | **mode** | **3 条/评估点** | 否 | `Trainer:295` |
| `metric_eval.bin` | 7 | `phase`, `depth`, `epsilon`, `mode`, `win%`, `draw%`, `lose%` | 评估 episode | **mode** | **3 条/评估点** | 否 | `Trainer:296` |
| `metric_memory_eval.bin` | 14 | 记忆权重调度评估 | 评估 episode | — | — | 否 | `memory_weight_scheduler.cpp:119` |
| `metric_skill_v4.bin` | 自有 magic | 技能测试结果 | — | — | — | — | `skill_metric_format.h` |

### 2.1 关键口径澄清（必须遵守）

1. **`metric_loss.bin` 的 `step` 不是训练步序号，而是 `replay_type`**（`metric_log_format.h:26` 注释明确）。
2. **`loss` / `q_value` / `target_q` 三者都是同一 batch 的均值**：写入值为 `avg_loss`、`q_sum/sample_count`、`tq_sum/sample_count`（`dqn_agent.cpp:652-662`）。
   - **已确认 `target_q` 就是相应 TD 目标**：`dqn_agent.cpp:598` 用
     `batch_qvals[b][batch_actions[b]] - batch_targets[b]` 作为 Huber 损失的误差项，
     而 `dqn_agent.cpp:656-662` 对**同一个被接受批** `b = 0..sample_count-1` 分别累加
     `batch_qvals[b][batch_actions[b]]` 与 `batch_targets[b]`。
   - 因此 `mean(Q) − mean(target) == mean(Q − target)`，即 **batch 平均 TD 误差**；
     分析器把该结论与出处证据一并输出（`is_batch_mean_td_error`）。
   - 但仍**不得称为「逐样本 TD 误差」**（逐样本值未落盘），也**不得**把不同总体
     （不同批/replay_type/时间线）的聚合均值相减。
   - 被拒样本（Huber 门 / raw-TQ 门，`dqn_agent.cpp:597-624`）已从该批剔除，
     故 D 只描述**被接受样本**的批均值。
3. **`metric_win.bin` / `metric_eval.bin` 的 win%/draw%/lose% 是黑白方平均**：
   `avg_win = (bw + ww) / 400`（`dqn_agent.cpp:1154-1158`），即每 mode 评估 200 黑 + 200 白共 400 局后取平均。
   - **分边（黑/白）历史胜率未被写入任何历史流**。
4. **replay_type 是类别量**：0=黑方回放缓冲、1=白方回放缓冲（`dqn_agent.cpp:652`），
   只存在于 `metric_loss.bin` 的 `step`。只能统计频数、比例、转移、连续区间，**不做斜率/回归**。
   全项目**不存在 prioritized replay**（`priorit` / `importance_sampling` 在 `DQN_Core` 与 Trainer 均无命中）。
5. `metric_win.bin` 的 `mode` 与 `metric_eval.bin` 的 `mode` 同为 0/1/2 评估模式，二者按 `(episode, mode)` 配对。

## 3. Episode 与训练步的关系

- 一局（episode）内会发生**多次梯度更新**，每次更新写一条 `metric_loss.bin` / `metric_epsilon.bin` 记录。
- 每次更新还会标注其 `replay_type`（黑/白缓冲），因此
  **同一 `(episode, replay_type)` 也可能出现多条记录**。
- 结论：**Episode 不是唯一主键**。样本身份必须用
  `(timeline, episode, step, record_index)` 四元组，其中 `record_index` 是文件内原始顺序号。
- **同局多条记录是正常情况，不得去重。**

## 4. 分组维度

| 维度 | 取值来源 | 说明 |
|---|---|---|
| 时间线 timeline | `active_timeline.txt` + 文件所在目录 | pretrain(0) / formal(1) / 回档分支 |
| 记录序号 record_index | 文件物理顺序 | 保序、不做全局排序 |
| 评估模式 mode | `metric_win/eval.fields[3]`（win 为 `fields[0]`） | 0/1/2 |
| 探索率 epsilon | `metric_eval.fields[2]`、`metric_epsilon*.bin` | [0,1] |
| 对手深度 depth | `metric_eval.fields[1]` | 整数 0..25 |
| 训练阶段 phase | `metric_eval.fields[0]` | 整数 0..5 |
| 黑白方 | **仅 `latest_side_win.txt`（最新快照）** | 历史不可用 |
| Replay Type | `metric_loss/epsilon` 的 `step` | 0=黑缓冲，1=白缓冲 |

## 5. 原始值 vs 聚合值

- **原始（逐记录）**：`metric_loss.bin`、`metric_epsilon.bin`、`metric_epsilon_episode.bin`、`metric_steps.bin`、`metric_rewards.bin`、`metric_episode_time.bin`、`metric_game.bin`。
- **已聚合**：`metric_win.bin` / `metric_eval.bin` 的比率字段已在一局 400 盘上聚合（且已黑白平均）；`metric_loss` 的三字段已在一 batch 上聚合。
- 分析器必须在每个输出项上标注 `raw` 或 `aggregate`，并给出聚合层级。

## 6. 回档分支与重启

- 回档备份命名：`<name>.rollback_<pid>_<GetTickCount64()>.bak`（`training_log_rollback.h:35`）。
  - 第二个数字是**系统 tick 毫秒数，不是 episode**；同一次回档事件的所有文件共享同一后缀。
  - 备份**从不被覆盖**（`CopyFileA(..., TRUE)`，`training_log_rollback.h:105`），因此每个后缀唯一对应一次回档事件。
  - 后缀中的 tick 可用于**事件排序**。
- 回档语义：备份 = 回档发生**前**的完整文件；活动文件 = 被回退（截断）后的续训。
  **二者不能拼成一条连续轨迹。**
- 原子性日志：`rollback.pending`（存在即表示回档中断，`training_log_rollback.h:34`）。
- 触发文件：`rollback_to.txt`（`Trainer:3052`）。
- 重启/故障证据：`training_fault.log` 每行
  `<时间> pid=<pid> reason=<n> component=<c> detail=<d> ep=<ep> step=<step> winerr=<n>`。

## 7. 文本日志

| 文件 | 格式 | 频率 |
|---|---|---|
| `train.dat` | `<global_step> <loss> <q> <tq> <epsilon> <replay_type>`；种群模式前置 `<seed_id>` | 每训练步；`global_step%100==0` flush |
| `game.dat` | `<ep> <step> <player> <action> <reward> <done>`；种群模式前置 `<seed_id>` | 每决策步；`done` 时 flush |
| `reward_episode.dat` | `<ep> <eps> <raw×7> <weights×7> <actual×7> <total>` | 每局 |
| `eval.dat` / `minimax_games.dat` | 评估文本与对局明细 | 每评估点 |
| `latest_side_win.txt` | `side_win_v1 <ep> <黑占mode0..2 %> <白占mode0..2 %>`（百分数） | **仅最新一次** |
| `phase0_skill_test.txt` | 技能测试文本报告 | 阶段 0 退出时一次 |
| `watchdog.log` | 看门狗事件（含阈值配置、重启决策） | 事件驱动 |
| `training_fault.log` | 故障事件 | 事件驱动 |

## 8. 已有 / 缺失的历史信息（诚实边界）

**已记录**
- 每训练步：loss、Q、Target-Q（同 batch 均值）、epsilon、replay_type
- 每局：epsilon、7 项奖励分量与总分、局长、墙钟耗时、逐决策步明细
- 每评估点：phase、depth、epsilon、3 个 mode 的 win/draw/lose（黑白平均）
- 回档事件（备份文件名）、故障事件（training_fault.log）、看门狗事件（watchdog.log）

**未记录（必须输出 unavailable）**
- **分边（黑/白）历史胜率**——历史流里不存在，只有 `latest_side_win.txt` 的最新一次
- **逐样本 TD 误差 / TD 目标**——只有 batch 均值 Q 与 target-Q
- **采样周期/评估周期**——没有周期配置字段；不得把 episode 间隔当成缺失记录
- **配置变更事件的完整历史**——没有专门配置日志；只有能从 epsilon/depth/phase 字段变化、
  `watchdog.log`、`training_fault.log` 取得**有证据**的那些
- **每次评估的对手身份/模型版本**——未记录
- **每局所属的黑白方**——`game.dat`/`metric_steps.bin` 只记 `player`（先行方），不等于训练方归属

## 8.1 补充核实（第二轮调查新增）

- **`params.txt` 由 Watchdog 写**（`Watchdog_NoTerminal.cpp:390-400`，`lr = 0.0004 × 0.5^restart_count`，下限 1e-5），
  仅在文件不存在时写入 → 可作为 **Watchdog 重启次数**的间接证据，**不是**随时间变化的 LR 序列。
- **`train_config.txt`**（`ipc_server.cpp:162-179`，`key = value` 带空格）在本仓库内**只读不写**；
  `dqn_config.txt` 由程序写成 `key=value`（无空格）。两者语法不对称，不要混用。
- **`metric_skill_v4.bin` 会有缺口**：写入门有三重条件（`(ep+1)%1000==0`、64 快照历史完整、多快照评测成功），
  实测仅 153 行而评估点有 204 个 → **必须按行内 `episode` 字段索引，不能按行号线性对应**。
  且行内 **15 个字段是跨快照统计、14 个是单模型结果**，`*_min_margin`/`*_max_margin` 与 `*_fixed_margin`（跨快照均值）
  **不可同列比较**。该文件**不会**出现 NaN（写入门拒绝非有限值）。
- **`metric_memory_eval.bin` 的 `fields[0..9]` 是 NaN**（`0x7FC00000`）是**合法的已落盘占位符**（调度器关闭时），
  与 skill_v4「拒绝 NaN」语义相反，不要混淆。
- **`metric_eval.bin` 的 `phase` 是训练器局部变量**，与 `get_curriculum_phase(ep)` 不一定一致
  （实测末条 `ep=204000` 记 `phase=3`，而课程函数给出 2）→ 当作训练器标签原样呈现，不做推断。
- **`metric_win/eval` 用 `row.episode > 0` 作有效性前提**（`metric_eval_reader.h:21,24`）→ **episode 0 的评估行会被丢弃**。
- **单局胜负未记录**：`minimax_games.dat` 的行是 `episode game_id step turn action 0.000000 done`，**不含 winner**；
  要还原胜负必须重放棋盘。
- **梯度范数未落盘**：只存在于 `neural_probe` 的共享内存 IPC 协议，不在任何日志文件中。
- **checkpoint 文件名与指标口径不一致**：checkpoint 文件名/头内 `episode` 用**阶段内局部** `ep+1`，
  而 `metric_skill_v4.bin` 用**全局** `metric_episode`；跨两者对齐只能用 `metric_episode` / `global_step`。
- **`global_step` 跨时间线连续、局号会重置**：`logs/pretrain/train.dat` 末行 `global_step=4813024`，
  `logs/train.dat` 首行 `4813056`（差 32 = 一个 batch），但正式时间线局号从 0 重新计。

## 9. 采样频率与缺口判定

- 各流采样频率**不同且非固定**：`metric_steps` 每决策步、`metric_loss` 每梯度更新、
  `metric_*_episode` 每局、`metric_win/eval` 每评估点（评估间隔由 `save_interval` 决定）。
- 因此：**Episode 间隔不得直接计为缺失记录**。
- 只有 schema 中 `per_episode == true` 的流，才允许「缺失 EP 即 +1」推断（`metric_log_format.h:22` 注释）。
- `metric_win.bin` / `metric_eval.bin` 是「检查点」流：一次评估产生**升序 mode 0/1/2** 三条，
  推断 EP 时必须借用同一无歧义组内的 EP，**不得猜测周期**（`Log_Vindicator.cpp:53-75`）。
