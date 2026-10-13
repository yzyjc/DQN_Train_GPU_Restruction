# 训练轨迹分析器（trajectory analyzer）使用说明

> 重构后的训练轨迹分析报告入口。目标：把「训练到底在往哪走」变成**有口径、可复现、可回溯**的统计输出，
> 而不是一段自然语言结论。

## 1. 文件与入口

| 角色 | 路径 |
|---|---|
| 数据模型与 API | `DQN_Core/include/trajectory_analyzer.h` |
| 实现（只读） | `DQN_Core/src/trajectory_analyzer.cpp` |
| 报告入口（TUI 命令） | `DQN_Main_TUI_No_Terminal/dqn_report.cpp` |
| 回归测试（合成数据） | `tests/trajectory_analyzer_regression.cpp` |
| 真实日志只读验证 | `diag/trajectory_real_validate.cpp`（`diag/` 被 .gitignore 忽略，属本地验证程序） |
| 指标清单（口径来源） | `TRAJECTORY_METRIC_INVENTORY.md` |

## 2. 怎么用

### 2.1 TUI 命令面板（推荐）

`Ctrl+P` 打开命令面板，输入：

```
trajectory
```

可选参数（空格分隔）：

```
trajectory length budget=2000        # 篇幅优先：限制摘要点数
trajectory error tol=0.01            # 误差优先：硬误差阈值（单位=该序列单位）
trajectory top=50                    # 紧凑文本里异常排行榜条数
trajectory nodetail                  # 不写明细文件
```

该命令走**现有后台报告线程**（`report_worker`），界面在分析期间保持响应；
但请注意：**分析耗时可达数十秒**（见第 7 节），期间查看器显示“正在读取日志并生成报告”。

### 2.2 独立运行（离线验证／批量）

```
hipcc -O2 -std=c++20 -fms-compatibility -D_CRT_SECURE_NO_WARNINGS ^
  -IDQN_Core\include diag\trajectory_real_validate.cpp DQN_Core\src\trajectory_analyzer.cpp ^
  -o diag\traj_real.exe
diag\traj_real.exe x64\Release\logs diag\traj_real\
```

### 2.3 运行回归测试

```
hipcc -O1 -std=c++20 -fms-compatibility -D_CRT_SECURE_NO_WARNINGS ^
  -IDQN_Core\include tests\trajectory_analyzer_regression.cpp DQN_Core\src\trajectory_analyzer.cpp ^
  -o diag\traj_test.exe
diag\traj_test.exe
```

当前结果：**PASS=155 / FAIL=0**。

## 3. 输出（同一份结果生成，口径必然一致）

| 输出 | 位置 | 用途 |
|---|---|---|
| 紧凑文本 | 查看器 / `trajectory_report.txt` | 粘贴给 AI、快速浏览（实测 ~0.2 MB） |
| 完整 JSON | `trajectory_report.json`（含 `schema_version`） | 结构化消费（实测 **6.7 MB**；含异常索引+区域+Top20，不内联全部异常） |
| 全部异常 JSONL | `trajectory_anomalies.jsonl` | **全部**异常，紧凑短键 + 触发位图，条数不设上限（实测 0.11 MB） |
| 全部关键点 JSONL | `trajectory_keypoints.jsonl` | 全部关键点（实测 40 MB） |
| 异常明细 CSV | `trajectory_anomalies.csv` | 表格工具消费 |
| 关键点 CSV | `trajectory_keypoints.csv` | 表格工具消费（每序列写入上限 20000） |
| Top X% 切片 CSV | `trajectory_top_slices.csv` | 极端分布与切片统计 |
| 逐记录 JSONL | `trajectory_metric_loss.jsonl` | **默认不生成**，用 `records` 参数开启 |

> **分层设计**：Level 1 = 主 JSON（索引/区域/Top）；Level 2 = `*_anomalies.jsonl` / `*_keypoints.jsonl`
> （全部明细，不设条数上限）；Level 3 = 原始日志本身（按 `episode` / `record_index` 回查）。
> 逐记录 JSONL 默认关闭：其内容与 `metric_loss.bin` 重复，文本体积约为二进制的 6 倍（实测 198 MB vs 100 MB），
> 而 Level 3 本就应该是原始日志。
>
> 实测：原始输入 171 MB → 全部结构化输出 **104 MB**，
> `core` 压缩比 **25.6×**（原始输入 / report.json）、`total` 压缩比 **1.7×**（原始输入 / 全部输出）。

明细目录默认写在**活动时间线目录下**的 `trajectory_report\`。该目录只放报告产物，不含训练数据。

> 三份输出由**同一个 `AnalyzeResult`** 序列化，不存在“两种报告各自计算导致口径不同”。

## 4. 统计口径（重要）

### 4.1 时间线与分支
- 活动时间线由 `active_timeline.txt`（`timeline_v1 0|1`）决定：`0 → logs/pretrain/`，`1 → logs/`。
- 回档分支按 `<name>.rollback_<pid>_<GetTickCount64()>.bak` 的 `pid_tick` **分组识别**
  （后缀第二段是**系统 tick，不是 episode**），按 tick 升序排列。
- **不同分支不会被拼成连续轨迹**；活动时间线单独分析，分支只列出并标注。

### 4.2 样本身份
Episode **不是唯一主键**：一局内有多次梯度更新，同一次更新再按黑/白缓冲分别记录。
样本身份 = `(timeline, episode, step, record_index)`，其中 `record_index` 是文件内原始物理序号。
**同局多条记录属于正常情况，不做去重。**

### 4.3 重复 / 缺口 / 回跳
- 完全重复：仅当流的 `(episode, step)` 应当唯一（schema `per_episode=true`，如 `metric_game.bin`）时，
  相邻且**全字段逐位相同**才计为 `records_duplicate`。
- 其它流的 `step` 是 `replay_type` 一类类别/常量，相邻记录同值属正常，单独计入
  `identical_adjacent_on_nonunique_key`，**不会**被误报为重复。
- 采样周期未记录，因此**不把 Episode 间隔当作缺失记录**；只报告可确认缺口、疑似间隙（EP 回跳）、
  截断尾部与非有限值排除数。
- 读取时**固定本次读取边界**（打开时的文件字节数），未写完的尾记录被忽略并计数，**不修改源文件**。

### 4.4 指标分离
严格分开、不混用：胜率/平率/败率、训练 loss、不同 epsilon 评估模式、对手深度、训练阶段、黑白方。
- `metric_win.bin` / `metric_eval.bin` 的比率字段是**黑白方平均**（每 mode 200 黑 + 200 白 = 400 盘）。
- **分边（黑/白）历史胜率未被记录**：历史流里不存在，只有 `latest_side_win.txt` 的**最新一次**快照。
  报告显式输出 unavailable，不提供分边时间序列/趋势/异常。

### 4.5 差值 D（Q − Target-Q）
- `metric_loss.bin` 的 `loss` / `q_value` / `target_q` 都是**同一被接受批的均值**，
  且 `target_q` **经代码确认为该批样本的 TD 目标**
  （`dqn_agent.cpp:598` 用 `batch_qvals[b][actions[b]] - batch_targets[b]` 作为 Huber 误差项；
  `dqn_agent.cpp:656-662` 对同一批分别求 Q 与 target 均值）。
- 因此 `q_value − target_q` = **该 batch 的平均 TD 误差**（分析器输出 `is_batch_mean_td_error: true`
  并附带上述出处证据）。
- 仍然禁止：(1) 称为「逐样本 TD 误差」——逐样本值未落盘；
  (2) 把来自不同总体（不同批 / 不同 replay_type / 不同时间线）的聚合均值相减。
- 实现上**先逐记录配对算 D，再对 D 聚合**，绝不用两个独立均值相减。

### 4.6 窗口（W5/W10/W20/W50）
**两种口径都输出**，各自声明：
- `records:<宽度>@<起始下标>` —— **样本窗口**（N 条记录）
- `episodes:<宽度>@<起始EP>` —— **EP 窗口**（N 局跨度）

每个窗口给出：真实 EP 跨度、有效样本数、最小有效样本要求（2）、均值、样本标准差（n-1 分母）、
min/max、四分位。不满足最小样本数时 `valid=false` + `null_reason`，**不伪造 0**。

窗口计算做了**有界化**：只对确定性的等距子集计算并输出，报告里给出
`理论窗口数 / 实际计算数 / 等距步长`，不声称覆盖每一个窗口。

### 4.7 差分
- 一阶差分 = 后一条有效记录 − 前一条有效记录（原始单位）。
- 归一化变化率 = 差分 / EP 距离（单位：值/局）。**EP 距离为 0 时不除零**，输出
  `rate_valid=false` + 原因「相邻记录属于同一 Episode」；EP 回跳处也不跨边界归一化。
- 二阶差分在**不均匀采样**下的定义已在输出中声明：它是相邻两个一阶（归一化）变化率之差，
  仅在 EP 距离相等时才有连续时间含义。
- 差分点存储有上限（默认 20000/序列），超出时按等距子集保存并在说明里给出实际总数。

### 4.8 类别指标（Replay Type）
`replay_type`（0=黑方回放缓冲，1=白方回放缓冲）只输出：**频数、比例、转移矩阵、连续区间**，
并显式声明**不做斜率或线性回归**。

### 4.9 异常
- 基线**只用此前有效样本**（滚动先验窗口 512 条），绝不把待检测点混入自己的基线；
  用滚动窗口而非全历史，是因为全历史基线在非平稳序列（如 epsilon 衰减）上会把正常趋势误报成异常。
- 两级筛选：
  1. **固定阈值触发**（方案 A）：`z > z_threshold`（默认 4）且 `|偏差| > 0.5×MAD`；`MAD-z > mad_z_threshold`（默认 6）。
  2. **按分位目标定阈**（方案 B，**默认启用** `keeprate=0.005`）：每条序列只保留分数最高的前 0.5%，
     使「异常」在定义上就是罕见事件。
- 阈值与规则名逐条输出；`std/MAD` 为 0、样本不足、零分母 → 输出 `null` + 原因，**不伪造有限分数**。
- **异常只表示统计规则被触发**，不会称其为「训练崩溃/恢复/成功」。
- 全部异常写入 `trajectory_anomalies.jsonl`（不设条数上限）；主 JSON 只放汇总 + 区域 + Top20。

#### 为什么默认启用分位定阈（实测对比）

活动正式时间线、1090 万条记录：

| 配置 | 保留异常 | 异常率 | 区域数 | critical 区域 | 区域结构 |
|---|---|---|---|---|---|
| `keeprate=0`（仅方案 A） | 192,487 | 1.77% | 105 | 46 | **退化成平铺**：EP 3..2003、2004..4004、4006..6006… |
| `z=8 madz=12` | 86,402 | 0.79% | 180 | 28 | **仍然平铺**：EP 4..2004、2005..4004… |
| `keeprate=1e-3` | 329 | 0.003% | 58 | 2 | 局部化 |
| **`keeprate=5e-3`（默认）** | **1,128** | **0.01%** | **96** | **2** | **局部化**（如 EP 24090..24328、EP 77180..77194） |

结论：**抬高固定阈值只减少每格条数，不改变「几乎每局都有异常」这个本质**；只有分位定阈让区域变成
真正的孤立事件（「这里发生了一场持续 N 局的异常」）。默认值取 `5e-3`，需要旧行为时用 `keeprate=0`。

### 4.10 异常区域聚类
- 合并条件 = **时间连续**（相邻异常 EP 间隔 ≤ `anomaly_region_gap`，默认 200）
  **且区域 EP 跨度 ≤ `anomaly_region_max_span`**（默认 2000，超过强制切分）。
- 区域带 `n / peak_ep / peak_series / peak_score / mean_score / metric_peaks / family_counts /
  dominant_family / span_limited / Top5 代表点 / 前中后均值对比`。
- 指标家族（PERFORMANCE / VALUE_NETWORK / OPTIMIZATION / REWARD / SCHEDULE / SYSTEM）
  作为**区内明细**输出，**不作为切分依据**——按家族硬切会过碎（实测 9.5k → 44k 个区域）。
- severity 按**分数分位**分层（`critical = max_score >= p99`、`major = >= p90`），
  而非绝对阈值；无有限分数的区域一律 minor。
- **实现要点（曾出错）**：聚类前必须按 **episode** 排序。`record_index` 是每个指标文件各自的物理序号，
  跨指标按它排序会交织错乱，使 `nxt_ep - cur_ep` 变负而不触发间隔退出，区域链无限延伸
  （这正是早期出现「EP 5..32k 超级区域」的根因）。

### 4.10 压缩摘要与「短 vs 准」的取舍
- 重建方式：自适应二分（垂距法，RDP）+ 线性重建；误差单位 = 该序列单位。
- 输出：原始点数、保留点数（完整值）、实测**最大误差**、实测 **RMSE**、容差、是否达标。
- 两种模式：
  - **误差优先**（`error`）：容差为硬阈值，允许关键点增多。
  - **篇幅优先**（`length`）：容差由点数预算反解，**误差非硬保证**。
- 报告**不会同时声称**「固定短摘要」与「任意数据上的严格误差保证」；未达标时以实测误差为准并说明。
- `points` 数组有存储上限（默认 20000/序列），但 **max_error/RMSE 始终按全部保留点计算**。
- 「可重建趋势」仅指声明的插值与误差范围，不保证保留原始数据全部信息。

### 4.11 相关与滞后相关
- 先按 EP 聚合（同 EP 取均值）再配对，**只配同一 EP 都存在的位置**；
  **不做前向填充、不用最近邻制造配对、不跨分支对齐**。
- 滞后定义显式给出：`corr(a(EP), b(EP+lag))`，lag 单位为「局」；正 lag 表示 b 滞后 a。
- 同时给出 `r`（带符号）与 `|r|`（绝对值），**不把负数写成 |r|**。
- 同时输出 raw / diff / detrended 三种变换，并注明样本对数与 EP 范围。
  零方差 → `null` + 原因（不给假 0）。

### 4.12 配置事件
- 事件**必须有记录证据**：来自 `metric_eval.bin` 字段变化（phase/depth/eval epsilon），
  或独立日志（`training_fault.log`）。证据字符串逐条输出。
- 缺失历史时输出 unavailable，**不根据相关变化猜测**发生过配置修改。
- 每个事件给出前/后窗口的实际覆盖 EP 范围、样本数、缺失数、均值；倍数仅在分母非零时计算，
  类别变化保留旧值/新值；同时发生的事件互相列出（`overlapping`），**不声称因果**。

## 5. 未记录的信息（诚实边界）

报告在 `unavailable` 段逐条声明，包括：

- **分边（黑/白）历史胜率** —— 只有最新一次快照。
- **逐样本 TD 误差 / TD 目标** —— 只有 batch 均值（D 已确认为 batch 平均 TD 误差）。
- **采样周期 / 评估周期配置** —— 因此不把 EP 间隔算作缺失记录。
- **配置变更完整历史** —— 无专门配置日志，只报告有证据的事件。
- **每次评估的对手模型版本/身份** —— 未记录。
- `metric_steps.bin`（每决策步）未纳入逐记录序列分析以控制开销；
  `metric_skill_v4.bin` / `metric_memory_eval.bin` 使用各自魔数/schema，未展开。

## 6. 已知近似与限制

| 项 | 声明 |
|---|---|
| 分析范围 | **全部可用历史**（流式读取，不沿用 TUI 尾部条数限制） |
| 流式统计量 | n / mean / std / min / max 由 Welford 流式累计，**覆盖全部记录，精确**；分位数基于保留点（见下） |
| 高频率序列点数 | **渐进抽稀**（不是等距、也不是硬上限）：每训练步序列前 300000 条原样保留，之后按 1/stride 抽稀且 stride 随已见条数增长 → **保留点前密后疏**；实测 503 万条的序列保留约 103 万点。每个序列的 `sampling_note` 会给出「retained X of Y」，如实声明 |
| 每局序列点数 | 同上，阈值 1000000 |
| 窗口 | 有界化：每个(序列,宽度)只对等距子集计算并输出，输出 `理论窗口数 / 实际计算数 / 等距步长` |
| 差分点 / 压缩点 | 每序列存储上限 20000，超出按等距子集保存并给出实际总数 |
| 关键点 CSV | 每序列写入上限 20000，超出写明省略数 |
| 异常 | 检测数全部统计；**存储上限 50000**，超出时显式报告 `detected / stored / omitted`（实测 155432 检测 / 50000 存 / 105432 省略） |
| 逐记录 JSONL | 覆盖**被保留**的训练记录（受上面的渐进抽稀影响），**不是全部 503 万条** |
| 紧凑文本 | 按节区展示（Top-N、前后若干项），其余注明省略数量；截断时报告省略字节数与完整结果位置 |
| 浮动 | 比例声明为 `[0,1]`（ratio）或百分数（percent），单位随项输出；变化量用 pp 时单独标注 |

## 7. 实测开销（真实日志只读验证）

数据：`x64\Release\logs`，活动正式时间线，**1,089 万条完整记录 / 176 MB 被读 / 67 个回档分支**，
分析期间训练进程仍在运行（边写边读）。

| 指标 | 数值 |
|---|---|
| Analyze 耗时 | 约 **28 s** |
| 写明细耗时 | 约 **5.6 s** |
| 峰值工作集 | 约 **906 MB** |
| 紧凑文本 | 约 **84 KB** |
| 完整 JSON | 约 **90 MB** |
| 明细合计 | 约 **321 MB**（其中逐记录 JSONL 207 MB） |

优化过程（同一数据）：耗时 62 s → 28 s；峰值内存 3.6 GB → 0.9 GB；JSON 594 MB → 90 MB。

## 8. 测试覆盖

`tests/trajectory_analyzer_regression.cpp` 用**合成数据**覆盖并断言：

常数 / 零 MAD、线性（斜率单位与 R²）、阶跃（位置与幅度）、尖峰（位置=record_index 150、基线不含自身）、
方差突变、已知滞后双序列（`corr(eps(EP), len(EP+1)) ≈ 1`）、不均匀采样与同局多记录（EP 距离 0 不除零）、
缺口/重复/乱序（EP 回跳须报疑似而非拼接）、回档分支枚举（两个 tick → 两条分支且按 tick 排序、声明不拼接）、
NaN/Inf/越界/空目录、小样本（n=1 时 std 必须为 null+原因）、多评估模式（三 mode 独立序列）、
类别指标（频数/比例/转移/连续区间、比例和为 1、声明不做回归）、
压缩误差与预算冲突（误差优先须达标；篇幅优先须如实报误差并说明取舍）、
文件尾部只写了一半（截断字节数精确、只用完整记录、源文件未被修改）、JSON 合法性（禁止 nan/inf 字面量）。

## 9. 接入方式

- 复用既有读取器：`metric_log_reader.h`、`metric_log_format.h`、`metric_eval_reader.h`、
  `training_log_timeline.h`、`training_log_rollback.h`（**未另写一套语义不同的解析**）。
- **不沿用 TUI 图表的尾部条数限制**（`HIST_MAX=25000`、`60`、`200`、`BR_MAX_RECORDS=8192` 等），
  改为流式读取**全部可用历史**。
- 报告缓冲区由 64 KiB 提升到 1 MiB；**截断时显式报告**省略字节数与完整结果位置（原实现是静默截断）。
- 仅修改分析器与展示入口；**未改动训练算法，未启停任何训练进程，未修改任何源日志**。
