# 主程序架构与修改约束

本页描述当前 NoTerminal 主链，不把历史终端版、测试夹具和备份当作同一个入口。
当前完整构建/回归入口为 `tests/run_regressions.ps1`，目标是 Release/x64。

## 模块职责

```text
DQN_Core
  网络/训练更新/棋盘/检查点/IPC 服务
  include/metric_log_format.h      磁盘布局、版本、字段数、episode 基准
  include/metric_log_reader.h      只读：扫描、增量、尾部、按局读取
  include/metric_log_writer.h      单写者：打开、头部创建、写前尾部修复
  include/metric_eval_reader.h     胜率/评估记录校验
  include/training_log_timeline.h  预训练与正式训练目录选择
  include/training_log_rollback.h  恢复存档前的日志回滚事务
      ↑               ↑                      ↑
Trainer               TUI                     Log_Vindicator
课程/评估调度          图表/分析/回放/指令       检查与离线修复
训练状态的所有者       观察者，不修改训练日志    不决定训练状态

Watchdog：进程保活/训练退化保护；必须尊重主动关闭标记和保存失败状态。
```

日志公用层位于 Core/include，但使用它不要求链接 GPU 或启动训练。
TUI 的 `bin_reader.h` 仅是兼容转发头，不能再往里面添加独立解析实现。
`dqn_report.cpp` 和 Dashboard 共用评估校验；分析报告可以按局筛选，
Dashboard 恢复最近 60 条记录（正常情况下为 20 个三模式检查点）。
多个进程之间通过 IPC 和文件协议通信，不能借用另一个程序的可变全局状态。

## 日志契约

普通 metric 文件采用小端 24 字节头、8 字节整数前缀和 N 个 32 位浮点数。
头部为 `magic, field_count, version, reserved[3]`。版本 0、1 可读；
未知版本不能按现有记录宽度截断或续写。布局由静态断言约束，未改磁盘格式。

| 文件 | 磁盘 episode | 前缀第二整数及数据含义 |
| --- | --- | --- |
| metric_steps.bin | 0-based | 本局 step；player/action/reward/done |
| metric_game.bin | 0-based | 单局汇总，曲线显示时 EP 加 1 |
| metric_loss.bin | 1-based | 第二整数是 replay_type，不是训练总步数；loss/Q/target-Q |
| metric_epsilon.bin | 1-based | 更新时记录 epsilon，同一 EP 可有多条 |
| metric_epsilon_episode.bin | 1-based | 每局 epsilon |
| metric_episode_time.bin | 1-based | 每局墙钟耗时，单位秒 |
| metric_rewards.bin | 1-based | 七项加权奖励及其合计，不等同于完整环境回报 |
| metric_win.bin | 1-based | mode/win%/draw%/lose%；三个 mode 共用一个 EP |
| metric_eval.bin | 1-based | phase/depth/epsilon/mode/win%/draw%/lose% |

新增普通 metric 时先改 `METRIC_LOG_SCHEMAS`，再接入写入、图表映射和测试；
维护器和回滚层直接复用表中的字段数和 EP 基准，避免各处自行猜测。
skill-v2/v3 采用独立 magic 和语义版本，由 `skill_metric_format.h` 定义、
`skill_metric_reader.h` 校验、`skill_metric_writer.h` 写入。
当前写 `metric_skill_v3.bin`，不将 v3 分数追加进 v2 文件。写者持有排他写句柄，
只截断已知格式的残缺记录尾；头部损坏/版本未知时拒绝写入并保留原文件。
不能加入普通 metric 表后当成普通文件修复。

读端只处理完整记录：`header + complete_count * stride` 是完整尾部，
不能用 `EOF - count * stride` 定位，因为 EOF 可能位于尚未写完的一条记录中。
`br_read_tail_checked` 的 -1 表示不可用/错误，0 表示有效空文件；界面应区分二者。
其他旧接口仍将不可用归为 0，新的需要保留快照的代码应使用 checked 接口。
NaN/-1 是维护器占位符，不是观测值；浮点数转 mode/phase 等整数前先校验。
读取器的偏移和记录总数为 64 位，旧 `br_read_header(int*)` 数量输出饱和到 INT_MAX。

## 所有权和生命周期

- 训练检查点选择写入时间线：Phase0 写 `logs/pretrain/`，正式训练写 `logs/`。
  `active_timeline.txt` 仅通知观察者，不是选择恢复模型的依据。
- `metric_log_prepare_append/open_append` 只允许训练写者在打开句柄前调用。
  已知格式可去掉未完成尾部；错误布局隔离保留；未知版本拒绝写入并保留原名原内容。
- TUI/分析只读。活动日志读取失败不能触发“自动修复”或删除。
- Vindicator 的 daemon 模式在训练存在时只检查结构；离线才修复。目录不可用是可重试诊断，
  不能让后台循环因为文件系统异常直接退出。
- 恢复检查点先暂存/验证模型，日志回滚先准备所有临时文件与备份再提交，
  成功后才打开写入句柄。不要把磁盘修改混进只读校验函数。
- 主动退出应走既有保存状态机：同步 GPU、确认存档成功后退出；失败则暂停等待重试。
  SIGINT/SIGTERM 处理器只写退出标志；训练线程在安全边界发布主动退出标记，
  禁止从信号上下文直接保存配置、刷新日志或遍历记忆缓存。

## 训练与考核边界（2026-09-01）

- `action_policy.h` 是实战和考核共同的 ε=0 合法动作/近似并列抽样策略。
  技能 v3 使用独立固定 PRNG：空盘 64 次，强制首手 25 个局面单列。
  不消耗训练/对局评估 PRNG，也不再用固定下标打破并列。
- checkpoint 磁盘结构仍为 V7；`SkillTracker.initialized` 的 0=无基线、
  1=旧严格 argmax 基线、3=v3 基线。旧基线在下次考核重测，不能混用两种口径。
  新程序读旧档；旧程序可能拒绝含 v3 基线的新档，不应混用不同轮次的 EXE。
- Core 评估返回 COMPLETE/CANCELLED/FAULT；输出数组只在完整成功时提交。
  Trainer 通过取消回调泵送消息并处理退出；不可在取消后发布胜率/升级难度。
  每次推理/搜索前后检查取消，不承诺中断正在执行的单次 GPU 调用或 minimax 搜索。
- `load_checkpoint_model` 只提取验证过的权重，候选在独立 CPU agent 上复验。
  未通过不得加载进 live agent，更不能恢复其 LR/局号/回放。通过后也只继承权重。
- 正式任务起点保留权重、重置 replay/首步反思/target/对手池，再保存 Phase1_0。
  恢复旧 Phase1_0 同样执行；其他旧正式档由 `training_task_boundary.h` 删除
  同一方连续行动的单人样本，并保持正式经验的时间顺序。Phase0 续训不做此清理。
- 评估日志和 latest_eval 使用 evaluated_depth，checkpoint 使用下一轮训练深度。
  换难度时清空 EMA/前次表现/最佳胜率，下一轮不与旧难度成绩作直接比较。

## 后续拆分边界

Trainer、TUI 入口仍偏大；本轮只抽取重复且已出现分歧的日志层，未重写所有入口。
后续可分别抽取训练会话生命周期、图表数据源、IPC 命令路由。拆分必须保持现有
回归覆盖，并逐步减少测试通过重命名 main 后包含整个 `.cpp` 的耦合。
历史终端版仍在解决方案中，本轮没有把它们迁移到 NoTerminal 的完整生命周期。
不要把“共用 Core”理解成所有历史入口已经同等验证。

测试不替代长时间 UI/IPC 压测或模型收敛实验；本次未调整奖励、更新频率或毕业阈值，
但明确隔离了预训练与正式训练的经验任务。
