# 黑白独立网络：基础交付与后续实施提示词

作者：GPT6-Astra，2026-09-20。

## 已完成的基础与明确边界

- 分支：`feature/independent-black-white-networks`，基于 `ead9171`。
- `DQN_Core/include/independent_networks.h`：黑白分别持有 online / target 网络，堆分配、禁止浅复制；各自拥有更新范数 EMA、预热计数、提交/拒绝计数和目标同步计数的基础结构。
- 构造函数可以从原共享 online / target 权重初始化双方；只迁移权重，尚未处理回放与训练语义迁移。
- `tests/independent_networks_regression.cpp`：对双方分别执行真实 CPU SGD 更新，检查当前方输出层参数变化、当前方 target 与另一方 online / target / 更新状态不变。
- **尚未接入 DQNAgent、Trainer、GPU、存档、评测、哨兵或 UI。正式程序仍共享权重。** 状态结构仅为后续接线提供归属，不代表保护门已经隔离。
- 当前实现是 SGD，无 Adam 动量或二阶矩；不要凭空新增优化器算法。未来若引入优化器状态，必须按方保存。
- 未改正式目录程序、配置、checkpoint、replay，未启动训练。
- 基础验证已通过：`build_verify/astra_white_draw_20260919/run_20260920_024903_482/independent_networks_regression/output.txt`。使用已有静态库是因为本轮仅新增头文件容器与测试，未修改任何生产实现。

## 以下内容可直接发给实施 agent

请在 `E:\source\repos\DQN_Train_GPU_Restruction` 完成黑白独立网络训练架构。先阅读本文件、`STEP_OUTCOME_SHAPING.md`、基础头文件和测试，然后以当前源码为准核实，不要把历史报告中的推测当作根因结论。

### 用户授权与目标

用户已授权黑白各自独立 online 网络、target 网络及训练状态，保留黑方进攻、白方防守的非对称奖励。用户也已授权扩展终局归因。目的：一方梯度更新不得修改另一方参数。不要改成仅独立输出头；不要为恢复零和而删除白和奖励。无需反复请求实施许可。

仅完成代码、隔离构建与测试；正式部署和长训另行报告，不自动覆盖当前正式版本或启动长跑。

### Git 与现场保护

1. 先执行 `git status`、`git branch --show-current`、`git log -5 --oneline`，检查现有基础提交。当前工作分支为 `feature/independent-black-white-networks`，可继续使用。
2. 保留不是你产生的修改和未跟踪文件，特别是 `PROJECT_OVERVIEW.md`、`ROCm_DEV_GUIDE.md`、`正式训练记录/`。不要 reset --hard、clean、改写旧提交、强推或清理训练数据。
3. 独立问题单独提交，明确 add 文件。提交前检查完整 diff、staged diff、相关测试；按实际 agent 身份署名，基础作者是 GPT6-Astra，不要冒用。
4. 不提交编译输出、diag、日志、checkpoint/replay、归档。新诊断在隔离目录运行；检查是否有正式 Trainer，避免命名管道、共享内存和路径干扰。

### 当前真实实现与切入位置

- `DQN_Core/include/core.h`：DQNAgent 当前只有一套 q_net / target_net，另有 opponent_net 和历史 opponent_pool。黑白回放分开但训练参数共享。不要误把 target 或 opponent 当作白方在线网络。
- `DQN_Core/src/neural_net.cpp`：实际生产 GPU facade 在此，使用静态 g_gpu_q / g_gpu_target、API 锁、CPU 参数打包缓冲。`gpu_training_session.cpp` 是另一层旧抽象，不能只改那里就声称生产生效。
- 同文件 `train_agent_batch`：安全更新、范数 EMA、哨兵、拒绝回滚及 LR 回退。当前 EMA/预热等有进程全局状态；需要明确各方归属，防止黑方统计限制白方更新。
- `DQN_Core/src/dqn_agent.cpp`：select_action / select_action_with_net、replay_from_buffer、目标计算、初始化、释放、target 更新、opponent pool。
- `DQN_Core/include/terminal_outcome_credit.h`：当前只把对手终局回应桥接到上一步；普通非终局仍为一步经验，训练仍取负对手 Q。
- `DQN_Train_Trainer_NoTerminal/DQN_Train_Trainer_NoTerminal.cpp`：play_episode、pretrain、在线方轮换、冻结对手、历史池、phase 切换、加载热换、target 同步、GPU 启停、评测、Probe 发布。
- `DQN_Core/src/checkpoint.cpp`：当前最高 V12。版本、结构长度、CRC、自动发现、model-only 加载、pool 偏移、暂存加载和 `.mem` 同代恢复必须一起检查。不能只改保存版本。
- 在整个仓库检索所有直接访问 q_net / target_net / opponent_net 的地方，包含战术测试、minimax 评测、Probe、TUI 对弈、导出和故障恢复；不要仅替换 Trainer 的选招。

### 实施顺序

**第一步：明确角色和所有权。**

接入基础容器或等价设计，提供显式物理棋色访问接口。黑白物理角色不能与“本局在线方/冻结对手方”混淆。维持现有预训练兼容路径；明确何时由共享预训练权重初始化两个独立网络，后续不能每局或每次恢复重新复制。

**第二步：先解决各方自己的目标与经验语义。**

建议正式对局改为连接“本方动作 → 对手真实回应 → 本方下一行动状态”，使用本方 online 选动作、本方 target 估值。一步己方终局与两步对手终局都停止 bootstrap，并保留该方结果奖励。

必须先写清楚折扣约定：若 gamma 仍表示每个落子的折扣，两步后 bootstrap 使用 gamma²，对手回应时才发生的终局奖励也应处理其时间距离；若采用每次本方决策折扣，应明确这是一项语义变更并统一测试。不得无说明地沿用两步经验 + 单步折扣，也不得机械混入对手自身收益。

普通己方非终局 shaping 是否保留、如何与延迟终局奖励组合，要遵循现有即时奖励语义并写出公式。每个实际动作只入库一次，无重复矛盾标签；截断/取消/异常不能伪造终局。验证两步状态变化恰为合法双方各一手，轮次正确，中间没有已结束比赛。

旧一步回放不能直接当成新两步经验。旧档迁移优先保留双方初始化权重、显式清空不兼容回放并记录原因；旧 opponent pool / 记忆 / 计数分别制定迁移规则，不默认静默丢弃。新档恢复必须保留新回放。Phase0 单人课程与战术锚点需专门处理：不可把非终局一步锚点错误当成两步样本，禁止无故删除防守训练信号。

**第三步：CPU 路由及按方状态。**

每个批次必须属于同一物理棋色；锚点混入也要符合该棋色。黑批更新黑 online，白批更新白 online。双方 target 更新、门控 EMA、预热、拒绝记录和必要学习率回退状态按方归属。训练故障导致整个程序停止可以共用，但不可混淆为共享模型健康统计。已启用的颜色保持旋转增强继续保留。

**第四步：GPU 四网络驻留。**

分别维护黑白 online / target 句柄，训练和前向必须显式选方；同步、快照、上传、下载、回滚都标明对象。不使用“临时交换 CPU q_net 字段”的隐式路由，避免 Probe/评测和 GPU 句柄串线。失败必须可见，禁止拿 CPU 回退结果冒充 GPU 验证。只读另一方推理不应改变其参数。

检查目标同步只更新指定方；保存时取得同一逻辑时点的两方完整状态。完整设备错误时保留现有停止策略，不临时启用未经验证的自动健康回滚。

**第五步：冻结对手、历史池与评测。**

当前训练会轮换在线方并使用 minimax / 历史池 / 冻结对手。必须定义双方对应的冻结网络和历史池来源，不能白方继续使用黑方网络快照而未说明。评测分别调用黑白模型，报告分色胜负和 Win-in-1 / Block-in-1；Probe 若暂只展示一方，明确显示棋色，不能误标为整个系统。

**第六步：新存档。**

为新增状态分配明确格式版本（预期 V13，但先查现分支是否被其他人占用）。同一 CRC 保护两套模型、目标与按方训练状态及兼容回放；不要直接 fwrite 含 unique_ptr 的容器。更新所有长度计算、发现校验、加载路径和测试断言。

旧 V11/V12 作为迁移初始化要显式记录；新格式必须精确恢复。`.mem` 同代 CRC、备份中断恢复和失败保存保护不能退化。截断第二方模型或状态时加载整体失败，不得半加载或静默复制黑方补白方。

### 必须完成的测试

1. 基础测试：`independent_networks_regression`。它仅证明容器 + CPU 单步隔离，不足以替代以下生产测试。
2. 真正 replay_from_buffer 和 train_agent_batch：只训练黑批，黑在线参数确实变化，白 online/target/按方状态逐字节或逐字段不变；反向同样成立。不得仅比较 loss、缓存或指针地址。
3. GPU batch=32、多动作、多连续批次，CPU/GPU loss 和参数更新比较；检查每项有限性、参数 delta 的 L2 相对误差、实际后端与返回码。双方各自的设备快照证明未被另一方更新。单样本 PASS 不算批量验证。
4. 同方两步目标：手算可控例子核对在线选动作/target估值、折扣指数、合法动作屏蔽、己方终局、对手终局、白和 +0.05、黑和0，无终局 bootstrap。
5. 旧档迁移和新档保存恢复：两方权重、target、训练状态、回放、随机状态一致；第二方损坏/截断整体拒绝；自动发现新档不误隔离；checkpoint+.mem 中断恢复回归通过。
6. 实际 Trainer 隔离短训，覆盖黑白在线方轮换、phase0→正式阶段、target hard/soft 更新、停止重启、评测和状态发布。记录各方提交/拒绝数；不能只证明总提交数非零。
7. 保留原步长奖励边界、TUI 配置、存档记忆与相关训练保护回归。所有失败先解释修复，不把未执行写成通过。

### 构建与已有证据

基础测试可复用已验证的当前共享版静态库：

```powershell
& tests/run_recheck2_regressions.ps1 -SkipBuild -Tests independent_networks_regression -OutputDirectory build_verify/astra_white_draw_20260919
```

开始修改生产 .cpp 后必须重新构建匹配的 DQN_Core.lib 和 Trainer/TUI/Watchdog，不能继续使用这个旧库证明新实现。隔离输出至新 build_verify 子目录。HIP 使用项目 gpu_build.targets，注意实际 ROCm DLL 来源；本机曾因 SYSTEM32 与 ROCm DLL 混装误判 GPU 故障。

### 最终交付要求

给出分支、实际署名、各 commit、修改文件、测试原始证据路径、当前 git status。明确“已接入哪些生产路径 / 哪些未接入 / 是否部署 / 是否启动训练”。给出一条隔离短训启动方法及其源码 hash，但不要宣称训练效果提升或历史根因已闭合。

无需为普通实现细节重复请求许可。若遇到真实设计冲突，先给出具体方案和影响，再询问；不要以“还有很多工作”为由把未完成当成已完成。
