# 按方收益传播契约、折扣与时序（第 4 步）

作者：deepseek-flash，2026-09-20。分支 `feature/independent-black-white-networks`。

本文取代本文早期版本的 `y_strict` 推导。**早期结论已撤回**，理由见第 1 节。
权威实现与注释在 `DQN_Core/include/experience_semantics.h`。

## 1. 撤回：`y_strict = R_p − γ·R_opp − γ²·maxQ` 不成立

两条可核验的理由：

1. **`R_opp` 不属于 p 的收益流。** 奖励按行动方生成并写入行动方自己的回放缓冲
   （`store_experience` 按 `player` 分黑白；Trainer 的直写路径同理）。
   按方收益的定义就是「我的每一个决策点上的奖励」；对手那一手的奖励记在对手缓冲里，
   从来不属于 p 的 return。把它加进 p 的目标是**收益定义错误**，不是折扣错误。
2. **终局桥接根本不 bootstrap。** `TerminalOutcomeCredit::observe` 把桥接样本置
   `done=1`，而 `replay_from_buffer` 对 `done` 样本直接跳过 bootstrap。
   所以 `y_now = R_p − γ·maxQ` 描述的是一个**不存在的样本**。

「差额与闭式 `γ(1−γ)qmax − γ·R_opp` 在 float32 内一致」只证明那段算术自洽，
不构成正确性证明——它描述的两个端点都不对应真实样本。

> 交叉印证：`INDEPENDENT_NETWORKS_HANDOFF.md:58` 本来就要求「旧一步回放不能直接
> 当成新两步经验；显式清空不兼容回放并记录原因」。撤回后的两 ply 契约才是交接文档原意。

## 2. 收益流定义

对物理棋色 p，每个 ply（半手）恰有一个行动方：

* 行动方是 p   → p 收到 `compute_reward(..., player=p)`
* 行动方是对手 → p 收到 0

```
V_p(s)   = E[ Σ_{t>=0} γ^t · r_p(ply t) | s_0 = s ]
Q_p(s,a) = r_p(s,a) + γ · V_p(s')      s' = p 落子之后（轮到对手）
V_p(s')  = γ · V_p(s'')                对手那一手，p 得 0
V_p(s'') = max_a Q_p(s'', a)           s'' = 对手回应之后，p 再次决策
```

## 3. 正确契约

```
y = r_p + γ² · max_a Q_p_target(s'')     s'' = 本方下一个决策点的棋盘
```

* **正号**：端点是**己方**价值，己方收益到己方价值之间没有任何取负。
* **指数 2**：按 ply 计时跨两手。按「己方决策」计时则 `γ_own = γ²`、指数 1，
  两种计时同值；本仓库显式采用 **ply 计时 + 指数 2**，因为 `Experience` 里存的是
  棋盘，`observed_plies()` 可直接取证。
* **只用己方目标网络**，不读另一方的网络。
* **终局不 bootstrap**，非对称奖励原样保留。

## 4. 为什么必须保留非对称收益

`WIN_REWARD = +0.5`（phase ≥ 1 双倍 = +1.0）与 `LOSE_REWARD = −0.5` **不构成零和**
（一手终局双方收益之和 = +0.5），`step_outcome.h` 的四个系数也天然不对称。
因此任何 negamax 式的「取负对手价值」都不成立，契约里不得出现 `−V_opp`。
回归里有一条断言直接钉死这点：同一终局棋盘 `player=0/1` 的奖励不互为相反数。

## 5. 经验形态

| 形态 | state | next_state | done | bootstrap |
|---|---|---|---|---|
| A 两 ply 非终局（独立网络正式形态） | p 决策点 | 对手真实回应之后、p 再次决策 | 0 | `γ²·max_a Q_p_target(next_state)`，正号 |
| B 终局桥接（对手回应结束比赛） | p 决策点 | 对手回应之后的最终棋盘 | 1 | 无 |
| B' 本方那一手直接终局 | p 决策点 | 一手之后的最终棋盘 | 1 | 无 |
| C 一 ply 非终局（仅遗留共享模式） | p 决策点 | 一手之后、轮到对手 | 0 | `±1·γ·maxQ`，符号按 turn |

A/B/B' 由 `TerminalOutcomeCredit::observe` 产出：`next_state` 恒写「对手回应之后」
（不再只在 `done` 时替换）。C 只在 `indep_active()==0` 时合法。

**颜色增强**：独立模式下 `store_experience` 强制走**同色旋转**，绝不颜色互换。
`flip_state` 是「180° 旋转 + 棋子取反 + turn 取反」即颜色互换，会把白方样本写进
黑方缓冲（`experience_side` 只看缓冲区指针），并让白和 `+0.05` 之类的非对称奖励
污染黑方。未开启按方路由时保持历史行为逐位不变。

## 6. 形态守卫与旧回放迁移

* `check_replay_span()`：独立模式下非终局样本必须跨 2 ply；终局样本接受 1 或 2 ply。
  违反即 `training_fail_detail(2, "replay_span", ...)`，**不静默通过**。
* 锚点在独立模式下补一个**真实**对手回应，使其满足两 ply 契约；锚点自身的 reward
  不变，防守训练信号没有被削弱或删除。
* 旧回放（预训练产物、V11/V12 阶段中途存档）属于 PLY1 契约。按 HANDOFF:58
  **显式清空并记录条数**，写 `logs/replay_semantics_migration.log`，
  网络权重不受影响；绝不重新标记、绝不按新公式重算旧样本目标。
* 清理用新函数 `discard_ply1_replay()`（谓词 = `done==0 && observed_plies==1`），
  **不能**复用 `discard_solitaire_replay()`：后者的谓词是 turn 相等，而两 ply 正式
  样本同样满足 turn 相等，会把合法新回放整批删掉，同时对一 ply 正式样本一条都不删。
* V13 独立块的块头带 `magic + struct_size + replay_semantics`。读端逐项校验：
  长度或版本不匹配即**整体拒绝加载**，不靠 `version >= 13` 推断块长。
* 启用独立网络后 `checkpoint_version` 恒为 13（`observe` 里按 `indep_active` 决定，
  不再无条件写 12）——否则 `save_version` 退化成 12，另一方会被静默丢弃，
  已载入的 V13 档还会被覆写降级。

## 7. 证据

| 测试 | 覆盖 |
|---|---|
| `tests/experience_semantics_regression.cpp` | 登记表、折扣指数、符号约定、真实生产者产出的样本过校验器；已删除被撤回的 `y_strict` 段 |
| `tests/reward_propagation_contract_regression.cpp` | 真实棋盘手算对照、`y ≠ r+γ·maxQ` 与 `y ≠ r−γ·maxQ` 反例、正号、非对称收益、白和 +0.05 / 黑和 0、遗留言义不变 |
| `tests/replay_span_contract_regression.cpp` | 跨度守卫的负面对照（两 ply 独立可训练 / 一 ply 遗留照常 / 一 ply 独立必须闩死） |
| `tests/checkpoint_v13_persistence_regression.cpp` | 走真实 `observe` 的存档持久性（V13 恒写、重载两方互不相同、不降级、遗留模式仍写 V12） |

## 8. 边界

* 未部署，未启动正式训练，未覆盖正式 checkpoint/EXE。
* 不声称训练效果提升，不声称 Freeze-1 根因闭合。
* `γ²` 是契约的一部分；改成按「己方决策」计时必须同时改计时口径与文档，
  不得只改指数。
