# 新版测试训练部署验收

协作署名：GPT6-Astra。日期：2026-10-13。

## 部署结果

正式目录 `x64/Release` 已替换 TUI 与 Trainer；Watchdog、Log_Vindicator 和 ROCm DLL 保持原样。实际编译源码提交为 `655bcfd923b0b7d59a6de596f8a2e30a28f14058`。本报告提交只增加部署记录，不改变已验收的程序源码。

- TUI SHA-256：`7BFB58C83D5A548D60761D21B64BE63909D2A264D76067A7B9445EC5A2BEAE5D`。
- Trainer SHA-256：`9B2697FA4E3C2CDEB5A144315CEE1C07D2A230D436CD0BE9D2657A97E3BFA19E`。
- 原 TUI、Trainer 和配置备份：`build_verify/deploy_backup_20261013_171326/`。
- 部署清单：`x64/Release/competency_deploy_manifest_GPT6_Astra.json`，包含程序和配置校验和。

正式配置启用 `curriculum_enabled=1`、`rehearsal_enabled=1`，技能评测间隔为 500。其余新课程参数使用保守默认值，学习率保留 AUTO；旧黑棋早败惩罚配置项已移除。没有把隔离测试中的 EP60 存档或记忆复制到正式目录。

## 实际端到端验收

证据目录：`build_verify/lifecycle_20261013_170829/`；操作日志：`build_verify/lifecycle_http_output.txt`。

验收通过真实 TUI 内置 HTTP 执行，不直接绕过 UI 启停 Trainer：

1. 从空目录启动，独立网络立即启用，Stage 0 课程真实更新 GPU 参数；正常保存退出，生成 EP20 V14 存档。
2. 再次启动，恢复 EP20，继续短训并正常保存退出为 EP40。
3. 再次启动，恢复 EP40，继续短训并正常保存退出为 EP60。

启动/停止返回 HTTP 202 后，脚本等待操作状态完成再发送下一次请求；早发的重启请求被 HTTP 409 正确拒绝，未重复启动 Trainer。最新存档为 V14、49,613,308 字节，CRC 正确，配套 `.mem` 为 V11、EP60、generation 与主体 CRC 一致。两次恢复保留课程局数，没有回到 Stage 0 的第 0 局。未记录训练 fault。所有验收进程已退出。

## 使用与边界

可以直接打开正式目录中的 TUI 开始新一轮测试训练。正式 `logs/`、`checkpoints/`、`checkpoints_temp/` 仍为空，新一轮从零开始；课程与复习已开启。用 System 面板或 `course status` 查看课程阶段与分色技能快照；课程 Stage 与旧 Phase 指标是不同的概念。

这次验收覆盖启动、真实短训、技能评测、保存退出和恢复；此前 15 项机制回归也通过。未验证整个 Stage 0–5 的长期推进或胜率提升，不据此保证学习效果。正式长训未由本次部署自动启动。

原 EP50002 训练资料已归档到 `正式训练记录/2026-9-20~2026-10-13/archived_run_2026-10-13_phase1_ep50002_165901/`，不会自动作为新训练起点。GPU Analyzer 开发继续暂停。
