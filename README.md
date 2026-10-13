# DQN Train GPU Restruction — 0.1

Windows 原生 5×5 连四强化学习训练实验项目。黑白棋使用独立 online/target 网络，GPU 同时驻留；TUI 提供训练、Probe、评估和本地 HTTP Agent 控制。

> **本项目当前只针对 AMD ROCm/HIP 后端做了优化。0.1 的验证环境为 Windows HIP SDK / ROCm 工具链 7.13.0、gfx1151。CUDA、DirectML、Vulkan 以及其他 ROCm 版本没有完成本版适配与性能验证；界面出现其他后端名称不代表已支持。**

## 运行范围

- 当前二进制：Windows x64、AMD **gfx1151**（在 Radeon 8060S 上验证）。
- 需要 AMD 驱动、HIP SDK / ROCm 工具链 **7.13.0** 和 Microsoft Visual C++ x64 运行库。随该工具链验证的 `amdhip64_7.dll` 文件版本为 **10.0.3581.0**；DLL 文件版本与 SDK 版本不是同一个编号。
- 本版不是通用 GPU 包；Linux、双 MI50/gfx906、Android 和网页客户端尚未发布。

下载 Release 中的 `DQN_Train_GPU_Restruction-0.1-windows-x64-gfx1151.zip`，解压至可写目录，运行 `Start.cmd`。启动脚本从本机 HIP SDK 准备匹配的运行库，然后打开 TUI；TUI 默认启动 Trainer。也可以先运行 `Start.cmd -NoTrain`，只打开界面。

HIP SDK 可通过环境变量 `HIP_PATH` 指定。启动器也检查 `D:\AMD_ROCm` 及 AMD 默认安装目录。SDK：<https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html>。VC++ 运行库：<https://aka.ms/vs/17/release/vc_redist.x64.exe>。

## 0.1 内容

- 独立黑白网络与按方训练状态。
- 不对称终局步长奖励；黑棋早败额外惩罚已删除。
- 技能门槛驱动课程、Minimax 干扰、冻结/历史快照对手。
- 分色技能回归检测、有限批次复习及冷却；复习结束恢复正常学习率。
- V14 checkpoint 和同代记忆恢复，兼容已有旧格式。
- Probe 拓扑、分色参数更新诊断；GPU Analyze 使用现有版本。
- 内置本地 HTTP：首选端口 17865，冲突时尝试 17866–17874。运行后在 `tui_http_endpoint.json` 查找实际地址和访问令牌；`GET /api/help` 返回使用说明。

默认启用新课程和复习，从空数据开始。课程门槛保持保守值。System 面板和 `course status` 显示课程状态。训练文件保存在程序目录下；不要覆盖现有存档安装新版本。

## 验证边界

15 项机制回归通过，另通过 TUI 界面回归和 HTTP 冒烟。真实 TUI HTTP 启动、GPU 短训、V14 保存退出和恢复到 EP20→40→60 已验证。完整课程长期推进及学习效果尚未验证；0.1 是实验发布。

详见 [机制说明](COMPETENCY_HTTP_UPGRADE.md)、[部署验收](TRAINING_DEPLOYMENT_20261013.md)、[架构说明](ARCHITECTURE.md)。

## 构建

当前项目使用 Visual Studio C++ v145、Windows SDK 和 HIP 工具链。默认开发路径在项目文件中，可覆盖 `DqnRocmRoot` 和 `DqnGpuArch`；当前验证架构为 `gfx1151`。运行 `tests/run_competency_regressions.ps1 -IncludeGpu` 执行隔离构建及回归；该脚本的本机工具路径需按环境调整。

公开仓库首次发布使用源码快照，不附带旧训练数据或本机诊断记录；原始提交见 `SOURCE_PROVENANCE.json`。第三方 ADLX 的许可文件位于其源码目录，HIP/驱动由用户从 AMD 安装。视频资源保留项目已有内容。本项目未新增开源许可授权。

发布整理协作署名：GPT6-Astra。
