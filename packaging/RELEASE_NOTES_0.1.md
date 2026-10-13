# 0.1

首个公开实验版本：Windows x64 / gfx1151。

**仅针对 AMD ROCm/HIP 做了优化，验证工具链版本为 Windows HIP SDK 7.13.0。其他后端和 ROCm 版本未完成适配与性能验证。**

包含黑白独立网络、技能课程与定向复习、终局步长奖励、V14 存档恢复、Probe 修正，以及 TUI 本地 HTTP 控制。15 项机制回归、TUI 冒烟和真实 GPU 短训保存恢复已通过。

下载 ZIP 后运行 `Start.cmd`；需要本机 HIP SDK 7.13.0、AMD 驱动和 VC++ x64 运行库。程序包不包含 checkpoint、记忆、日志、个人配置或 AMD 驱动/SDK。

本版本用于测试训练。长期学习效果未验证；Linux/MI50 等后续平台不属于此二进制的支持范围。

协作署名：GPT6-Astra。
