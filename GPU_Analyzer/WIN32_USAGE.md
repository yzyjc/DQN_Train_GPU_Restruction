# GPU Analyzer Win32 / TUI integration

协作署名：GPT6-Astra。

- 直接运行 `GPU_Analyzer.exe` 打开原生窗口；`--hidden` 在后台采样。
- 关闭窗口或 Hide 只隐藏；Stop 按钮或 TUI `gpu stop` 正常停止并保存日志。
- TUI GPU Analyze 页提供 Start / Stop / Show / Hide；命令面板提供 `gpu start`、`gpu stop`、`gpu show`、`gpu hide`。
- GPU Analyze 页的 `[Modules 8]` 按钮可一键预约下一训练会话前 8 批低开销 host 模块采集；预约后按钮变为 `[Cancel]`。也可用命令面板 `gpu profile modules 1..8` 自选批次数，或 `gpu profile events 1..8` 启用短时 HIP event 深度诊断。输出自动写入 `logs\gpu_capture_<timestamp>_<pid>\`，页内显示采集进度与结果。深度事件模式会额外创建 HIP events，并在每个捕获批次结束时同步排空，**会干扰被测训练，只建议短时诊断，不应用于正式训练数据**。页面分别显示 host 时间、HIP event 有效覆盖和严格时间线覆盖，不将 event envelope 冒称 GPU 纯执行时间、GPU idle 或真实 DRAM 带宽。`gpu profile status` 查看状态，`gpu profile off` 取消下一次采集。设置只在新 Trainer/Watchdog 进程启动时生效，不会改变正在运行的训练。
- GPU Analyze 页的 `[Copy Report]` 按钮或命令 `gpu report copy` 会把当前遥测、采集文件/有效性、阶段与事件摘要复制为 UTF-8 文本；报告自带测量口径和不可推断项，便于直接粘贴分享。没有采集记录时仍会生成当前状态报告并附上采集命令。
- TUI 的 console 显示开关也控制 Analyzer 窗口。TUI 退出不停止 Analyzer；操作 Analyzer 不停止 Trainer。
- 重复启动不创建第二个采样写入器；TUI 每两秒尝试重连遥测。
- TUI OS 遥测刷新限制为 4 Hz，修复旧缓存永久不更新；这不是对硬件采样频率的保证。
- 保留 `--duration`、`--mode`、`--no-log`、`--log-dir`、`--report`、`--csv`、`--compare`。
- 原生窗口显示当前采样摘要；完整指标仍在 TUI。文本诊断写入工作目录 `gpu_analyzer.runtime.log`。

## 验证

构建 Release/x64 的 GPU_Analyzer 和 TUI。隔离目录须放置兼容的 HIP runtime DLL。

```powershell
./tests/gpu_analyzer_window_smoke.ps1 -Exe ./build_verify/gpu_analyzer_win32/GPU_Analyzer.exe
```

覆盖隐藏启动、显示、点叉隐藏、重复启动拒绝、停止并等待采样线程、完整二进制记录及 CSV 导出。
测试拒绝在已有 Analyzer 窗口时运行，不启动或控制 Trainer。

2026-09-27：已接入 ADLX GPU 温度、GPU 功耗和核心频率；启动 Analyzer 后 TUI 显示，未连接、超过 5 秒或读取失败显示 N/A。功耗不是整颗 APU 或整机功耗。当前仅接受单 GPU 且名称匹配；双卡精确匹配、显存频率、热点及 WGP 数据仍未完成。未验证 Linux/MI50，未改变训练算法。
