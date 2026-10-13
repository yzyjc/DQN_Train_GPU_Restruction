#include "independent_networks.h"
#define _CRT_SECURE_NO_WARNINGS
#include "analyzer_cpu_topology.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <time.h>
#include "close_schedule.h"
#include <process.h>
#include <ctype.h>
#include <mmsystem.h>
#include <xinput.h>
#include <stdint.h>
#include <errno.h>
#include <io.h>
#include <sys/stat.h>
#include <tlhelp32.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <dxgi1_2.h>
#include "dqn_replay.h"
#include "playground.h"
#include "skill_test.h"
#include "neural_probe_protocol.h"
#include "network_input.h"
#include "probe_diagnostics.h"
#include "neural_viewer_3d.h"
#include "skill_metric_reader.h"
#include "local_http_server.h"
#include "trainer_http_bridge.h"
#include "competency_view.h"
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "virtual_terminal.h"
#include "ui_controls.h"
#include "status_frames.h"
#include "gpu_telemetry_client.h"
#include "bin_reader.h"
#include "../DQN_Core/include/metric_eval_reader.h"
#include "terminal_window.h"
#include "transition_overlay.h"
#include <imm.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "imm32.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "dxgi.lib")

#pragma comment(lib, "winmm.lib")

/* 终端解耦：所有 printf 输出重定向到虚拟终端 cell buffer */
#define printf vt_printf
#pragma comment(lib, "xinput.lib")

/* ============================================================================
   DQN Main TUI - 训练系统总控前端
   ============================================================================
   设计：分页 + 面板 + 可交互控件
   - 顶部固定导航栏，底部固定快捷键栏，中间分页内容
   - UI 风格/颜色与 Controller v4.0 一致
   - 交互主要鼠标，键盘为辅（复用 Controller 热键 + 前台检测）
   - 命令面板（Ctrl+P）+ 彩蛋（Bad Apple BAV）
   ============================================================================ */

#define PIPE_NAME "\\\\.\\pipe\\dqn_control_pipe"
#define CONFIG_FILENAME "dqn_config.txt"
/* Three fixed pages plus eight single-function pages on the narrowest layout.
   This used to be 10, so building "11 Help" wrote past every tab-state array
   and produced the corrupted Help caption seen after entering Skill Test. */
#define MAX_TABS 11
#define TAB_BAR_Y 1
#define CONTENT_Y 3

/* ---------- 颜色 ---------- */
enum Color {
    COLOR_BLACK = 0, COLOR_DARK_BLUE = 1, COLOR_DARK_GREEN = 2, COLOR_DARK_CYAN = 3,
    COLOR_DARK_RED = 4, COLOR_DARK_MAGENTA = 5, COLOR_DARK_YELLOW = 6, COLOR_LIGHT_GRAY = 7,
    COLOR_DARK_GRAY = 8, COLOR_BLUE = 9, COLOR_GREEN = 10, COLOR_CYAN = 11,
    COLOR_RED = 12, COLOR_MAGENTA = 13, COLOR_YELLOW = 14, COLOR_WHITE = 15
};

/* ---------- 状态机（进程编排） ---------- */
typedef enum {
    MASTER_IDLE, MASTER_STARTING, MASTER_IPC_READY,
    MASTER_WATCHDOG_READY, MASTER_RUNNING, MASTER_SHUTDOWN
} MasterState;
const char* state_name(MasterState s);

typedef struct { HANDLE hProcess; DWORD pid; const wchar_t* name; } WorkerProc;
#define TRAINER_EXE L"DQN_Train_Trainer_NoTerminal.exe"   /* 无终端版：日志写 trainer.log，后台运行 */
#define WATCHDOG_EXE L"Watchdog_NoTerminal.exe"
#define VINDICATOR_EXE L"Log_Vindicator.exe"
#define CKPT_DIR "checkpoints\\"   /* 正式存档点目录（与训练程序一致） */
#define LOG_DIR "logs\\"           /* 通信兼日志文件目录（train.dat/game.dat/eval.dat 等） */
static std::string g_sampled_directory;
static std::string tui_metric_log_path(const char* name) {
    return (g_sampled_directory.empty() ? active_training_log_dir(LOG_DIR) : g_sampled_directory) + name;
}

/* ---------- 全局 ---------- */
static int g_console_width = 0;
static int g_console_height = 0;
static int g_active_tab = 0;          // 0-8
static int need_tab = 1, need_content = 1, need_help = 1;  /* 细粒度刷新标志 */
static TransitionState g_transition = {};   /* 页面切换动画 */
static int g_delay_3d_show = 0;             /* 延迟显示 3D 窗口（等页面动画完成） */
static MasterState g_state = MASTER_IDLE;
static WorkerProc g_trainer = { NULL, 0, TRAINER_EXE };
static WorkerProc g_watchdog = { NULL, 0, WATCHDOG_EXE };
static WorkerProc g_vindicator = { NULL, 0, VINDICATOR_EXE };
static const char* INTENTIONAL_SHUTDOWN_FILE = "intentional_shutdown.flag";
static const char* TRAINING_FAULT_FILE = "training_fault.flag";
#include "dashboard_workers.h"
static DashboardWorkers g_dashboard_workers;
static int& g_guard_user_enabled = g_dashboard_workers.guard_enabled;
static bool g_start_watchdog_for_session = true;
static DashboardWorkerHit g_worker_hits[DASH_WORKER_COUNT];
static int g_worker_hit_cols = 0, g_worker_hit_rows = 0;
static HANDLE g_pipe = INVALID_HANDLE_VALUE;
static int g_pipe_readonly = 0;       /* 兼容由不同 Windows 账户启动的旧 Trainer */
int g_running = 1;
static int g_memory_enabled = 0;
// GPU profiling is armed from the Analyzer command panel and applied only
// when a new Trainer/Watchdog process is created. It never changes a live run.
static int g_gpu_profile_arm_batches = 0;
static int g_gpu_profile_arm_mode = 1; /* 1=modules (low overhead), 2=deep HIP events */
static int g_gpu_profile_session_batches = 0;
static int g_gpu_profile_session_mode = 1;
static wchar_t g_gpu_profile_session_dir[MAX_PATH] = L"";
static int g_gpu_profile_last_files = 0;
static int g_gpu_profile_last_batches = 0;
static int g_gpu_profile_last_mode = 0;
static wchar_t g_gpu_profile_last_dir[MAX_PATH] = L"";
static ULONGLONG g_gpu_report_copy_notice_until = 0;
static int g_ontop = 0;              /* 窗口置顶开关（右上角置顶按钮） */

/* ---------- 本机 GPU 设备信息（DXGI 动态探测，禁止在 UI 中写死型号） ---------- */
typedef struct {
    char name[160];
    char backend[32];
    unsigned vendor_id;
    unsigned device_id;
    uint64_t dedicated_bytes;
    uint64_t shared_bytes;
    int valid;
} GpuDeviceInfo;
static GpuDeviceInfo g_gpu_device = { "未检测到 GPU", "CPU", 0, 0, 0, 0, 0 };
static IDXGIFactory1* g_gpu_factory = NULL;

static int file_exists_w(const wchar_t* path) {
    DWORD a = GetFileAttributesW(path);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static void detect_gpu_device(void) {
    /* DXGI factory 内部持有驱动/适配器通知句柄。旧代码每 10 秒创建再释放
       一个 factory，部分驱动不会立即回收底层句柄，长时间后句柄持续增长。
       复用 factory；只有 DXGI 明确报告失效时才重建。 */
    if (g_gpu_factory && !g_gpu_factory->IsCurrent()) {
        g_gpu_factory->Release();
        g_gpu_factory = NULL;
    }
    if (!g_gpu_factory &&
        FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&g_gpu_factory))) return;
    IDXGIFactory1* factory = g_gpu_factory;
    DXGI_ADAPTER_DESC1 best = {};
    uint64_t best_score = 0;
    int found = 0;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* adapter = NULL;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d = {};
        if (SUCCEEDED(adapter->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            uint64_t score = (uint64_t)d.DedicatedVideoMemory + (uint64_t)d.SharedSystemMemory;
            /* 训练器当前优先 ROCm：有 AMD 硬件时优先显示它。 */
            if (d.VendorId == 0x1002) score += (UINT64_C(1) << 63);
            if (!found || score > best_score) { best = d; best_score = score; found = 1; }
        }
        adapter->Release();
    }
    if (!found) return;
    WideCharToMultiByte(CP_UTF8, 0, best.Description, -1,
        g_gpu_device.name, (int)sizeof(g_gpu_device.name), NULL, NULL);
    g_gpu_device.name[sizeof(g_gpu_device.name) - 1] = '\0';
    g_gpu_device.vendor_id = best.VendorId;
    g_gpu_device.device_id = best.DeviceId;
    g_gpu_device.dedicated_bytes = (uint64_t)best.DedicatedVideoMemory;
    g_gpu_device.shared_bytes = (uint64_t)best.SharedSystemMemory;
    if (best.VendorId == 0x1002 && file_exists_w(L"amdhip64_7.dll"))
        snprintf(g_gpu_device.backend, sizeof g_gpu_device.backend, "ROCm");
    else if (best.VendorId == 0x10de && GetModuleHandleW(L"nvcuda.dll"))
        snprintf(g_gpu_device.backend, sizeof g_gpu_device.backend, "CUDA");
    else
        snprintf(g_gpu_device.backend, sizeof g_gpu_device.backend, "DirectX");
    g_gpu_device.valid = 1;
}

static char status_lines[10][128] = { 0 };
static int g_eval_episode = 0, g_eval_phase = 0, g_eval_depth = 0;
static float g_eval_win = 0, g_eval_draw = 0, g_eval_lose = 0;
static int g_side_win_episode = 0;
static float g_black_eval_win[3] = {}, g_white_eval_win[3] = {};
#define MAX_WIN_POINTS 100
#define NUM_WIN_MODES 3   /* eval.dat 三种评估模式：mode0=Greedy(ε=0) 1=Stable(ε=0.01) 2=Robust(ε=0.02) */
static float g_win_history[NUM_WIN_MODES][MAX_WIN_POINTS] = { { 0 } };
static int g_ep_history[NUM_WIN_MODES][MAX_WIN_POINTS] = { { 0 } };
static int g_win_count[NUM_WIN_MODES] = { 0 };
static int g_win_view = 72;  /* Training 图表显示点数（←→ 缩放） */
static DWORD g_session_start = 0;   /* 会话开始时间（S 启动时记录） */
static int g_session_ep_start = 0;  /* 会话开始时的 episode */
static float g_ep_time = 0, g_elapsed = 0, g_remaining = 0;  /* 训练程序推送的真实时间 */
static float g_phase_forward_us = 0, g_phase_backward_us = 0, g_phase_target_us = 0;
static int g_phase_steps = 0;
static float g_side_forward_us[2] = {0, 0};
static float g_side_backward_us[2] = {0, 0};
static int g_side_steps[2] = {0, 0};
static float g_replay_sample_us = 0, g_replay_forward_us = 0, g_replay_target_us = 0;
static float g_replay_train_us = 0, g_replay_sync_us = 0;
static int g_replay_count = 0;
static float g_bwd_fc4_us = 0, g_bwd_fc3_us = 0, g_bwd_fc2_us = 0, g_bwd_fc1_us = 0;
static float g_bwd_pool_us = 0, g_bwd_attn_us = 0, g_bwd_embed_us = 0;
static int g_bwd_count = 0;
static float g_ts_setup_us = 0, g_ts_memset_us = 0, g_ts_fwd_us = 0, g_ts_hubler_us = 0, g_ts_bwd_us = 0;
static float g_ts_validate_us = 0;
static int g_ts_count = 0;
static int g_pretrain_active = 0, g_pretrain_steps = 0, g_pretrain_stable = 0;
static float g_pretrain_avg_steps = 0, g_pretrain_success = 0;
static int g_ui_eval_active = 0, g_ui_eval_completed = 0, g_ui_eval_total = 0;
static StageProgress g_ui_stage = {};   /* Minimax 后的阶段进度（技能测试/多快照/保存） */
#define MAX_LOG_DISPLAY 512
static char g_log_display[MAX_LOG_DISPLAY][160] = { 0 };
static int g_log_display_count = 0;

/* ---------- 多指标历史缓冲（Training 页折线图数据源） ---------- */
#define HIST_MAX         25000   /* 全局时间轴：20 个 Win 检查点约覆盖 2 万局 */
#define MAX_BUCKETS      8
#define TRAIN_TAIL_SIZE  65536
typedef struct {
    float v[HIST_MAX];   /* Y 值 */
    float ep[HIST_MAX];  /* BUG-021: 每个点对应的平均 episode 号（bin 数据用，文本数据填 0） */
    int n;
    int base;            /* 文本回退路径仍用 base；bin 路径用 ep[] 计算真实 X 轴 */
} HistoryBuf;
typedef enum {
    METRIC_Q = 0, METRIC_LOSS, METRIC_TARGET_Q, METRIC_EPSILON,
    METRIC_EP_TIME, METRIC_GAME_LEN, METRIC_WIN0, METRIC_WIN1, METRIC_WIN2, METRIC_DEPTH,
    METRIC_R_LIVE2, METRIC_R_OPEN3, METRIC_R_THREAT3, METRIC_R_FOUR,
    METRIC_R_DOUBLE3, METRIC_R_BLOCK_WIN, METRIC_R_BLOCK_OPEN3, METRIC_R_TOTAL,
    METRIC_COUNT
} MetricId;
static HistoryBuf g_hist[METRIC_COUNT];
static const char* g_metric_labels[METRIC_COUNT] = {
    "Q", "Loss", "T-Q", "Eps", "t/ep", "Steps", "Win@0", "Win@1", "Win@2", "Depth",
    "R-Live2", "R-Open3", "R-Threat3", "R-Four", "R-Double3", "R-BlockWin",
    "R-Block3", "R-Total"
};
static int g_metric_colors[METRIC_COUNT];

/* 高优先级指标先取得辨识度最高且不重复的颜色；颜色不足时仅低优先级指标复用。 */
static void init_metric_colors(void) {
    static const int palette[] = {
        COLOR_CYAN, COLOR_RED, COLOR_MAGENTA, COLOR_YELLOW, COLOR_GREEN,
        COLOR_BLUE, COLOR_WHITE, COLOR_DARK_CYAN, COLOR_DARK_YELLOW,
        COLOR_DARK_MAGENTA, COLOR_DARK_GREEN, COLOR_DARK_BLUE,
        COLOR_LIGHT_GRAY, COLOR_DARK_RED, COLOR_DARK_GRAY
    };
    static const MetricId priority[] = {
        METRIC_Q, METRIC_LOSS, METRIC_TARGET_Q, METRIC_EPSILON, METRIC_R_TOTAL,
        METRIC_WIN0, METRIC_WIN1, METRIC_WIN2, METRIC_DEPTH, METRIC_R_BLOCK_WIN,
        METRIC_R_FOUR, METRIC_R_DOUBLE3, METRIC_R_OPEN3, METRIC_R_THREAT3,
        METRIC_R_BLOCK_OPEN3, METRIC_R_LIVE2, METRIC_EP_TIME, METRIC_GAME_LEN
    };
    const int color_count = (int)(sizeof(palette) / sizeof(palette[0]));
    for (int rank = 0; rank < METRIC_COUNT; rank++)
        g_metric_colors[priority[rank]] = palette[rank % color_count];
}
/* 折线图主题：0=方块(柱状) 1=圆角折线 2=直角折线；默认圆角 */
static int g_line_theme = 3;   /* 0=block 1=round 2=right 3=curve(真实曲线，默认) */
static int64_t g_train_offset = -1;    /* BUG-024: 64 位文件偏移 */
/* game.dat 增量采样（训练对局每局步数） */
static int64_t g_tg_offset = -1;       /* BUG-024: 64 位文件偏移 */
#define TG_SCAN_LIMIT (32 * 1024 * 1024)  /* 首次尽量覆盖完整 game.dat，供 Win 混合图回看长区间 */
#define MM_WINDOW 65536
static float g_last_ep_time_sampled = 0;
static int g_last_ep_time_episode = -1;   /* t/ep 必须按局采样，不能用浮点值变化代替新局判定 */
/* 图表格：每格独立选一个指标，Tab 切焦点格，←→ 轮换该格指标 */
#define MAX_CELLS 9      /* 最多 3×3 图表框 */
#define METRIC_MENU_VISIBLE 10
#define MIN_CELL_W 70   /* 框最小宽（阈值：决定一排放几个框） */
#define MIN_CELL_H 12   /* 框最小高（阈值：决定竖排几个框） */

/* 根据窗口大小算图表网格：阈值决定 cols/rows，每框宽高 = 可用区/格数（填满不留白），
   总数不超过 MAX_CELLS（指标数） */
void compute_chart_layout(int avail_w, int avail_h, int* cols, int* rows, int* cw, int* ch) {
    int c = 1;
    if (avail_w >= 2 * MIN_CELL_W) c = 2;
    if (avail_w >= 3 * MIN_CELL_W) c = 3;
    int r = 1;
    if (avail_h >= 2 * MIN_CELL_H) r = 2;
    if (avail_h >= 3 * MIN_CELL_H) r = 3;
    while (c * r > MAX_CELLS && r > 1) r--;   /* 优先减行，保持框更宽 */
    while (c * r > MAX_CELLS && c > 1) c--;
    *cols = c;
    *rows = r;
    *cw = avail_w / c;
    *ch = avail_h / r;
}
static int g_cell_metric[MAX_CELLS] = { METRIC_Q, METRIC_LOSS, METRIC_TARGET_Q, METRIC_EPSILON, METRIC_EP_TIME, METRIC_GAME_LEN, METRIC_R_TOTAL, METRIC_R_BLOCK_WIN, METRIC_R_OPEN3 };
/* 勾选集合：每格可勾选多个指标，曲线模式叠加显示（下拉勾选式） */
static int g_cell_checks[MAX_CELLS][METRIC_COUNT] = {
    { 1,0,0,0,0,0,0,0,0 },
    { 0,1,0,0,0,0,0,0,0 },
    { 0,0,1,0,0,0,0,0,0 },
    { 0,0,0,1,0,0,0,0,0 },
    { 0,0,0,0,1,0,0,0,0 },
    { 0,0,0,0,0,1,0,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 },
    { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0 },
    { 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0 },
};
static int g_cell_focus = 0;   /* 当前焦点格 */
static int g_cell_dropdown = -1;  /* 单框下拉选择打开时 = 框索引，-1=关闭 */
static int g_cell_drop_cur = 0;   /* 下拉高亮指标 id */
/* 每格 x 轴查看窗口（距最新点偏移；start=-1 全量）；滑块点击设置，只重绘该格 */
static int g_cell_view_start[MAX_CELLS] = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
static int g_cell_view_end[MAX_CELLS] = { 0 };
static int g_cell_dirty[MAX_CELLS] = { 0 };
static int g_cell_global_axis[MAX_CELLS] = { 0 }; /* 全局复位后：按 episode 自动管理可见范围 */
static int g_cell_reset_press = -1;   /* 复位按钮按下高亮的格（点击反馈，短暂显示） */
static DWORD g_reset_press_time = 0;

/* 轮换焦点格显示的指标 */
void cycle_cell_metric(int dir) {
    if (g_cell_focus < 0 || g_cell_focus >= MAX_CELLS) return;
    int cur = g_cell_metric[g_cell_focus];
    if (cur < 0) cur = 0;
    int nw = (cur + dir + METRIC_COUNT) % METRIC_COUNT;
    g_cell_metric[g_cell_focus] = nw;
    memset(g_cell_checks[g_cell_focus], 0, sizeof g_cell_checks[g_cell_focus]);   /* ←→ 轮换重置为单指标 */
    g_cell_checks[g_cell_focus][nw] = 1;
}

static void history_push(MetricId id, float v) {
    HistoryBuf* h = &g_hist[id];
    if (h->n >= HIST_MAX) {
        memmove(&h->v[0], &h->v[1], (HIST_MAX - 1) * sizeof(float));
        memmove(&h->ep[0], &h->ep[1], (HIST_MAX - 1) * sizeof(float));
        h->n = HIST_MAX - 1;
        h->base++;   /* 丢弃最旧点：绝对起始 +1 */
    }
    h->v[h->n] = v;
    h->ep[h->n] = 0;   /* 文本路径无 episode 信息，填 0（X 轴回退到 base+idx） */
    h->n++;
}
/* 推入带 episode 的数据点（bin 路径） */
static void history_push_ep(MetricId id, float v, float episode) {
    HistoryBuf* h = &g_hist[id];
    /* 日志可能在恢复较早 checkpoint 后继续追加。不要把两个训练时间线连成
       一条折返曲线；同一 episode 的新记录则以最新值覆盖。 */
    if (h->n > 0 && episode > 0 && h->ep[h->n - 1] > 0) {
        if (episode == h->ep[h->n - 1]) {
            h->v[h->n - 1] = v;
            return;
        }
        if (episode < h->ep[h->n - 1]) {
            /* 增量日志、IPC 状态和维护器可能存在短暂先后差。单个迟到点
               绝不能清空整条历史；若历史里已有该局就更新，否则忽略。
               真正的文件重建/回滚由 sample_bin_metrics 的 identity/tail
               检测按文件精确清理。 */
            for (int i = h->n - 2; i >= 0; --i) {
                if (h->ep[i] == episode) { h->v[i] = v; return; }
                if (h->ep[i] < episode) break;
            }
            return;
        }
    }
    if (h->n >= HIST_MAX) {
        memmove(&h->v[0], &h->v[1], (HIST_MAX - 1) * sizeof(float));
        memmove(&h->ep[0], &h->ep[1], (HIST_MAX - 1) * sizeof(float));
        h->n = HIST_MAX - 1;
        h->base++;
    }
    h->v[h->n] = v;
    h->ep[h->n] = episode;
    h->n++;
}

/* 状态栏显示最近 N 局的真实墙钟均值。数据来自持久化的
   metric_episode_time.bin，因此 TUI 重启后仍能恢复，而不是只统计本次连接。 */
static float recent_episode_time_average(int max_episodes, int* sample_count) {
    const HistoryBuf* h = &g_hist[METRIC_EP_TIME];
    double sum = 0.0;
    int count = 0;
    for (int i = h->n - 1; i >= 0 && count < max_episodes; --i) {
        const float value = h->v[i];
        if (!isfinite(value) || value < 0.0f || value >= 1e30f) continue;
        sum += value;
        ++count;
    }
    if (sample_count) *sample_count = count;
    return count ? (float)(sum / count) : 0.0f;
}

/* ========== bin 文件采样（bin_reader.h 抽象层）：自动发现、增量读取、映射到 HistoryBuf ========== */
static BRFile g_bin_files[BR_MAX_FILES];
static int g_bin_file_count = 0;
#include "episode_average.h"
static EpisodeAccumulator g_episode_average[METRIC_COUNT] = {};

/* 只有确认某个 bin 文件被重建/回滚时才清除它负责的曲线。 */
static void clear_histories_for_bin(const char* name) {
    if (!name) return;
    if (strstr(name, "metric_loss.bin")) {
        g_episode_average[METRIC_LOSS] = {};
        g_episode_average[METRIC_Q] = {};
        g_episode_average[METRIC_TARGET_Q] = {};
        memset(&g_hist[METRIC_LOSS], 0, sizeof(g_hist[METRIC_LOSS]));
        memset(&g_hist[METRIC_Q], 0, sizeof(g_hist[METRIC_Q]));
        memset(&g_hist[METRIC_TARGET_Q], 0, sizeof(g_hist[METRIC_TARGET_Q]));
    } else if (strstr(name, "metric_epsilon_episode.bin") || strstr(name, "metric_epsilon.bin")) {
        memset(&g_hist[METRIC_EPSILON], 0, sizeof(g_hist[METRIC_EPSILON]));
    } else if (strstr(name, "metric_episode_time.bin")) {
        memset(&g_hist[METRIC_EP_TIME], 0, sizeof(g_hist[METRIC_EP_TIME]));
    } else if (strstr(name, "metric_game.bin")) {
        memset(&g_hist[METRIC_GAME_LEN], 0, sizeof(g_hist[METRIC_GAME_LEN]));
    } else if (strstr(name, "metric_win.bin")) {
        for (int id = METRIC_WIN0; id <= METRIC_WIN2; ++id)
            memset(&g_hist[id], 0, sizeof(g_hist[id]));
    } else if (strstr(name, "metric_eval.bin")) {
        memset(&g_hist[METRIC_DEPTH], 0, sizeof(g_hist[METRIC_DEPTH]));
    } else if (strstr(name, "metric_rewards.bin")) {
        for (int id = METRIC_R_LIVE2; id <= METRIC_R_TOTAL; ++id)
            memset(&g_hist[id], 0, sizeof(g_hist[id]));
    }
}

/* 分桶平均降采样：把 n 个值分成 max_buckets 桶，每桶取平均后推入 history */
static void bin_bucket_push(MetricId id, const float* vals, int n, int max_buckets, const int* episodes) {
    if (n <= 0) return;
    int buckets = (n < max_buckets) ? n : max_buckets;
    int per = n / buckets;
    if (per == 0) { buckets = n; per = 1; }
    for (int b = 0; b < buckets; b++) {
        int i0 = b * per, i1 = (b == buckets - 1) ? n : (b + 1) * per;
        if (i0 >= i1) break;
        double sum = 0; double ep_sum = 0;
        int valid = 0;
        for (int i = i0; i < i1; i++) {
            /* Log Vindicator 用 NaN 表示字段缺失。占位符必须保留在日志里，
               但不能参与平均，否则一个坏点会让整桶曲线消失。 */
            if (!(vals[i] == vals[i]) || vals[i] <= -1e30f || vals[i] >= 1e30f) continue;
            if (episodes && episodes[i] < 0) continue;
            sum += vals[i];
            if (episodes) ep_sum += episodes[i];
            valid++;
        }
        if (valid == 0) continue;
        float avg = (float)(sum / valid);
        float avg_ep = episodes ? (float)(ep_sum / valid) : 0;
        history_push_ep(id, avg, avg_ep);
    }
}

/* 从 BRRecord 数组提取某个 field 并分桶推入（同时传入 episode 用于 X 轴） */
static void bin_extract_and_push(const BRRecord* buf, int n, int field_idx, MetricId id, int max_buckets) {
    if (n <= 0 || field_idx < 0) return;
    float* arr = (float*)malloc(n * sizeof(float));
    int* eps = (int*)malloc(n * sizeof(int));
    if (!arr || !eps) { free(arr); free(eps); return; }
    for (int i = 0; i < n; i++) { arr[i] = buf[i].fields[field_idx]; eps[i] = buf[i].episode; }
    bin_bucket_push(id, arr, n, max_buckets, eps);
    free(arr); free(eps);
}

/* Loss/Q/T-Q 每局有数十到数百条 replay 更新。图表横轴是 episode，
   因而每局只需要一个均值点；逐训练记录塞入 HistoryBuf 会在 25000 点
   后触发每点两次大 memmove，造成运行越久刷新越慢。 */
static void bin_extract_episode_average(const BRRecord* buf, int n, int field_idx, MetricId id) {
    if (!buf || n <= 0 || field_idx < 0 || field_idx >= BR_MAX_FIELDS) return;
    EpisodeAccumulator& a = g_episode_average[id];
    for (int i = 0; i < n; ++i) {
        float v = buf[i].fields[field_idx];
        int ep = buf[i].episode;
        if (ep <= 0 || !isfinite(v) || fabsf(v) >= 1e30f) continue;
        if (a.count && ep < a.ep) continue; // True rewinds reset the accumulator with the reader.
        if (!a.count || ep != a.ep) {
            if (a.count) history_push_ep(id, (float)(a.sum / a.count), (float)a.ep);
            a = {ep, 0, 0.0};
        }
        a.add(ep, v);
    }
    if (a.count) history_push_ep(id, (float)(a.sum / a.count), (float)a.ep);
}

/* 标记：每个 bin 文件是否已完成首次全量读取（首次=br_read_all 整体分桶，之后=br_read_incremental 增量分桶） */
static int g_bin_first_read_done[BR_MAX_FILES] = {0};

/* Loss/Q/T-Q 的旧格式把 global_step 放在 episode 字段中。通过两份 epsilon
   日志建立 step -> epsilon -> episode 桥接，使所有 Training 图统一使用局数。 */
typedef struct { int step; float epsilon; } StepEpsilon;
typedef struct { int episode; float epsilon; } EpisodeEpsilon;
static StepEpsilon* g_step_eps_map = NULL;
static EpisodeEpsilon* g_episode_eps_map = NULL;
static int g_step_eps_count = 0, g_episode_eps_count = 0;
static int64_t g_step_eps_size = -1, g_episode_eps_size = -1;

static int64_t file_size_shared(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    _fseeki64(f, 0, SEEK_END);
    int64_t size = _ftelli64(f);
    fclose(f);
    return size;
}

static void refresh_step_episode_bridge(void) {
    const std::string step_storage = tui_metric_log_path("metric_epsilon.bin");
    const std::string episode_storage = tui_metric_log_path("metric_epsilon_episode.bin");
    const char* step_path = step_storage.c_str();
    const char* episode_path = episode_storage.c_str();
    int64_t ss = file_size_shared(step_path), es = file_size_shared(episode_path);
    if (ss >= (int64_t)sizeof(BRHeader) && ss != g_step_eps_size) {
        FILE* f = fopen(step_path, "rb"); BRHeader h;
        if (f && fread(&h, sizeof(h), 1, f) == 1 && h.magic == BR_MAGIC && h.field_count == 1) {
            int n = (int)((ss - sizeof(BRHeader)) / (8 + 4));
            int old_n = (g_step_eps_size >= (int64_t)sizeof(BRHeader) && ss > g_step_eps_size) ? g_step_eps_count : 0;
            if (!old_n) { free(g_step_eps_map); g_step_eps_map = NULL; }
            StepEpsilon* map = (StepEpsilon*)realloc(old_n ? g_step_eps_map : NULL, (size_t)n * sizeof(*map));
            if (map) {
                g_step_eps_map = map;
                _fseeki64(f, (int64_t)sizeof(BRHeader) + (int64_t)old_n * 12, SEEK_SET);
                BRRecord rec; int got = old_n;
                while (got < n && fread(&rec, 12, 1, f) == 1) {
                    map[got].step = rec.episode; map[got].epsilon = rec.fields[0]; got++;
                }
                g_step_eps_count = got; g_step_eps_size = (int64_t)sizeof(BRHeader) + (int64_t)got * 12;
            }
        }
        if (f) fclose(f);
    }
    if (es >= (int64_t)sizeof(BRHeader) && es != g_episode_eps_size) {
        FILE* f = fopen(episode_path, "rb"); BRHeader h;
        if (f && fread(&h, sizeof(h), 1, f) == 1 && h.magic == BR_MAGIC && h.field_count == 1) {
            int n = (int)((es - sizeof(BRHeader)) / (8 + 4));
            int old_n = (g_episode_eps_size >= (int64_t)sizeof(BRHeader) && es > g_episode_eps_size) ? g_episode_eps_count : 0;
            if (!old_n) { free(g_episode_eps_map); g_episode_eps_map = NULL; }
            EpisodeEpsilon* map = (EpisodeEpsilon*)realloc(old_n ? g_episode_eps_map : NULL, (size_t)n * sizeof(*map));
            if (map) {
                g_episode_eps_map = map;
                _fseeki64(f, (int64_t)sizeof(BRHeader) + (int64_t)old_n * 12, SEEK_SET);
                BRRecord rec; int got = old_n;
                while (got < n && fread(&rec, 12, 1, f) == 1) {
                    map[got].episode = rec.episode; map[got].epsilon = rec.fields[0]; got++;
                }
                g_episode_eps_count = got; g_episode_eps_size = (int64_t)sizeof(BRHeader) + (int64_t)got * 12;
            }
        }
        if (f) fclose(f);
    }
}

static int loss_step_to_episode(int step) {
    if (!g_step_eps_map || !g_episode_eps_map || g_step_eps_count <= 0 || g_episode_eps_count <= 0) return 0;
    int lo = 0, hi = g_step_eps_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (g_step_eps_map[mid].step < step) lo = mid + 1;
        else if (g_step_eps_map[mid].step > step) hi = mid - 1;
        else { lo = mid; break; }
    }
    int si = lo;
    if (si >= g_step_eps_count) si = g_step_eps_count - 1;
    if (si > 0 && abs(g_step_eps_map[si - 1].step - step) < abs(g_step_eps_map[si].step - step)) si--;
    float eps = g_step_eps_map[si].epsilon;
    /* episode epsilon 总体单调不增；相等区段取最新项，避免固定 EPS 映射到最早局。 */
    int el = 0, eh = g_episode_eps_count - 1;
    while (el <= eh) {
        int mid = el + (eh - el) / 2;
        if (g_episode_eps_map[mid].epsilon > eps) el = mid + 1;
        else if (g_episode_eps_map[mid].epsilon < eps) eh = mid - 1;
        else { el = mid; break; }
    }
    int best = el;
    if (best >= g_episode_eps_count) best = g_episode_eps_count - 1;
    if (best > 0 && fabs(g_episode_eps_map[best - 1].epsilon - eps) <=
        fabs(g_episode_eps_map[best].epsilon - eps)) best--;
    /* 相同 epsilon（手动固定或平台期）取该连续区段的最新 episode。 */
    while (best + 1 < g_episode_eps_count &&
        fabs(g_episode_eps_map[best + 1].epsilon - g_episode_eps_map[best].epsilon) < 1e-7f) best++;
    return g_episode_eps_map[best].episode;
}

void sample_bin_metrics(void) {
    const std::string directory = active_training_log_dir(LOG_DIR);
    if (directory != g_sampled_directory) {
        memset(g_hist, 0, sizeof(g_hist));
        memset(g_episode_average, 0, sizeof(g_episode_average));
        memset(g_bin_files, 0, sizeof(g_bin_files));
        memset(g_bin_first_read_done, 0, sizeof(g_bin_first_read_done));
        g_bin_file_count = 0;
        g_train_offset = g_tg_offset = -1;
        g_step_eps_size = g_episode_eps_size = -1;
        g_step_eps_count = g_episode_eps_count = 0;
        g_last_ep_time_episode = -1;
        for (int c = 0; c < MAX_CELLS; ++c) {
            g_cell_view_start[c] = -1; g_cell_view_end[c] = 0; g_cell_global_axis[c] = 1;
            g_cell_dirty[c] = 1;
        }
        g_sampled_directory = directory;
        need_content = 1;
    }
    /* 每次扫描只发现文件；按文件名合并，绝不能覆盖已有文件的 offset/首次读取状态。
       Windows 目录枚举顺序不稳定，按数组下标复用状态会造成 win 消失和 epsilon 重放锯齿。 */
    BRFile scanned[BR_MAX_FILES];
    int scanned_count = br_scan(directory.c_str(), scanned, BR_MAX_FILES);
    for (int s = 0; s < scanned_count; s++) {
        int existing = br_find(g_bin_files, g_bin_file_count, scanned[s].name);
        if (existing >= 0) {
            BRFile* old = &g_bin_files[existing];
            /* 同名文件可能被清理器删除重建；也可能原地截断后在两秒采样间隔内
               又长到旧 offset 以上。单看 size 无法识别这两种情况，会从文件
               中部继续读，表现为大量历史曲线永久不出现。 */
            int timeline_reset =
                (old->identity != 0 && scanned[s].identity != old->identity) ||
                scanned[s].file_size < old->offset ||
                (old->tail_episode > 0 && scanned[s].tail_episode > 0 &&
                 scanned[s].tail_episode < old->tail_episode);
            if (timeline_reset) {
                clear_histories_for_bin(scanned[s].name);
                *old = scanned[s];
                g_bin_first_read_done[existing] = 0;
            } else {
                /* 保留 offset，只刷新可能增长的文件状态。 */
                old->file_size = scanned[s].file_size;
                old->record_count = scanned[s].record_count;
                old->field_count = scanned[s].field_count;
                old->tail_episode = scanned[s].tail_episode;
                old->identity = scanned[s].identity;
            }
        } else if (g_bin_file_count < BR_MAX_FILES) {
            g_bin_files[g_bin_file_count] = scanned[s];
            g_bin_first_read_done[g_bin_file_count] = 0;
            g_bin_file_count++;
        }
    }
    /* Loss 每局可能有多次 replay 更新。启动时若只读 65536 条，和 Win
       混合显示时通常覆盖不了 20 个检查点（约两万局）。扩大尾部缓冲，
       再按 episode 压缩进固定历史，不增加长期内存增长。 */
    static BRRecord buf[262144];
    for (int i = 0; i < g_bin_file_count; i++) {
        const char* name = g_bin_files[i].name;
        int n;
        int initial_read = !g_bin_first_read_done[i];
        if (!g_bin_first_read_done[i]) {
            /* 启动即采用"全局复位"窗口：普通逐局指标只取最近 200 点；
               Win 每检查点有 3 个 mode，因此读取最近 60 条 = 每条曲线 20 个检查点。
               Loss/Q 是逐训练步记录，需要较长连续尾部才能覆盖约 200 局。 */
            if (strstr(name, "metric_loss.bin")) {
                g_episode_average[METRIC_LOSS] = {};
                g_episode_average[METRIC_Q] = {};
                g_episode_average[METRIC_TARGET_Q] = {};
                // Seek by real episode, not a fixed number of replay rows.
                // Preserve up to HIST_MAX episode means for Win overlays; process
                // all records in bounded chunks, retaining the last accumulator.
                int latest = g_bin_files[i].tail_episode;
                int eps_index = br_find(g_bin_files, g_bin_file_count, "metric_epsilon_episode.bin");
                int anchor = eps_index >= 0 ? g_bin_files[eps_index].tail_episode : 0;
                bool legacy_axis = anchor > 0 && latest > anchor + 100;
                if (!legacy_axis && latest > 0 && g_bin_files[i].field_count == 3 &&
                    br_seek_episode_window(&g_bin_files[i], latest > HIST_MAX ? latest - HIST_MAX + 1 : 1)) {
                    const int64_t end = g_bin_files[i].offset +
                        ((g_bin_files[i].file_size - g_bin_files[i].offset) / 20) * 20;
                    while (g_bin_files[i].offset < end) {
                        int take = (int)((end - g_bin_files[i].offset) / 20);
                        if (take > 8192) take = 8192;
                        int got = br_read_incremental(&g_bin_files[i], buf, take);
                        if (got <= 0) break;
                        bin_extract_episode_average(buf, got, 0, METRIC_LOSS);
                        bin_extract_episode_average(buf, got, 1, METRIC_Q);
                        bin_extract_episode_average(buf, got, 2, METRIC_TARGET_Q);
                    }
                    g_bin_first_read_done[i] = 1;
                    continue;
                }
                n = br_read_tail(&g_bin_files[i], buf, 262144);
            }
            else if (strstr(name, "metric_win.bin") || strstr(name, "metric_eval.bin"))
                n = br_read_tail(&g_bin_files[i], buf, 60);
            else if (strstr(name, "metric_epsilon_episode.bin") ||
                     strstr(name, "metric_episode_time.bin") ||
                     strstr(name, "metric_rewards.bin") ||
                     strstr(name, "metric_game.bin")) n = br_read_tail(&g_bin_files[i], buf, HIST_MAX);
            else n = br_read_tail(&g_bin_files[i], buf, 200);
            g_bin_first_read_done[i] = 1;
        } else {
            /* 后续：增量读取新数据，分桶推入 */
            n = br_read_incremental(&g_bin_files[i], buf, 262144);
        }
        if (n <= 0) continue;
        /* BUG-030: 验证 field_count 是否匹配预期（防止旧 schema 文件混入） */
        int expected_fc = 0;
        if (strstr(name, "metric_loss.bin")) expected_fc = 3;
        else if (strstr(name, "metric_epsilon_episode.bin")) expected_fc = 1;
        else if (strstr(name, "metric_episode_time.bin")) expected_fc = 1;
        else if (strstr(name, "metric_epsilon.bin")) expected_fc = 1;
        else if (strstr(name, "metric_game.bin")) expected_fc = 1;
        else if (strstr(name, "metric_win.bin")) expected_fc = 4;
        else if (strstr(name, "metric_eval.bin")) expected_fc = 7;
        else if (strstr(name, "metric_rewards.bin")) expected_fc = 8;
        else if (strstr(name, "metric_steps.bin")) expected_fc = 4;
        if (expected_fc > 0 && g_bin_files[i].field_count != expected_fc) continue;   /* field_count 不匹配，跳过 */
        int buckets = (n > 100) ? 200 : n;
        if (strstr(name, "metric_loss.bin")) {
            /* 新格式已直接写真实 episode；旧格式才把 global_step 填在 episode。
               若文件中发生"大 step -> 小 episode"回落，说明 Trainer 已升级，
               丢弃回落前的旧轴数据，避免一个桶内混合两种坐标。 */
            /* 日志格式判断必须使用日志自己的局号。旧只读管道可能把多条
               状态 JSON 粘包，IPC episode 一度落后数百局；拿它判断会把
               新格式真实 episode 错当 global_step，继而让三条曲线回退。
               每局 epsilon 日志与 loss 同时写入，是最可靠的时间轴锚。 */
            int cur_ep = 0;
            int eps_file = br_find(g_bin_files, g_bin_file_count, "metric_epsilon_episode.bin");
            if (eps_file >= 0) cur_ep = g_bin_files[eps_file].tail_episode;
            if (cur_ep <= 0 && g_hist[METRIC_EPSILON].n > 0)
                cur_ep = (int)g_hist[METRIC_EPSILON].ep[g_hist[METRIC_EPSILON].n - 1];
            if (cur_ep <= 0) cur_ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
            int new_start = 0;
            for (int j = 1; j < n; j++)
                if (buf[j].episode > 0 && buf[j - 1].episode > buf[j].episode + 100) new_start = j;
            if (new_start > 0) {
                memmove(buf, buf + new_start, (size_t)(n - new_start) * sizeof(buf[0]));
                n -= new_start;
                buckets = (n > 100) ? 200 : n;
            }
            int legacy_step_axis = n > 0 && cur_ep > 0 && buf[n - 1].episode > cur_ep + 100;
            if (legacy_step_axis && cur_ep > 0) {
                /* 用总训练记录/局数估算尾部连续记录的局数跨度，
                   比把 65536 条强行铺到 25000 局更接近真实时间线。 */
                int span = HIST_MAX;
                if (initial_read && cur_ep > 0 && g_bin_files[i].record_count > 0) {
                    double records_per_ep = (double)g_bin_files[i].record_count / cur_ep;
                    if (records_per_ep > 0.1) span = (int)(n / records_per_ep + 0.5);
                    if (span < 200) span = 200; if (span > HIST_MAX) span = HIST_MAX;
                }
                int first_ep = cur_ep - span + 1; if (first_ep < 1) first_ep = 1;
                for (int j = 0; j < n; j++)
                    buf[j].episode = initial_read && n > 1 ? first_ep + (int)((long long)(cur_ep - first_ep) * j / (n - 1)) : cur_ep;
            }
            bin_extract_episode_average(buf, n, 0, METRIC_LOSS);
            bin_extract_episode_average(buf, n, 1, METRIC_Q);
            bin_extract_episode_average(buf, n, 2, METRIC_TARGET_Q);
        } else if (strstr(name, "metric_epsilon_episode.bin")) {
            /* 每局一个点，不做桶平均，保留 epsilon 阶段变化。 */
            bin_extract_and_push(buf, n, 0, METRIC_EPSILON, (n > HIST_MAX) ? HIST_MAX : n);
        } else if (strstr(name, "metric_episode_time.bin")) {
            /* 每局真实墙钟耗时；重启 TUI 后可恢复最近 200 局，不再依赖 IPC 抽样。 */
            bin_extract_and_push(buf, n, 0, METRIC_EP_TIME,
                initial_read ? ((n > HIST_MAX) ? HIST_MAX : n) : buckets);
        } else if (strstr(name, "metric_epsilon.bin")) {
            /* 有每局日志时忽略旧的每训练步 epsilon，避免重复及过度平滑。 */
            if (br_find(g_bin_files, g_bin_file_count, "metric_epsilon_episode.bin") >= 0) continue;
            bin_extract_and_push(buf, n, 0, METRIC_EPSILON, buckets);
        } else if (strstr(name, "metric_game.bin")) {
            /* 每局仅一条最终步数记录，是 Steps 图的权威来源。写入端已禁止
               seed agent 进入无 seed 字段的 bin；这里把 0-based 局号统一成
               Training 页使用的 1-based episode。 */
            for (int j = 0; j < n; j++) {
                if (buf[j].episode >= 0 && buf[j].episode < INT_MAX)
                    buf[j].episode++;
            }
            bin_extract_and_push(buf, n, 0, METRIC_GAME_LEN,
                initial_read ? ((n > HIST_MAX) ? HIST_MAX : n) : buckets);
        } else if (strstr(name, "metric_win.bin")) {
            for (int mode = 0; mode < 3; mode++) {
                float* w_arr = (float*)malloc(n * sizeof(float));
                int* w_eps = (int*)malloc(n * sizeof(int));
                int cnt = 0;
                if (w_arr && w_eps) {
                    for (int j = 0; j < n; j++)
                        if ((int)buf[j].fields[0] == mode) { w_arr[cnt] = buf[j].fields[1]; w_eps[cnt] = buf[j].episode; cnt++; }
                }
                bin_bucket_push((MetricId)(METRIC_WIN0 + mode), w_arr, cnt, (cnt > 20) ? 20 : cnt, w_eps);
                free(w_arr); free(w_eps);
            }
        } else if (strstr(name, "metric_eval.bin")) {
            /* Depth 与 Win 同为每个评估检查点一条有效曲线点。它是离散的搜索
               深度，不能跨检查点求平均，否则会产生不存在的小数深度并扭曲
               时间轴。启动读取最近 60 条原始记录，即 mode=0 的最近 20 点。 */
            float* d_arr = (float*)malloc(n * sizeof(float));
            int* d_eps = (int*)malloc(n * sizeof(int));
            int d_cnt = 0;
            if (d_arr && d_eps) {
                for (int j = 0; j < n; j++)
                    if (metric_eval_row_valid(buf[j]) && (int)buf[j].fields[3] == 0) {
                        d_arr[d_cnt] = buf[j].fields[1];
                        d_eps[d_cnt] = buf[j].episode;
                        d_cnt++;
                    }
            }
            bin_bucket_push(METRIC_DEPTH, d_arr, d_cnt, d_cnt, d_eps);
            free(d_arr); free(d_eps);
        } else if (strstr(name, "metric_rewards.bin")) {
            for (int field = 0; field < 8; field++) {
                bin_extract_and_push(buf, n, field,
                    (MetricId)(METRIC_R_LIVE2 + field), (n > HIST_MAX) ? HIST_MAX : n);
            }
        }
    }
    /* BUG-021: bin 路径 X 轴由 ep[] 实时计算，不再用 base 伪偏移 */
}

/* 只读 train.dat 自上次以来新增的行，分桶平均后推入历史（降采样） */
void sample_train_metrics(void) {
    /* bin 是权威数据源；对应文件存在时不要再把无 episode 的文本点混入同一历史。 */
    int has_bin_loss = br_find(g_bin_files, g_bin_file_count, "metric_loss.bin") >= 0;
    int has_bin_eps = br_find(g_bin_files, g_bin_file_count, "metric_epsilon_episode.bin") >= 0 ||
        br_find(g_bin_files, g_bin_file_count, "metric_epsilon.bin") >= 0;
    if (has_bin_loss && has_bin_eps) return;
    FILE* f = fopen(tui_metric_log_path("train.dat").c_str(), "rb");
    if (!f) return;
    _fseeki64(f, 0, SEEK_END);
    int64_t size = _ftelli64(f);
    if (g_train_offset < 0 || g_train_offset > size) { g_train_offset = size; fclose(f); return; }
    int64_t len = size - g_train_offset;
    if (len <= 0) { fclose(f); return; }
    if (len > TRAIN_TAIL_SIZE) { g_train_offset = size - TRAIN_TAIL_SIZE; len = TRAIN_TAIL_SIZE; }
    _fseeki64(f, g_train_offset, SEEK_SET);
    static char buf[TRAIN_TAIL_SIZE + 1];
    size_t got = fread(buf, 1, (size_t)len, f);
    buf[got] = '\0';
    /* 末行可能不完整：回退偏移，下轮补齐 */
    int64_t consumed = (int64_t)got;
    if (got > 0 && buf[got - 1] != '\n') {
        char* last_nl = strrchr(buf, '\n');
        consumed = last_nl ? (int64_t)(last_nl - buf + 1) : 0;
    }
    g_train_offset += consumed;

    static struct { long long step; float loss, q, tq, eps; } ts[MAX_BUCKETS * 64];
    int n = 0;
    char* p = buf;
    while (*p && n < (int)(sizeof(ts) / sizeof(ts[0]))) {
        char* line = p;
        while (*p && *p != '\n') p++;
        if (*p == '\n') { *p = '\0'; p++; } else break;
        /* 用空格数区分列格式：6 空格=7列(带seed), 5 空格=6列(主训练) ——
           避免 6 列行被 7 列 sscanf 误解析（%d 把 step 当 seed 导致被当作种群跳过） */
        int spaces = 0;
        for (const char* q = line; *q; q++) if (*q == ' ') spaces++;
        int seed; long long gs; float loss, q, tq, eps; int rt;
        if (spaces == 6) {
            if (sscanf(line, "%d %lld %f %f %f %f %d", &seed, &gs, &loss, &q, &tq, &eps, &rt) != 7) continue;
            if (seed >= 1) continue;   /* 跳过种群训练记录（高 eps 会污染采样） */
        } else if (spaces == 5) {
            if (sscanf(line, "%lld %f %f %f %f %d", &gs, &loss, &q, &tq, &eps, &rt) != 6) continue;
        } else continue;
        if (loss > 1.0f || eps < 0.0f || eps > 1.0f) continue;
        ts[n].step = gs; ts[n].loss = loss; ts[n].q = q; ts[n].tq = tq; ts[n].eps = eps;
        n++;
    }
    fclose(f);
    if (n == 0) return;

    int buckets = (n < MAX_BUCKETS) ? n : MAX_BUCKETS;
    int per = n / buckets;
    if (per == 0) { buckets = n; per = 1; }
    for (int b = 0; b < buckets; b++) {
        int i0 = b * per, i1 = (b == buckets - 1) ? n : (b + 1) * per;
        if (i0 >= i1) break;
        double sl = 0, sq = 0, st = 0, se = 0;
        for (int i = i0; i < i1; i++) { sl += ts[i].loss; sq += ts[i].q; st += ts[i].tq; se += ts[i].eps; }
        int m = i1 - i0;
        if (!has_bin_loss) {
            history_push(METRIC_LOSS, (float)(sl / m));
            history_push(METRIC_Q, (float)(sq / m));
            history_push(METRIC_TARGET_Q, (float)(st / m));
        }
        if (!has_bin_eps) history_push(METRIC_EPSILON, (float)(se / m));
    }
    /* 统一 x 轴口径为训练局号：Q/Loss/T-Q/Eps/t-ep 以当前局号为锚（train.dat 无局号字段，近似每采样点约一局） */
    {
        int cur_ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
        if (cur_ep > 0) {
            for (int mm = METRIC_Q; mm <= METRIC_EP_TIME; mm++) {
                if ((mm <= METRIC_TARGET_Q && has_bin_loss) || (mm == METRIC_EPSILON && has_bin_eps)) continue;
                int nn = g_hist[mm].n;
                if (nn > 0) g_hist[mm].base = cur_ep - (nn - 1);   /* 最新刻度 = 当前局号 */
            }
        }
    }
}

/* 解析 game.dat 的字节块：每行一步，done=1 的行（该局结束）step+1 = 双方总步数，直接推入历史。
   支持主训练 6 列（episode step player action reward done）与种群 7 列（seed 开头）。 */
static void parse_train_steps(char* buf, size_t len) {
    /* game.dat 的旧版本中曾出现"6 列但属于另一条训练时间线"的记录，不能只靠列数判别。
       以状态文件中的主训练局号为锚，拒绝明显超前/过旧的 episode，避免 Step 横轴跳线。 */
    int cur_main_ep = status_lines[0][0] ? atoi(status_lines[0]) : 0;
    int min_main_ep = cur_main_ep > HIST_MAX ? cur_main_ep - HIST_MAX : 1;
    buf[len] = '\0';
    char* p = buf;
    while (*p) {
        char* line = p;
        while (*p && *p != '\n') p++;
        if (*p == '\n') { *p = '\0'; p++; } else break;
        int spaces = 0;
        for (const char* q = line; *q; q++) if (*q == ' ') spaces++;
        int step, ep, player, act; float rew; int done;
        if (spaces == 5) {            /* 主训练 6 列 */
            if (sscanf(line, "%d %d %d %d %f %d", &ep, &step, &player, &act, &rew, &done) != 6) continue;
        } else if (spaces == 6) {     /* 种群 7 列：不进入主训练图表 */
            int seed;
            if (sscanf(line, "%d %d %d %d %d %f %d", &seed, &ep, &step, &player, &act, &rew, &done) != 7) continue;
            continue;
        } else continue;
        /* game.dat 的 episode 是 0-based，状态/奖励/bin 指标均是 1-based。
           过滤时仍按原始编号比较，入图时统一转换，避免 Step 比其他曲线错开一局。 */
        if (cur_main_ep > 0 && (ep + 1 < min_main_ep || ep + 1 > cur_main_ep + 100)) continue;
        if (done == 1) {
            history_push_ep(METRIC_GAME_LEN, (float)(step + 1), (float)(ep + 1));
        }
    }
}

/* 只读 game.dat 采样训练对局步数（agent 与对手 agent 自对弈，每局双方总步数）。
   每局（done=1）推一个点进 METRIC_GAME_LEN，滚动窗口显示最近局步数。
   首次扫描文件尾部最多 TG_SCAN_LIMIT 重建最近窗口；之后增量读新增。 */
void sample_train_steps(void) {
    /* 有每局二进制日志时不再扫描 game.dat。旧文本路径只作兼容回退；
       否则长日志启动扫描会重复数据并触发 HistoryBuf 大量 memmove。 */
    if (br_find(g_bin_files, g_bin_file_count, "metric_game.bin") >= 0) return;
    FILE* f = fopen(tui_metric_log_path("game.dat").c_str(), "rb");
    if (!f) return;
    _fseeki64(f, 0, SEEK_END);
    int64_t size = _ftelli64(f);
    if (size <= 0) { fclose(f); return; }

    /* 首次（offset 失效）：扫描文件尾部最多 TG_SCAN_LIMIT，重建最近步数窗口 */
    if (g_tg_offset < 0 || g_tg_offset > size) {
        int64_t start = (size > TG_SCAN_LIMIT) ? size - TG_SCAN_LIMIT : 0;
        if (start > 0) {              /* 跳到首个完整行（start 处可能是半行） */
            _fseeki64(f, start, SEEK_SET);
            char c;
            while (fread(&c, 1, 1, f) == 1 && c != '\n');
            start = _ftelli64(f);
        }
        char buf[MM_WINDOW + 1];
        int64_t pos = start;
        while (pos < size) {
            int64_t want = (size - pos > MM_WINDOW) ? MM_WINDOW : (size - pos);
            _fseeki64(f, pos, SEEK_SET);
            size_t got = fread(buf, 1, (size_t)want, f);
            if (got == 0) break;
            size_t use = got;
            if (pos + (int64_t)got < size) {   /* 非文件末尾：回退不完整尾行 */
                char* last_nl = strrchr(buf, '\n');
                if (last_nl) use = (size_t)(last_nl - buf) + 1;
            }
            parse_train_steps(buf, use);
            pos += (int64_t)use;
        }
        /* 普通 Step 图仍由全局轴只展示最近 200 局，但内存中保留长历史；
           与 Win 同格时才能覆盖最近 20 个检查点，而不是只剩右侧一小截。 */
        g_tg_offset = size;
        fclose(f);
        return;
    }

    /* 增量：读自上次偏移的新数据 */
    int64_t len = size - g_tg_offset;
    if (len <= 0) { fclose(f); return; }
    if (len > MM_WINDOW) { g_tg_offset = size - MM_WINDOW; len = MM_WINDOW; }
    _fseeki64(f, g_tg_offset, SEEK_SET);
    static char buf[MM_WINDOW + 1];
    size_t got = fread(buf, 1, (size_t)len, f);
    buf[got] = '\0';
    int64_t consumed = (int64_t)got;
    if (got > 0 && buf[got - 1] != '\n') {
        char* last_nl = strrchr(buf, '\n');
        consumed = last_nl ? (int64_t)(last_nl - buf + 1) : 0;
    }
    g_tg_offset += consumed;
    parse_train_steps(buf, (size_t)consumed);
    fclose(f);
}

void read_train_logs(void) {
    /* BUG-019 修复：读取文件尾部（不限于 BR_MAX_RECORDS），取最后 MAX_LOG_DISPLAY 条 */
    FILE* f = fopen(tui_metric_log_path("metric_loss.bin").c_str(), "rb");
    if (!f) return;
    BRHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != BR_MAGIC || hdr.field_count == 0) { fclose(f); return; }
    _fseeki64(f, 0, SEEK_END);
    int64_t fsize = _ftelli64(f);
    int rec_size = 8 + hdr.field_count * 4;
    int total = (int)((fsize - (int64_t)sizeof(BRHeader)) / rec_size);
    g_log_display_count = 0;
    if (total <= 0) { fclose(f); return; }
    int read_n = (total > MAX_LOG_DISPLAY * 2) ? MAX_LOG_DISPLAY * 2 : total;
    int64_t start_off = fsize - (int64_t)read_n * rec_size;
    if (start_off < (int64_t)sizeof(BRHeader)) start_off = (int64_t)sizeof(BRHeader);
    _fseeki64(f, start_off, SEEK_SET);
    BRRecord* buf = (BRRecord*)malloc(read_n * sizeof(BRRecord));
    if (!buf) { fclose(f); return; }
    int n = 0;
    while (n < read_n && _ftelli64(f) + rec_size <= fsize) {
        if (fread(&buf[n], rec_size, 1, f) != 1) break;
        n++;
    }
    int start = (n > MAX_LOG_DISPLAY) ? n - MAX_LOG_DISPLAY : 0;
    /* BUG-017/018: 先批量读取 epsilon 文件，按 global_step 索引（避免每条记录都全量读） */
    /* 只读最近的 epsilon。旧代码每次刷新日志页都从文件开头读取 8192 条，
       训练时间一长就无法匹配最新 loss；随后还对每条 loss 线性扫描整块
       epsilon，令日志页和图表刷新一起变慢。 */
    static BRRecord eps_buf[BR_MAX_RECORDS];
    BRFile eps_file = {};
    snprintf(eps_file.path, sizeof(eps_file.path), "%s", tui_metric_log_path("metric_epsilon.bin").c_str());
    int eps_n = br_read_tail(&eps_file, eps_buf, BR_MAX_RECORDS);
    for (int i = start; i < n && g_log_display_count < MAX_LOG_DISPLAY; i++) {
        /* metric_loss.bin: episode=global_step, step=replay_type, fields={loss,qv,tq} */
        float eps_val = 0.0f;
        int lo = 0, hi = eps_n - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (eps_buf[mid].episode < buf[i].episode) lo = mid + 1;
            else if (eps_buf[mid].episode > buf[i].episode) hi = mid - 1;
            else { eps_val = eps_buf[mid].fields[0]; break; }
        }
        snprintf(g_log_display[g_log_display_count], 160,
            "gs=%d loss=%.6f qv=%.6f tq=%.6f eps=%.4f rt=%d",
            (int)buf[i].episode, buf[i].fields[0], buf[i].fields[1], buf[i].fields[2],
            eps_val, (int)buf[i].step);
        g_log_display_count++;
    }
    free(buf);
    /* BUG: 正常路径过去漏掉 fclose(f)，每次日志刷新泄漏一个文件句柄。
       长时间运行后会同时拖慢图表/日志并导致 analyze 难以启动。 */
    fclose(f);
}

/* ---------- 命令面板 ---------- */
static int g_palette_open = 0;
static char g_palette_buf[64] = "";
static char g_palette_msg[64] = "";   /* 命令执行反馈（未知命令/状态机拒绝时显示，面板保持打开） */
static int g_palette_doc_scroll = 0;  /* 命令教程滚动偏移（滚轮） */
/* 命令教程（render_palette 展示：命令格式 + 说明） */
typedef struct { const char* cmd; const char* desc; } PaletteDoc;
static const PaletteDoc g_palette_docs[] = {
    { "course status", "查看分色课程、技能回归、复习和冷却快照" },
    { "http status / on / off", "本地 Agent 接口：查看端口或启停服务，不停止训练" },
    { "gpu start/stop/show/hide", "GPU监控：启动/停止/显示/隐藏，不影响训练" },
    { "gpu report copy", "将 GPU Analyze 当前状态和采集结果复制到剪贴板" },
    { "gpu profile modules/events <1-8>", "配置下一会话采集；GPU Analyze 有 Modules 8 按钮" },
    { "gpu profile status / off", "查看采集状态 / 取消下一次采集；当前训练不受影响" },
    { "start / start session",       "启动训练会话（拉起训练子进程并连接 IPC）" },
    { "shutdown / shutdown session", "关闭当前训练会话（训练进程退出）" },
    { "close at e <ep>",           "到指定局数保存退出；兼容 close at <ep>" },
    { "close at t HHMM",           "本机时间关闭，如 0730；已过则次日；TUI须开着" },
    { "close off",                 "取消预约；新预约替换旧预约" },
    { "tab <n>",                     "切换页面：1 Training  2 PLAYGROUND  3+ 合并功能页" },
    { "report",                      "训练摘要报告：胜负/胜率/步数趋势（结果自动复制到剪贴板）" },
    { "analyze [opts]",               "训练轨迹分析（重构）：全量历史/口径声明/异常/压缩/相关；opts: length budget=N tol=X top=N toppct=X z=X madz=X keeprate=X nodetail" },
    { "analyze-legacy",              "旧的综合分析实现（回归/残差/尾部 8192 条），保留用于回溯对比" },
    { "trajectory [opts]",           "与 analyze 相同的重构报告（别名）" },
    { "trainstat <step>",            "查看训练参数：step 为间隔训练步数（如 trainstat 200）" },
    { "eval [ep]",                   "查看评估记录：eval.dat 三种评估模式胜率（ep 可省略）" },
    { "play / p",                    "PLAYGROUND：与当前模型人机对战（模型执黑 X 先手）" },
    { "g <ep>",                      "回放自我对弈第 ep 局棋谱（空格/→ 前进，Esc 退出）" },
    { "m <ep> <mode> <c> <idx>",     "回放 Minimax 对局：mode 0~2(ε档) c=b/w idx 0~199" },
    { "theme <type>",                "折线图主题：block | round | right | curve（默认 curve）" },
    { "console",                     "切换训练/watchdog 自有窗口显示/隐藏（立即生效，不影响主程序）" },
};

/* ---------- 报告查看器（logreader 分析结果展示） ---------- */
static int g_viewer_open = 0;
static int g_viewer_scroll = 0;
static int g_viewer_line_count = 0;
static char g_viewer_lines[512][120];
static const char* g_viewer_title = "";
/* Checkpoint 列表滚动（Checkpoint 管理器页，滚轮/↑↓/PgUp/PgDn 滚动查看全部检查点） */
static int g_ckpt_scroll = 0;
/* close at <ep>：指定局数到达后自动关闭训练程序+看门狗（-1 = 未设置） */
static int g_close_at = -1;
static time_t g_close_time = 0; // GPT6-Astra: one-shot local deadline, owned by TUI.
/* 异步报告生成（analyze/report 读大文件，后台线程避免阻塞渲染闪烁） */
/* dqn_report.cpp 接口 */
void tui_report_exec(const char* cmd);
void tui_report_prepare(void);
void tui_report_request_cancel(void);
const char* tui_report_get_text(void);
const char* tui_report_get_title(void);
static HANDLE g_report_thread = NULL;
static volatile LONG g_report_running = 0;
static volatile LONG g_report_ready = 0;
static char g_report_command[256] = "";

/* ---------- 对局回放（g/m 命令） ---------- */
static int g_replay_open = 0;
static int g_replay_cur = 0;

/* ---------- PLAYGROUND 人机对战（play 命令） ---------- */
static int g_play_open = 0;
static int g_play_board[25];   /* 0空/1黑(X)用户/2白(O)AI */
static int g_play_cur = 12;    /* 光标 */
static int g_play_moves = 0;
static int g_play_over = -1;   /* -1 进行中 / 0 平 / 1 黑胜 / 2 白胜 */
static int g_play_manual_step = 0;   /* 0=AI自动跟, 1=手动按空格AI再动 */
static int g_play_ai_pending = 0;    /* AI 等待手动触发 */
static int g_play_hist[25];    /* 走棋历史（位置） */
static int g_play_hist_n = 0;
static int g_play_analysis_on = 1;   /* AI 分析显示开关 */
static PlayCand g_play_cands[5];     /* top 候选着法 */
static int g_play_cand_n = 0;
static double g_play_infer_ms = 0;
static int g_play_sidebar_tab = 0;   /* 小屏侧边栏展开项 0=MODEL 1=SETTINGS 2=HIST 3=EVAL */
static int g_play_btn = -1;          /* 操作栏按钮按下高亮：-1=无, 0=Move 1=Place 2=Undo 3=Analysis 4=Restart 5=Snapshot 6=Back */
static char g_play_ckpt[32] = "CP_0015";   /* 最近 checkpoint 名 */
static int g_play_init = 0;                /* tab 首次进入 PLAYGROUND 时加载快照 */
static int g_play_last_board[25];    /* 分析缓存匹配 */
static int g_play_last_turn = -1;
/* PLAYGROUND 面板布局 */
typedef struct { int x, y, w, h; } PlayPanel;
enum { PLY_BOARD = 0, PLY_MODEL, PLY_AI, PLY_INFO, PLY_SETTINGS, PLY_HIST, PLY_EVAL, PLY_TEST, PLY_COUNT };

/* ---------- 函数声明 ---------- */
void set_color(int fg, int bg);
void gotoxy(int x, int y);
void clear_screen(void);
int get_console_width(void);
int get_console_height(void);
int get_display_width(const char* str);
void draw_border(int x, int y, int w, int h, const char* title);

void draw_tab_bar(void);
void draw_help_bar(void);
void render_content(void);
static char g_tab_names[MAX_TABS][64];
static int g_tab_x[MAX_TABS];   /* 每 tab 屏幕 x（draw_tab_bar 填充，鼠标点击用） */
static int g_tab_w[MAX_TABS];   /* 每 tab 显示宽 */
void build_tab_names(void);
void render_merged(int page);
void render_playground(void);
void full_refresh(void);
void render_dashboard(int ox, int oy, int pw, int ph);
void render_training(void);
void render_config(int ox, int oy, int pw, int ph);
void render_model(int ox, int oy, int pw, int ph);
void render_gpu(int ox, int oy, int pw, int ph);
void render_gpu_analyze(void);
void render_logs(int ox, int oy, int pw, int ph);
void render_checkpoint(int ox, int oy, int pw, int ph);
void render_system(int ox, int oy, int pw, int ph);
void toggle_palette(void);
void render_palette(void);
void run_palette_command(const char* cmd);

int process_running(WorkerProc* wp);
int start_process(WorkerProc* wp);
void stop_process(WorkerProc* wp);
static int stop_watchdog_gracefully(void);
void state_transition(MasterState next);
void start_session(void);
void shutdown_session(void);
int pipe_connect(void);
int pipe_send(const char* cmd);
void pipe_poll(void);
void parse_status_push(const char* json);
void read_eval(void);

int window_focused(void);
void handle_mouse_click(int x, int y);
static void progress_btn_geom(int* x0, int* x1, int* y);
static int  progress_btn_click(int mx, int my);

/* ---------- BAV 彩蛋 ---------- */
void play_bav(void);

/* ============================================================================
   UI 基建（复用 Controller）
   ============================================================================ */
void set_color(int fg, int bg) {
    vt_set_color(fg, bg);
}
void gotoxy(int x, int y) {
    vt_gotoxy(x, y);
}
void clear_screen(void) {
    vt_clear_screen();
}
int get_console_width(void) {
    return vt_get_width();
}
int get_console_height(void) {
    return vt_get_height();
}
int get_display_width(const char* str) {
    int w = 0;
    while (*str) {
        if ((*str & 0x80) && (*(str + 1) & 0x80)) { w += 2; str += 2; }
        else { w += 1; str++; }
    }
    return w;
}
/* UTF-8 多字节解码（1-4 字节），返回字节数 */
static int tui_utf8_dec(const char* s, wchar_t* out) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *out = (wchar_t)c; return 1; }
    int n; wchar_t cp;
    if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
    else { *out = (wchar_t)c; return 1; }
    for (int i = 1; i < n; i++)
        if (((unsigned char)s[i] & 0xC0) == 0x80) cp = (wchar_t)((cp << 6) | ((unsigned char)s[i] & 0x3F));
    *out = cp; return n;
}
/* 精确显示宽度：按完整字符解码 + vt_char_width（CJK/全角 2 列），与渲染对齐 */
static int tui_disp_width(const char* s) {
    int col = 0;
    while (*s) { wchar_t cp; int adv = tui_utf8_dec(s, &cp); col += vt_char_width(cp); s += adv; }
    return col;
}
/* 限宽打印：从 (x,y) 打印，总显示宽度 ≤ max_w；超宽按完整 UTF-8 字符截断并补 "…"。
   绝不切开多字节字符（否则乱码）。用于窄面板（侧边栏/合并页小格/命令教程）。 */
static void print_clip(int x, int y, int max_w, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (max_w <= 0) return;
    if (tui_disp_width(buf) <= max_w) { gotoxy(x, y); printf("%s", buf); return; }
    /* 截断：保留 ≤ max_w-1 列（完整字符），末尾补 "…"（1 列） */
    int col = 0, cut = 0;
    const char* p = buf;
    while (*p) {
        wchar_t cp; int adv = tui_utf8_dec(p, &cp);
        int w = vt_char_width(cp);
        if (col + w > max_w - 1) break;
        col += w; p += adv; cut = (int)(p - buf);
    }
    if (cut == 0) { gotoxy(x, y); printf("…"); return; }   /* 极端窄：只放省略号 */
    gotoxy(x, y);
    char save = buf[cut]; buf[cut] = '\0';
    printf("%s", buf);
    buf[cut] = save;
    printf("…");
}
static void print_roll_clip(int x, int y, int max_w, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (max_w <= 0) return;
    if (tui_disp_width(buf) <= max_w) { gotoxy(x, y); ui_number_printf("%s", buf); return; }
    /* 截断：保留 ≤ max_w-1 列（完整字符），末尾补 "…"（1 列） */
    int col = 0, cut = 0;
    const char* p = buf;
    while (*p) {
        wchar_t cp; int adv = tui_utf8_dec(p, &cp);
        int w = vt_char_width(cp);
        if (col + w > max_w - 1) break;
        col += w; p += adv; cut = (int)(p - buf);
    }
    if (cut == 0) { gotoxy(x, y); ui_number_printf("…"); return; }   /* 极端窄：只放省略号 */
    gotoxy(x, y);
    char save = buf[cut]; buf[cut] = '\0';
    ui_number_printf("%s", buf);
    buf[cut] = save;
    ui_number_printf("…");
}
// All discrete controls share visual state; handlers retain their existing actions.
static void control_here(const char* label, bool active, bool toggle, int key = 0) {
    int x, y; vt_get_cursor(&x, &y);
    UiButtonStyle style = toggle ? ui_toggle_button() : ui_push_button();
    bool pressed = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) &&
        UiControlRect{x, y, ui_button_width(label, style)}.contains(g_mouse_click_x, g_mouse_click_y);
    if (key && (GetAsyncKeyState(key) & 0x8000)) pressed = true;
    ui_button(x, y, label, {active, pressed}, style);
}
static UiControlRect control_clip(int x, int y, int width, const char* text, bool active, bool toggle, int key = 0) {
    char label[256]; snprintf(label, sizeof(label), "%s", text);
    int used = 0, end = 0;
    while (label[end]) {
        wchar_t glyph; int n = tui_utf8_dec(label + end, &glyph);
        int cells = vt_char_width(glyph);
        if (used + cells > width) break;
        used += cells; end += n;
    }
    label[end] = 0; gotoxy(x, y); control_here(label, active, toggle, key);
    return {x, y, used};
}
void clear_rect(int x, int y, int w, int h) {
    vt_clear_rect(x, y, w, h);
}
void draw_border(int x, int y, int w, int h, const char* title) {
    gotoxy(x, y); printf("┌"); for (int i = 0; i < w - 2; i++) printf("─"); printf("┐");
    if (title) { gotoxy(x + 2, y); set_color(COLOR_CYAN, COLOR_BLACK); printf("%s", title); set_color(COLOR_WHITE, COLOR_BLACK); }
    for (int r = 1; r < h - 1; r++) { gotoxy(x, y + r); printf("│"); gotoxy(x + w - 1, y + r); printf("│"); }
    gotoxy(x, y + h - 1); printf("└"); for (int i = 0; i < w - 2; i++) printf("─"); printf("┘");
}

/* ---------- 滑块（Controller 式） ---------- */
typedef struct {
    int x, y, width;
    float value, min_val, max_val, step;
    const char* label;
    const char* key;
} Slider;

static Slider g_config_sliders[] = {
    { 0, 0, 30, 4, 1, 20, 1, "训练频率", "train_repeat" },
    { 0, 0, 30, 0.0004f, 0.0001f, 0.005f, 0.0001f, "学习率 x1000", "learning_rate" },
    { 0, 0, 30, 1.0f, 0.1f, 10.0f, 0.1f, "封堵缩放", "block_win_scale" },
    { 0, 0, 30, 1.0f, 0.1f, 10.0f, 0.1f, "高阶缩放", "advanced_scale" },
    { 0, 0, 30, 0.5f, 0.0f, 1.0f, 0.05f, "记忆强度", "memory_scale" },
    { 0, 0, 30, 0.3f, 0.0f, 1.0f, 0.05f, "记忆置信", "memory_confidence" },
    { 0, 0, 30, 0, 0, 1, 1, "终局质量开关", "step_outcome_enabled" },
    { 0, 0, 30, .10f, 0, .15f, .01f, "黑快胜加分", "black_fast_win_scale" },
    { 0, 0, 30, .05f, 0, .15f, .01f, "白存活减罚", "white_survival_scale" },
    { 0, 0, 30, WHITE_DRAW_LENGTH_SCALE, 0, .15f, .01f, "白和棋加分", "white_draw_length_scale" },
};
#define CONFIG_SLIDER_COUNT (sizeof(g_config_sliders) / sizeof(g_config_sliders[0]))
// GPT6-Astra: persistent viewport; wheel scrolling never changes parameter values.
static int g_config_scroll = 0, g_config_last_focus = -1;
static int g_config_x = 0, g_config_y = 0, g_config_w = 0, g_config_h = 0;
static int g_config_visible = 1, g_config_rows = 0;
static int g_config_focus = 0;  /* Config 页选中滑块 */
/* 学习率默认不是固定的 0.0004，而是由 checkpoint + 衰减计划维护。
   数值滑块仅在用户主动调整后才成为覆盖值。 */
static int g_learning_rate_auto = 1;
/* IND-01 修复：记录每个滑块是否被用户主动修改过。「应用全部」只提交这些
   dirty 字段，避免用写死的默认值覆盖 Trainer 里未被编辑的真实参数。 */
static int g_slider_dirty[CONFIG_SLIDER_COUNT] = { 0 };

void draw_slider(Slider* s, int selected) {
    gotoxy(s->x - 2, s->y);
    set_color(selected ? COLOR_CYAN : COLOR_WHITE, COLOR_BLACK);
    printf("|");
    gotoxy(s->x, s->y);
    set_color(COLOR_LIGHT_GRAY, COLOR_BLACK);
    /* 标签按显示宽度补齐到统一 12 列（最宽「学习率 x1000」），保证滑块起点对齐 */
    printf("%s:", s->label);
    int pad = 12 - get_display_width(s->label);
    if (pad < 0) pad = 0;
    for (int i = 0; i < pad; i++) printf(" ");
    float ratio = (s->value - s->min_val) / (s->max_val - s->min_val);
    if (ratio < 0) ratio = 0; if (ratio > 1) ratio = 1;
    int pos = (int)(ratio * s->width);
    gotoxy(s->x + 14, s->y);   /* label(12) + 冒号(1) + 空格(1) */
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("[");
    for (int i = 0; i < s->width; i++) {
        set_color((i == pos) ? COLOR_YELLOW : COLOR_CYAN, COLOR_BLACK);
        printf(i == pos ? "●" : "─");   /* Controller 圆点 */
    }
    printf("] ");
    if (strcmp(s->key, "learning_rate") == 0) {
        if (g_learning_rate_auto) printf("AUTO");
        else ui_number_printf("%.2f", s->value * 1000.0f);   /* ×1000 显示，与 Controller 一致 */
    }
    else if (strcmp(s->key, "train_repeat") == 0) ui_number_printf("%.0f", s->value);
    else ui_number_printf("%.2f", s->value);
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 发送滑块参数到 Trainer */
void apply_slider(int idx) {
    if (idx < 0 || idx >= CONFIG_SLIDER_COUNT) return;
    Slider* s = &g_config_sliders[idx];
    char cmd[128];
    if (strcmp(s->key, "train_repeat") == 0) snprintf(cmd, 128, "SET train_repeat %.0f", s->value);
    else if (strcmp(s->key, "learning_rate") == 0) {
        if (g_learning_rate_auto) snprintf(cmd, 128, "SET learning_rate -1");
        else snprintf(cmd, 128, "SET learning_rate %.6f", s->value);
    }
    else snprintf(cmd, 128, "SET %s %.2f", s->key, s->value);
    pipe_send(cmd);
}

/* IND-01：只发送被用户修改过的滑块，发送后清除 dirty 标记。 */
static void apply_dirty_sliders(void) {
    for (int k = 0; k < CONFIG_SLIDER_COUNT; k++)
        if (g_slider_dirty[k]) apply_slider(k);
    memset(g_slider_dirty, 0, sizeof(g_slider_dirty));
}

/* IND-01：RESET 恢复默认值后，学习率已发 -1（AUTO），其余字段标记为待应用。 */
static void reset_sliders_to_default(void) {
    float defs[] = { 4, 0.0004f, 1.0f, 1.0f, 0.5f, 0.3f, 0, .10f, .05f, WHITE_DRAW_LENGTH_SCALE };
    for (int k = 0; k < CONFIG_SLIDER_COUNT; k++) g_config_sliders[k].value = defs[k];
    g_learning_rate_auto = 1;
    pipe_send("SET learning_rate -1");   /* 恢复学习率自动衰减（不写死） */
    for (int k = 0; k < CONFIG_SLIDER_COUNT; k++) g_slider_dirty[k] = (k == 1) ? 0 : 1;
}

/* ---------- 按钮（Controller 式） ---------- */
typedef struct {
    int x, y;
    const char* text;
    const char* command;
} Button;

static UiButtonStyle config_button_style(const Button* b) {
    UiButtonStyle style = strcmp(b->command, "MEMORY_TOGGLE") == 0 ? ui_lamp_button() : ui_push_button();
    style.padding = 1;
    return style;
}
void draw_button(Button* b, int selected) {
    bool mouse_pressed = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) &&
        UiControlRect{b->x, b->y, ui_button_width(b->text, config_button_style(b))}.contains(g_mouse_click_x, g_mouse_click_y);
    bool pressed = mouse_pressed || (selected && (GetAsyncKeyState(VK_RETURN) & 0x8000));
    ui_button(b->x, b->y, b->text, {g_memory_enabled != 0, pressed}, config_button_style(b));
}
int button_hit(Button* b, int mx, int my) {
    return UiControlRect{b->x, b->y, ui_button_width(b->text, config_button_style(b))}.contains(mx, my);
}

/* Config 页按钮 */
static Button g_config_buttons[] = {
    { 0, 0, "应用全部", "APPLY" },
    { 0, 0, "记忆开关", "MEMORY_TOGGLE" },
    { 0, 0, "重置", "RESET" },
};
#define CONFIG_BTN_COUNT (sizeof(g_config_buttons) / sizeof(g_config_buttons[0]))

/* 页面边框只画一次（细粒度刷新，参考 Controller g_status_drawn） */
static int g_page_drawn[MAX_TABS] = { 0 };
static int g_last_tab = -1;
static int g_force_redraw = 0;   /* 合并页每帧重画边框（多面板共享 tab 号，防丢边框） */

/* ============================================================================
   进程管理 + 状态机
   ============================================================================ */
static DWORD g_trainer_exit_code = STILL_ACTIVE;
static int g_trainer_shutdown_state = 0;
static int g_trainer_sentinel_paused = 0;
const char* state_name(MasterState s) {
    switch (s) {
    case MASTER_IDLE: return "IDLE";
    case MASTER_STARTING: return "STARTING";
    case MASTER_IPC_READY: return "IPC_READY";
    case MASTER_WATCHDOG_READY: return "WATCHDOG_READY";
    case MASTER_RUNNING:
        if (g_trainer_sentinel_paused) return "SAFE_HOLD";
        if (g_trainer_shutdown_state == 2) return "SAVING";
        if (g_trainer_shutdown_state == 3) return "SAVE_FAILED";
        if (g_trainer_shutdown_state == 5) return "FAULT";
        return "RUNNING";
    case MASTER_SHUTDOWN: return "SHUTDOWN";
    default: return "?";
    }
}
int process_running(WorkerProc* wp) {
    if (wp->hProcess) {
        DWORD code = STILL_ACTIVE;
        if (GetExitCodeProcess(wp->hProcess, &code) && code == STILL_ACTIVE) return 1;
        if (wp == &g_trainer) g_trainer_exit_code = code;
        CloseHandle(wp->hProcess); wp->hProcess = NULL; wp->pid = 0;
        return 0;
    }
    /* BUG-039/040: 按名称查找进程，找到后打开句柄（避免后续查找到不同进程） */
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32W pe; pe.dwSize = sizeof(pe);
    DWORD found = 0;
    if (Process32FirstW(snap, &pe)) {
        do { if (_wcsicmp(pe.szExeFile, wp->name) == 0) { found = pe.th32ProcessID; break; } }
        while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
    if (found) {
        wp->pid = found;
        /* 打开进程句柄，后续调用直接用句柄检查（不再按名称查找） */
        wp->hProcess = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION |
            PROCESS_TERMINATE, FALSE, found);
        /* 某些安全策略不授予终止权时，仍可用只读句柄监视；stop_process()
           会按已锁定的 PID 再申请 PROCESS_TERMINATE 并检查结果。 */
        if (!wp->hProcess)
            wp->hProcess = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, found);
        return wp->hProcess != NULL;
    }
    return 0;
}
static int g_trainer_console = 1;   /* 训练/watchdog 终端显示开关（默认显示；命令面板 console / 进度条按钮 动态切换） */
static HWND g_trainer_console_hwnd = NULL;   /* 训练进程控制台窗口（CREATE_NEW_CONSOLE 后由 PID/标题定位） */
static HWND g_watchdog_console_hwnd = NULL;
static int  g_progress_btn_press = 0;   /* 进度条控制台按钮按下反馈 */
static DWORD g_progress_btn_time = 0;
static DWORD g_progress_btn_note_time = 0;   /* 无窗口提示（重启训练生效）显示计时 */

/* 控制台窗口查找：类名 ConsoleWindowClass，且窗口进程 PID 匹配或标题含 exe 名 */
typedef struct { DWORD pid; const wchar_t* exe; HWND hwnd; } ConsoleFind;
static BOOL CALLBACK find_console_enum(HWND h, LPARAM lp) {
    ConsoleFind* cf = (ConsoleFind*)lp;
    wchar_t cls[64];
    if (!GetClassNameW(h, cls, 64) || _wcsicmp(cls, L"ConsoleWindowClass") != 0) return TRUE;
    DWORD wpid = 0;
    GetWindowThreadProcessId(h, &wpid);
    if (wpid == cf->pid) { cf->hwnd = h; return FALSE; }
    if (cf->exe) {
        wchar_t title[260] = L"";
        if (GetWindowTextW(h, title, 260) && wcsstr(title, cf->exe)) { cf->hwnd = h; return FALSE; }
    }
    return TRUE;
}
static HWND find_process_console(DWORD pid, const wchar_t* exe) {
    ConsoleFind cf = { pid, exe, NULL };
    EnumWindows(find_console_enum, (LPARAM)&cf);
    return cf.hwnd;
}
/* 取某 worker 的控制台窗口句柄（缓存失效则重找） */
static HWND worker_console_hwnd(WorkerProc* wp) {
    HWND* cached = (wp == &g_trainer) ? &g_trainer_console_hwnd : &g_watchdog_console_hwnd;
    if (*cached && IsWindow(*cached)) return *cached;
    *cached = find_process_console(wp->pid, wp->name);
    return *cached;
}

struct EnvironmentNameLess {
    bool operator()(const std::wstring& a, const std::wstring& b) const {
        return _wcsicmp(a.c_str(), b.c_str()) < 0;
    }
};

static bool build_gpu_profile_environment(std::vector<wchar_t>& block) {
    if (g_gpu_profile_session_batches <= 0 || !g_gpu_profile_session_dir[0]) return false;
    LPWCH current = GetEnvironmentStringsW();
    if (!current) return false;
    std::map<std::wstring, std::wstring, EnvironmentNameLess> entries;
    for (const wchar_t* p = current; *p; p += wcslen(p) + 1) {
        const wchar_t* separator = wcschr(p + (p[0] == L'=' ? 1 : 0), L'=');
        if (!separator) continue;
        entries[std::wstring(p, separator)] = separator + 1;
    }
    FreeEnvironmentStringsW(current);
    entries[L"DQN_GPU_PROFILE_MODE"] = g_gpu_profile_session_mode == 2 ? L"deep" : L"modules";
    wchar_t count[16];
    swprintf_s(count, L"%d", g_gpu_profile_session_batches);
    entries[L"DQN_GPU_PROFILE_BATCHES"] = count;
    entries[L"DQN_GPU_PROFILE_DIR"] = g_gpu_profile_session_dir;

    size_t total = 1;
    for (const auto& item : entries) total += item.first.size() + item.second.size() + 2;
    block.clear();
    block.reserve(total);
    for (const auto& item : entries) {
        block.insert(block.end(), item.first.begin(), item.first.end());
        block.push_back(L'=');
        block.insert(block.end(), item.second.begin(), item.second.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return true;
}

static bool prepare_gpu_profile_session() {
    if (g_gpu_profile_arm_batches <= 0) return true;
    CreateDirectoryW(L"logs", nullptr);
    SYSTEMTIME now{};
    GetLocalTime(&now);
    swprintf_s(g_gpu_profile_session_dir, L"logs\\gpu_capture_%04u%02u%02u_%02u%02u%02u_%lu",
        now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
        GetCurrentProcessId());
    if (!CreateDirectoryW(g_gpu_profile_session_dir, nullptr)) {
        g_gpu_profile_session_dir[0] = L'\0';
        return false;
    }
    g_gpu_profile_session_batches = g_gpu_profile_arm_batches;
    g_gpu_profile_session_mode = g_gpu_profile_arm_mode;
    g_gpu_profile_arm_batches = 0;
    return true;
}

static int count_gpu_capture_files(const wchar_t* directory, int mode) {
    if (!directory || !directory[0]) return 0;
    wchar_t pattern[MAX_PATH + 32];
    if (swprintf_s(pattern, mode == 2 ? L"%s\\gpu_profile_*.csv" : L"%s\\gpu_modules_*.csv", directory) < 0) return 0;
    WIN32_FIND_DATAW data{};
    HANDLE find = FindFirstFileW(pattern, &data);
    if (find == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    do { if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) ++count; }
    while (FindNextFileW(find, &data));
    FindClose(find);
    return count;
}

static const char* gpu_profile_mode_name(int mode) {
    return mode == 2 ? "events" : "modules";
}

static void cancel_gpu_profile_session(bool rearm) {
    if (rearm && g_gpu_profile_session_batches > 0)
        g_gpu_profile_arm_batches = g_gpu_profile_session_batches;
    if (!rearm && g_gpu_profile_session_batches > 0 && g_gpu_profile_session_dir[0]) {
        wcscpy_s(g_gpu_profile_last_dir, g_gpu_profile_session_dir);
        g_gpu_profile_last_files = count_gpu_capture_files(g_gpu_profile_last_dir, g_gpu_profile_session_mode);
        g_gpu_profile_last_batches = g_gpu_profile_session_batches;
        g_gpu_profile_last_mode = g_gpu_profile_session_mode;
    }
    g_gpu_profile_session_batches = 0;
    g_gpu_profile_session_dir[0] = L'\0';
}

int start_process(WorkerProc* wp) {
    if (process_running(wp)) return 1;   /* [console 功能暂禁用：不再查找/缓存窗口] */
    wchar_t cur[MAX_PATH];
    GetCurrentDirectoryW(MAX_PATH, cur);
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%s\\%s", cur, wp->name);
    wchar_t cmd_line[MAX_PATH + 8];
    swprintf(cmd_line, MAX_PATH + 8, L"\"%s\"", path);
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    /* [console 功能暂禁用：恢复后台无窗口启动（CREATE_NO_WINDOW）] */
    DWORD flags = CREATE_NO_WINDOW;
    std::vector<wchar_t> profile_environment;
    LPVOID environment = NULL;
    if ((wp == &g_trainer || wp == &g_watchdog) && g_gpu_profile_session_batches > 0) {
        if (!build_gpu_profile_environment(profile_environment)) return 0;
        environment = profile_environment.data();
        flags |= CREATE_UNICODE_ENVIRONMENT;
    }
    if (!CreateProcessW(NULL, cmd_line, NULL, NULL, FALSE, flags, environment, cur, &si, &pi)) return 0;
    wp->hProcess = pi.hProcess;
    wp->pid = pi.dwProcessId;
    CloseHandle(pi.hThread);
    /* [console 功能暂禁用]
    Sleep(100);
    HWND chwnd = find_process_console(wp->pid, wp->name);
    if (wp == &g_trainer) g_trainer_console_hwnd = chwnd;
    else g_watchdog_console_hwnd = chwnd;
    if (chwnd) ShowWindow(chwnd, g_trainer_console ? SW_SHOW : SW_HIDE);
    */
    return 1;
}
/* 旧 Trainer 管道只读时，让同会话 Watchdog 转发白名单内的窗口命令。 */
static int request_console_via_watchdog(const char* command) {
    if (g_pipe != INVALID_HANDLE_VALUE) CloseHandle(g_pipe);
    g_pipe = INVALID_HANDLE_VALUE;
    g_pipe_readonly = 0;
    remove("wd_trainer_result.txt");
    remove("wd_trainer_command.txt");
    FILE* f = fopen("wd_trainer_command.tmp", "wb");
    if (!f) return 0;
    fprintf(f, "%s\n", command);
    fclose(f);
    if (!MoveFileExA("wd_trainer_command.tmp", "wd_trainer_command.txt",
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return 0;
    int applied = 0;
    for (int wait = 0; wait < 120; ++wait) {
        FILE* result = fopen("wd_trainer_result.txt", "rb");
        if (result) {
            char line[96] = "";
            fgets(line, sizeof(line), result);
            fclose(result);
            applied = strncmp(line, "OK", 2) == 0;
            break;
        }
        win_pump_messages();
        Sleep(50);
    }
    remove("wd_trainer_result.txt");
    pipe_connect();
    return applied;
}

/* 切换训练/watchdog 自有窗口显示/隐藏（立即应用，不影响主程序）；返回是否生效。
   训练走 pipe 消息（训练自己 ShowWindow 自有窗口），watchdog 写 wd_cmd.txt 由它轮询执行。 */
static int toggle_console_now(void) {
    g_trainer_console = 1 - g_trainer_console;
    int applied = 0;
    HWND analyzer_window = FindWindowW(L"DQNGpuAnalyzerWindow", nullptr);
    if (analyzer_window) { PostMessageW(analyzer_window, WM_APP + 41, g_trainer_console ? 1 : 2, 0); applied = 1; }
    const char* cmd = g_trainer_console ? "CONSOLE_SHOW" : "CONSOLE_HIDE";
    if (process_running(&g_trainer)) {
        if (g_pipe != INVALID_HANDLE_VALUE && !g_pipe_readonly && pipe_send(cmd))
            applied = 1;   /* 新 Trainer：直接控制 */
        else if (process_running(&g_watchdog) && request_console_via_watchdog(cmd))
            applied = 1;   /* 已运行旧 Trainer：由同会话 Watchdog 转发 */
    }
    if (process_running(&g_watchdog)) {
        /* BUG-038: 先写临时文件再 rename（NTFS 原子操作），避免 watchdog 读到半写文件 */
        FILE* f = fopen("wd_cmd.tmp", "w");
        if (f) { fprintf(f, "%s\n", g_trainer_console ? "SHOW" : "HIDE"); fclose(f); }
        remove("wd_cmd.txt");
        if (rename("wd_cmd.tmp", "wd_cmd.txt") == 0) applied = 1;
    }
    return applied;
}
void stop_process(WorkerProc* wp) {
    DWORD pid = wp->pid;
    if (wp->hProcess) {
        if (!TerminateProcess(wp->hProcess, 0) && pid) {
            HANDLE kill = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
            if (kill) {
                TerminateProcess(kill, 0);
                WaitForSingleObject(kill, 3000);
                CloseHandle(kill);
            }
        } else {
            WaitForSingleObject(wp->hProcess, 3000);
        }
        CloseHandle(wp->hProcess); wp->hProcess = NULL; wp->pid = 0;
    } else if (wp->pid) {
        HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, wp->pid);
        if (h) { TerminateProcess(h, 0); WaitForSingleObject(h, 3000); CloseHandle(h); }
        wp->pid = 0;
    }
}

/* Watchdog 有自己的关闭协议；关闭它不触碰 Trainer，也不依赖强制终止。 */
static int request_watchdog_stop(void) {
    remove("watchdog_shutdown.ack");
    FILE* f = fopen("watchdog_shutdown.request.tmp", "wb");
    if (!f) return 0;
    fputs("STOP\n", f);
    fflush(f);
    fclose(f);
    if (!MoveFileExA("watchdog_shutdown.request.tmp", "watchdog_shutdown.request",
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return 0;
    return 1;
}
static int stop_watchdog_gracefully(void) {
    if (!process_running(&g_watchdog)) return 1;
    if (!request_watchdog_stop()) return 0;
    for (int i = 0; i < 50; ++i) {
        if (!process_running(&g_watchdog)) break;
        win_pump_messages();
        Sleep(100);
    }
    if (process_running(&g_watchdog)) return 0;
    if (g_watchdog.hProcess) {
        CloseHandle(g_watchdog.hProcess);
        g_watchdog.hProcess = NULL;
    }
    g_watchdog.pid = 0;
    return 1;
}

/* ============================================================================
   系统监控（CPU/内存/GPU，Windows 性能计数器 + 系统 API）
   ============================================================================ */
static DWORD64 g_cpu_idle = 0, g_cpu_kernel = 0, g_cpu_user = 0;

/* ---------- 遥测（后台线程采样，UI 只读缓存，绝不阻塞渲染） ---------- */
typedef struct {
    volatile float cpu_usage;      /* % */
    volatile float cpu_temp;       /* °C，无 backend 时 -1 */
    volatile int    cpu_temp_valid;
    volatile float gpu_usage;      /* % */
    volatile float gpu_copy;       /* Copy Engine %, -1 if unavailable */
    volatile float gpu_temp;       /* °C，无 backend 时 -1 */
    volatile int    gpu_temp_valid;
    volatile float gpu_mem_used;   /* GB */
    volatile float gpu_mem_total;  /* GB */
    volatile float ram_used;       /* GB */
    volatile float ram_total;      /* GB */
    volatile float cpu_max_core;   /* % (per-processor 最大值) */
    volatile int    ready;
} Telemetry;
static Telemetry g_tel = { 0 };
static HANDLE g_tel_thread = NULL;
static volatile int g_tel_running = 0;
static PDH_HQUERY g_pdh_query = NULL;
static PDH_HCOUNTER g_pdh_gpu_util = NULL;
static PDH_HCOUNTER g_pdh_gpu_copy = NULL;  /* Copy Engine 利用率 */
static PDH_HCOUNTER g_pdh_gpu_mem = NULL;
static PDH_HCOUNTER g_pdh_gpu_mem_total = NULL;
static PDH_HCOUNTER g_pdh_cpu_util = NULL;  /* per-logical-processor 利用率 */

static void close_pdh(void) {
    if (g_pdh_query) PdhCloseQuery(g_pdh_query);
    g_pdh_query = NULL;
    g_pdh_gpu_util = g_pdh_gpu_copy = g_pdh_gpu_mem = g_pdh_gpu_mem_total = NULL;
    g_pdh_cpu_util = NULL;
}

/* CPU 使用率采样（两次 GetSystemTimes 差值，供后台遥测线程调用） */
static float cpu_sample_now(void) {
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) return 0;
    DWORD64 i = ((DWORD64)idle.dwHighDateTime << 32) | idle.dwLowDateTime;
    DWORD64 k = ((DWORD64)kernel.dwHighDateTime << 32) | kernel.dwLowDateTime;
    DWORD64 u = ((DWORD64)user.dwHighDateTime << 32) | user.dwLowDateTime;
    if (g_cpu_idle) {
        DWORD64 di = i - g_cpu_idle;
        DWORD64 dk = (k - g_cpu_kernel) - di;
        DWORD64 du = u - g_cpu_user;
        DWORD64 total = dk + du;
        g_cpu_idle = i; g_cpu_kernel = k; g_cpu_user = u;
        return (total + di > 0) ? (float)(total) / (float)(total + di) * 100.0f : 0;
    }
    g_cpu_idle = i; g_cpu_kernel = k; g_cpu_user = u;
    return 0;
}
/* UI 只读缓存，不触碰硬件 */
float get_cpu_usage(void) { return g_tel.cpu_usage; }

/* 内存使用采样（GB，供后台遥测线程调用） */
static void mem_sample_now(float* used_gb, float* total_gb) {
    MEMORYSTATUSEX ms = { sizeof(ms) };
    if (!GlobalMemoryStatusEx(&ms)) { *used_gb = *total_gb = 0; return; }
    *total_gb = (float)(ms.ullTotalPhys / (1024.0 * 1024 * 1024));
    *used_gb = (float)((ms.ullTotalPhys - ms.ullAvailPhys) / (1024.0 * 1024 * 1024));
}
/* UI 只读缓存 */
void get_mem_usage(float* used_gb, float* total_gb) {
    *used_gb = g_tel.ram_used;
    *total_gb = g_tel.ram_total;
}

/* 初始化 PDH（GPU Engine 计数器） */
int init_pdh(void) {
    if (g_pdh_query) return 1;
    if (PdhOpenQuery(NULL, 0, &g_pdh_query) != ERROR_SUCCESS) return 0;
    PdhAddEnglishCounterA(g_pdh_query, "\\GPU Engine(*)\\Utilization Percentage", 0, &g_pdh_gpu_util);
    PdhAddEnglishCounterA(g_pdh_query, "\\GPU Engine(*)\\Utilization Percentage", 0, &g_pdh_gpu_copy);
    PdhAddEnglishCounterA(g_pdh_query, "\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &g_pdh_gpu_mem);
    PdhAddEnglishCounterA(g_pdh_query, "\\GPU Adapter Memory(*)\\Total Dedicated Usage", 0, &g_pdh_gpu_mem_total);
    /* per-logical-processor 利用率 (取所有逻辑处理器的最大值) */
    PdhAddEnglishCounterA(g_pdh_query, "\\Processor Information(*)\\% Processor Time", 0, &g_pdh_cpu_util);
    PdhCollectQueryData(g_pdh_query);
    return 1;
}

/* PDH 多实例计数器求和（显存等总量用） */
static double pdh_sum_counter(PDH_HCOUNTER counter) {
    DWORD size = 0, count = 0;
    PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, NULL);
    if (size == 0) return 0;
    PDH_FMT_COUNTERVALUE_ITEM_A* items = (PDH_FMT_COUNTERVALUE_ITEM_A*)malloc(size);
    if (!items) return 0;
    double sum = 0;
    if (PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < count; i++) sum += items[i].FmtValue.doubleValue;
    }
    free(items);
    return sum;
}
/* PDH 多实例计数器取最大值（GPU 引擎利用率：各引擎(3D/Copy/Compute)是独立百分比，
   求和会超 100%（如 120/150），取最忙引擎的值作为整体占用率，不超 100%。忽略负值/无效采样） */
static double pdh_max_counter(PDH_HCOUNTER counter) {
    DWORD size = 0, count = 0;
    PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, NULL);
    if (size == 0) return 0;
    PDH_FMT_COUNTERVALUE_ITEM_A* items = (PDH_FMT_COUNTERVALUE_ITEM_A*)malloc(size);
    if (!items) return 0;
    double mx = 0;
    if (PdhGetFormattedCounterArrayA(counter, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < count; i++)
            if (items[i].FmtValue.doubleValue > mx) mx = items[i].FmtValue.doubleValue;
    }
    free(items);
    return mx;
}

/* Copy Engine 利用率采样 (遍历所有 GPU Engine 实例, 取 engtype_Copy 的最大值) */
static float gpu_copy_sample_now(void) {
    if (!g_pdh_query || !g_pdh_gpu_copy) return -1;
    DWORD size = 0, count = 0;
    PdhGetFormattedCounterArrayA(g_pdh_gpu_copy, PDH_FMT_DOUBLE, &size, &count, NULL);
    if (size == 0) return -1;
    PDH_FMT_COUNTERVALUE_ITEM_A* items = (PDH_FMT_COUNTERVALUE_ITEM_A*)malloc(size);
    if (!items) return -1;
    float mx = 0;
    int found = 0;
    if (PdhGetFormattedCounterArrayA(g_pdh_gpu_copy, PDH_FMT_DOUBLE, &size, &count, items) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < count; i++) {
            /* 只取 Copy 引擎实例 */
            if (items[i].szName && strstr(items[i].szName, "engtype_Copy")) {
                if (items[i].FmtValue.doubleValue > mx) mx = (float)items[i].FmtValue.doubleValue;
                found = 1;
            }
        }
    }
    free(items);
    return found ? mx : -1;
}

/* CPU Max Logical Core 利用率 (遍历所有逻辑处理器, 取最大值) */
static float cpu_max_core_sample_now(void) {
    return analyzer_cpu::sample(g_pdh_cpu_util);
}

/* GPU 利用率采样（PDH 两次采样差值，间隔 250ms，供后台遥测线程调用） */
static float gpu_sample_now(void) {
    if (!g_pdh_query && !init_pdh()) return 0;
    PdhCollectQueryData(g_pdh_query);
    Sleep(250);
    PdhCollectQueryData(g_pdh_query);
    return (float)pdh_max_counter(g_pdh_gpu_util);   /* 取最忙引擎，避免多引擎求和超 100% */
}
/* UI 只读缓存，不触碰硬件 */
float get_gpu_usage(void) { return g_tel.gpu_usage; }
float get_gpu_copy(void) { return g_tel.gpu_copy; }
float get_cpu_max_core(void) { return g_tel.cpu_max_core; }

/* GPU 显存使用（GB，后台遥测线程缓存） */
float get_gpu_mem_used(void) { return g_tel.gpu_mem_used; }
float get_gpu_mem_total(void) { return g_tel.gpu_mem_total; }

// GPT6-Astra: sensor access belongs to Analyzer; remove unsafe legacy ADL ABI.
static float gpu_temp_sample(void) { return -1.0f; }
/* UI 只读缓存（不可用时 -1，UI 显示 N/A） */
float get_gpu_temp(void) {
    return g_tel.gpu_temp_valid ? g_tel.gpu_temp : -1.0f;
}

/* ---------- 遥测后台线程（每 ~1s 采样一次，UI 只读 g_tel，绝不阻塞渲染） ---------- */
static unsigned __stdcall telemetry_thread(void* arg) {
    cpu_sample_now();              /* 建立 CPU 差值基准 */
    init_pdh();
    if (g_pdh_query) PdhCollectQueryData(g_pdh_query);
    DWORD last_pdh_recycle = GetTickCount();
    while (g_tel_running) {
        Sleep(1000);
        g_tel.cpu_usage = cpu_sample_now();
        float ru, rt;
        mem_sample_now(&ru, &rt);
        g_tel.ram_used = ru; g_tel.ram_total = rt;
        if (g_pdh_query) {
            g_tel.gpu_usage = gpu_sample_now();   /* 内部 Sleep(250) 做两次采样差值 */
            g_tel.gpu_copy = gpu_copy_sample_now();
            g_tel.cpu_max_core = cpu_max_core_sample_now();
            PdhCollectQueryData(g_pdh_query);
            g_tel.gpu_mem_used = (float)pdh_sum_counter(g_pdh_gpu_mem) / (1024.0 * 1024 * 1024);
            g_tel.gpu_mem_total = (float)pdh_sum_counter(g_pdh_gpu_mem_total) / (1024.0 * 1024 * 1024);
        }
        /* GPU Engine(*) 是动态通配实例。PDH 会缓存已经消失的进程/引擎
           实例及其内核句柄；长时间运行后句柄持续增长。周期回收整个
           query 可释放缓存，下一轮自动重建，遥测只会短暂缺一帧。 */
        if (GetTickCount() - last_pdh_recycle >= 30000) {
            close_pdh();
            init_pdh();
            if (g_pdh_query) PdhCollectQueryData(g_pdh_query);
            last_pdh_recycle = GetTickCount();
        }
        /* Legacy local temperature stays unavailable; Analyzer owns ADLX sampling. */
        float t = gpu_temp_sample();
        if (t > 0) { g_tel.gpu_temp = t; g_tel.gpu_temp_valid = 1; }
        else { g_tel.gpu_temp = -1; g_tel.gpu_temp_valid = 0; }
        /* CPU 温度：暂无 Windows 原生可靠 backend，保持 N/A */
        g_tel.cpu_temp = -1; g_tel.cpu_temp_valid = 0;
        g_tel.ready = 1;
    }
    return 0;
}
void start_telemetry(void) {
    if (g_tel_thread) return;
    g_tel_running = 1;
    /* 尝试连接 GPU_Analyzer 共享内存 (失败不影响 TUI 自有遥测) */
    gpu_telem_client_open();
    uintptr_t thread_handle = _beginthreadex(NULL, 0, telemetry_thread, NULL, 0, NULL);
    g_tel_thread = thread_handle ? (HANDLE)thread_handle : NULL;
    if (!g_tel_thread) g_tel_running = 0;
}
void stop_telemetry(void) {
    g_tel_running = 0;
    if (g_tel_thread) {
        WaitForSingleObject(g_tel_thread, 3000);
        CloseHandle(g_tel_thread);
        g_tel_thread = NULL;
    }
    gpu_telem_client_close();
    close_pdh();
}
static tui_http::Server g_http_server;
static std::mutex g_http_action_lock;
static int g_http_pending_action=0;
static bool g_http_action_running=false;
static uint64_t g_http_action_id=0;
static std::string g_http_action_status="{\"state\":\"idle\"}";
static tui_http::Response http_controller(const std::string& command){
    if(command=="SESSION_ACTION_STATUS"){std::lock_guard<std::mutex> lock(g_http_action_lock);return {200,g_http_action_status};}
    if(command=="SESSION_START" || command=="SESSION_STOP"){
        std::lock_guard<std::mutex> lock(g_http_action_lock);
        if(g_http_pending_action || g_http_action_running)return {409,"A session operation is already pending."};
        g_http_pending_action=command=="SESSION_START"?1:2;++g_http_action_id;
        g_http_action_status="{\"state\":\"queued\",\"operation_id\":"+std::to_string(g_http_action_id)+"}";
        return {202,g_http_action_status};
    }
    return tui_http::trainer_command(command);
}
static void http_start(){
    int enabled=1,preferred=17865;FILE* f=fopen("tui_http_config.txt","r");
    if(f){char line[128];while(fgets(line,sizeof(line),f)){int v;if(sscanf(line,"enabled=%d",&v)==1)enabled=v!=0;if(sscanf(line,"port=%d",&v)==1 && v>=1024 && v<=65526)preferred=v;}fclose(f);}
    if(enabled)g_http_server.start(http_controller,preferred);
}
static void http_tick(){
    int action=0;uint64_t id=0;{std::lock_guard<std::mutex> lock(g_http_action_lock);action=g_http_pending_action;
        if(!action)return;g_http_pending_action=0;g_http_action_running=true;id=g_http_action_id;}
    if(action==1)start_session();else shutdown_session();
    {std::lock_guard<std::mutex> lock(g_http_action_lock);g_http_action_running=false;
        g_http_action_status="{\"state\":\"dispatched\",\"operation_id\":"+std::to_string(id)+",\"note\":\"Query Trainer status to verify completion\"}";}
    need_content=1;
}
static std::vector<std::string> course_saved_lines(){
    static ULONGLONG last=0;static std::vector<std::string> cached;
    if(last && GetTickCount64()-last<1000)return cached;last=GetTickCount64();
    std::string json;FILE* f=fopen("logs\\competency_status.json","rb");
    if(f){char b[8192];size_t n=fread(b,1,sizeof(b),f);if(n<sizeof(b))json.assign(b,n);fclose(f);}
    cached=competency_view::lines(json);return cached;
}
void state_transition(MasterState next) { g_state = next; }
void start_session(void) {
    if (g_state != MASTER_IDLE && g_state != MASTER_SHUTDOWN) return;
    if (!prepare_gpu_profile_session()) {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "GPU采集目录创建失败，已取消启动");
        return;
    }
    g_start_watchdog_for_session = g_dashboard_workers.watchdog_enabled != 0;
    g_trainer_exit_code = STILL_ACTIVE;
    g_trainer_shutdown_state = 0;
    /* 新会话显式解除主动停机锁。Watchdog 只有看到锁已解除，才允许拉起 Trainer。 */
    remove(INTENTIONAL_SHUTDOWN_FILE);
    /* A user-started session is the acknowledgement boundary for a prior
       protected fault stop.  The detailed history remains in logs. */
    remove(TRAINING_FAULT_FILE);
    g_session_start = GetTickCount();
    g_session_ep_start = atoi(status_lines[0][0] ? status_lines[0] : "0");
    g_train_offset = -1;   /* 训练重启后重置采样偏移，避免与旧数据错位 */
    state_transition(MASTER_STARTING);
    if (g_guard_user_enabled) start_process(&g_vindicator);   /* 常驻日志巡检器，无窗口后台运行 */
    if (!start_process(&g_trainer)) {
        cancel_gpu_profile_session(true);
        state_transition(MASTER_IDLE);
        return;
    }
}
void shutdown_session(void) {
    if (g_state == MASTER_IDLE || g_state == MASTER_SHUTDOWN) return;
    state_transition(MASTER_SHUTDOWN);
    /* 必须先落主动停机标记，再停止 Watchdog/Trainer。即使 TUI 是后来附加、
       Watchdog 正处于退避睡眠或终止失败，它醒来后也不会重新拉起训练。 */
    {
        FILE* marker = fopen("intentional_shutdown.tmp", "wb");
        if (marker) {
            fputs("user requested shutdown\n", marker);
            fflush(marker);
            fclose(marker);
            MoveFileExA("intentional_shutdown.tmp", INTENTIONAL_SHUTDOWN_FILE,
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        }
    }
    /* 先停维护器，避免 Trainer 优雅退出后的数百毫秒内被 watchdog 当作
       崩溃重新拉起，形成 shutdown 与 restart 竞态。 */
    if (!stop_watchdog_gracefully()) {
        MessageBoxW(NULL,
            L"看门狗未在 5 秒内确认退出。已保留其进程，主动停机标记仍有效；不会强制终止。",
            L"看门狗关闭超时", MB_OK | MB_ICONWARNING);
    }
    /* BUG-008: 先通过 IPC 发 SAVE_AND_EXIT 让训练程序优雅退出（保存检查点），等待最多 30 秒 */
    if (g_pipe != INVALID_HANDLE_VALUE && process_running(&g_trainer)) {
        g_trainer_exit_code = STILL_ACTIVE;
        g_trainer_shutdown_state = 0;
        const char* cmd = "SAVE_AND_EXIT";
        DWORD written;
        WriteFile(g_pipe, cmd, (DWORD)strlen(cmd), &written, NULL);
        /* 等待训练程序自行退出（最多 30 秒） */
        for (int w = 0; w < 300 && process_running(&g_trainer); w++) {
            pipe_poll();
            if (g_trainer_shutdown_state == 3 || g_trainer_shutdown_state == 5) break;
            win_pump_messages();  /* 保存大 checkpoint 时 TUI 仍可移动、重绘和关闭 */
            Sleep(100);
        }
    }
    // A timeout/disconnected pipe is not permission to discard unsaved training.
    if (process_running(&g_trainer)) {
        state_transition(MASTER_STARTING);
        MessageBoxW(NULL, g_trainer_shutdown_state == 3 ?
            L"存档失败，训练已暂停，模型仍保留在内存中。请检查磁盘空间、权限和文件占用，然后再次点击关闭重试。未强制结束训练。" :
            L"训练程序尚未确认保存退出，未强制结束。请稍后重试关闭，或检查训练窗口。保活已停止。",
            L"等待训练保存", MB_OK | MB_ICONWARNING);
        return;
    }
    stop_process(&g_vindicator);
    if (g_trainer_exit_code != 0 && g_trainer_shutdown_state != 4)
        MessageBoxW(NULL, L"训练程序已退出，但未确认本次进度成功保存。请检查训练错误；现有旧存档未删除。",
            L"退出未确认保存", MB_OK | MB_ICONERROR);
    stop_process(&g_trainer);
    if (g_pipe != INVALID_HANDLE_VALUE) { CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE; }
    cancel_gpu_profile_session(false);
    state_transition(MASTER_IDLE);
}

static WorkerProc* dashboard_worker_process(int id) {
    return id == DASH_TRAINER ? &g_trainer : (id == DASH_WATCHDOG ? &g_watchdog : &g_vindicator);
}
struct DashboardWorkerBackend {
    bool running(int id) { return process_running(dashboard_worker_process(id)) != 0; }
    bool protect_trainer_stop() {
        FILE* f = fopen("intentional_shutdown.tmp", "wb");
        if (!f) return false;
        bool ok = fputs("dashboard trainer stop\n", f) >= 0;
        if (fflush(f) != 0 || _commit(_fileno(f)) != 0) ok = false;
        if (fclose(f) != 0) ok = false;
        return ok && MoveFileExA("intentional_shutdown.tmp", INTENTIONAL_SHUTDOWN_FILE,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    }
    bool request_trainer_exit() {
        if (!pipe_connect() || g_pipe_readonly || !pipe_send("SAVE_AND_EXIT")) return false;
        g_trainer_exit_code = STILL_ACTIVE;
        g_trainer_shutdown_state = 1;
        return true;
    }
    bool start(int id) {
        if (id == DASH_TRAINER) {
            if (remove(INTENTIONAL_SHUTDOWN_FILE) != 0 && errno != ENOENT) return false;
            if (remove(TRAINING_FAULT_FILE) != 0 && errno != ENOENT) return false;
            if (!prepare_gpu_profile_session()) return false;
            g_trainer_exit_code = STILL_ACTIVE;
            g_trainer_shutdown_state = 0;
            g_session_start = GetTickCount();
            g_session_ep_start = atoi(status_lines[0]);
            g_train_offset = -1;
            g_start_watchdog_for_session = false; // This switch starts only Trainer.
        } else if (id == DASH_WATCHDOG) {
            // Running a watchdog while Trainer is stopped means standby, not
            // an implicit request to launch Trainer as a second side effect.
            if (!running(DASH_TRAINER) && !protect_trainer_stop()) return false;
            if (remove("watchdog_shutdown.request") != 0 && errno != ENOENT) return false;
        }
        if (!start_process(dashboard_worker_process(id))) {
            if (id == DASH_TRAINER) {
                protect_trainer_stop();
                cancel_gpu_profile_session(true);
            }
            return false;
        }
        if (id == DASH_TRAINER) state_transition(MASTER_STARTING);
        return true;
    }
    bool stop(int id) {
        if (id == DASH_WATCHDOG) return request_watchdog_stop() != 0;
        if (id != DASH_LOGGUARD) return false;
        stop_process(&g_vindicator);
        return !running(DASH_LOGGUARD);
    }
};
static void dashboard_worker_click(int id) {
    DashboardWorkerBackend backend;
    if (!dashboard_toggle_worker(g_dashboard_workers, id, backend))
        MessageBoxW(NULL, L"操作未完成。请检查控制管道或文件权限后重试；Trainer 不会被强制结束。",
            L"程序开关", MB_OK | MB_ICONWARNING);
    need_content = 1;
}

/* ============================================================================
   管道
   ============================================================================ */
int pipe_connect(void) {
    if (g_pipe != INVALID_HANDLE_VALUE) return 1;
    g_pipe = CreateFileA(PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (g_pipe == INVALID_HANDLE_VALUE && GetLastError() == ERROR_ACCESS_DENIED) {
        /* 旧 Trainer 使用默认管道 ACL：其他登录账户通常只有读取权。
           先以只读方式接入，至少保证实时状态/图表可用；新版 Trainer
           重启后会显式授予交互用户读写权，届时自动恢复完整控制。 */
        g_pipe = CreateFileA(PIPE_NAME, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (g_pipe != INVALID_HANDLE_VALUE) g_pipe_readonly = 1;
    }
    if (g_pipe != INVALID_HANDLE_VALUE) {
        /* 服务端按消息写 JSON；客户端显式切到消息读取模式，避免多个 JSON
           在字节模式下粘包后总是解析到较旧的第一条状态。 */
        DWORD mode = PIPE_READMODE_MESSAGE;
        if (!g_pipe_readonly && !SetNamedPipeHandleState(g_pipe, &mode, NULL, NULL)) {
            CloseHandle(g_pipe);
            g_pipe = INVALID_HANDLE_VALUE;
            g_pipe_readonly = 0;
        }
    }
    return (g_pipe != INVALID_HANDLE_VALUE);
}
int pipe_send(const char* cmd) {
    if (g_pipe == INVALID_HANDLE_VALUE) return 0;
    if (g_pipe_readonly) return 0;
    DWORD n;
    if (!WriteFile(g_pipe, cmd, (DWORD)strlen(cmd), &n, NULL) || n == 0) {
        CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE;
        return 0;
    }
    return 1;
}
void pipe_poll(void) {
    static StatusFrames frames;
    static HANDLE previous_pipe = INVALID_HANDLE_VALUE;
    if (g_pipe != previous_pipe || g_pipe == INVALID_HANDLE_VALUE) frames.reset();
    previous_pipe = g_pipe;
    if (g_pipe == INVALID_HANDLE_VALUE) return;
    for (int drain = 0; drain < 256; ++drain) {
        DWORD available = 0;
        bool ok = PeekNamedPipe(g_pipe, NULL, 0, NULL, &available, NULL) != 0;
        if (ok && !available) return;
        char chunk[4096]; DWORD count = 0;
        if (ok) {
            BOOL read = ReadFile(g_pipe, chunk, sizeof(chunk), &count, NULL);
            ok = (read || GetLastError() == ERROR_MORE_DATA) && count > 0;
        }
        if (!ok) {
            frames.reset(); previous_pipe = INVALID_HANDLE_VALUE;
            CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE; g_pipe_readonly = 0;
            if (process_running(&g_trainer)) state_transition(MASTER_STARTING);
            return;
        }
        frames.feed(chunk, count, [](const char* json) {
            parse_status_push(json);
            need_content = 1;
        });
    }
}
/* 读 dqn_config.txt 的静态配置（train_repeat / 奖励权重）。
   trainer 的 build_status_json 不推送这些字段，从配置文件补齐显示。 */
static void load_static_config_values(void) {
    FILE* f = fopen("dqn_config.txt", "r");
    if (!f) return;
    char line[128];
    int train_rep = 0; float bws = 0, advs = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "train_repeat=%d", &train_rep) == 1) continue;
        if (sscanf(line, "block_win_scale=%f", &bws) == 1) continue;
        if (sscanf(line, "advanced_scale=%f", &advs) == 1) continue;
    }
    fclose(f);
    snprintf(status_lines[7], sizeof(status_lines[7]), "%d", train_rep);
    snprintf(status_lines[8], sizeof(status_lines[8]), "%.2f/%.2f", bws, advs);
}

void parse_status_push(const char* json) {
    // GPT6-Astra: read back only fields present, and never overwrite pending edits.
    for(int i=6;i<CONFIG_SLIDER_COUNT;++i) {
        char key[80]; snprintf(key,sizeof(key),"\"%s\":",g_config_sliders[i].key);
        const char* field=strstr(json,key);
        if(field && !g_slider_dirty[i]) {
            float value=(float)atof(field+strlen(key));
            if(isfinite(value) && value>=g_config_sliders[i].min_val && value<=g_config_sliders[i].max_val)
                g_config_sliders[i].value=value;
        }
    }
    if (const char* shutdown = strstr(json, "\"shutdown_state\":"))
        sscanf(shutdown, "\"shutdown_state\":%d", &g_trainer_shutdown_state);
    if (const char* paused = strstr(json, "\"sentinel_paused\":"))
        sscanf(paused, "\"sentinel_paused\":%d", &g_trainer_sentinel_paused);
    int ep = 0, ph = 0, dp = 0;
    float wr = 0, lr2 = 0, dr = 0, epc = 0, lr = 0;
    int mem_enabled = 0, mem_cache = 0, mem_kb = 0;
    int mem_cache_b = 0, mem_kb_b = 0, mem_cache_w = 0, mem_kb_w = 0;
    float mem_scale = 0, mem_conf = 0;
    int has_mem_scale = 0, has_mem_conf = 0;
    int train_rep = 0; float bws = 0, advs = 0;
    const char* p;
    /* 训练分阶段计时 */
    if ((p = strstr(json, "\"fwd_us\":"))) g_phase_forward_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"bwd_us\":"))) g_phase_backward_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"tgt_us\":"))) g_phase_target_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"ph_steps\":"))) g_phase_steps = atoi(p + 11);
    /* 按方分阶段计时 */
    if ((p = strstr(json, "\"bfwd_us\":"))) g_side_forward_us[0] = (float)atof(p + 10);
    if ((p = strstr(json, "\"wfwd_us\":"))) g_side_forward_us[1] = (float)atof(p + 10);
    if ((p = strstr(json, "\"bbwd_us\":"))) g_side_backward_us[0] = (float)atof(p + 10);
    if ((p = strstr(json, "\"wbwd_us\":"))) g_side_backward_us[1] = (float)atof(p + 10);
    if ((p = strstr(json, "\"bsteps\":"))) g_side_steps[0] = atoi(p + 9);
    if ((p = strstr(json, "\"wsteps\":"))) g_side_steps[1] = atoi(p + 9);
    /* replay_from_buffer 细分计时 */
    if ((p = strstr(json, "\"rspl_us\":"))) g_replay_sample_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"rfwd_us\":"))) g_replay_forward_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"rtgt_us\":"))) g_replay_target_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"rtrn_us\":"))) g_replay_train_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"rsyn_us\":"))) g_replay_sync_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"rcnt\":"))) g_replay_count = atoi(p + 7);
    /* backward_update 分段计时 */
    if ((p = strstr(json, "\"fc4_us\":"))) g_bwd_fc4_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"fc3_us\":"))) g_bwd_fc3_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"fc2_us\":"))) g_bwd_fc2_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"fc1_us\":"))) g_bwd_fc1_us = (float)atof(p + 9);
    if ((p = strstr(json, "\"pool_us\":"))) g_bwd_pool_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"attn_us\":"))) g_bwd_attn_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"embed_us\":"))) g_bwd_embed_us = (float)atof(p + 11);
    if ((p = strstr(json, "\"bwdcnt\":"))) g_bwd_count = atoi(p + 9);
    /* train_step_batch per-sample 开销 */
    if ((p = strstr(json, "\"tssetup\":"))) g_ts_setup_us = (float)atof(p + 10);
    if ((p = strstr(json, "\"tsmemset\":"))) g_ts_memset_us = (float)atof(p + 11);
    if ((p = strstr(json, "\"tsfwd\":"))) g_ts_fwd_us = (float)atof(p + 8);
    if ((p = strstr(json, "\"tshubler\":"))) g_ts_hubler_us = (float)atof(p + 11);
    if ((p = strstr(json, "\"tsbwd\":"))) g_ts_bwd_us = (float)atof(p + 8);
    if ((p = strstr(json, "\"tsvalidate\":"))) g_ts_validate_us = (float)atof(p + 13);
    if ((p = strstr(json, "\"tscount\":"))) g_ts_count = atoi(p + 10);
    if ((p = strstr(json, "\"ep\":"))) ep = atoi(p + 5);
    if ((p = strstr(json, "\"ph\":"))) ph = atoi(p + 5);
    if ((p = strstr(json, "\"wr\":"))) wr = atof(p + 5);
    if ((p = strstr(json, "\"lr2\":"))) lr2 = atof(p + 6);
    if ((p = strstr(json, "\"dr\":"))) dr = atof(p + 5);
    if ((p = strstr(json, "\"dp\":"))) dp = atoi(p + 5);
    if ((p = strstr(json, "\"epc\":"))) epc = atof(p + 6);
    if ((p = strstr(json, "\"lr\":"))) lr = atof(p + 5);
    int has_live_config = 0;
    if ((p = strstr(json, "\"train_repeat\":"))) { train_rep = atoi(p + 15); has_live_config++; }
    if ((p = strstr(json, "\"block_win_scale\":"))) { bws = atof(p + 18); has_live_config++; }
    if ((p = strstr(json, "\"advanced_scale\":"))) { advs = atof(p + 17); has_live_config++; }
    if ((p = strstr(json, "\"memory_enabled\":"))) mem_enabled = atoi(p + 17);
    if ((p = strstr(json, "\"memory_cache\":"))) mem_cache = atoi(p + 15);
    if ((p = strstr(json, "\"memory_kb\":"))) mem_kb = atoi(p + 12);
    if ((p = strstr(json, "\"memory_cache_b\":"))) mem_cache_b = atoi(p + 17);
    if ((p = strstr(json, "\"memory_kb_b\":"))) mem_kb_b = atoi(p + 14);
    if ((p = strstr(json, "\"memory_cache_w\":"))) mem_cache_w = atoi(p + 17);
    if ((p = strstr(json, "\"memory_kb_w\":"))) mem_kb_w = atoi(p + 14);
    if ((p = strstr(json, "\"memory_scale\":"))) { mem_scale = atof(p + 15); has_mem_scale = 1; }
    if ((p = strstr(json, "\"memory_confidence\":"))) { mem_conf = atof(p + 20); has_mem_conf = 1; }   /* 前缀20字符，+19 会指向引号导致恒 0 */
    if ((p = strstr(json, "\"t\":"))) g_ep_time = atof(p + 4);      /* 当前局时间 */
    if ((p = strstr(json, "\"e\":"))) g_elapsed = atof(p + 4);      /* 总运行时间 */
    if ((p = strstr(json, "\"r\":"))) g_remaining = atof(p + 4);    /* 剩余时间 */
    if ((p = strstr(json, "\"pretrain\":"))) g_pretrain_active = atoi(p + 11);
    if ((p = strstr(json, "\"pretrain_steps\":"))) g_pretrain_steps = atoi(p + 17);
    if ((p = strstr(json, "\"pretrain_avg_steps\":"))) g_pretrain_avg_steps = atof(p + 21);
    if ((p = strstr(json, "\"pretrain_success\":"))) g_pretrain_success = atof(p + 19);
    if ((p = strstr(json, "\"pretrain_stable\":"))) g_pretrain_stable = atoi(p + 18);
    if ((p = strstr(json, "\"eval_active\":"))) g_ui_eval_active = atoi(p + 14);
    if ((p = strstr(json, "\"eval_completed\":"))) g_ui_eval_completed = atoi(p + 17);
    if ((p = strstr(json, "\"eval_total\":"))) g_ui_eval_total = atoi(p + 13);
    /* Minimax 后的阶段进度（旧训练器不带这些字段：g_ui_stage 保持 0/IDLE，
       此时沿用既有的 Minimax/段内进度显示）。 */
    if ((p = strstr(json, "\"stage_id\":"))) g_ui_stage.stage = atoi(p + 11);
    if ((p = strstr(json, "\"stage_completed\":"))) g_ui_stage.completed = atoi(p + 18);
    if ((p = strstr(json, "\"stage_total\":"))) g_ui_stage.total = atoi(p + 14);
    if ((p = strstr(json, "\"stage_unit\":"))) g_ui_stage.unit = atoi(p + 13);
    if ((p = strstr(json, "\"stage_snap_done\":"))) g_ui_stage.snapshot_completed = atoi(p + 18);
    if ((p = strstr(json, "\"stage_snap_total\":"))) g_ui_stage.snapshot_total = atoi(p + 19);
    if ((p = strstr(json, "\"stage_indet\":"))) g_ui_stage.indeterminate = atoi(p + 14);
    if ((p = strstr(json, "\"stage_state\":"))) g_ui_stage.state = atoi(p + 14);
    /* 正式训练耗时按 IPC 的 episode 变化立即采样。旧代码每 2 秒才看一次，
       会漏局；断连恢复后再把相隔上百局的两个点误连成大斜坡。 */
    if (!g_pretrain_active && ep > 0 && g_ep_time > 0.0f &&
        ep != g_last_ep_time_episode) {
        history_push_ep(METRIC_EP_TIME, g_ep_time, (float)ep);
        g_last_ep_time_sampled = g_ep_time;
        g_last_ep_time_episode = ep;
    }
    /* 预训练没有 game.dat 对局步日志；把 IPC 的每局步数直接送入 Steps 曲线。
       IPC 可能同一局推送多次，按 episode 去重。 */
    {
        static int s_last_pretrain_step_ep = -1;
        if (g_pretrain_active && ep > 0 && g_pretrain_steps > 0 &&
            ep != s_last_pretrain_step_ep) {
            history_push_ep(METRIC_GAME_LEN, (float)g_pretrain_steps, (float)ep);
            s_last_pretrain_step_ep = ep;
        }
        if (!g_pretrain_active) s_last_pretrain_step_ep = -1;
    }
    snprintf(status_lines[0], sizeof(status_lines[0]), "%d", ep);
    snprintf(status_lines[1], sizeof(status_lines[1]), "Phase %d", ph);
    snprintf(status_lines[2], sizeof(status_lines[2]), "%.1f%% W / %.1f%% L / %.1f%% D", wr, lr2, dr);
    snprintf(status_lines[3], sizeof(status_lines[3]), "%d", dp);
    snprintf(status_lines[4], sizeof(status_lines[4]), "%.4f", epc);
    snprintf(status_lines[5], sizeof(status_lines[5]), "%.6f", lr);
    snprintf(status_lines[6], sizeof(status_lines[6]), "B:c%d/kb%d W:c%d/kb%d",
        mem_cache_b, mem_kb_b, mem_cache_w, mem_kb_w);
    snprintf(status_lines[7], sizeof(status_lines[7]), "%d", train_rep);
    snprintf(status_lines[8], sizeof(status_lines[8]), "%.2f/%.2f", bws, advs);
    snprintf(status_lines[9], sizeof(status_lines[9]), "%.2f/%.2f", mem_scale, mem_conf);
    g_memory_enabled = mem_enabled;
    /* train_repeat/奖励缩放已由同一条 IPC 状态推送；不能再用可能滞后的
       dqn_config.txt 覆盖，否则滑块刚应用就会在 UI 中跳回旧值。
       但允许连接本轮升级前仍在运行的 Trainer：旧推送完全不带这三个字段，
       此时只做兼容性回退，避免界面临时显示为 0。 */
    if (has_live_config != 3) load_static_config_values();
    /* IND-01 修复：用 Trainer 推送的真实值初始化「未编辑」滑块，否则点击
       「应用」会用写死的默认值覆盖实际训练参数。用户主动改过的滑块保持原值。 */
    if (has_live_config == 3) {
        if (!g_slider_dirty[0]) g_config_sliders[0].value = (float)train_rep;
        if (!g_slider_dirty[2]) g_config_sliders[2].value = bws;
        if (!g_slider_dirty[3]) g_config_sliders[3].value = advs;
    }
    if (has_mem_scale && !g_slider_dirty[4]) {
        float v = mem_scale; if (v < 0) v = 0; if (v > 1) v = 1;
        g_config_sliders[4].value = v;
    }
    if (has_mem_conf && !g_slider_dirty[5]) {
        float v = mem_conf; if (v < 0) v = 0; if (v > 1) v = 1;
        g_config_sliders[5].value = v;
    }
}
static int read_dashboard_metric_tail(const char* name, uint32_t fields, BRRecord* out, int capacity) {
    BRFile file{};
    const std::string path = tui_metric_log_path(name);
    if (path.size() >= sizeof(file.path)) return -1;
    memcpy(file.path, path.c_str(), path.size() + 1);
    return br_read_tail_checked(&file, out, capacity, fields);
}
static void read_latest_side_win(void) {
    const std::string path = tui_metric_log_path("latest_side_win.txt");
    FILE* f = _fsopen(path.c_str(), "rb", _SH_DENYNO);
    if (!f) return; /* Keep the last valid snapshot across a transient rename. */
    char magic[24] = {};
    int episode = 0;
    float black[3] = {}, white[3] = {};
    const int fields = fscanf(f, "%23s %d %f %f %f %f %f %f", magic, &episode,
        &black[0], &black[1], &black[2], &white[0], &white[1], &white[2]);
    fclose(f);
    if (fields != 8 || strcmp(magic, "side_win_v1") != 0 || episode <= 0) return;
    for (int i = 0; i < 3; ++i)
        if (!isfinite(black[i]) || !isfinite(white[i]) ||
            black[i] < 0 || black[i] > 100 || white[i] < 0 || white[i] > 100)
            return;
    g_side_win_episode = episode;
    memcpy(g_black_eval_win, black, sizeof(black));
    memcpy(g_white_eval_win, white, sizeof(white));
}
void read_eval(void) {
    read_latest_side_win();
    // 20 checkpoints x 3 modes. Shared reader ignores unfinished append bytes;
    // a transient read error keeps the previous Dashboard snapshot intact.
    BRRecord win_buf[60]{}, eval_buf[60]{};
    const int win_n = read_dashboard_metric_tail("metric_win.bin", 4, win_buf, 60);
    if (win_n < 0) return;
    const int eval_n = read_dashboard_metric_tail("metric_eval.bin", 7, eval_buf, 60);
    g_eval_episode = g_eval_phase = g_eval_depth = 0;
    g_eval_win = g_eval_draw = g_eval_lose = 0;
    for (int m = 0; m < NUM_WIN_MODES; m++) g_win_count[m] = 0;
    for (int i = 0; i < win_n; i++) {
        if (!metric_win_row_valid(win_buf[i])) continue;
        int ep = win_buf[i].episode;
        int mode = (int)win_buf[i].fields[0];
        float wr = win_buf[i].fields[1];
        float dr = win_buf[i].fields[2];
        float lr = win_buf[i].fields[3];
        if (mode < 0 || mode >= NUM_WIN_MODES) continue;
        /* 从 metric_eval.bin 查找同 episode 的 phase/depth/epsilon（字段：phase,depth,eps,mode,win%,draw%,lose%） */
        int ph = 0, dep = 0; float eps = 0;
        for (int j = 0; j < eval_n; j++) {
            if (!metric_eval_row_valid(eval_buf[j])) continue;
            if (eval_buf[j].episode == ep && (int)eval_buf[j].fields[3] == mode) {
                ph = (int)eval_buf[j].fields[0];
                dep = (int)eval_buf[j].fields[1];
                eps = eval_buf[j].fields[2];
                break;
            }
        }
        if (mode == 0) {   /* Dashboard 总览仍取 Greedy(ε=0) */
            g_eval_episode = ep; g_eval_phase = ph; g_eval_depth = dep; g_eval_win = wr; g_eval_draw = dr; g_eval_lose = lr;
        }
        /* 三种评估模式各存一条独立历史（滚动窗口，记录对应 episode） */
        int m = mode;
        if (g_win_count[m] < MAX_WIN_POINTS) {
            g_win_history[m][g_win_count[m]] = wr;
            g_ep_history[m][g_win_count[m]] = ep;
            g_win_count[m]++;
        } else {
            for (int i = 0; i < MAX_WIN_POINTS - 1; i++) { g_win_history[m][i] = g_win_history[m][i + 1]; g_ep_history[m][i] = g_ep_history[m][i + 1]; }
            g_win_history[m][MAX_WIN_POINTS - 1] = wr;
            g_ep_history[m][MAX_WIN_POINTS - 1] = ep;
        }
    }
    /* Training 页的三条胜率只由 sample_bin_metrics() 维护。
       这里是 Dashboard 摘要读取路径，文件正被 Trainer 追加时可能短暂读取失败；
       若再同步 g_hist，会把已经加载好的三条曲线错误覆盖成"无数据"。 */
}

/* ============================================================================
   导航栏 / 帮助栏 / 页面渲染
   ============================================================================ */
static const char* TAB_NAMES[MAX_TABS] = {
    "1 Dashboard", "2 Training", "3 Config", "4 Model", "5 GPU",
    "6 Logs", "7 Checkpoint", "8 System", "9 Help"
};
/* 两排导航：第一排 5 个，第二排 4 个 */
static const int TAB_ROW_1[] = { 0, 1, 2, 3, 4 };
static const int TAB_ROW_2[] = { 5, 6, 7, 8 };
#define TAB_BAR_Y2 2

/* ---------- 功能合并系统（大屏自适应 tab 数） ----------
   Training 独立；其余 8 个功能按优先级合并成更少 tab，
   每个合并页内网格排布多个功能面板（列优先：左上 左下 右上 右下） */
enum {
    FUN_DASHBOARD = 0, FUN_CONFIG, FUN_LOGS, FUN_GPU,
    FUN_MODEL, FUN_SYSTEM, FUN_CHECKPOINT, FUN_HELP, FUN_COUNT
};
static_assert(MAX_TABS >= 3 + FUN_COUNT,
    "MAX_TABS must hold all fixed and unmerged function pages");
static const char* FUN_NAMES[FUN_COUNT] = {
    "Dashboard", "Config", "Logs", "GPU", "Model", "System", "Checkpoint", "Help"
};
#define MIN_MERGE_W 64   /* 面板最小宽（阈值：更早多列合并） */
#define MIN_MERGE_H 16   /* 面板最小高（阈值：更早多行合并） */
#define MERGE_GAP 1      /* 面板间距 */
static int g_merge_n = 1;        /* 每合并页显示的功能数 */
static int g_merge_pages = 8;    /* 合并页数 */
static int g_total_tabs = 9;     /* 总 tab 数（含 Training） */

/* 计算合并网格：面板宽高填满可用区，但保证 ≥ 功能固定尺寸（84×19），否则减列/行 */
void compute_merge_grid(int avail_w, int avail_h, int* cols, int* rows, int* pw, int* ph) {
    int c = avail_w / (MIN_MERGE_W + MERGE_GAP); if (c < 1) c = 1;
    int r = avail_h / (MIN_MERGE_H + MERGE_GAP); if (r < 1) r = 1;
    int w = (avail_w - MERGE_GAP * (c - 1)) / c;
    int h = (avail_h - MERGE_GAP * (r - 1)) / r;
    while (w < 84 && c > 1) { c--; w = (avail_w - MERGE_GAP * (c - 1)) / c; }   /* 防功能溢出 */
    while (h < 19 && r > 1) { r--; h = (avail_h - MERGE_GAP * (r - 1)) / r; }
    *cols = c; *rows = r; *pw = w; *ph = h;
}

/* 根据窗口大小计算合并布局：面板网格容量 → 每页功能数 → 合并页数 → 总 tab 数 */
void compute_merge_layout(void) {
    int avail_w = g_console_width - 4;
    int avail_h = (g_console_height - 4) - (CONTENT_Y + 1);
    int cols, rows, pw, ph;
    compute_merge_grid(avail_w, avail_h, &cols, &rows, &pw, &ph);
    int n = cols * rows;
    if (n > FUN_COUNT) n = FUN_COUNT;
    g_merge_n = n;
    g_merge_pages = (FUN_COUNT + n - 1) / n;
    g_total_tabs = 4 + g_merge_pages;   /* Training + PLAYGROUND + PROBE + GPU_Analyze + 合并页 */
    if (g_total_tabs > MAX_TABS) g_total_tabs = MAX_TABS;
    if (g_active_tab >= g_total_tabs) g_active_tab = g_total_tabs - 1;
    build_tab_names();
}

/* 动态排布 tab：x 累加（每个 tab 后间隔 2 空格），记录 g_tab_x/g_tab_w 供鼠标点击 */
void draw_tab_bar(void) {
    clear_rect(0, TAB_BAR_Y, g_console_width, 2);
    int n = g_total_tabs;
    int row1 = (n > 5) ? 5 : n;
    int row2 = n - row1;
    char dn[64];
    int x;
    /* 第一排：左对齐 + 往右缩进 */
    x = 4;
    for (int i = 0; i < row1; i++) {
        int len = get_display_width(g_tab_names[i]);
        int avail = g_console_width - x - 2;
        if (len > avail) len = avail;
        if (len < 0) len = 0;
        g_tab_x[i] = x;
        g_tab_w[i] = len;
        gotoxy(x, TAB_BAR_Y);
        snprintf(dn, sizeof(dn), "%.*s", len, g_tab_names[i]);
        control_here(dn, i == g_active_tab, true);
        set_color(COLOR_WHITE, COLOR_BLACK);
        x += len + 2;   /* 间隔 2 空格 */
    }
    /* 第二排：左对齐 + 往右缩进 */
    x = 4;
    for (int i = 0; i < row2; i++) {
        int idx = row1 + i;
        int len = get_display_width(g_tab_names[idx]);
        int avail = g_console_width - x - 2;
        if (len > avail) len = avail;
        if (len < 0) len = 0;
        g_tab_x[idx] = x;
        g_tab_w[idx] = len;
        gotoxy(x, TAB_BAR_Y2);
        snprintf(dn, sizeof(dn), "%.*s", len, g_tab_names[idx]);
        control_here(dn, idx == g_active_tab, true);
        set_color(COLOR_WHITE, COLOR_BLACK);
        x += len + 2;
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 右侧状态（第二排末尾，避开 tab） */
    int status_x = g_console_width - 14;
    if (status_x > x + 2) {
        gotoxy(status_x, TAB_BAR_Y2);
        set_color(COLOR_YELLOW, COLOR_BLACK);
        printf("[%s]", state_name(g_state));
        ui_button(status_x, TAB_BAR_Y, "置顶", {g_ontop != 0, false}, ui_lamp_button());
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 窗口置顶开关：TOPMOST / NOTOPMOST，保持位置尺寸不变 */
static void toggle_ontop(void) {
    g_ontop = 1 - g_ontop;
    SetWindowPos(g_hwnd, g_ontop ? HWND_TOPMOST : HWND_NOTOPMOST,
        0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
}

void draw_help_bar(void) {
    int y = g_console_height - 3;
    set_color(COLOR_WHITE, COLOR_BLACK);
    clear_rect(0, y, g_console_width, 3); // Footer owns the full width, including both corners.
    gotoxy(2, y);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    int len = g_console_width - 4;
    if (len < 10) len = 10;
    for (int i = 0; i < len; i++) printf("━");
    gotoxy(2, y + 1);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf(" [1-9]切页 [Ctrl+P]命令 [S]启动会话 [Q]退出  鼠标:点击导航/控件  (关闭: 命令面板 shutdown)");
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* ---------- 各页面（ox,oy = 面板偏移，pw,ph = 面板宽高；独立 tab 传 0,0 和内容区尺寸） ---------- */
void render_dashboard(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    int merge = 1; // All layouts use the supplied available panel dimensions.
    int top_h = merge ? ((ph - 1) / 2) : 12;   /* SESSION/TRAINING 拉高撑满上半 */
    if (top_h < 6) top_h = 6;
    int sw = merge ? ((pw - 2) / 2) : 40;      /* SESSION 宽 */
    if (sw < 30) sw = 30;
    if (!g_page_drawn[tab] || g_force_redraw) {
        if (merge) {
            /* 三框拉长拉高，撑满面板（TRAINING 吸收宽余数，右边/下边不留误差） */
            int sw2 = pw - 2 - sw;
            draw_border(x, y, sw, top_h, " SESSION ");
            draw_border(x + sw + 2, y, sw2, top_h, " TRAINING ");
            draw_border(x, y + top_h + 1, pw, ph - top_h - 1, " EVALUATION ");   /* 宽 pw 到底 */
        } else {
            int evh = ph - 13;
            if (evh < 7) evh = 7;
            draw_border(x, y, 40, 12, " SESSION ");
            draw_border(x + 42, y, 40, 12, " TRAINING ");
            draw_border(x, y + 13, 82, evh, " EVALUATION ");
        }
        g_page_drawn[tab] = 1;
    }
    /* 内容行（每次更新，尾随空格清残留） */
    gotoxy(x + 2, y + 1); ui_number_printf("状态: %s          ", state_name(g_state));
    /* 状态文字本身是开关；按实际字符宽度换行并记录点击范围。 */
    {
        clear_rect(x + 2, y + 2, sw - 4, 3);
        const char* names[] = { "Trainer", "Watchdog", "LogGuard" };
        int cx = x + 2, cy = y + (sw - 4 < 36 ? 2 : 3);
        for (int id = 0; id < DASH_WORKER_COUNT; ++id) {
            bool running = process_running(dashboard_worker_process(id)) != 0;
            bool pending = running && g_dashboard_workers.stopping[id];
            const char* status = pending ? (id == DASH_TRAINER ?
                (g_trainer_shutdown_state == 3 ? "保存失败" : "保存中") : "关闭中") :
                (running ? "运行" : "停止");
            char label[80]; snprintf(label, sizeof(label), "%s:%s", names[id], status);
            int width = ui_button_width(label, ui_lamp_button());
            if (cx + width > x + sw - 2) { cx = x + 2; ++cy; }
            g_worker_hits[id] = { cx, cy, width };
            ui_button(cx, cy, label, {running, false}, ui_lamp_button());
            cx += width + 1;
        }
        g_worker_hit_cols = g_console_width; g_worker_hit_rows = g_console_height;
    }
    if (top_h > 6) {
        clear_rect(x + 2, y + 5, sw - 4, 1);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        print_roll_clip(x + 2, y + 5, sw - 4, "点击状态：启动 / 关闭");
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    int x2 = x + sw + 2;
    gotoxy(x2 + 2, y + 1); ui_number_printf("Episode: %s          ", status_lines[0][0] ? status_lines[0] : "--");
    gotoxy(x2 + 2, y + 2); ui_number_printf("阶段: %s              ", status_lines[1][0] ? status_lines[1] : "--");
    /* 胜负行可能很长，截断到面板内避免顶掉右边框 */
    {
        char wr_disp[28];
        snprintf(wr_disp, sizeof(wr_disp), "%.23s", status_lines[2][0] ? status_lines[2] : "--");
        gotoxy(x2 + 2, y + 3); ui_number_printf("胜/负/和: %-23s", wr_disp);
    }
    gotoxy(x2 + 2, y + 4); ui_number_printf("深度: %s  学习率: ", status_lines[3][0] ? status_lines[3] : "--");
    ui_number_printf("%s", status_lines[5][0] ? status_lines[5] : "--");
    gotoxy(x2 + 2, y + 5); ui_number_printf("Epsilon: %s          ", status_lines[4][0] ? status_lines[4] : "--");
    /* EVALUATION 内容（合并时从面板下半开始） */
    int evc = merge ? (y + top_h + 2) : (y + 14);
    gotoxy(x + 2, evc); ui_number_printf("Ep:%d", g_eval_episode); ui_number_printf("  Phase:%d  深度:%d   ", g_eval_phase, g_eval_depth);
    gotoxy(x + 2, evc + 1); ui_number_printf("Win:%.1f%%  Draw:%.1f%%  Lose:%.1f%%   ", g_eval_win, g_eval_draw, g_eval_lose);
    gotoxy(x + 2, evc + 2); printf("提示: Ctrl+P 打开命令面板");
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 值格式化：Eps 多显示几位小数，其他按大小自适应 */
/* 值格式化：固定 7 字符宽（%.3f），保证 y 轴刻度等宽对齐（避免变位数/科学计数法） */
static void fmt_metric(float v, MetricId id, char* out, int sz) {
    /* BUG-050: 自适应宽度，避免大数值/NaN/Infinity 溢出覆盖绘图区 */
    if (v != v) { snprintf(out, sz, " NaN"); return; }   /* NaN */
    if (v > 1e30f || v < -1e30f) { snprintf(out, sz, " Inf"); return; }   /* BUG-051: Infinity 检测 */
    if (id == METRIC_GAME_LEN) snprintf(out, sz, "%7.0f", v);
    else if (v > 99999 || v < -9999) snprintf(out, sz, "%7.0f", v);   /* 大值用整数 */
    else if (v > 999 || v < -99) snprintf(out, sz, "%7.1f", v);
    else snprintf(out, sz, "%7.3f", v);
}

/* 文本折线图（带边框框 + 左轴刻度）：圆角拐角折线 ╭╮╰╯│─，3 点滑动平均去锯齿 */
/* 计算某指标的归一化曲线点（窗口 [view_s,view_e] 区间，自动缩放） */
static void build_curve_points(MetricId id, const float* data, int count, int view_s, int view_e,
    int pw, int ph, float x_min, float x_max, float scale_min, float scale_max, int fixed_scale,
    float pts[VT_CURVE_MAX][2], uint8_t break_before[VT_CURVE_MAX], int* n) {
    *n = 0;
    memset(break_before, 0, VT_CURVE_MAX * sizeof(break_before[0]));
    int ds = 0, de = count - 1;
    if (view_s >= 0) {
        ds = count - 1 - view_s; if (ds < 0) ds = 0;
        de = count - 1 - view_e; if (de < ds) de = ds; if (de > count - 1) de = count - 1;
    }
    int vn = de - ds + 1;
    if (vn < 2 || ph < 2) { *n = 0; return; }
    float mn = 0.0f, mx = 0.0f;
    int have_range = 0;
    for (int i = ds; i <= de; i++) {
        float v = data[i];
        if (!(v == v) || v <= -1e30f || v >= 1e30f) continue;
        if (!have_range) { mn = mx = v; have_range = 1; }
        else { if (v < mn) mn = v; if (v > mx) mx = v; }
    }
    if (!have_range) { *n = 0; return; }
    if (fixed_scale) { mn = scale_min; mx = scale_max; }
    if (mx - mn < 1e-6f) { mn -= 1.0f; mx += 1.0f; }
    /* 叠加曲线按真实 episode 定位 X，不再按数组下标对齐。
       Win 每 1000 局一点、Reward 每局一点也能共用同一时间轴。 */
    /* 不能只检查首点：旧 game.dat 的第 0 局会留下 ep=0，但后续点都有
       有效局号。首点污染曾导致整条 Steps 退化成数组索引轴。 */
    int use_episode_x = 0;
    if (x_max > x_min) {
        for (int i = ds; i <= de; i++) {
            if (g_hist[id].ep[i] > 0) { use_episode_x = 1; break; }
        }
    }
    if (use_episode_x) {
        int first = ds, last = de;
        while (first <= de && (g_hist[id].ep[first] <= 0 || g_hist[id].ep[first] < x_min)) first++;
        while (last >= first && g_hist[id].ep[last] > x_max) last--;
        if (last - first + 1 < 2) { *n = 0; return; }
        ds = first; de = last; vn = de - ds + 1;
        have_range = 0;
        for (int i = ds; i <= de; i++) {
            float v = data[i];
            if (!(v == v) || v <= -1e30f || v >= 1e30f) continue;
            if (!have_range) { mn = mx = v; have_range = 1; }
            else { if (v < mn) mn = v; if (v > mx) mx = v; }
        }
        if (!have_range) { *n = 0; return; }
        if (fixed_scale) { mn = scale_min; mx = scale_max; }
        if (mx - mn < 1e-6f) { mn -= 1.0f; mx += 1.0f; }
        /* 只提交日志里真实存在的 episode 点，并按真实 X 坐标连线。
           旧实现为每个像素列复制"最近值"，把所有曲线强行画成方块阶梯，
           同时制造了并不存在的数值点。点过密时等距抽取真实样本。 */
        int valid_n = 0;
        for (int i = ds; i <= de; i++) {
            float v = data[i];
            if (g_hist[id].ep[i] > 0 && v == v && v > -1e30f && v < 1e30f) valid_n++;
        }
        int out_n = valid_n;
        if (out_n > pw) out_n = pw;
        if (out_n > VT_CURVE_MAX) out_n = VT_CURVE_MAX;
        if (out_n < 2) { *n = 0; return; }
        int written = 0, scan = ds, valid_ord = -1;
        float prev_episode = -1.0f;
        /* 真实逐局曲线的正常抽样间距可能因像素降采样大于 1；只有远超
           当前显示密度的间隔才算数据缺口。Win 检查点本来就稀疏，不断线。 */
        float gap_threshold = (x_max - x_min) * 3.0f / (float)out_n;
        if (gap_threshold < 3.0f) gap_threshold = 3.0f;
        for (int c = 0; c < out_n; c++) {
            int target_ord = (int)((long long)c * (valid_n - 1) / (out_n - 1));
            int pick = -1;
            while (scan <= de) {
                float v = data[scan];
                if (g_hist[id].ep[scan] > 0 && v == v && v > -1e30f && v < 1e30f) {
                    valid_ord++;
                    if (valid_ord == target_ord) { pick = scan++; break; }
                }
                scan++;
            }
            if (pick < 0) break;
            float xr = (g_hist[id].ep[pick] - x_min) / (x_max - x_min);
            if (xr < 0.0f) xr = 0.0f;
            if (xr > 1.0f) xr = 1.0f;
            float val = data[pick];
            float norm = (val - mn) / (mx - mn);
            if (norm < 0) norm = 0; if (norm > 1) norm = 1;
            pts[written][0] = xr; pts[written][1] = norm;
            if (written > 0 && id != METRIC_WIN0 && id != METRIC_WIN1 && id != METRIC_WIN2 &&
                g_hist[id].ep[pick] - prev_episode > gap_threshold)
                break_before[written] = 1;
            prev_episode = g_hist[id].ep[pick];
            written++;
        }
        *n = written;
        return;
    }
    int cnt = vn < pw ? vn : pw; if (cnt > VT_CURVE_MAX) cnt = VT_CURVE_MAX;
    for (int c = 0; c < cnt; c++) {
        int a = ds + (int)((double)c * vn / cnt);
        int b = ds + (int)((double)(c + 1) * vn / cnt);
        if (b == a) b = a + 1;
        if (b > de + 1) b = de + 1;
        double sum = 0;
        int _valid = 0;
        for (int i = a; i < b; i++) { if (data[i] == data[i] && data[i] > -1e30f && data[i] < 1e30f) { sum += data[i]; _valid++; } }   /* BUG-051: 跳过 NaN/Inf */
        float val = (_valid > 0) ? (float)(sum / _valid) : 0.0f;
        float norm = (val - mn) / (mx - mn);
        int level = (int)(norm * (ph - 1) + 0.5);
        if (level > ph - 1) level = ph - 1;
        if (level < 0) level = 0;
        if (_valid > 0) {
            pts[*n][0] = (float)c / (cnt > 1 ? cnt - 1 : 1);
            pts[*n][1] = (float)level / (ph > 1 ? ph - 1 : 1);
            (*n)++;
        }
    }
}

/* 全局复位后的统一 episode 范围。Win 和 Depth 都是稀疏的检查点指标：
   任一存在时由其最近 20 个真实检查点定义；否则固定最近 200 局。
   若把 Depth 当逐局指标，只会得到最近 200 局窗口，而评估间隔通常是
   1000 局，窗口内不足两个点，曲线便会看似消失。 */
static int cell_global_episode_range(int cell_idx, float* out_min, float* out_max) {
    if (cell_idx < 0 || cell_idx >= MAX_CELLS || !g_cell_global_axis[cell_idx]) return 0;
    int sparse_id = -1;
    for (int k = METRIC_WIN0; k <= METRIC_WIN2; k++)
        if (g_cell_checks[cell_idx][k] && g_hist[k].n > 0) { sparse_id = k; break; }
    if (sparse_id < 0 && g_cell_checks[cell_idx][METRIC_DEPTH] && g_hist[METRIC_DEPTH].n > 0)
        sparse_id = METRIC_DEPTH;
    if (sparse_id >= 0) {
        HistoryBuf* wh = &g_hist[sparse_id];
        int last = wh->n - 1;
        int first = last - 19; if (first < 0) first = 0;
        float xmax = wh->ep[last] > 0 ? wh->ep[last] : (float)(wh->base + last);
        float xmin = wh->ep[first] > 0 ? wh->ep[first] : (float)(wh->base + first);
        if (xmax <= xmin) xmin = xmax - 1;
        *out_min = xmin; *out_max = xmax; return 1;
    }
    float latest = 0;
    for (int k = 0; k < METRIC_COUNT; k++) if (g_cell_checks[cell_idx][k] && g_hist[k].n > 0) {
        HistoryBuf* h = &g_hist[k]; int last = h->n - 1;
        float ep = h->ep[last] > 0 ? h->ep[last] : (float)(h->base + last);
        if (ep > latest) latest = ep;
    }
    if (latest <= 0) return 0;
    *out_max = latest;
    *out_min = latest - 199.0f; if (*out_min < 1) *out_min = 1;
    return 1;
}

/* Y 轴量纲族：同族可以共享真实刻度；异族叠加时只能比较相对形状。 */
static int metric_y_family(MetricId id) {
    if (id == METRIC_Q || id == METRIC_TARGET_Q) return 0;
    if (id == METRIC_LOSS) return 1;
    if (id == METRIC_EPSILON) return 2;
    if (id == METRIC_EP_TIME) return 3;
    if (id == METRIC_GAME_LEN) return 4;
    if (id >= METRIC_WIN0 && id <= METRIC_WIN2) return 5;
    return 6; /* 所有奖励分量与 R_Total */
}

void draw_line_series(int x, int y, int w, int h, MetricId id, const float* data, int count, int focus, int view_s, int view_e, int cell_idx) {
    char title[24];
    snprintf(title, sizeof(title), " ");   /* 标题由下方逐勾选指标绘制（各指标色） */
    clear_rect(x, y, w, h);   /* 清框内旧内容，防残影 */
    draw_border(x, y, w, h, title);
    /* 标题行：勾选指标名用 + 连接，每个名字用其曲线颜色 */
    {
        int tx = x + 2, first = 1;
        if (focus) { gotoxy(tx, y); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("v "); tx += 2; }
        for (int k = 0; k < METRIC_COUNT; k++) {
            if (!g_cell_checks[cell_idx][k]) continue;
            if (!first) { gotoxy(tx, y); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("+"); tx++; }
            gotoxy(tx, y);
            set_color(g_metric_colors[k], COLOR_BLACK);
            printf("%s", g_metric_labels[k]);
            tx += (int)strlen(g_metric_labels[k]);
            first = 0;
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 最新值放标题行右侧（最大 6 位小数，能看出缓慢变化趋势；步数为整数） */
    if (count > 0) {
        char val[24];
        if (id == METRIC_GAME_LEN) snprintf(val, sizeof(val), "%d", (int)data[count - 1]);
        else snprintf(val, sizeof(val), "%.6f", data[count - 1]);
        gotoxy(x + w - 2 - (int)strlen(val), y);
        set_color(g_metric_colors[id], COLOR_BLACK);
        printf("%s", val);
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    if (count == 0) {
        gotoxy(x + 2, y + h / 2);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("(无数据)");
        set_color(COLOR_WHITE, COLOR_BLACK);
        return;
    }
    /* x 轴查看窗口：view_s/view_e 为"距最新点的偏移"（-1 全量）。窗口数据区间 [ds, de] */
    int ds = 0, de = count - 1;
    if (view_s >= 0) {
        ds = count - 1 - view_s; if (ds < 0) ds = 0;
        de = count - 1 - view_e; if (de < ds) de = ds; if (de > count - 1) de = count - 1;
    }
    float global_x_min = 0, global_x_max = 0;
    int global_axis = cell_global_episode_range(cell_idx, &global_x_min, &global_x_max);
    if (global_axis) {
        ds = 0; de = count - 1;
        while (ds < count && g_hist[id].ep[ds] > 0 && g_hist[id].ep[ds] < global_x_min) ds++;
        while (de >= ds && g_hist[id].ep[de] > global_x_max) de--;
        /* 主指标在该范围无点时保留安全单点，Win 由 sample-and-hold 绘制。 */
        if (ds >= count || de < ds) { ds = count - 1; de = count - 1; }
    }
    int vn = de - ds + 1;                 /* 窗口内点数 */
    /* Y 轴协调：同量纲曲线共享真实范围；不同量纲叠加改为相对百分比轴。 */
    float mn = data[ds], mx = data[ds];
    for (int i = ds + 1; i <= de; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    int y_family = metric_y_family(id), mixed_y_units = 0;
    for (int k = 0; k < METRIC_COUNT; k++) if (g_cell_checks[cell_idx][k]) {
        if (metric_y_family((MetricId)k) != y_family) mixed_y_units = 1;
    }
    if (!mixed_y_units) {
        /* 汇总格内所有同族曲线；全局 episode 轴时只统计当前可见区间。 */
        int have = 0;
        for (int k = 0; k < METRIC_COUNT; k++) if (g_cell_checks[cell_idx][k]) {
            HistoryBuf* hh = &g_hist[k];
            for (int j = 0; j < hh->n; j++) {
                if (global_axis && hh->ep[j] > 0 && (hh->ep[j] < global_x_min || hh->ep[j] > global_x_max)) continue;
                float v = hh->v[j]; if (!(v == v) || v <= -1e30f || v >= 1e30f) continue;
                if (!have) { mn = mx = v; have = 1; }
                else { if (v < mn) mn = v; if (v > mx) mx = v; }
            }
        }
        if (y_family == 5) { mn = 0.0f; mx = 100.0f; }       /* Win 百分比固定刻度 */
        else if (y_family == 2) { mn = 0.0f; mx = 1.0f; }    /* epsilon 固定刻度 */
        else {
            if (y_family == 1 || y_family == 3 || y_family == 4 || y_family == 6) {
                if (mn > 0) mn = 0; if (mx < 0) mx = 0;      /* Loss/耗时/步数/奖励保留零基线 */
            }
            float span = mx - mn;
            if (span < 1e-6f) { span = fabsf(mx) * 0.1f; if (span < 0.01f) span = 1.0f; }
            /* Loss/耗时/步数是非负量，零基线不能再被 padding 推成负数。 */
            if (y_family == 1 || y_family == 3 || y_family == 4) {
                if (mn < 0.0f) mn = 0.0f;
                mx += span * 0.05f;
            } else {
                mn -= span * 0.05f; mx += span * 0.05f;
            }
        }
    }
    float shared_mn = mn, shared_mx = mx;
    if (mixed_y_units) { /* 曲线各自归一化；左轴明确显示相对百分比 */
        mn = 0.0f; mx = 100.0f;
    }
    if (mx - mn < 1e-6f) { mn -= 1.0f; mx += 1.0f; }
    /* 布局：标题 y，绘图区 y+1..y+h-4，底部轴 y+h-3，滑块 y+h-2，边框底 y+h-1 */
    int plot_top = y + 1;
    int plot_bottom = y + h - 4;
    if (plot_bottom < plot_top) plot_bottom = plot_top + 1;
    int ph = plot_bottom - plot_top;      /* 绘图行数 */
    int axis_w = 9;                       /* 左轴列宽（刻度值 7 + 空格 + ┤） */
    int pw = w - axis_w - 3;              /* 绘图宽（右侧留 1 列，不顶边框） */
    if (pw < 10) pw = 10;
    if (pw > w - axis_w - 3) pw = w - axis_w - 3;   /* 小窗口钳制 */

    /* 每列 y 像素行（每列取窗口内该区间均值） */
    int* rows_arr = (int*)malloc((size_t)pw * sizeof(int));
    if (!rows_arr) return;
    for (int c = 0; c < pw; c++) {
        int a = ds + (int)((double)c * vn / pw);
        int b = ds + (int)((double)(c + 1) * vn / pw);
        if (b == a) b = a + 1;
        if (b > de + 1) b = de + 1;
        double sum = 0;
        int _valid = 0;
        for (int i = a; i < b; i++) { if (data[i] == data[i] && data[i] > -1e30f && data[i] < 1e30f) { sum += data[i]; _valid++; } }   /* BUG-051: 跳过 NaN/Inf */
        float val = (_valid > 0) ? (float)(sum / _valid) : 0.0f;
        float row_mn = mixed_y_units ? shared_mn : mn;
        float row_mx = mixed_y_units ? shared_mx : mx;
        if (row_mx - row_mn < 1e-6f) { row_mn -= 1.0f; row_mx += 1.0f; }
        float norm = (val - row_mn) / (row_mx - row_mn);
        int level = (int)(norm * (ph - 1) + 0.5);
        if (level > ph - 1) level = ph - 1;
        if (level < 0) level = 0;
        rows_arr[c] = level;
    }
    /* 3 点滑动平均（去除高频锯齿，避免"鞭炮"状密集斜杠） */
    int* sr = (id == METRIC_EPSILON) ? NULL : (int*)malloc((size_t)pw * sizeof(int));
    if (sr) {
        for (int c = 0; c < pw; c++) {
            int v0 = rows_arr[c];
            int v1 = (c > 0) ? rows_arr[c - 1] : v0;
            int v2 = (c < pw - 1) ? rows_arr[c + 1] : v0;
            sr[c] = (v0 + v1 + v2) / 3;
        }
        memcpy(rows_arr, sr, (size_t)pw * sizeof(int));
        free(sr);
    }
    /* 渲染：左轴刻度（固定 7 字符宽，y 轴等宽对齐） */
    char tv[16];
    for (int r = 0; r < ph; r++) {
        gotoxy(x + 1, plot_top + r);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        int is_tick = 0;
        for (int k = 0; k < 5; k++) if (r == k * (ph - 1) / 4) is_tick = 1;
        if (is_tick) {
            float tick_v = mx - (mx - mn) * r / (ph - 1);
            if (mixed_y_units) snprintf(tv, sizeof(tv), "%6.0f%%", tick_v);
            else fmt_metric(tick_v, id, tv, sizeof(tv));
            printf("%s", tv);            /* 固定 7 字符宽，┤ 位置恒定 */
            printf("┤");
        } else {
            printf("%*s", 7, "");
            printf("│");
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 所有指标服从用户选择的全局主题；epsilon 也按真实采样点连续绘制。 */
    int selected_series = 0;
    for (int k = 0; k < METRIC_COUNT; k++) selected_series += g_cell_checks[cell_idx][k] ? 1 : 0;
    int series_has_episode_x = 0;
    for (int i = ds; i <= de; ++i) {
        if (g_hist[id].ep[i] > 0) { series_has_episode_x = 1; break; }
    }
    /* 字符折线按数组列采样，无法正确表达稀疏 Win 点和逐局指标的真实
       episode 间距，也不会绘制附加曲线。全局轴或多曲线时强制走真实坐标层。 */
    int effective_theme = (global_axis || selected_series > 1 || series_has_episode_x) ? 3 : g_line_theme;
    /* 主题：3=真实曲线（D2D overlay），0=方块，1=圆角，2=直角 */
    if (effective_theme == 3) {
        VTCurve c = {};
        c.x = x + 1 + axis_w;
        c.y = plot_top;
        c.w = pw;
        c.h = ph;
        float x_min = global_axis ? global_x_min : ((g_hist[id].ep[ds] > 0) ? g_hist[id].ep[ds] : (float)(g_hist[id].base + ds));
        float x_max = global_axis ? global_x_max : ((g_hist[id].ep[de] > 0) ? g_hist[id].ep[de] : (float)(g_hist[id].base + de));
        build_curve_points(id, data, count, global_axis ? -1 : view_s, global_axis ? 0 : view_e,
            pw, ph, x_min, x_max, shared_mn, shared_mx, !mixed_y_units,
            c.pts, c.break_before, &c.n);   /* 主指标 */
        if (c.n > 1) { c.color = vt_palette_rgb(g_metric_colors[id]); vt_curve_add(&c); }
        for (int k = 0; k < METRIC_COUNT; k++) {   /* 叠加勾选的额外指标（同表多曲线） */
            if (k == (int)id || !g_cell_checks[cell_idx][k]) continue;
            /* 额外指标使用主指标的 episode 可视区间，采样密度不同也不错位。 */
            build_curve_points((MetricId)k, g_hist[k].v, g_hist[k].n, -1, 0, pw, ph, x_min, x_max,
                shared_mn, shared_mx, !mixed_y_units, c.pts, c.break_before, &c.n);
            if (c.n > 1) { c.color = vt_palette_rgb(g_metric_colors[k]); vt_curve_add(&c); }
        }
    } else if (effective_theme == 0) {
        /* Block：柱状图，每列 █ 从底向上画到 level 高度（柱子高，占满绘图区） */
        for (int c = 0; c < pw; c++) {
            int hgt = rows_arr[c] + 1;
            if (hgt > ph) hgt = ph;
            for (int r = 0; r < hgt; r++) {
                gotoxy(x + 1 + axis_w + c, plot_bottom - r);
                set_color(g_metric_colors[id], COLOR_BLACK);
                printf("█");
            }
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
    } else {
        /* Round/Rigid：拐角折线（字符码 grid，避免多字节存不进 char） */
        enum { GC_EMPTY=0, GC_H, GC_V, GC_R1, GC_R2, GC_R3, GC_R4, GC_S1, GC_S2, GC_S3, GC_S4 };
        unsigned short* grid = (unsigned short*)malloc((size_t)ph * pw * sizeof(unsigned short));
        if (!grid) { free(rows_arr); return; }
        for (int i = 0; i < ph * pw; i++) grid[i] = GC_EMPTY;
        for (int c = 0; c < pw - 1; c++) {
            int y0 = rows_arr[c], y1 = rows_arr[c + 1];
            int a = y0 < y1 ? y0 : y1, b = y0 > y1 ? y0 : y1;
            int round = (effective_theme == 1);
            for (int r = a; r <= b; r++) {
                unsigned short ch;
                if (y0 == y1) ch = GC_H;
                else if (y0 < y1)  /* 上升（值增大，视觉向上）：起点╭┌ 顶端╯┘ */
                    ch = (r == y0) ? (round ? GC_R1 : GC_S1) : (r == y1) ? (round ? GC_R3 : GC_S3) : GC_V;
                else               /* 下降（值减小，视觉向下）：起点╮┐ 终点╰└ */
                    ch = (r == y0) ? (round ? GC_R2 : GC_S2) : (r == y1) ? (round ? GC_R4 : GC_S4) : GC_V;
                grid[r * pw + c] = ch;
            }
        }
        for (int r = 0; r < ph; r++) {
            for (int c = 0; c < pw; c++) {
                unsigned short ch = grid[r * pw + c];
                if (ch == GC_EMPTY) continue;
                gotoxy(x + 1 + axis_w + c, plot_top + (ph - 1 - r));   /* rows_arr 值大靠上，与刻度一致 */
                set_color(g_metric_colors[id], COLOR_BLACK);
                switch (ch) {
                case GC_H: printf("-"); break;
                case GC_V: printf("|"); break;
                case GC_R1: printf("╯"); break;   /* 上升起点（左下） */
                case GC_R2: printf("╮"); break;   /* 下降起点（左上） */
                case GC_R3: printf("╭"); break;   /* 上升顶端（右上） */
                case GC_R4: printf("╰"); break;   /* 下降低端（右下） */
                case GC_S1: printf("┘"); break;
                case GC_S2: printf("┐"); break;
                case GC_S3: printf("┌"); break;
                case GC_S4: printf("└"); break;
                default: break;
                }
                set_color(COLOR_WHITE, COLOR_BLACK);
            }
        }
        free(grid);
    }
    /* 底部轴：└ 与 y 轴 ┤ 对齐（x+1+axis_w-2），横线到 x+w-2 不顶边框。
       x 轴刻度：绝对序号（base+索引）均匀覆盖整条横线（滚动窗口标号），右侧留 n=count */
    gotoxy(x + 1 + axis_w - 2, plot_bottom + 1);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("└");
    for (int c = 0; c < w - axis_w - 2; c++) printf("─");
    printf("┘");
    {
        int line_x0 = x + 1 + axis_w;
        int line_len = w - axis_w - 2 - 8;   /* 右侧留 8 字符空隙 */
        int K = 5;                            /* 刻度数（含两端） */
        if (!global_axis && vn < 6) K = vn < 2 ? 1 : 3; /* 全局轴始终等分 5 个 episode 刻度 */
        int base = g_hist[id].base;
        float real_axis_min = 0.0f, real_axis_max = 0.0f;
        if (series_has_episode_x) {
            for (int i = ds; i <= de; ++i) if (g_hist[id].ep[i] > 0) {
                if (real_axis_min <= 0.0f) real_axis_min = g_hist[id].ep[i];
                real_axis_max = g_hist[id].ep[i];
            }
        }
        /* bin 数据用 ep[] 真实序号；历史中若残留旧文本点，单点回退而不是显示 ep=0。 */
        for (int k = 0; k < K; k++) {
            int pos, idx;
            if (K == 1) { pos = line_x0; idx = ds; }
            else {
                pos = line_x0 + (int)((long long)line_len * k / (K - 1));
                idx = ds + (int)((long long)vn * k / (K - 1));
                if (idx > de) idx = de;
            }
            char s[16];
            int x_value = global_axis ? (int)(global_x_min + (global_x_max - global_x_min) * k / (K > 1 ? K - 1 : 1)) : base + idx;
            int has_real_x = 0;
            if (global_axis) {
                has_real_x = 1;
            } else if (series_has_episode_x && real_axis_max >= real_axis_min) {
                /* 曲线本身按真实 episode 坐标放置，刻度也必须按相同坐标
                   均匀划分；按数组下标挑标签会把稀疏点标在错误位置。 */
                x_value = (int)(real_axis_min + (real_axis_max - real_axis_min) *
                    k / (K > 1 ? K - 1 : 1));
                has_real_x = 1;
            } else if (idx >= 0 && idx < g_hist[id].n && g_hist[id].ep[idx] > 0) {
                x_value = (int)g_hist[id].ep[idx];
                has_real_x = 1;
            } else if (idx >= 0 && idx < g_hist[id].n) {
                /* 从最近的真实 X 点按索引距离推算，兼容升级前已混入的 ep=0 点。 */
                int near_idx = -1;
                for (int d = 1; d < g_hist[id].n; d++) {
                    int l = idx - d, r = idx + d;
                    if (l >= 0 && g_hist[id].ep[l] > 0) { near_idx = l; break; }
                    if (r < g_hist[id].n && g_hist[id].ep[r] > 0) { near_idx = r; break; }
                }
                if (near_idx >= 0) {
                    x_value = (int)g_hist[id].ep[near_idx] + (idx - near_idx);
                    has_real_x = 1;
                }
            }
            if (!has_real_x && base <= 0) {
                int cur_ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
                if (cur_ep > 0) x_value = cur_ep - (g_hist[id].n - 1 - idx);
                if (x_value <= 0) x_value = idx + 1;
            }
            snprintf(s, sizeof(s), "%d", x_value);
            gotoxy(pos - (int)strlen(s) / 2, plot_bottom + 1);
            printf("%s", s);
        }
        /* Training 页统一以 episode 为横轴；旧 Loss 日志已通过 epsilon 桥接换算。 */
        gotoxy(x + w - 3, plot_bottom + 1);   /* 右边框内恰好容纳 "ep"，不覆盖刻度/边框 */
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("ep");
    }
    if (focus) {   /* 复位按钮在 y 轴数字标号下方（x 轴行左端空位） */
        gotoxy(x + 1, plot_bottom + 1);
        control_here("[R]复位", false, false, 'R');
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 滑块行（focus 显示：高亮 = 当前窗口位置，点击可移；非 focus 清空防残影） */
    {
        int sl_y = plot_bottom + 2;
        clear_rect(x + 1, sl_y, w - 2, 1);
        if (focus) {
            int sl_x0 = x + 1 + axis_w;
            int sl_w = w - axis_w - 2; if (sl_w < 4) sl_w = 4;
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            for (int c = 0; c < sl_w; c++) { gotoxy(sl_x0 + c, sl_y); printf("·"); }
            /* 高亮块固定长度（轨道 1/4），位置 = 窗口右端 de（复位/最新时贴最右） */
            int block = sl_w / 4; if (block < 3) block = 3;
            /* 位置 = 期望窗口中心偏移 (vs+ve)/2：0=最新贴右，count-1=最老贴左。
               用期望值而非 clamp 后的 ds/de，点两端时滑块精确贴边，行程不受窗口大小挤压 */
            int ctr_off = (view_s < 0) ? 0 : (view_s + view_e) / 2;
            if (ctr_off > count - 1) ctr_off = count - 1;
            int pos_c = (count > 1) ? sl_x0 + sl_w - (int)((long long)sl_w * ctr_off / (count - 1)) : sl_x0 + sl_w;
            int b0 = pos_c - block / 2; if (b0 < sl_x0) b0 = sl_x0;
            int b1 = b0 + block; if (b1 > sl_x0 + sl_w) b1 = sl_x0 + sl_w;
            set_color(COLOR_CYAN, COLOR_BLACK);
            for (int c = b0; c < b1; c++) { gotoxy(c, sl_y); printf("█"); }
            gotoxy(b0, sl_y); printf("◄");
            gotoxy(b1 - 1, sl_y); printf("►");
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    free(rows_arr);
}

/* 进度条框绘制（Training 页底部 3 行；主循环跑马灯定时器可单独调用，只刷新进度条行，
   避免整页渲染反复重画图格/曲线/十字准线导致鼠标小框闪烁） */
static void draw_progress_bar(int x, int w) {
    int pbar_h = 3;
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    int grid_top = CONTENT_Y + 1;
    if (py_pbar < grid_top + 3 + 3) py_pbar = grid_top + 3 + 3;
    clear_rect(x, py_pbar, w, pbar_h);
    draw_border(x, py_pbar, w, pbar_h, " Progress ");
    int ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
    int in_seg = ep % 1000;
    float pct = (float)in_seg / 1000.0f;
    const int stage_active = (g_ui_stage.state == TASK_RUNNING && g_ui_stage.stage != STG_NONE);
    if (g_pretrain_active && !g_ui_eval_active && !stage_active) {
        /* 单人课程无固定总局数：以平均步数从棋盘上限25收敛到目标4作为进度。 */
        float avg = g_pretrain_avg_steps > 0 ? g_pretrain_avg_steps : 25.0f;
        pct = (25.0f - avg) / 21.0f;
        if (pct < 0) pct = 0; if (pct > 1) pct = 1;
    } else if (g_ui_eval_active && g_ui_eval_total > 0) {
        pct = (float)g_ui_eval_completed / (float)g_ui_eval_total;
        if (pct < 0) pct = 0; if (pct > 1) pct = 1;
    } else if (stage_active && !g_ui_stage.indeterminate && g_ui_stage.total > 0) {
        /* 后评估阶段（技能测试/多快照）：按已完成/总量显示真实百分比。 */
        pct = (float)g_ui_stage.completed / (float)g_ui_stage.total;
        if (pct < 0) pct = 0; if (pct > 1) pct = 1;
    } else if (stage_active) {
        pct = 0;   /* 未知总量阶段：条形图交给跑马灯，不虚构百分比 */
    }
    int bar_w = w - 36; if (bar_w < 12) bar_w = 12;   /* 右端留空间给控制台开关按钮 */
    int filled = (int)(pct * bar_w);
    gotoxy(x + 2, py_pbar + 1);
    printf("Progress: [");
    /* 跑马灯高光：进度条 ≥3 个等号才显示；固定 3 格宽（中间最亮，两边渐变）；
       固定 2 秒从左到右移动，到头后消失等 0.8 秒再从左出现 */
    if (filled >= 3) {
        const DWORD slide_ms = 2000, pause_ms = 800;
        DWORD tick = GetTickCount() % (slide_ms + pause_ms);
        float center = -2.0f + (filled + 4.0f) * (float)tick / slide_ms;
        for (int i = 0; i < bar_w; i++) {
            if (i < filled) {
                float distance = (float)i - center;
                if (distance < 0) distance = -distance;
                float light = tick < slide_ms ? 1.0f - distance / 2.0f : 0.0f;
                if (light < 0) light = 0;
                light = light * light * (3.0f - 2.0f * light);
                vt_set_truecolor_fg((int)(0x39 + (0x9E - 0x39) * light),
                    (int)(0xC5 + (0xFF - 0xC5) * light),
                    (int)(0xBB + (0xFF - 0xBB) * light));
                printf("=");
            } else if (i == filled) {
                set_color(COLOR_CYAN, COLOR_BLACK);
                printf(">");
            } else {
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                printf(" ");
            }
        }
    } else {
        /* 进度 <5 个等号：无高光 */
        for (int i = 0; i < bar_w; i++) {
            if (i < filled) set_color(COLOR_CYAN, COLOR_BLACK), printf("=");
            else if (i == filled) set_color(COLOR_CYAN, COLOR_BLACK), printf(">");
            else set_color(COLOR_DARK_GRAY, COLOR_BLACK), printf(" ");
        }
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    ui_number_printf("] %4.1f%%  ", pct * 100.0f);
    if (!status_lines[0][0]) { set_color(COLOR_YELLOW, COLOR_BLACK); printf("(等待训练会话... S 启动)"); set_color(COLOR_WHITE, COLOR_BLACK); }
    /* 控制台开关按钮（progress 旁边隔 1-2 列，仅会话存在时显示）：指示灯 ● + 英文 CONSOLE，按下帧高亮 */
    if (status_lines[0][0]) {
        int bx0, bx1, by;
        progress_btn_geom(&bx0, &bx1, &by);
        ui_button(bx0, by, "CONSOLE", {g_trainer_console != 0, false}, ui_lamp_button());
        if (GetTickCount() - g_progress_btn_note_time < 2500) {
            gotoxy(bx0, by + 1);
            set_color(COLOR_YELLOW, COLOR_BLACK);
            print_clip(bx0, by + 1, g_console_width - 4 - bx0, "重启训练后生效");
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    /* 进度条第二行：段内进度 + 对局步数（最新局 + 近30局平均） */
    {
        float last = 0; float avg30 = 0; int n = g_hist[METRIC_GAME_LEN].n;
        if (n > 0) {
            last = g_hist[METRIC_GAME_LEN].v[n - 1];
            int take = n < 30 ? n : 30;
            double sum = 0;
            for (int i = n - 1; i >= n - take; i--) sum += g_hist[METRIC_GAME_LEN].v[i];
            avg30 = take ? (float)(sum / take) : 0;
        }
        gotoxy(x + 2, py_pbar + 2);
        if (g_pretrain_active && !g_ui_eval_active && !stage_active) {
            ui_number_printf("课程四连  Steps:%2d  Avg100:%5.2f  成功:%5.1f%%  稳定:%d/3",
                g_pretrain_steps, g_pretrain_avg_steps, g_pretrain_success * 100.0f,
                g_pretrain_stable);
        } else if (g_ui_eval_active && g_ui_eval_total > 0) {
            ui_number_printf("Minimax 评估: %d/%d 局  完成后继续训练",
                g_ui_eval_completed, g_ui_eval_total);
        } else if (stage_active) {
            const char* name = stage_progress_name(g_ui_stage.stage);
            if (g_ui_stage.indeterminate) {
                ui_number_printf("%s...", name);
            } else if (g_ui_stage.snapshot_total > 0) {
                /* 多快照：同时展示快照进度 + 当前阶段累计完成/总量。 */
                ui_number_printf("%s  快照 %d/%d · 总计 %d/%d %s",
                    name, g_ui_stage.snapshot_completed, g_ui_stage.snapshot_total,
                    g_ui_stage.completed, g_ui_stage.total,
                    stage_progress_unit(g_ui_stage.unit));
            } else {
                int spct = g_ui_stage.total > 0 ?
                    (int)(g_ui_stage.completed * 100.0f / g_ui_stage.total) : 0;
                ui_number_printf("%s  %d/%d %s  %d%%",
                    name, g_ui_stage.completed, g_ui_stage.total,
                    stage_progress_unit(g_ui_stage.unit), spct);
            }
        } else {
            ui_number_printf("Seg:%d/1000  距下次评估:%d局  Steps:%5.1f  Avg30:%5.1f",
                in_seg, in_seg ? 1000 - in_seg : 1000, last, avg30);
        }
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* Training 页：顶部总览 + 大框图表网格（每格独立指标）+ 页内底部进度条框 */
void render_training(void) {
    int tab = g_active_tab;
    int x = 2, y = CONTENT_Y;
    int w = g_console_width - 4;
    /* 刚切回 Training 时页面会先被清空，此帧必须完整画完监控值和进度条。
       不能因为某个 chart 仍带 dirty 标志就走下面的局部重绘并提前 return，
       否则状态区会空白到下一次两秒日志轮询。 */
    const int full_page_redraw = !g_page_drawn[tab] || g_force_redraw;
    if (full_page_redraw) {
        clear_rect(0, CONTENT_Y, g_console_width, (g_console_height - 4) - CONTENT_Y + 1);
        g_page_drawn[tab] = 1;
    }
    /* 顶部一行操作提示（监控值在下方监控值区，避免重复） */
    int mem_cache_b = 0, mem_kb_b = 0, mem_cache_w = 0, mem_kb_w = 0;
    sscanf(status_lines[6], "B:c%d/kb%d W:c%d/kb%d", &mem_cache_b, &mem_kb_b, &mem_cache_w, &mem_kb_w);
    gotoxy(x, y);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("[Tab]切框 [Enter]下拉选指标 [←→]轮换 (焦点框标题带v)");
    if (g_console_width >= 165)
        printf("   Win@0/1/2 = 三种评估探索率 Greedy/Stable/Robust");
    set_color(COLOR_WHITE, COLOR_BLACK);
    if (g_console_width >= 110) {
        gotoxy(g_console_width - 18, y);
        control_here("[ 全局复位 ]", false, false, 0);
        set_color(COLOR_WHITE, COLOR_BLACK);
    }

    /* 布局：阈值决定 cols/rows，每框宽高 = 可用区/格数（填满不留白）；
       进度条框固定 3 行高在页内底部 */
    int pbar_h = 3;
    int mon_h = 3;                        /* 信息层行数（网格与进度条之间，在空隙中垂直居中） */
    int grid_top = y + 1;
    int avail_w = g_console_width - 4;
    /* 进度条框固定贴底：底到 g_console_height-4（帮助栏在 H-3） */
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
    /* 网格高度：预留信息层 + 上下各 ≥1 行空隙 */
    int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
    int cols, rows, cw, ch;
    compute_chart_layout(avail_w, avail_h, &cols, &rows, &cw, &ch);
    int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
    if (g_cell_focus >= N) g_cell_focus = N - 1;
    int rem_w = avail_w - cols * cw; if (rem_w < 0) rem_w = 0;   /* 网格铺满：余数分配到前列，右边缘与进度条框对齐（避免奇数宽错位） */
    /* 信息层在网格底部与进度条顶之间的空隙中垂直居中 */
    int grid_bottom = grid_top + rows * ch;
    int gap = py_pbar - grid_bottom;
    int mon_top = grid_bottom + (gap >= mon_h ? (gap - mon_h) / 2 : 1);

    /* 部分重绘：滑块改窗口后只重绘对应格，不动监控/进度条/其他格（避免整页闪烁） */
    if (!full_page_redraw) {
        int any = 0;
        for (int i = 0; i < N; i++) if (g_cell_dirty[i]) { any = 1; break; }
        if (any) {
            for (int i = 0; i < N; i++) {
                if (!g_cell_dirty[i] && g_line_theme != 3) continue;   /* 曲线模式：所有格重绘以重注册曲线 */
                int r = i / cols, c = i % cols;
                int w_i = cw + (c < rem_w ? 1 : 0);
                int cx = x + c * cw + (c < rem_w ? c : rem_w);
                int cy = grid_top + r * ch;
                int mi = g_cell_metric[i];
                if (mi >= 0 && mi < METRIC_COUNT)
                    draw_line_series(cx, cy, w_i, ch, (MetricId)mi, g_hist[mi].v, g_hist[mi].n, i == g_cell_focus, g_cell_view_start[i], g_cell_view_end[i], i);
                g_cell_dirty[i] = 0;
            }
            return;
        }
    }

    /* 画每个图表框（每框自清自画；列宽分配余数，右边缘与进度条框对齐） */
    for (int i = 0; i < N; i++) {
        int r = i / cols, c = i % cols;
        int w_i = cw + (c < rem_w ? 1 : 0);
        int cx = x + c * cw + (c < rem_w ? c : rem_w);
        int cy = grid_top + r * ch;
        int mi = g_cell_metric[i];
        if (mi >= 0 && mi < METRIC_COUNT)
            draw_line_series(cx, cy, w_i, ch, (MetricId)mi, g_hist[mi].v, g_hist[mi].n, i == g_cell_focus, g_cell_view_start[i], g_cell_view_end[i], i);
        else {
            clear_rect(cx, cy, w_i, ch);   /* 全不勾：直接黑框 */
            draw_border(cx, cy, w_i, ch, " - ");
        }
    }
    /* 单框下拉选择面板（Enter 打开，覆盖在焦点框标题上方） */
    if (g_cell_dropdown >= 0 && g_cell_dropdown < N) {
        int dr = g_cell_dropdown / cols, dc = g_cell_dropdown % cols;
        int dcx = x + dc * cw + (dc < rem_w ? dc : rem_w);
        int dcy = grid_top + dr * ch;
        int visible = (METRIC_COUNT < METRIC_MENU_VISIBLE) ? METRIC_COUNT : METRIC_MENU_VISIBLE;
        int menu_start = g_cell_drop_cur - visible / 2;
        if (menu_start < 0) menu_start = 0;
        if (menu_start > METRIC_COUNT - visible) menu_start = METRIC_COUNT - visible;
        int mw = 22, mh = visible + 2;
        int mx = dcx + 2, my = dcy;
        if (mx + mw > x + w) mx = x + w - mw;
        if (my < grid_top) my = grid_top;
        if (my + mh > g_console_height - 3) my = g_console_height - 3 - mh;
        clear_rect(mx, my, mw, mh);   /* 清背景，屏蔽背后的折线/文字 */
        draw_border(mx, my, mw, mh, " Metric ");
        for (int row = 0; row < visible; row++) {
            int k = menu_start + row;
            gotoxy(mx + 2, my + 1 + row);
            char option[128]; snprintf(option, sizeof(option), "%s %s", g_cell_checks[g_cell_dropdown][k] ? "[x]" : "[ ]", g_metric_labels[k]);
            control_here(option, g_cell_checks[g_cell_dropdown][k] != 0, true);
        }
        vt_mask_add(mx, my, mw, mh);   /* 面板遮罩：该区域曲线被重画 cell 覆盖，不整个隐藏曲线 */
    }
    /* 监控值区（网格与进度条之间，直接打印所有监控值，无框） */
    clear_rect(x, mon_top, w, mon_h);
    gotoxy(x + 2, mon_top);
    set_color(COLOR_CYAN, COLOR_BLACK);
    ui_number_printf("EP:%s", status_lines[0][0] ? status_lines[0] : "--");
    ui_number_printf("  PH:%s  D:%s  ε:%s  LR:",
        status_lines[1][0] ? status_lines[1] : "--",
        status_lines[3][0] ? status_lines[3] : "--",
        status_lines[4][0] ? status_lines[4] : "--");
    ui_number_printf("%s", status_lines[5][0] ? status_lines[5] : "--");
    ui_number_printf("  REP:%s  t/ep:%.2fs",
        status_lines[7][0] ? status_lines[7] : "--",
        g_ep_time);
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 第二行：胜率 + B记忆库 + 黑方评估 */
    const int side_win_col = x + 80;
    gotoxy(x + 2, mon_top + 1);
    ui_number_printf("%s", status_lines[2][0] ? status_lines[2] : "W:-- L:-- D:--");
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    gotoxy(x + 60, mon_top + 1);
    ui_number_printf("B c%d kb:%d", mem_cache_b, mem_kb_b);
    set_color(COLOR_WHITE, COLOR_BLACK);
    if (g_side_win_episode > 0) {
        gotoxy(side_win_col, mon_top + 1);
        ui_number_printf("  黑@0:%.1f 黑@1:%.1f 黑@2:%.1f",
            g_black_eval_win[0], g_black_eval_win[1], g_black_eval_win[2]);
    }
    if (status_lines[8][0]) {
        gotoxy(x + w - 54, mon_top + 1);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        ui_number_printf("BW/Adv:%s", status_lines[8]);
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 第三行：时间 + W记忆库 + 白方评估 */
    gotoxy(x + 2, mon_top + 2);
    int avg_time_count = 0;
    const float avg_time_100 = recent_episode_time_average(100, &avg_time_count);
    if (avg_time_count > 0)
        ui_number_printf("100ave time:%.2fs  El:%.1fs  Rem:%.1fs  Mem:%s",
            avg_time_100, g_elapsed, g_remaining,
            g_memory_enabled ? "ON" : "OFF");
    else
        ui_number_printf("100ave time:--  El:%.1fs  Rem:%.1fs  Mem:%s",
            g_elapsed, g_remaining,
            g_memory_enabled ? "ON" : "OFF");
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    gotoxy(x + 60, mon_top + 2);
    ui_number_printf("W c%d kb:%d", mem_cache_w, mem_kb_w);
    set_color(COLOR_WHITE, COLOR_BLACK);
    if (g_side_win_episode > 0) {
        gotoxy(side_win_col, mon_top + 2);
        ui_number_printf("  白@0:%.1f 白@1:%.1f 白@2:%.1f",
            g_white_eval_win[0], g_white_eval_win[1], g_white_eval_win[2]);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 左右竖线连接图表框与进度条框，包住信息层 */
    {
        for (int yy = grid_bottom; yy < py_pbar; yy++) {
            gotoxy(x, yy); printf("│");
            gotoxy(x + w - 1, yy); printf("│");
        }
    }
    /* 进度条框（贴底）——绘制已抽到 draw_progress_bar，跑马灯定时器可单独调用 */
    draw_progress_bar(x, w);
}

void render_config(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    int w = pw;
    if (w < 80) w = 80;
    int by = CONTENT_Y + oy + ph - 1;   /* 面板底部，嵌入底部按钮 */
    if (by < y + 3) by = y + 3;
    if (!g_page_drawn[tab] || g_force_redraw) {
        /* 自画边框（顶部 + 左右，不含底部——底部每次画嵌入按钮） */
        gotoxy(x, y);
        set_color(COLOR_WHITE, COLOR_BLACK);
        printf("┌");
        for (int i = 0; i < w - 2; i++) printf("─");
        printf("┐");
        gotoxy(x + 2, y);
        set_color(COLOR_CYAN, COLOR_BLACK);
        printf("TRAINING CONFIG");
        set_color(COLOR_WHITE, COLOR_BLACK);
        for (int r = 1; r < by - y; r++) {
            gotoxy(x, y + r); printf("│");
            gotoxy(x + w - 1, y + r); printf("│");
        }
        gotoxy(x, by); printf("│");
        gotoxy(x + w - 1, by); printf("│");
        g_page_drawn[tab] = 1;
    }
    int two_col = (pw >= 160);   /* 长滑块优先：很宽才两列 */
    // GPT6-Astra: keep every new control reachable in a short terminal window.
    for(int i=0;i<CONFIG_SLIDER_COUNT;++i) g_config_sliders[i].y=-1000;
    clear_rect(x+1,y+1,w-2,by-y-1);
    const int split = ((int)CONFIG_SLIDER_COUNT + 1) / 2;
    const int visible = (by - y - 3) / 2 > 0 ? (by - y - 3) / 2 : 0;
    const int rows = two_col ? split : (int)CONFIG_SLIDER_COUNT;
    g_config_x = x; g_config_y = y; g_config_w = w; g_config_h = by - y + 1;
    g_config_visible = visible; g_config_rows = rows;
    // Follow keyboard selection only when it changes, not on every refresh.
    if (g_config_focus != g_config_last_focus && g_config_focus >= 0 &&
        g_config_focus < (int)CONFIG_SLIDER_COUNT && visible > 0) {
        const int row = two_col ? g_config_focus % split : g_config_focus;
        if (row < g_config_scroll) g_config_scroll = row;
        if (row >= g_config_scroll + visible) g_config_scroll = row - visible + 1;
    }
    g_config_last_focus = g_config_focus;
    const int max_scroll = rows > visible ? rows - visible : 0;
    if (g_config_scroll < 0) g_config_scroll = 0;
    if (g_config_scroll > max_scroll) g_config_scroll = max_scroll;
    if (two_col) {
        int col1 = x + 4, col2 = x + 4 + (w - 8) / 2;
        int sw = (w - 8) / 2 - 40;
        if (sw < 12) sw = 12; if (sw > 30) sw = 30;
        const int row_start = g_config_scroll;
        for (int i = 0; i < CONFIG_SLIDER_COUNT; i++) {
            const int row=i%split;
            if(row<row_start || row>=row_start+visible)continue;
            Slider* s = &g_config_sliders[i];
            s->width = sw;
            s->x=i<split ? col1 : col2;
            s->y=y+2+(row-row_start)*2;
            draw_slider(s, (g_config_focus == i));
        }
    } else {
        int col1 = x + 4;
        int sw = w - 34;
        if (sw < 20) sw = 20; if (sw > 40) sw = 40;
        const int start = g_config_scroll;
        for (int i = 0; i < CONFIG_SLIDER_COUNT; i++) {
            if(i<start || i>=start+visible)continue;
            Slider* s = &g_config_sliders[i];
            s->width = sw;
            s->x = col1; s->y = y + 2 + (i-start) * 2;
            draw_slider(s, (g_config_focus == i));
        }
    }
    /* 底部边框嵌入按钮（Controller 式：┗ ─ 按钮 ─ ┛，紧贴边框） */
    clear_rect(x, by, w, 1);   /* 清掉原底部，避免残留 */
    gotoxy(x, by);
    set_color(COLOR_WHITE, COLOR_BLACK);
    printf("┗");
    /* 按钮实际显示宽（含指示灯），推进 = 宽 + 1 间距 */
    auto btn_w = [](Button* b) -> int {
        return ui_button_width(b->text, config_button_style(b));
    };
    /* 先画完整底部横线（x+1 到 x+w-2） */
    gotoxy(x, by);
    set_color(COLOR_WHITE, COLOR_BLACK);
    printf("┗");
    for (int i = x + 1; i < x + w - 1; i++) printf("─");
    /* 按钮居中覆盖横线（两侧空格留出，按钮间自然保留连续横线） */
    int total = 0;
    for (int i = 0; i < CONFIG_BTN_COUNT; i++) total += btn_w(&g_config_buttons[i]);
    int sx = x + (w - total) / 2;
    if (sx < x + 2) sx = x + 2;
    for (int i = 0; i < CONFIG_BTN_COUNT; i++) {
        g_config_buttons[i].x = sx;
        g_config_buttons[i].y = by;
        draw_button(&g_config_buttons[i], (g_config_focus >= CONFIG_SLIDER_COUNT && g_config_focus - CONFIG_SLIDER_COUNT == i));
        sx += btn_w(&g_config_buttons[i]);
    }
    gotoxy(x + w - 1, by);   /* 显式定位 ┛，与右边框同列 */
    printf("┛");
    gotoxy(x + 2, by - 1);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("Tab:焦点 ↑↓:选择 ←→:调整 Enter:执行 滚轮:滚动 %d-%d/%d",
        visible ? g_config_scroll + 1 : 0,
        g_config_scroll + visible < rows ? g_config_scroll + visible : rows, rows);
    set_color(COLOR_WHITE, COLOR_BLACK);
}

void render_model(int ox, int oy, int pw, int ph) {
    int x = 2 + ox, y = CONTENT_Y + oy;
    draw_border(x, y, pw, ph, " MODEL ");
    gotoxy(x + 2, y + 1);
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("NETWORK ARCHITECTURE");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 3); printf("输入: 25 (5x5 棋盘) + 1 (轮到谁)");
    gotoxy(x + 4, y + 4); printf("  ↓  Token Embedding      25 x 16");
    gotoxy(x + 4, y + 5); printf("  ↓  Multi-Head Attention 4头 x 8维");
    gotoxy(x + 4, y + 6); printf("  ↓  Learnable Pooling    → 128");
    gotoxy(x + 4, y + 7); printf("  ↓  Dense 1024 (ReLU)");
    gotoxy(x + 4, y + 8); printf("  ↓  Dense 512 (ReLU)");
    gotoxy(x + 4, y + 9); printf("  ↓  Dense 256 (ReLU)");
    gotoxy(x + 4, y + 10); printf("  ↓  Output 25 (Q values)");
    gotoxy(x + 2, y + 12);
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("PARAMETERS");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 13); printf("约 0.85M 参数 (FP32 ~3.4MB)");
    gotoxy(x + 4, y + 15); printf("层性能计时 (前向/反向 ms) 待接入");
    set_color(COLOR_WHITE, COLOR_BLACK);
}

void render_logs(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, pw, ph, " LOG VIEWER (train.dat) ");
        gotoxy(x + 2, y + 1);
        set_color(COLOR_CYAN, COLOR_BLACK);
        printf("%-14s%-10s%-10s%-10s%-10s%s", "Step", "Loss", "Q", "TargetQ", "Eps", "Type");
        set_color(COLOR_WHITE, COLOR_BLACK);
        g_page_drawn[tab] = 1;
    }
    int visible_rows = ph > 3 ? ph - 3 : 0; // Top border, column header, bottom border.
    int shown = g_log_display_count < visible_rows ? g_log_display_count : visible_rows;
    int first = g_log_display_count - shown; // Keep the latest records visible.
    clear_rect(x + 1, y + 2, pw - 2, visible_rows);
    for (int i = 0; i < shown; i++) {
        const char* entry = g_log_display[first + i];
        long long gs; float loss, q, tq, eps; int rt;
        if (sscanf(entry, "gs=%lld loss=%f qv=%f tq=%f eps=%f rt=%d", &gs, &loss, &q, &tq, &eps, &rt) >= 6) {
            print_clip(x + 2, y + 2 + i, pw - 4,
                "%-14lld%-10.4f%-10.4f%-10.4f%-10.4f%d", gs, loss, q, tq, eps, rt);
        } else {
            print_clip(x + 2, y + 2 + i, pw - 4, "%s", entry);
        }
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

void render_checkpoint(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    int bw = pw;   /* 实际边框尺寸（列表必须按此计算，否则画出框外"兜不住"） */
    int bh = ph;
    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, bw, bh, " CHECKPOINT MANAGER ");
        g_page_drawn[tab] = 1;
    }
    /* 枚举全部 checkpoint_*.bin，按 phase/ep 数值排序（1000/20000/3000 保证数字序，非字符串序） */
    static char ck_name[512][64];
    static double ck_mb[512];
    static int ck_key[512];
    int n = 0;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(CKPT_DIR "checkpoint_*.bin", &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (n >= 512) break;
            snprintf(ck_name[n], sizeof ck_name[n], "%s", fd.cFileName);
            ck_mb[n] = fd.nFileSizeLow / (1024.0 * 1024.0);
            int ph_ = 0, ep = 0;
            if (sscanf(fd.cFileName, "checkpoint_phase%d_%d.bin", &ph_, &ep) != 2)
                sscanf(fd.cFileName, "checkpoint_seed%d_%d.bin", &ph_, &ep);   /* 多 agent 种子格式 */
            ck_key[n] = ph_ * 1000000 + ep;
            n++;
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    /* 冒泡排序（n 通常几十个） */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (ck_key[j] < ck_key[i]) {
                int tk = ck_key[i]; ck_key[i] = ck_key[j]; ck_key[j] = tk;
                char ts[64]; memcpy(ts, ck_name[i], sizeof ts); memcpy(ck_name[i], ck_name[j], sizeof ts); memcpy(ck_name[j], ts, sizeof ts);
                double td = ck_mb[i]; ck_mb[i] = ck_mb[j]; ck_mb[j] = td;
            }
    int list_h = bh - 3; if (list_h < 1) list_h = 1;
    int max_scroll = n - list_h; if (max_scroll < 0) max_scroll = 0;
    if (g_ckpt_scroll > max_scroll) g_ckpt_scroll = max_scroll;
    if (g_ckpt_scroll < 0) g_ckpt_scroll = 0;
    int mw = bw - 4; if (mw < 4) mw = 4;
    clear_rect(x + 2, y + 1, bw - 4, list_h);   /* 清列表区，防滚动后短行残留 */
    if (n == 0) {
        gotoxy(x + 2, y + 1); printf("无检查点");
    } else {
        for (int i = 0; i < list_h && g_ckpt_scroll + i < n; i++) {
            int idx = g_ckpt_scroll + i;
            set_color(COLOR_LIGHT_GRAY, COLOR_BLACK);
            print_clip(x + 2, y + 1 + i, mw, "%-45s %8.1f MB", ck_name[idx], ck_mb[idx]);
        }
    }
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    if (n > list_h) {
        int last = g_ckpt_scroll + list_h; if (last > n) last = n;
        print_roll_clip(x + 2, y + bh - 2, mw, "%d-%d / %d   [↑↓/滚轮]滚动", g_ckpt_scroll + 1, last, n);
    } else {
        gotoxy(x + 2, y + bh - 2); printf("%d 个检查点", n);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

void render_gpu(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, pw, ph, " GPU MONITOR ");
        g_page_drawn[tab] = 1;
    }
    gotoxy(x + 2, y + 1); set_color(COLOR_CYAN, COLOR_BLACK); printf("DEVICE"); set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 2); ui_number_printf("%s                              ", g_gpu_device.name);
    gotoxy(x + 4, y + 3); ui_number_printf("Backend: %s   PCI: %04X:%04X              ",
        g_gpu_device.backend, g_gpu_device.vendor_id, g_gpu_device.device_id);
    gotoxy(x + 4, y + 4); ui_number_printf("专用显存: %.1f GB   共享上限: %.1f GB          ",
        g_gpu_device.dedicated_bytes / 1073741824.0,
        g_gpu_device.shared_bytes / 1073741824.0);
    gotoxy(x + 2, y + 6); set_color(COLOR_CYAN, COLOR_BLACK); printf("后端选择"); set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 7); ui_number_printf("当前检测: %s                         ", g_gpu_device.backend);
    gotoxy(x + 2, y + 9); set_color(COLOR_CYAN, COLOR_BLACK); printf("实时遥测"); set_color(COLOR_WHITE, COLOR_BLACK);
    float gpu_util = get_gpu_usage();
    float gpu_mem_used = get_gpu_mem_used();
    float gpu_mem_total = get_gpu_mem_total();
    float cpu = get_cpu_usage();
    float mem_used, mem_total;
    get_mem_usage(&mem_used, &mem_total);
    /* 温度显示：暂缺可用传感器（ADL 对 RDNA4 失效，ADLX/WMI 待接入），恢复时取消注释 */
    //float gpu_temp = get_gpu_temp();
    gotoxy(x + 4, y + 10); ui_number_printf("GPU 利用率: %.0f%%            ", gpu_util);
    double detected_total = g_gpu_device.dedicated_bytes / 1073741824.0;
    if (detected_total <= 0.0) detected_total = g_gpu_device.shared_bytes / 1073741824.0;
    gotoxy(x + 4, y + 11); ui_number_printf("GPU 显存: %.1f/%.1f GB       ", gpu_mem_used,
        gpu_mem_total > 0 ? gpu_mem_total : (float)detected_total);
    gotoxy(x + 4, y + 12); ui_number_printf("CPU: %.0f%%   内存: %.1f/%.1f GB  ", cpu, mem_used, mem_total);

    /* GPU_Analyzer 遥测仪表盘 */
    int telem_y = y + 14;
    if (gpu_telem_client_is_connected() && gpu_telem_client_is_alive()) {
        GpuTelemSample latest;
        if (gpu_telem_client_get_latest(&latest) == 0) {
            gotoxy(x + 2, telem_y); set_color(COLOR_CYAN, COLOR_BLACK);
            printf("GPU ANALYZER"); set_color(COLOR_WHITE, COLOR_BLACK);

            gotoxy(x + 4, telem_y + 1);
            ui_number_printf("GPU: %.1f%%  VRAM: %.0f MB  CPU: %.1f%%  ",
                latest.gpu_util_pct, latest.gpu_mem_used_mb, latest.cpu_usage_pct);

            float ema_util = 0, ema_mem = 0, peak_util = 0, peak_mem = 0;
            uint32_t sample_count = 0;
            gpu_telem_client_get_stats(&ema_util, &ema_mem, &peak_util, &peak_mem, &sample_count);

            gotoxy(x + 4, telem_y + 2);
            ui_number_printf("EMA: %.1f%%  Peak: %.1f%%  Samples: %u  ",
                ema_util, peak_util, sample_count);

            gotoxy(x + 4, telem_y + 3);
            if (latest.gpu_temp_c > 0)
                ui_number_printf("温度: %.1f°C  功耗: %.1f W  时钟: %.0f MHz  ",
                    latest.gpu_temp_c, latest.gpu_power_w, latest.gpu_clock_mhz);
            else
                ui_number_printf("温度/功耗/时钟: N/A (ADLX 不可用)          ");
        }
    } else {
        gotoxy(x + 2, telem_y); set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("GPU ANALYZER: 未连接 (运行 GPU_Analyzer.exe 启用)");
        set_color(COLOR_WHITE, COLOR_BLACK);
    }

    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* ========== GPU Analyze 自适应双栏布局 (tab 3) ========== */

/* section 渲染函数类型: 返回实际占用行数 */
typedef int (*SectionRenderer)(int col, int row, int max_rows, void* ctx);

/* 渲染一个 section 到指定位置, 返回实际行数 */
static int sec_device(int col, int row, int max_rows, void* ctx);
static int sec_realtime(int col, int row, int max_rows, void* ctx);
static int sec_training_basic(int col, int row, int max_rows, void* ctx);
static int sec_training_side(int col, int row, int max_rows, void* ctx);
static int sec_training_replay(int col, int row, int max_rows, void* ctx);
static int sec_training_bwd(int col, int row, int max_rows, void* ctx);
static int sec_training_ts(int col, int row, int max_rows, void* ctx);
static int sec_gpu_timing(int col, int row, int max_rows, void* ctx);
static int sec_statistics(int col, int row, int max_rows, void* ctx);
static int sec_capabilities(int col, int row, int max_rows, void* ctx);
static int sec_capture_result(int col, int row, int max_rows, void* ctx);

/* section 渲染函数类型: 返回实际占用行数 */
typedef int (*SectionRenderer)(int col, int row, int max_rows, void* ctx);

/* 渲染一个 section 到指定位置, 返回实际行数 */
static int sec_device(int col, int row, int max_rows, void* ctx);
static int sec_realtime(int col, int row, int max_rows, void* ctx);
static int sec_training_basic(int col, int row, int max_rows, void* ctx);
static int sec_training_side(int col, int row, int max_rows, void* ctx);
static int sec_training_replay(int col, int row, int max_rows, void* ctx);
static int sec_training_bwd(int col, int row, int max_rows, void* ctx);
static int sec_training_ts(int col, int row, int max_rows, void* ctx);
static int sec_gpu_timing(int col, int row, int max_rows, void* ctx);
static int sec_statistics(int col, int row, int max_rows, void* ctx);
static int sec_capabilities(int col, int row, int max_rows, void* ctx);

/* section 定义 (每个小栏独立参与双栏分配) */
typedef struct { const char* name; SectionRenderer render; int est_height; } SectionDef;
static const SectionDef g_sections[] = {
    { "DEVICE",              sec_device,          10 },
    { "REALTIME",            sec_realtime,        18 },
    { "TRAINING PERF",       sec_training_basic,   8 },
    { "BREAKDOWN BY SIDE",   sec_training_side,    4 },
    { "REPLAY BREAKDOWN",    sec_training_replay,  7 },
    { "BACKWARD BREAKDOWN",  sec_training_bwd,    10 },
    { "TRAIN STEP OVERHEAD", sec_training_ts,      9 },
    { "GPU TIMING",          sec_gpu_timing,       7 },
    { "STATISTICS",          sec_statistics,      12 },
    { "CAPTURE RESULT",      sec_capture_result,  14 },
};
#define SECTION_COUNT (sizeof(g_sections) / sizeof(g_sections[0]))

// GPT6-Astra: Analyzer owns its lifetime; TUI sends only explicit controls.
static bool analyzer_control(const char* action) {
    HWND window = FindWindowW(L"DQNGpuAnalyzerWindow", nullptr);
    if (!strcmp(action, "start")) {
        if (window) return true;
        wchar_t path[MAX_PATH];
        DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
        if (!n || n >= MAX_PATH) return false;
        wchar_t* slash = wcsrchr(path, L'\\');
        if (!slash) return false;
        *(slash + 1) = 0;
        wchar_t command[MAX_PATH + 64];
        swprintf_s(command, L"\"%sGPU_Analyzer.exe\" --hidden", path);
        STARTUPINFOW si{}; si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        if (!CreateProcessW(nullptr, command, nullptr, nullptr, FALSE, 0,
            nullptr, path, &si, &pi)) return false;
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        return true;
    }
    if (!window) return false;
    WPARAM op = !strcmp(action, "stop") ? 0 : (!strcmp(action, "show") ? 1 : 2);
    return PostMessageW(window, WM_APP + 41, op, 0) != FALSE;
}
// GPT6-Astra: small screens omit the schematic; all telemetry remains scrollable.
static int g_analyze_scroll=0, g_analyze_scroll_max=0, g_analyze_view_rows=1;
// GPT6-Astra: 100 discrete load colors, green -> yellow -> red; missing data stays gray.
static void analyzer_load_color(float busy) {
    if(!std::isfinite(busy) || busy<0) {set_color(COLOR_DARK_GRAY,COLOR_BLACK);return;}
    int level=(int)((std::min)(busy,100.0f)*99.0f/100.0f+0.5f);
    float t=level/99.0f;
    int red=(int)(64+176*(std::min)(1.0f,2*t)+0.5f);
    int green=(int)(224-160*(std::max)(0.0f,2*t-1)+0.5f);
    vt_set_truecolor_fg(red,green,64);
}
static void draw_analyzer_chip(int x,int y,int w,int h) {
    static const auto cores=analyzer_cpu::topology();
    static char brand[49]={}; if(!brand[0]) analyzer_cpu::brand(brand);
    auto loads=analyzer_cpu::snapshot();
    // GPT6-Astra: compact rendering of the user's Die shot.txt, not a verified die map.
    if(w>=92 && h>=49 && strstr(brand,"395") && cores.size()==16) {
        const int width=92; x+=(w-width)/2;
        vt_text_clip(x,y,width,h);
        draw_border(x,y,width,49," RYZEN AI MAX+ 395 / STRIX HALO ");
        gotoxy(x+2,y+2);set_color(COLOR_DARK_GRAY,COLOR_BLACK);
        printf("OS core numbering; physical positions / CCD mapping unverified");
        gotoxy(x+2,y+3);printf("T1 / T2 = threads within each core; green -> red = 0-100%");
        for(int die=0;die<2;++die) {
            int dx=x+2+die*45,dy=y+5;
            draw_border(dx,dy,43,20,die?" CCD 1 (template) ":" CCD 0 (template) ");
            draw_border(dx+16,dy+2,11,16," L3 ");
            gotoxy(dx+18,dy+8);printf("32 MB");
            for(int half=0;half<2;++half)for(int r=0;r<4;++r) {
                int c=die*8+half*4+r,cx=dx+1+half*28,cy=dy+2+r*4;
                char title[16];snprintf(title,sizeof(title)," CORE %02d ",c);
                draw_border(cx,cy,13,4,title);
                for(size_t t=0;t<cores[c].threads.size() && t<2;++t) {
                    const auto id=cores[c].threads[t];float busy=-1;
                    for(auto v:loads)if(v.group==id.group && v.number==id.number){busy=v.busy;break;}
                    gotoxy(cx+1,cy+1+(int)t);
                    analyzer_load_color(busy);
                    if(busy>=0)printf("T%u   %5.1f%%",(unsigned)t+1,busy);
                    else printf("T%u      --",(unsigned)t+1);
                }
                set_color(COLOR_WHITE,COLOR_BLACK);
            }
        }
        int gy=y+26;
        draw_border(x+2,gy,88,19," SOC / GPU DIE ");
        const int columns[]={4,22,58,76};
        draw_border(x+40,gy+2,15,15," GPU CTRL ");
        gotoxy(x+42,gy+7);printf("Aggregate");
        gotoxy(x+42,gy+8);float gpu=get_gpu_usage();
        if(std::isfinite(gpu)&&gpu>=0)printf("%5.1f%%",gpu);else printf("--");
        for(int col=0;col<4;++col)for(int r=0;r<5;++r) {
            int wgp=col*5+r,wx=x+columns[col],wy=gy+2+r*3;
            char title[16];snprintf(title,sizeof(title)," WGP %02d ",wgp+1);
            draw_border(wx,wy,13,3,title);
            gotoxy(wx+2,wy+1);printf("CU %02d/%02d",wgp*2,wgp*2+1);
        }
        gotoxy(x+3,y+46);set_color(COLOR_DARK_GRAY,COLOR_BLACK);
        printf("WGP load hidden: no independent counters. NPU / IO omitted to fit.");
        vt_text_clip_reset();set_color(COLOR_WHITE,COLOR_BLACK);return;
    }
    const int box_w=w>54?54:w;
    x+=(w-box_w)/2;
    vt_text_clip(x,y,box_w,h);
    draw_border(x,y,box_w,h," CPU / GPU SCHEMATIC ");
    gotoxy(x+2,y+2);set_color(COLOR_CYAN,COLOR_BLACK);
    printf("%.*s",box_w-4,brand);
    gotoxy(x+2,y+3);set_color(COLOR_DARK_GRAY,COLOR_BLACK);
    printf("Logical layout (not physical die placement)");
    gotoxy(x+2,y+5);set_color(COLOR_WHITE,COLOR_BLACK);
    printf("CPU busy time / OS-reported SMT pairing");
    int row=y+7;
    const int room=h-16;
    for(size_t c=0;c<cores.size() && (int)c<room;++c) {
        gotoxy(x+2,row++);set_color(COLOR_CYAN,COLOR_BLACK);printf("C%02u",(unsigned)c);
        for(size_t t=0;t<cores[c].threads.size() && t<2;++t) {
            auto id=cores[c].threads[t];float busy=-1;
            for(auto v:loads)if(v.group==id.group && v.number==id.number){busy=v.busy;break;}
            analyzer_load_color(busy);
            if(busy<0) printf("  T%u     N/A     ",(unsigned)t+1);
            else {
                char bar[6]=".....";int n=(int)(busy/20.0f+0.5f);if(n>5)n=5;
                for(int k=0;k<n;++k)bar[k]='#';
                printf("  T%u   %5.1f%% %s",(unsigned)t+1,busy,bar);
            }
        }
    }
    if(cores.empty()){gotoxy(x+2,row++);printf("CPU topology unavailable");}
    else if((int)cores.size()>room){gotoxy(x+2,row++);printf("%u additional cores not shown",(unsigned)cores.size()-room);}
    row+=2;
    gotoxy(x+2,row++);set_color(COLOR_CYAN,COLOR_BLACK);printf("+---------------- GPU ----------------+");
    gotoxy(x+2,row++);set_color(COLOR_WHITE,COLOR_BLACK);printf("| %-35.35s |",g_gpu_device.name);
    gotoxy(x+2,row++);printf("| Overall engine: %5.1f %%             |",get_gpu_usage());
    gotoxy(x+2,row++);printf("| Per-WGP: N/A (no counter source)     |");
    gotoxy(x+2,row++);printf("+-------------------------------------+");
    gotoxy(x+2,row++);set_color(COLOR_DARK_GRAY,COLOR_BLACK);printf("No estimated / duplicated WGP utilization");
    vt_text_clip_reset();set_color(COLOR_WHITE,COLOR_BLACK);
}
static int analyzer_wheel(int mx,int my,int delta) {
    if(g_active_tab!=3 || !delta || mx<2 || mx>=g_console_width-2 ||
        my<CONTENT_Y+2 || my>=g_console_height-4) return 0;
    int n=abs(delta)/WHEEL_DELTA;if(n<1)n=1;
    g_analyze_scroll+=delta>0?-n*2:n*2;
    if(g_analyze_scroll<0)g_analyze_scroll=0;
    if(g_analyze_scroll>g_analyze_scroll_max)g_analyze_scroll=g_analyze_scroll_max;
    return 1;
}

static void analyze_begin_frame();

void render_gpu_analyze(void) {
    analyze_begin_frame();
    static ULONGLONG reconnect_at = 0;
    if (!gpu_telem_client_is_alive() && GetTickCount64() >= reconnect_at) {
        reconnect_at = GetTickCount64() + 2000;
        gpu_telem_client_close(); gpu_telem_client_open();
    }
    int tab = g_active_tab;
    int x = 2, y = CONTENT_Y;
    int w = g_console_width - 4;
    int h = g_console_height - CONTENT_Y - 3;
    if (h < 20) h = 20;

    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, w, h, " GPU ANALYZE ");
        g_page_drawn[tab] = 1;
    }

    const int connected = gpu_telem_client_is_connected() && gpu_telem_client_is_alive();
    clear_rect(x + 1, y + 1, w - 2, 1);
    gotoxy(x + 2, y + 1);
    set_color(connected ? COLOR_GREEN : COLOR_RED, COLOR_BLACK);
    printf("%s", connected ? "CONNECTED" : "DISCONNECTED");
    gotoxy(x + 18, y + 1); control_here("[Start]", false, false, 0);
    gotoxy(x + 28, y + 1); control_here("[Stop]", false, false, 0);
    gotoxy(x + 38, y + 1); control_here("[Show]", false, false, 0);
    gotoxy(x + 48, y + 1); control_here("[Hide]", false, false, 0);
    gotoxy(x + 58, y + 1);
    control_here(GetTickCount64() < g_gpu_report_copy_notice_until ? "[Copied]" : "[Copy Report]", false, false, 0);
    set_color(COLOR_WHITE, COLOR_BLACK);

    clear_rect(x + 1, y + 2, w - 2, 1);
    gotoxy(x + 2, y + 2);
    control_here(g_gpu_profile_arm_batches > 0 ? "[Cancel]" : "[Modules 8]", false, false, 0);
    gotoxy(x + 16, y + 2);
    set_color(g_gpu_profile_session_batches > 0 ? COLOR_YELLOW :
        (g_gpu_profile_arm_batches > 0 ? COLOR_CYAN : COLOR_DARK_GRAY), COLOR_BLACK);
    vt_text_clip(x + 16, y + 2, w - 18, 1);
    if (g_gpu_profile_session_batches > 0) {
        char directory[MAX_PATH] = "";
        WideCharToMultiByte(CP_ACP, 0, g_gpu_profile_session_dir, -1, directory, sizeof(directory), NULL, NULL);
        int files = count_gpu_capture_files(g_gpu_profile_session_dir, g_gpu_profile_session_mode);
        printf("%s capture: %d/%d batch files -> %s",
            g_gpu_profile_session_mode == 2 ? "HIP EVENT" : "MODULE",
            files, g_gpu_profile_session_batches, directory);
    } else if (g_gpu_profile_arm_batches > 0) {
        printf("NEXT SESSION: %s, %d batches (starts with next Trainer session)",
            gpu_profile_mode_name(g_gpu_profile_arm_mode), g_gpu_profile_arm_batches);
    } else if (g_gpu_profile_last_dir[0]) {
        char directory[MAX_PATH] = "";
        WideCharToMultiByte(CP_ACP, 0, g_gpu_profile_last_dir, -1, directory, sizeof(directory), NULL, NULL);
        printf("LAST %s capture: %d batch files -> %s",
            g_gpu_profile_last_mode == 2 ? "HIP EVENT" : "MODULE",
            g_gpu_profile_last_files, directory);
    } else {
        printf("CAPTURE: off; click Modules 8 to arm low-overhead collection");
    }
    vt_text_clip_reset();
    set_color(COLOR_WHITE, COLOR_BLACK);

    const int top=y+4, visible=h-6;
    if(visible<1) return;
    g_analyze_view_rows=visible;
    const bool chip=w>=224 && h>=54;
    const bool two=w>=140;
    const int side=chip?62:(two?(w-6)/2:w-4);
    const int right=chip?x+w-2-side:x+3+side;
    static int old_width=0,old_height=0;
    if(old_width!=w || old_height!=h){g_analyze_scroll=0;old_width=w;old_height=h;}
    clear_rect(x+1,y+3,w-2,h-4);
    int column_totals[2]={0,0};
    vt_text_clip(0,0,0,0);
    for(int i=0;i<SECTION_COUNT;++i) {
        int used=g_sections[i].render(0,0,1000,nullptr);
        column_totals[two && i>=6?1:0]+=used>0?used+1:0;
    }
    vt_text_clip_reset();
    int heights[2]={0,0};
    for(int i=0;i<SECTION_COUNT;++i) {
        int column=two && i>=6?1:0;
        int col=column?right:x+2;
        vt_text_clip(col,top,side,visible);
        int limit=column_totals[column]>visible?column_totals[column]-visible:0;
        int offset=g_analyze_scroll<limit?g_analyze_scroll:limit;
        int used=g_sections[i].render(col,top+heights[column]-offset,1000,nullptr);
        vt_text_clip_reset();
        heights[column]+=used>0?used+1:0;
    }
    int total=heights[0]>heights[1]?heights[0]:heights[1];
    g_analyze_scroll_max=total>visible?total-visible:0;
    if(g_analyze_scroll>g_analyze_scroll_max){g_analyze_scroll=g_analyze_scroll_max;need_content=1;}
    if(chip) draw_analyzer_chip(x+side+4,top,right-(x+side+4)-2,visible);
    gotoxy(x+2,y+h-2);set_color(COLOR_DARK_GRAY,COLOR_BLACK);
    printf("Wheel/drag: scroll  PgUp/PgDn  %d/%d%s",g_analyze_scroll,g_analyze_scroll_max,
        chip?"":"  (compact: chip hidden)");
    set_color(COLOR_WHITE,COLOR_BLACK);

}

/* ==================== Section 渲染函数 ==================== */

/* 数据源缓存 (每帧刷新一次) */
static struct {
    float gpu_util, gpu_mem_used_gb, gpu_mem_total_gb, cpu, mem_used, mem_total;
    float ep_time, ep_per_sec, gpu_temp;
    int cur_ep, cur_phase, connected;
    float ema_util, peak_util;
    uint32_t sample_count;
    float gpu_copy, cpu_max;
    double wddm_dedicated, wddm_shared;
    GpuTelemSample latest;
    float s_tui_peak_util;
    int inited;
} g_analyze;

// Limit OS telemetry queries to 4 Hz; never freeze the first snapshot forever.
static void analyze_begin_frame() {
    static ULONGLONG last = 0;
    ULONGLONG now = GetTickCount64();
    if (now - last >= 250) { g_analyze.inited = 0; last = now; }
}

/* 刷新数据源 (每帧只做一次) */
static void analyze_refresh(void) {
    if (g_analyze.inited) return;
    g_analyze.inited = 1;
    g_analyze.connected = gpu_telem_client_is_connected() && gpu_telem_client_is_alive();
    g_analyze.gpu_util = get_gpu_usage();
    g_analyze.gpu_mem_used_gb = get_gpu_mem_used();
    g_analyze.gpu_mem_total_gb = get_gpu_mem_total();
    g_analyze.cpu = get_cpu_usage();
    get_mem_usage(&g_analyze.mem_used, &g_analyze.mem_total);
    g_analyze.cur_ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
    g_analyze.cur_phase = -1;
    sscanf(status_lines[1], "Phase %d", &g_analyze.cur_phase);
    g_analyze.ep_time = g_ep_time;
    g_analyze.ep_per_sec = (g_analyze.ep_time > 0.01f) ? (1.0f / g_analyze.ep_time) : 0.0f;
    g_analyze.gpu_temp = -1.0f;
    g_analyze.gpu_copy = get_gpu_copy();
    g_analyze.cpu_max = get_cpu_max_core();
    g_analyze.wddm_dedicated = g_gpu_device.dedicated_bytes / 1073741824.0;
    g_analyze.wddm_shared = g_gpu_device.shared_bytes / 1073741824.0;
    g_analyze.latest = {};
    g_analyze.latest.gpu_temp_c=g_analyze.latest.gpu_power_w=g_analyze.latest.gpu_clock_mhz=-1;
    g_analyze.ema_util = g_analyze.peak_util = 0;
    g_analyze.sample_count = 0;
    if (g_analyze.connected) {
        if(gpu_telem_client_get_latest(&g_analyze.latest)!=0 || !g_analyze.latest.timestamp_ms ||
           GetTickCount64()-g_analyze.latest.timestamp_ms>5000) {
            g_analyze.latest.gpu_temp_c=g_analyze.latest.gpu_power_w=g_analyze.latest.gpu_clock_mhz=-1;
        }
        if(std::isfinite(g_analyze.latest.gpu_temp_c) && g_analyze.latest.gpu_temp_c>=0)
            g_analyze.gpu_temp=g_analyze.latest.gpu_temp_c;
        gpu_telem_client_get_stats(&g_analyze.ema_util, nullptr, &g_analyze.peak_util, nullptr, &g_analyze.sample_count);
    }
    static float s_peak = 0;
    if (g_analyze.gpu_util > s_peak) s_peak = g_analyze.gpu_util;
    g_analyze.s_tui_peak_util = s_peak;
}

struct CaptureModuleAggregate { char name[64]; unsigned count; double total_us; };
static CaptureModuleAggregate g_capture_module_stats[32]{};
static int g_capture_module_stat_count = 0;
static int g_capture_module_file_count = 0;
static wchar_t g_capture_module_cache_dir[MAX_PATH] = L"";
static ULONGLONG g_capture_module_cache_time = 0;

struct CaptureEventAggregate {
    unsigned calls = 0, valid = 0;
    unsigned long long bytes = 0, valid_bytes = 0;
    double host_us = 0, event_us = 0;
};
struct CaptureStageAggregate {
    char name[32]{};
    unsigned calls = 0, valid = 0;
    double host_us = 0, event_us = 0;
};
static CaptureEventAggregate g_capture_events[4]{}; // kernel, memset, copy, sync
static CaptureEventAggregate g_capture_copy_directions[3]{}; // H2D, D2H, D2D
static std::vector<double> g_capture_kernel_durations;
static std::vector<double> g_capture_kernel_gaps;
static std::vector<double> g_capture_operation_gaps;
static unsigned g_capture_adjacent_operation_pairs = 0;
static unsigned g_capture_valid_operation_pairs = 0;
static CaptureStageAggregate g_capture_stages[12]{};
static int g_capture_stage_count = 0;
static int g_capture_event_file_count = 0;
static int g_capture_event_complete_files = 0;
static int g_capture_event_timeline_files = 0;
static int g_capture_event_invalid_files = 0;
static wchar_t g_capture_event_cache_dir[MAX_PATH] = L"";
static ULONGLONG g_capture_event_cache_time = 0;

static int capture_event_kind_index(const char* kind) {
    if (!strcmp(kind, "kernel")) return 0;
    if (!strcmp(kind, "memset")) return 1;
    if (!strcmp(kind, "copy")) return 2;
    if (!strcmp(kind, "sync")) return 3;
    return -1;
}

static void capture_distribution(const std::vector<double>& sorted, double* mean,
    double* median, double* p95, double* peak) {
    if (sorted.empty()) { *mean = *median = *p95 = *peak = 0; return; }
    double sum = 0;
    for (double x : sorted) sum += x;
    *mean = sum / sorted.size();
    *median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) * 0.5;
    const size_t p95_index = (size_t)ceil(0.95 * sorted.size()) - 1;
    *p95 = sorted[p95_index];
    *peak = sorted.back();
}

static void refresh_capture_event_stats() {
    const wchar_t* directory = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_dir : g_gpu_profile_last_dir;
    if (!directory || !directory[0]) return;
    ULONGLONG now = GetTickCount64();
    if (now - g_capture_event_cache_time < 1000 &&
        _wcsicmp(directory, g_capture_event_cache_dir) == 0) return;
    g_capture_event_cache_time = now;
    wcscpy_s(g_capture_event_cache_dir, directory);
    memset(g_capture_events, 0, sizeof(g_capture_events));
    memset(g_capture_copy_directions, 0, sizeof(g_capture_copy_directions));
    memset(g_capture_stages, 0, sizeof(g_capture_stages));
    g_capture_stage_count = 0;
    g_capture_kernel_durations.clear();
    g_capture_kernel_gaps.clear();
    g_capture_operation_gaps.clear();
    g_capture_adjacent_operation_pairs = g_capture_valid_operation_pairs = 0;
    g_capture_event_file_count = g_capture_event_complete_files = 0;
    g_capture_event_timeline_files = g_capture_event_invalid_files = 0;

    wchar_t pattern[MAX_PATH + 32];
    if (swprintf_s(pattern, L"%s\\gpu_profile_*.csv", directory) < 0) return;
    WIN32_FIND_DATAW data{};
    HANDLE find = FindFirstFileW(pattern, &data);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ++g_capture_event_file_count;
        wchar_t path[MAX_PATH + 64];
        if (swprintf_s(path, L"%s\\%s", directory, data.cFileName) < 0) continue;
        FILE* file = _wfopen(path, L"r");
        if (!file) { ++g_capture_event_invalid_files; continue; }
        char line[512];
        int committed = 0, overflow = 1, drained = 0, metadata = 0;
        bool trace_file_good = true;
        bool timeline_file_good = true;
        double previous_end = -1.0;
        int previous_device_kind = -1;
        bool previous_timeline_valid = false;
        bool has_device_row = false;
        while (fgets(line, sizeof(line), file)) {
            if (line[0] == '#') {
                int version = 0, batch = 0;
                double wall = 0;
                if (sscanf(line, "# version=%d,batch=%d,committed=%d,overflow=%d,drained=%d,wall_us=%lf",
                    &version, &batch, &committed, &overflow, &drained, &wall) == 6 &&
                    version == 1 && batch > 0 && std::isfinite(wall) && wall > 0) metadata = 1;
                continue;
            }
            if (!strncmp(line, "kind,", 5)) continue;
            char kind[24]{}, stage[32]{}, name[96]{};
            unsigned long long bytes = 0;
            double host_us = -1, start_us = -1, end_us = -1, duration_us = -1;
            int api_ok = 0, event_ok = 0, timeline_ok = 0;
            if (sscanf(line, "%23[^,],%31[^,],%95[^,],%llu,%lf,%lf,%lf,%lf,%d,%d,%d",
                kind, stage, name, &bytes, &host_us, &start_us, &end_us, &duration_us,
                &api_ok, &event_ok, &timeline_ok) != 11) {
                trace_file_good = timeline_file_good = false; continue;
            }
            int index = capture_event_kind_index(kind);
            if (index < 0) continue;
            int direction_index = -1;
            if (index == 2) {
                if (!strcmp(name, "H2D")) direction_index = 0;
                else if (!strcmp(name, "D2H")) direction_index = 1;
                else if (!strcmp(name, "D2D")) direction_index = 2;
            }
            auto& aggregate = g_capture_events[index];
            CaptureEventAggregate* direction = direction_index >= 0 ? &g_capture_copy_directions[direction_index] : nullptr;
            ++aggregate.calls;
            if (direction) ++direction->calls;
            int stage_index = 0;
            for (; stage_index < g_capture_stage_count; ++stage_index)
                if (!strcmp(g_capture_stages[stage_index].name, stage)) break;
            if (stage_index == g_capture_stage_count && g_capture_stage_count < 12) {
                strncpy_s(g_capture_stages[stage_index].name, stage, _TRUNCATE);
                ++g_capture_stage_count;
            }
            CaptureStageAggregate* stage_aggregate = stage_index < 12 ? &g_capture_stages[stage_index] : nullptr;
            if (stage_aggregate && index != 3) ++stage_aggregate->calls;
            if (!std::isfinite(host_us) || host_us < 0 || api_ok != 1) trace_file_good = timeline_file_good = false;
            else {
                aggregate.host_us += host_us;
                if (direction) direction->host_us += host_us;
                if (stage_aggregate) stage_aggregate->host_us += host_us;
            }
            aggregate.bytes += bytes;
            if (direction) direction->bytes += bytes;
            if (index == 3) continue; // host-side synchronization has no device event.
            has_device_row = true;
            const bool event_valid = api_ok == 1 && event_ok == 1 &&
                std::isfinite(duration_us) && duration_us >= 0;
            if (event_valid) {
                ++aggregate.valid;
                aggregate.valid_bytes += bytes;
                aggregate.event_us += duration_us;
                if (direction) { ++direction->valid; direction->valid_bytes += bytes; direction->event_us += duration_us; }
                if (index == 0) g_capture_kernel_durations.push_back(duration_us);
                if (stage_aggregate) { ++stage_aggregate->valid; stage_aggregate->event_us += duration_us; }
            } else trace_file_good = timeline_file_good = false;
            const bool row_timeline_valid = timeline_ok == 1 && std::isfinite(start_us) &&
                std::isfinite(end_us) && start_us >= 0 && end_us >= start_us;
            if (previous_device_kind >= 0) {
                ++g_capture_adjacent_operation_pairs;
                if (previous_timeline_valid && row_timeline_valid && start_us >= previous_end) {
                    ++g_capture_valid_operation_pairs;
                    g_capture_operation_gaps.push_back(start_us - previous_end);
                    if (previous_device_kind == 0 && index == 0)
                        g_capture_kernel_gaps.push_back(start_us - previous_end);
                } else timeline_file_good = false;
            }
            if (!row_timeline_valid) timeline_file_good = false;
            previous_device_kind = index;
            previous_timeline_valid = row_timeline_valid;
            if (row_timeline_valid) previous_end = end_us;
        }
        fclose(file);
        const bool complete = metadata && committed == 1 && overflow == 0 && drained == 1 &&
            has_device_row && trace_file_good;
        if (complete) ++g_capture_event_complete_files;
        else ++g_capture_event_invalid_files;
        if (complete && timeline_file_good) ++g_capture_event_timeline_files;
    } while (FindNextFileW(find, &data));
    FindClose(find);
    std::sort(g_capture_kernel_durations.begin(), g_capture_kernel_durations.end());
    std::sort(g_capture_kernel_gaps.begin(), g_capture_kernel_gaps.end());
    std::sort(g_capture_operation_gaps.begin(), g_capture_operation_gaps.end());
}

static void refresh_capture_module_stats() {
    const wchar_t* directory = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_dir : g_gpu_profile_last_dir;
    if (!directory || !directory[0]) {
        g_capture_module_stat_count = g_capture_module_file_count = 0;
        g_capture_module_cache_dir[0] = L'\0';
        return;
    }
    ULONGLONG now = GetTickCount64();
    if (now - g_capture_module_cache_time < 1000 &&
        _wcsicmp(directory, g_capture_module_cache_dir) == 0) return;
    g_capture_module_cache_time = now;
    wcscpy_s(g_capture_module_cache_dir, directory);
    memset(g_capture_module_stats, 0, sizeof(g_capture_module_stats));
    g_capture_module_stat_count = g_capture_module_file_count = 0;

    wchar_t pattern[MAX_PATH + 32];
    if (swprintf_s(pattern, L"%s\\gpu_modules_*.csv", directory) < 0) return;
    WIN32_FIND_DATAW data{};
    HANDLE find = FindFirstFileW(pattern, &data);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ++g_capture_module_file_count;
        wchar_t path[MAX_PATH + 64];
        if (swprintf_s(path, L"%s\\%s", directory, data.cFileName) < 0) continue;
        FILE* file = _wfopen(path, L"r");
        if (!file) continue;
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            char name[64]{}; double us = 0;
            if (sscanf(line, "%63[^,],%lf", name, &us) != 2 || !std::isfinite(us) || us < 0) continue;
            int index = 0;
            for (; index < g_capture_module_stat_count; ++index)
                if (!strcmp(g_capture_module_stats[index].name, name)) break;
            if (index == g_capture_module_stat_count) {
                if (g_capture_module_stat_count >= (int)(sizeof(g_capture_module_stats) / sizeof(g_capture_module_stats[0]))) continue;
                strncpy_s(g_capture_module_stats[index].name, name, _TRUNCATE);
                ++g_capture_module_stat_count;
            }
            ++g_capture_module_stats[index].count;
            g_capture_module_stats[index].total_us += us;
        }
        fclose(file);
    } while (FindNextFileW(find, &data));
    FindClose(find);
}

static int sec_capture_result(int col, int row, int, void*) {
    if (!g_gpu_profile_session_batches && !g_gpu_profile_last_dir[0]) return 0;
    const int mode = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_mode : g_gpu_profile_last_mode;
    if (mode == 2) {
        refresh_capture_event_stats();
        const int expected = g_gpu_profile_session_batches > 0 ?
            g_gpu_profile_session_batches : g_gpu_profile_last_batches;
        int r = 0;
        gotoxy(col, row + r); set_color(COLOR_CYAN, COLOR_BLACK);
        printf("HIP EVENT PROFILE (SHORT DIAGNOSTIC)"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
        gotoxy(col + 2, row + r); printf("Files %d/%d  complete %d  invalid %d",
            g_capture_event_file_count, expected, g_capture_event_complete_files,
            g_capture_event_invalid_files); r++;
        gotoxy(col + 2, row + r); printf("Strict timeline: %d/%d complete batches",
            g_capture_event_timeline_files, g_capture_event_complete_files); r++;
        const char* labels[] = { "KERNEL", "MEMSET", "COPY", "SYNC" };
        for (int i = 0; i < 4; ++i) {
            const auto& a = g_capture_events[i];
            gotoxy(col + 2, row + r);
            if (i == 3) {
                printf("%-6s calls %4u  host wait %.3f ms", labels[i], a.calls, a.host_us / 1000.0);
            } else {
                printf("%-6s calls %4u  event %u/%u  host %.2f ms",
                    labels[i], a.calls, a.valid, a.calls, a.host_us / 1000.0);
            }
            r++;
            if ((i == 1 || i == 2) && a.calls) {
                gotoxy(col + 4, row + r);
                printf("requested %.2f MB  valid %.2f MB", a.bytes / 1e6, a.valid_bytes / 1e6); r++;
                gotoxy(col + 4, row + r);
                if (a.valid == a.calls && g_capture_event_complete_files == g_capture_event_file_count &&
                    g_capture_event_file_count > 0 && a.event_us > 0)
                    printf("valid event-envelope BW %.2f GB/s", a.valid_bytes / a.event_us / 1000.0);
                else
                    printf("event BW withheld: incomplete/invalid coverage");
                r++;
            }
        }
        double mean = 0, median = 0, p95 = 0, peak = 0;
        gotoxy(col + 2, row + r); printf("KERNEL duration us  mean / p50 / p95 / peak"); r++;
        capture_distribution(g_capture_kernel_durations, &mean, &median, &p95, &peak);
        gotoxy(col + 4, row + r);
        if (!g_capture_kernel_durations.empty())
            printf("%7.2f / %7.2f / %7.2f / %7.2f  n=%u", mean, median, p95, peak,
                (unsigned)g_capture_kernel_durations.size());
        else printf("withheld: no valid kernel events");
        r++;
        const bool strict_gaps = g_capture_event_file_count > 0 &&
            g_capture_event_complete_files == g_capture_event_file_count &&
            g_capture_event_timeline_files == g_capture_event_complete_files &&
            g_capture_valid_operation_pairs == g_capture_adjacent_operation_pairs;
        gotoxy(col + 2, row + r); printf("ADJACENT KERNEL gap us  mean / p50 / p95 / peak"); r++;
        capture_distribution(g_capture_kernel_gaps, &mean, &median, &p95, &peak);
        gotoxy(col + 4, row + r);
        if (strict_gaps && !g_capture_kernel_gaps.empty())
            printf("%7.2f / %7.2f / %7.2f / %7.2f  n=%u", mean, median, p95, peak,
                (unsigned)g_capture_kernel_gaps.size());
        else printf("withheld: incomplete or invalid timeline");
        r++;
        gotoxy(col + 2, row + r); printf("ALL-OP gap us  mean / p50 / p95 / peak"); r++;
        capture_distribution(g_capture_operation_gaps, &mean, &median, &p95, &peak);
        gotoxy(col + 4, row + r);
        if (strict_gaps && !g_capture_operation_gaps.empty())
            printf("%7.2f / %7.2f / %7.2f / %7.2f  n=%u", mean, median, p95, peak,
                (unsigned)g_capture_operation_gaps.size());
        else printf("withheld: incomplete or invalid timeline");
        r++;
        const char* copy_labels[] = { "H2D", "D2H", "D2D" };
        for (int i = 0; i < 3; ++i) {
            const auto& a = g_capture_copy_directions[i];
            gotoxy(col + 2, row + r);
            printf("COPY %-3s calls %4u  event %u/%u  %.2f/%.2f MB",
                copy_labels[i], a.calls, a.valid, a.calls,
                a.bytes / 1e6, a.valid_bytes / 1e6);
            r++;
            gotoxy(col + 4, row + r);
            if (a.valid == a.calls && g_capture_event_complete_files == g_capture_event_file_count &&
                g_capture_event_file_count > 0 && a.event_us > 0)
                printf("host %.3f ms  event %.3f ms  envelope %.2f GB/s",
                    a.host_us / 1000.0, a.event_us / 1000.0,
                    a.valid_bytes / a.event_us / 1000.0);
            else
                printf("host %.3f ms  event timing withheld: incomplete coverage", a.host_us / 1000.0);
            r++;
        }
        gotoxy(col + 2, row + r); set_color(COLOR_CYAN, COLOR_BLACK);
        printf("STAGE TOTALS (host call / HIP event ms)"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
        for (int i = 0; i < g_capture_stage_count && r < 38; ++i) {
            const auto& stage = g_capture_stages[i];
            gotoxy(col + 2, row + r);
            printf("%-15.15s %5.2f / %5.2f  event %u/%u",
                stage.name, stage.host_us / 1000.0, stage.event_us / 1000.0,
                stage.valid, stage.calls);
            r++;
        }
        gotoxy(col + 2, row + r); set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("Sums across files; host wait overlaps events; no GPU duty/DRAM BW"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
        return r;
    }
    refresh_capture_module_stats();
    int r = 0;
    gotoxy(col, row + r); set_color(COLOR_CYAN, COLOR_BLACK);
    printf("CAPTURED HOST MODULE PROFILE"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col + 2, row + r);
    printf("Files %d / %d", g_capture_module_file_count,
        g_gpu_profile_session_batches > 0 ? g_gpu_profile_session_batches : g_gpu_profile_last_files); r++;
    double total = 0;
    for (int i = 0; i < g_capture_module_stat_count; ++i) total += g_capture_module_stats[i].total_us;
    if (g_capture_module_stat_count == 0 || total <= 0) {
        gotoxy(col + 2, row + r); printf("Waiting for module CSV batches..."); r++;
    } else {
        for (int i = 0; i < g_capture_module_stat_count && r < 13; ++i) {
            const auto& stat = g_capture_module_stats[i];
            gotoxy(col + 2, row + r);
            printf("%-20.20s %5.1f%% %5.2f ms n=%u", stat.name,
                stat.total_us * 100.0 / total,
                stat.total_us / (double)stat.count / 1000.0, stat.count); r++;
        }
    }
    gotoxy(col + 2, row + r); set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("Host intervals; not GPU execution or DRAM bandwidth"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    return r;
}

static std::string build_gpu_analyze_report() {
    analyze_refresh();
    const int mode = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_mode : g_gpu_profile_last_mode;
    const wchar_t* directory = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_dir : g_gpu_profile_last_dir;
    const int expected = g_gpu_profile_session_batches > 0 ?
        g_gpu_profile_session_batches : g_gpu_profile_last_batches;
    const int files = count_gpu_capture_files(directory, mode);
    if (mode == 2) refresh_capture_event_stats();
    else if (mode == 1 && directory && directory[0]) refresh_capture_module_stats();

    std::ostringstream out;
    SYSTEMTIME now{}; GetLocalTime(&now);
    out << "GPU Analyze report\n"
        << "Generated: " << now.wYear << '-' << std::setfill('0') << std::setw(2) << now.wMonth
        << '-' << std::setw(2) << now.wDay << ' ' << std::setw(2) << now.wHour << ':'
        << std::setw(2) << now.wMinute << ':' << std::setw(2) << now.wSecond << " local\n"
        << "Device: " << (g_gpu_device.name[0] ? g_gpu_device.name : "N/A") << "\n"
        << "Training: episode=" << (status_lines[0][0] ? status_lines[0] : "N/A")
        << ", phase=" << (status_lines[1][0] ? status_lines[1] : "N/A") << "\n"
        << "Live telemetry: GPU current/EMA/peak=";
    if (std::isfinite(g_analyze.gpu_util) && g_analyze.gpu_util >= 0)
        out << std::fixed << std::setprecision(1) << g_analyze.gpu_util << '/' << g_analyze.ema_util << '/' << g_analyze.peak_util << "%";
    else out << "N/A";
    if (std::isfinite(g_analyze.cpu_max) && g_analyze.cpu_max >= 0)
        out << ", CPU max logical core=" << g_analyze.cpu_max << "%";
    else out << ", CPU max logical core=N/A";
    if (std::isfinite(g_analyze.ep_per_sec) && g_analyze.ep_per_sec >= 0)
        out << ", episodes/s=" << std::setprecision(3) << g_analyze.ep_per_sec;
    else out << ", episodes/s=N/A";
    const auto cpu_stats = analyzer_cpu::busy_stats();
    if (cpu_stats.count)
        out << "\nCPU max-core samples=" << cpu_stats.count << ", mean=" << cpu_stats.mean
            << "%, p95=" << cpu_stats.p95 << "%, peak=" << cpu_stats.peak << "%";
    out << '\n';

    if (!directory || !directory[0]) {
        out << "Capture: no completed or active capture in this TUI session\n"
            << "To collect: click Modules 8 on GPU Analyze (low overhead), or use gpu profile events 1-8 for a short diagnostic; then start a new Trainer session.\n";
    } else {
        char path[MAX_PATH * 4] = "";
        WideCharToMultiByte(CP_UTF8, 0, directory, -1, path, sizeof(path), nullptr, nullptr);
        out << "Capture mode: " << (mode == 2 ? "HIP events (deep diagnostic)" : "HOST modules") << "\n"
            << "Files: " << files << '/' << expected << ", output: " << path << "\n";
        if (mode == 2) {
            out << "Complete traces: " << g_capture_event_complete_files
                << ", invalid/incomplete: " << g_capture_event_invalid_files
                << ", strict timelines: " << g_capture_event_timeline_files << "\n";
            const char* labels[] = { "kernel", "memset", "copy", "sync" };
            for (int i = 0; i < 4; ++i) {
                const auto& a = g_capture_events[i];
                out << labels[i] << ": calls=" << a.calls;
                if (i != 3) out << ", event_valid=" << a.valid << '/' << a.calls;
                out << ", host_us=" << a.host_us;
                if (i != 3) out << ", event_us=" << a.event_us;
                if (i == 1 || i == 2) out << ", requested_bytes=" << a.bytes << ", valid_bytes=" << a.valid_bytes;
                out << '\n';
            }
            double mean = 0, median = 0, p95 = 0, peak = 0;
            capture_distribution(g_capture_kernel_durations, &mean, &median, &p95, &peak);
            out << "Kernel duration us (valid subset): n=" << g_capture_kernel_durations.size()
                << ", mean=" << mean << ", median=" << median << ", p95=" << p95 << ", peak=" << peak << '\n';
            const bool strict_gaps = g_capture_event_file_count > 0 &&
                g_capture_event_complete_files == g_capture_event_file_count &&
                g_capture_event_timeline_files == g_capture_event_complete_files &&
                g_capture_valid_operation_pairs == g_capture_adjacent_operation_pairs;
            if (strict_gaps) {
                capture_distribution(g_capture_kernel_gaps, &mean, &median, &p95, &peak);
                out << "Adjacent kernel gap us: n=" << g_capture_kernel_gaps.size()
                    << ", mean=" << mean << ", median=" << median << ", p95=" << p95 << ", peak=" << peak << '\n';
                capture_distribution(g_capture_operation_gaps, &mean, &median, &p95, &peak);
                out << "All-operation gap us: n=" << g_capture_operation_gaps.size()
                    << ", mean=" << mean << ", median=" << median << ", p95=" << p95 << ", peak=" << peak << '\n';
            } else out << "Gap distributions: withheld (trace incomplete or timeline invalid)\n";
            const char* copy_labels[] = { "H2D", "D2H", "D2D" };
            for (int i = 0; i < 3; ++i) {
                const auto& a = g_capture_copy_directions[i];
                out << "copy " << copy_labels[i] << ": calls=" << a.calls << ", valid=" << a.valid << '/' << a.calls
                    << ", requested_bytes=" << a.bytes << ", host_us=" << a.host_us << ", event_us=" << a.event_us;
                if (a.valid == a.calls && g_capture_event_complete_files == g_capture_event_file_count &&
                    g_capture_event_file_count > 0 && a.event_us > 0)
                    out << ", event_envelope_GB_s=" << (a.valid_bytes / a.event_us / 1000.0);
                else out << ", event_envelope_GB_s=withheld_incomplete_coverage";
                out << '\n';
            }
            out << "Memset event-envelope GB/s=";
            const auto& memset = g_capture_events[1];
            if (memset.valid == memset.calls && g_capture_event_complete_files == g_capture_event_file_count &&
                g_capture_event_file_count > 0 && memset.event_us > 0)
                out << (memset.valid_bytes / memset.event_us / 1000.0) << '\n';
            else out << "withheld_incomplete_coverage\n";
            out << "Stage totals (sum across files; host_us / HIP_event_us):\n";
            for (int i = 0; i < g_capture_stage_count; ++i) {
                const auto& stage = g_capture_stages[i];
                out << "  " << stage.name << ": " << stage.host_us << " / " << stage.event_us
                    << "; valid_events=" << stage.valid << '/' << stage.calls << '\n';
            }
        } else {
            double total = 0;
            for (int i = 0; i < g_capture_module_stat_count; ++i) total += g_capture_module_stats[i].total_us;
            out << "Host module totals (not GPU execution):\n";
            for (int i = 0; i < g_capture_module_stat_count; ++i) {
                const auto& a = g_capture_module_stats[i];
                out << "  " << a.name << ": calls=" << a.count << ", total_us=" << a.total_us
                    << ", share_pct=" << (total > 0 ? a.total_us * 100.0 / total : 0)
                    << ", mean_us=" << (a.count ? a.total_us / a.count : 0) << '\n';
            }
        }
    }
    out << "Interpretation limits:\n"
        << "- HIP event envelopes include dispatch/runtime effects; they are not pure GPU execution time.\n"
        << "- Host synchronization wait overlaps device work; do not add host wait and event time.\n"
        << "- Inter-operation gaps are not a direct GPU-idle measurement.\n"
        << "- Requested copy/memset bytes and event-envelope rates are not physical DRAM traffic/bandwidth.\n"
        << "- WGP, VALU, occupancy, memory stalls, L2 and hardware DRAM counters are unavailable in this capture.\n";
    return out.str();
}

static bool copy_gpu_analyze_report_to_clipboard() {
    const std::string report = build_gpu_analyze_report();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, report.data(), (int)report.size(), nullptr, 0);
    if (wlen <= 0) return false;
    HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, (size_t)(wlen + 1) * sizeof(wchar_t));
    if (!memory) return false;
    wchar_t* dst = (wchar_t*)GlobalLock(memory);
    if (!dst) { GlobalFree(memory); return false; }
    MultiByteToWideChar(CP_UTF8, 0, report.data(), (int)report.size(), dst, wlen);
    dst[wlen] = L'\0';
    GlobalUnlock(memory);
    if (!OpenClipboard(g_hwnd)) { GlobalFree(memory); return false; }
    const bool emptied = EmptyClipboard() != FALSE;
    HANDLE transferred = emptied ? SetClipboardData(CF_UNICODETEXT, memory) : nullptr;
    CloseClipboard();
    if (!transferred) { GlobalFree(memory); return false; }
    g_gpu_report_copy_notice_until = GetTickCount64() + 3000;
    need_content = 1;
    return true;
}

static int sec_device(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("DEVICE"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("%s", g_gpu_device.name); r++;
    gotoxy(col+2, row+r); printf("PCI: %04X:%04X", g_gpu_device.vendor_id, g_gpu_device.device_id); r++;
    gotoxy(col+2, row+r); set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("Compute:  HIP / ROCm"); r++;
    gotoxy(col+2, row+r); printf("Telemetry: Windows PDH"); r++;
    gotoxy(col+2, row+r); printf("AMD sensors: ADLX (per-field validity)"); r++;
    gotoxy(col+2, row+r);
    if (g_gpu_profile_session_batches > 0) printf("Profiler:  HOST %s CAPTURE", gpu_profile_mode_name(g_gpu_profile_session_mode));
    else if (g_gpu_profile_arm_batches > 0) printf("Profiler:  %s ARMED NEXT RUN", gpu_profile_mode_name(g_gpu_profile_arm_mode));
    else printf("Profiler:  OFF");
    r++;
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(col+2, row+r); printf("WDDM capacity: %.1f GB + %.1f GB", g_analyze.wddm_dedicated, g_analyze.wddm_shared); r++;
    return r + 1;
}

static int sec_realtime(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int c = g_analyze.connected;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("REALTIME"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("GPU Compute Engine     %5.1f %%", g_analyze.gpu_util); r++;
    gotoxy(col+2, row+r); printf("GPU Compute EMA        %5.1f %%", c ? g_analyze.ema_util : g_analyze.gpu_util); r++;
    gotoxy(col+2, row+r); printf("GPU Compute Peak       %5.1f %%", c ? g_analyze.peak_util : g_analyze.s_tui_peak_util); r++;
    gotoxy(col+2, row+r);
    if (g_analyze.gpu_copy >= 0) printf("GPU Copy Engine        %5.1f %%", g_analyze.gpu_copy);
    else printf("GPU Copy Engine          N/A"); r++;
    gotoxy(col+2, row+r); printf("HIP Device Alloc     %6.0f MB", c ? g_analyze.latest.gpu_mem_used_mb : (g_analyze.gpu_mem_used_gb * 1024.0f)); r++;
    gotoxy(col+2, row+r); printf("WDDM dedicated cap.  %.1f GB", g_analyze.wddm_dedicated); r++;
    gotoxy(col+2, row+r); printf("WDDM shared limit    %.1f GB", g_analyze.wddm_shared); r++;
    // GPT6-Astra: unavailable sensor rows do not consume the live dashboard.
    if(c && std::isfinite(g_analyze.latest.gpu_clock_mhz) && g_analyze.latest.gpu_clock_mhz>=0) {
        gotoxy(col+2,row+r++);printf("GFX Clock          %7.0f MHz",g_analyze.latest.gpu_clock_mhz);
    }
    if(c && std::isfinite(g_analyze.latest.gpu_memory_clock_mhz) && g_analyze.latest.gpu_memory_clock_mhz>=0) {
        gotoxy(col+2,row+r++);printf("Memory Clock       %7.0f MHz",g_analyze.latest.gpu_memory_clock_mhz);
    }
    if(c && std::isfinite(g_analyze.gpu_temp) && g_analyze.gpu_temp>=0) {
        gotoxy(col+2,row+r++);printf("GPU Temperature        %5.1f °C  (ADLX)",g_analyze.gpu_temp);
    }
    if(c && std::isfinite(g_analyze.latest.gpu_power_w) && g_analyze.latest.gpu_power_w>=0) {
        gotoxy(col+2,row+r++);printf("GPU Power          %7.1f W",g_analyze.latest.gpu_power_w);
    }
    gotoxy(col+2, row+r);
    if (c && g_analyze.latest.cpu_usage_pct >= 0.0f) printf("CPU System           %4.1f %%", g_analyze.latest.cpu_usage_pct);
    else if (g_analyze.cpu >= 0.0f) printf("CPU System           %4.1f %%", g_analyze.cpu);
    else printf("CPU System             N/A"); r++;
    gotoxy(col+2, row+r);
    if (g_analyze.cpu_max >= 0) printf("CPU Max Logical Core  %4.1f %%", g_analyze.cpu_max);
    else printf("CPU Max Logical Core    N/A"); r++;
    auto core_stats=analyzer_cpu::busy_stats();
    if(core_stats.count){gotoxy(col+2,row+r++);printf("MaxCore last %zu: mean %.1f%%",core_stats.count,core_stats.mean);
        gotoxy(col+2,row+r++);printf("  P95 %.1f%% / peak %.1f%%",core_stats.p95,core_stats.peak);}
    gotoxy(col+2, row+r);
    float mu = g_analyze.mem_used;
    float mt = g_analyze.mem_total;
    if (mu > 0.01f || mt > 0.01f) printf("System RAM           %.1f / %.1f GB", mu, mt);
    else printf("System RAM             N/A"); r++;
    return r;
}

static int sec_training_basic(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("TRAINING PERF"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("Episode              %d", g_analyze.cur_ep); r++;
    gotoxy(col+2, row+r); printf("Phase                %d", g_analyze.cur_phase); r++;
    gotoxy(col+2, row+r); printf("Episodes/s           %6.2f  (%.2f s/ep)", g_analyze.ep_per_sec, g_analyze.ep_time); r++;
    if (g_phase_steps > 0) {
        float fwd_ms = g_phase_forward_us / g_phase_steps / 1000.0f;
        float bwd_ms = g_phase_backward_us / g_phase_steps / 1000.0f;
        float tgt_ms = g_phase_target_us / g_phase_steps / 1000.0f;
        float total_ms = fwd_ms + bwd_ms + tgt_ms;
        float sps = (g_analyze.ep_time > 0.01f) ? (float)g_phase_steps / g_analyze.ep_time : 0;
        gotoxy(col+2, row+r); printf("Steps/s              %6.2f", sps); r++;
        gotoxy(col+2, row+r); printf("ms/step              %6.2f", total_ms); r++;
        gotoxy(col+2, row+r); printf("Forward              %6.2f ms  %4.1f%%", fwd_ms, total_ms > 0 ? fwd_ms / total_ms * 100 : 0); r++;
        gotoxy(col+2, row+r); printf("Backward             %6.2f ms  %4.1f%%", bwd_ms, total_ms > 0 ? bwd_ms / total_ms * 100 : 0); r++;
    }
    return r;
}

static int sec_training_side(int col, int row, int max_rows, void*) {
    if (g_phase_steps <= 0) return 0;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("BREAKDOWN BY SIDE"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    for (int s = 0; s < 2; s++) {
        float sf = g_side_steps[s] > 0 ? g_side_forward_us[s] / g_side_steps[s] / 1000.0f : 0;
        float sb = g_side_steps[s] > 0 ? g_side_backward_us[s] / g_side_steps[s] / 1000.0f : 0;
        gotoxy(col+2, row+r); printf("%s  Fwd %6.2f ms  Bwd %6.2f ms  (%d steps)",
            s == 0 ? "Black" : "White", sf, sb, g_side_steps[s]); r++;
    }
    return r;
}

static int sec_training_replay(int col, int row, int max_rows, void*) {
    if (g_replay_count <= 0) return 0;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("REPLAY BREAKDOWN (%d batches)", g_replay_count); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    float rs = g_replay_sample_us / g_replay_count / 1000.0f;
    float rf = g_replay_forward_us / g_replay_count / 1000.0f;
    float rt = g_replay_target_us / g_replay_count / 1000.0f;
    float rn = g_replay_train_us / g_replay_count / 1000.0f;
    float ry = g_replay_sync_us / g_replay_count / 1000.0f;
    gotoxy(col+2, row+r); printf("Sample     %6.2f ms", rs); r++;
    gotoxy(col+2, row+r); printf("Forward    %6.2f ms", rf); r++;
    gotoxy(col+2, row+r); printf("Target     %6.2f ms", rt); r++;
    gotoxy(col+2, row+r); printf("Train(GPU) %6.2f ms", rn); r++;
    if (ry > 0.01f) { gotoxy(col+2, row+r); printf("Sync/Wait  %6.2f ms", ry); r++; }
    return r;
}

static int sec_training_bwd(int col, int row, int max_rows, void*) {
    if (g_bwd_count <= 0) return 0;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("BACKWARD BREAKDOWN (%d calls)", g_bwd_count); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    float bc4 = g_bwd_fc4_us / g_bwd_count / 1000.0f;
    float bc3 = g_bwd_fc3_us / g_bwd_count / 1000.0f;
    float bc2 = g_bwd_fc2_us / g_bwd_count / 1000.0f;
    float bc1 = g_bwd_fc1_us / g_bwd_count / 1000.0f;
    float bp  = g_bwd_pool_us / g_bwd_count / 1000.0f;
    float ba  = g_bwd_attn_us / g_bwd_count / 1000.0f;
    float be  = g_bwd_embed_us / g_bwd_count / 1000.0f;
    gotoxy(col+2, row+r); printf("FC4        %6.2f ms", bc4); r++;
    gotoxy(col+2, row+r); printf("FC3        %6.2f ms", bc3); r++;
    gotoxy(col+2, row+r); printf("FC2        %6.2f ms", bc2); r++;
    gotoxy(col+2, row+r); printf("FC1        %6.2f ms", bc1); r++;
    gotoxy(col+2, row+r); printf("Pool       %6.2f ms", bp); r++;
    gotoxy(col+2, row+r); printf("Attention  %6.2f ms", ba); r++;
    gotoxy(col+2, row+r); printf("Embedding  %6.2f ms", be); r++;
    return r;
}

static int sec_training_ts(int col, int row, int max_rows, void*) {
    if (g_ts_count <= 0) return 0;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("HOST BATCH BREAKDOWN (%d batches)", g_ts_count); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    float ts_setup = g_ts_setup_us / g_ts_count / 1000.0f;
    float ts_memset = g_ts_memset_us / g_ts_count / 1000.0f;
    float ts_fwd = g_ts_fwd_us / g_ts_count / 1000.0f;
    float ts_hubler = g_ts_hubler_us / g_ts_count / 1000.0f;
    float ts_bwd = g_ts_bwd_us / g_ts_count / 1000.0f;
    float ts_valid = g_ts_validate_us / g_ts_count / 1000.0f;
    float ts_total = ts_setup + ts_memset + ts_fwd + ts_hubler + ts_bwd + ts_valid;
    gotoxy(col+2, row+r); printf("Setup(copy) %6.2f ms", ts_setup); r++;
    gotoxy(col+2, row+r); printf("Input+clear %6.2f ms", ts_memset); r++;
    gotoxy(col+2, row+r); printf("Forward     %6.2f ms", ts_fwd); r++;
    gotoxy(col+2, row+r); printf("Huber       %6.2f ms", ts_hubler); r++;
    gotoxy(col+2, row+r); printf("Backward    %6.2f ms", ts_bwd); r++;
    gotoxy(col+2, row+r); printf("Validate    %6.2f ms", ts_valid); r++;
    gotoxy(col+2, row+r); printf("Total       %6.2f ms", ts_total); r++;
    return r;
}

static int sec_gpu_timing(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("GPU TIMING"); set_color(COLOR_WHITE, COLOR_BLACK); r++;

    /* 尝试连接 kernel timer */
    static int s_kt_tried = 0;
    if (!s_kt_tried && !g_kt_client_valid) {
        gpu_kt_client_open();
        s_kt_tried = 1;
    }

    if (g_kt_client_valid && gpu_kt_client_is_alive()) {
        float fwd_ms = 0, train_ms = 0, sync_ms = 0, io_ms = 0;
        uint64_t fwd_cnt = 0, train_cnt = 0;
        if (gpu_kt_client_get_stats(&fwd_ms, &fwd_cnt, &train_ms, &train_cnt, &sync_ms, &io_ms) == 0) {
            float total_gpu_ms = fwd_ms + train_ms + sync_ms + io_ms;
            float fwd_avg = fwd_cnt > 0 ? fwd_ms / fwd_cnt : 0;
            float train_avg = train_cnt > 0 ? train_ms / train_cnt : 0;
            gotoxy(col+2, row+r); printf("GPU Forward Total    %8.1f ms (%llu calls)", fwd_ms, fwd_cnt); r++;
            gotoxy(col+2, row+r); printf("GPU Train Interval      %8.1f ms (%llu calls)", train_ms, train_cnt); r++;
            gotoxy(col+2, row+r); printf("GPU Soft Update      %8.1f ms", sync_ms); r++;
            gotoxy(col+2, row+r); printf("GPU Upload/Download  %8.1f ms", io_ms); r++;
            gotoxy(col+2, row+r); printf("Avg GPU Forward      %8.3f ms", fwd_avg); r++;
            gotoxy(col+2, row+r); printf("Avg Train Interval        %8.3f ms", train_avg); r++;
        } else {
            gotoxy(col+2, row+r); printf("Kernel timer read failed"); r++;
        }
    } else {
        gotoxy(col+2, row+r); printf("Kernel Timer           N/A  (Trainer 未启用)"); r++;
        gotoxy(col+2, row+r); printf("GPU Duty Cycle         N/A"); r++;
        gotoxy(col+2, row+r); printf("GPU Idle Ratio         N/A"); r++;
        gotoxy(col+2, row+r); printf("Avg Kernel Time        N/A"); r++;
        gotoxy(col+2, row+r); printf("Inter-Kernel Gap       N/A"); r++;
    }
    return r;
}

static int sec_statistics(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int c = g_analyze.connected;
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("STATISTICS"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("Samples          %u", g_analyze.sample_count); r++;
    gotoxy(col+2, row+r); printf("Elapsed          %.0f s", g_elapsed); r++;
    gotoxy(col+2, row+r); printf("Remaining        %.0f s", g_remaining); r++;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("GPU Compute"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("Current  %5.1f %%", g_analyze.gpu_util); r++;
    gotoxy(col+2, row+r); printf("EMA      %5.1f %%", c ? g_analyze.ema_util : g_analyze.gpu_util); r++;
    gotoxy(col+2, row+r); printf("Peak     %5.1f %%", c ? g_analyze.peak_util : g_analyze.s_tui_peak_util); r++;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("Training"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("Episode  %d", g_analyze.cur_ep); r++;
    gotoxy(col+2, row+r); printf("Phase    %d", g_analyze.cur_phase); r++;
    gotoxy(col+2, row+r); printf("Ep/s     %.2f", g_analyze.ep_per_sec); r++;
    gotoxy(col+2, row+r); printf("s/ep     %.2f", g_analyze.ep_time); r++;
    if (g_phase_steps > 0) {
        float total_ms = (g_phase_forward_us + g_phase_backward_us + g_phase_target_us) / g_phase_steps / 1000.0f;
        gotoxy(col+2, row+r); printf("Step/s   %.2f", g_analyze.ep_time > 0.01f ? (float)g_phase_steps / g_analyze.ep_time : 0); r++;
        gotoxy(col+2, row+r); printf("ms/step  %.2f", total_ms); r++;
    }
    return r;
}

static int sec_capabilities(int col, int row, int max_rows, void*) {
    analyze_refresh();
    int r = 0;
    gotoxy(col, row+r); set_color(COLOR_CYAN, COLOR_BLACK); printf("CAPABILITIES"); set_color(COLOR_WHITE, COLOR_BLACK); r++;
    gotoxy(col+2, row+r); printf("Compute Engine   AVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("Copy Engine      %s", g_analyze.gpu_copy >= 0 ? "AVAILABLE" : "UNAVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("VRAM (HIP)       AVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("VRAM (WDDM)      AVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("CPU / RAM        AVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("Temperature      %s", g_analyze.gpu_temp >= 0 ? "AVAILABLE (ADLX)" : "UNAVAILABLE"); r++;
    gotoxy(col+2, row+r); printf("Clock / Power    %s / %s",g_analyze.connected && g_analyze.latest.gpu_clock_mhz>=0?"AVAILABLE":"N/A",g_analyze.connected && g_analyze.latest.gpu_power_w>=0?"AVAILABLE":"N/A"); r++;
    gotoxy(col+2, row+r); printf("Kernel Timeline  PARTIAL"); r++;
    gotoxy(col+2, row+r); printf("VALU / DRAM BW   N/A (no profiler)"); r++;
    gotoxy(col+2, row+r); printf("Occupancy        N/A (no profiler)"); r++;
    return r;
}

void render_system(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, pw, ph, " SYSTEM ");
        g_page_drawn[tab] = 1;
    }
    gotoxy(x + 2, y + 1); printf("TUI v0.1 | Windows 控制台 | 后端: %s        ", g_gpu_device.backend);
    gotoxy(x + 2, y + 3); printf("IPC: Named Pipe (%s)", PIPE_NAME);
    gotoxy(x + 2, y + 5); printf("状态机: %s", state_name(g_state));
    if(ph>8){gotoxy(x+2,y+7);print_clip(x+2,y+7,pw-4,g_http_server.status().c_str());}
    const auto course=course_saved_lines();for(size_t i=0;i<course.size() && 9+(int)i<ph-1;++i)print_clip(x+2,y+9+(int)i,pw-4,"%s",course[i].c_str());
    set_color(COLOR_WHITE, COLOR_BLACK);
}

void render_help(int ox, int oy, int pw, int ph) {
    int tab = g_active_tab;
    int x = 2 + ox, y = CONTENT_Y + oy;
    int hh = ph;
    if (hh < 20) hh = 20;
    if (!g_page_drawn[tab] || g_force_redraw) {
        draw_border(x, y, pw, hh, " HELP ");
        g_page_drawn[tab] = 1;
    }
    set_color(COLOR_CYAN, COLOR_BLACK);
    gotoxy(x + 2, y + 1); printf("页面导航");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 2); printf("1 Training 曲线  2 PLAYGROUND 人机对战  3+ 合并功能页");
    gotoxy(x + 4, y + 3); printf("功能: Dashboard/Config/Logs/GPU/Model/System/Checkpoint/Help (随窗口合并)");
    set_color(COLOR_CYAN, COLOR_BLACK);
    gotoxy(x + 2, y + 5); printf("操作");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 6); printf("[1-9]切页  [Ctrl+P]命令  [S]启动会话  [Q]退出 鼠标:点击导航/控件");
    gotoxy(x + 4, y + 7); printf("Config页: [Tab]焦点 [↑↓]选择 [←→]调整 [Enter]执行 [R]重置 | Training页: [←→]缩放曲线");
    gotoxy(x + 4, y + 8); printf("命令面板: start session | shutdown session | tab <1-9>");
    set_color(COLOR_CYAN, COLOR_BLACK);
    gotoxy(x + 2, y + 10); printf("参数指南（Config 页调整，点「应用全部」或 Enter 生效）");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 4, y + 11); printf("训练频率  每步批次数，越大学越快但易不稳      默认4  (1~8)");
    gotoxy(x + 4, y + 12); printf("学习率    权重步长，过大震荡过小收敛慢      默认0.4 (-1=自动衰减)");
    gotoxy(x + 4, y + 13); printf("活二奖励  已由课程自动调度，不再单独缩放        （无手动项）");
    gotoxy(x + 4, y + 14); printf("封堵缩放  封堵获胜奖励倍数，提升防守            默认1  (1~2)");
    gotoxy(x + 4, y + 15); printf("高阶缩放  活三冲四奖励倍数，提升进攻            默认1  (1~3)");
    gotoxy(x + 4, y + 16); printf("记忆强度  记忆对Q影响度，0无影响1强影响         默认0.5 (0~1)");
    gotoxy(x + 4, y + 17); printf("记忆置信  记忆使用门槛，越高越保守              默认0.3 (0.1~0.5)");
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* ---------- 报告查看器（分析命令结果，分页滚动显示） ---------- */
static void viewer_load(const char* text) {
    g_viewer_line_count = 0;
    const char* p = text;
    while (*p && g_viewer_line_count < 512) {
        char* dst = g_viewer_lines[g_viewer_line_count];
        int len = 0;
        while (*p && *p != '\n' && len < 119) dst[len++] = *p++;
        dst[len] = '\0';
        if (len > 0 && dst[len - 1] == '\r') dst[len - 1] = '\0';
        g_viewer_line_count++;
        if (*p == '\n') p++;
    }
    g_viewer_scroll = 0;
}

/* ---------- 合并页：按功能优先级渲染面板网格（列优先：左上 左下 右上 右下） ---------- */
static void render_func(int fun, int ox, int oy, int pw, int ph) {
    switch (fun) {
    case FUN_DASHBOARD: render_dashboard(ox, oy, pw, ph); break;
    case FUN_SYSTEM: render_system(ox, oy, pw, ph); break;
    case FUN_GPU: render_gpu(ox, oy, pw, ph); break;
    case FUN_MODEL: render_model(ox, oy, pw, ph); break;
    case FUN_CONFIG: render_config(ox, oy, pw, ph); break;
    case FUN_LOGS: render_logs(ox, oy, pw, ph); break;
    case FUN_CHECKPOINT: render_checkpoint(ox, oy, pw, ph); break;
    case FUN_HELP: render_help(ox, oy, pw, ph); break;
    }
}

void render_merged(int page) {
    int avail_w = g_console_width - 4;
    int avail_h = (g_console_height - 4) - (CONTENT_Y + 1) + 1;
    int first = page * g_merge_n;
    if (g_merge_n == 1) {
        /* 单功能页 = 全屏显示（保留独立 tab 的完整视图） */
        int full_h = (g_console_height - 4) - CONTENT_Y + 1;
        render_func(first, 0, 0, avail_w, full_h);
        return;
    }
    int cols, rows, pw, ph;
    compute_merge_grid(avail_w, avail_h, &cols, &rows, &pw, &ph);
    /* 合并页标题（显示含哪些功能） */
    gotoxy(2, CONTENT_Y);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("合并页 %d/%d  功能:", page + 1, g_merge_pages);
    for (int k = 0; k < g_merge_n && first + k < FUN_COUNT; k++) {
        set_color(g_metric_colors[k % METRIC_COUNT], COLOR_BLACK);
        printf(" %s", FUN_NAMES[first + k]);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    // Use actual occupancy on this page, not the maximum page capacity.
    int count = FUN_COUNT - first;
    if (count > g_merge_n) count = g_merge_n;
    if (count <= 0) return;
    if (cols > count) cols = count;
    /* 列优先排布面板 */
    g_force_redraw = 1;   /* 合并页每帧重画边框（多个面板共享 tab 号，防丢边框） */
    int k = 0;
    int usable_w = avail_w - MERGE_GAP * (cols - 1);
    int ox = 0;
    for (int c = 0; c < cols; ++c) {
        int column_rows = count / cols + (c < count % cols ? 1 : 0);
        int column_w = usable_w / cols + (c < usable_w % cols ? 1 : 0);
        int usable_h = avail_h - MERGE_GAP * (column_rows - 1);
        int oy = 1; // Keep the merged-page title above the panels.
        for (int r = 0; r < column_rows; ++r, ++k) {
            int panel_h = usable_h / column_rows + (r < usable_h % column_rows ? 1 : 0);
            clear_rect(2 + ox, CONTENT_Y + oy, column_w, panel_h);
            render_func(first + k, ox, oy, column_w, panel_h);
            oy += panel_h + MERGE_GAP;
        }
        ox += column_w + MERGE_GAP;
    }
    g_force_redraw = 0;
}

/* 当前 tab 是否包含指定功能（合并页内可能有多个功能面板） */
int current_tab_has_fun(int fun) {
    if (g_active_tab < 4) return 0;   /* tab 0=Training, 1=PLAYGROUND, 2=Probe, 3=GPU_Analyze 不参与合并 */
    int page = g_active_tab - 4;
    for (int k = 0; k < g_merge_n && page * g_merge_n + k < FUN_COUNT; k++)
        if (page * g_merge_n + k == fun) return 1;
    return 0;
}

/* 动态 tab 名（PLAYGROUND 固定第 2 位；合并页 = 被合并功能名 + 号连接） */
void build_tab_names(void) {
    memset(g_tab_names, 0, sizeof(g_tab_names));
    snprintf(g_tab_names[0], 64, "1 Training");
    snprintf(g_tab_names[1], 64, "2 PLAYGROUND");
    snprintf(g_tab_names[2], 64, "3 Probe");
    snprintf(g_tab_names[3], 64, "4 GPU Analyze");
    for (int p = 0; p < g_merge_pages && 4 + p < MAX_TABS; p++) {
        int first = p * g_merge_n;
        char tmp[56] = "";
        for (int k = 0; k < g_merge_n && first + k < FUN_COUNT; k++) {
            if (k > 0) strncat(tmp, "+", sizeof(tmp) - strlen(tmp) - 1);
            strncat(tmp, FUN_NAMES[first + k], sizeof(tmp) - strlen(tmp) - 1);
        }
        snprintf(g_tab_names[4 + p], 64, "%d %s", 5 + p, tmp);
    }
}

void render_viewer(void) {
    draw_tab_bar();      /* 查看器是模态，但保留导航栏/帮助栏，避免整屏黑 */
    draw_help_bar();
    int vh = g_console_height - CONTENT_Y - 3;
    if (vh < 4) vh = 4;
    clear_rect(0, CONTENT_Y, g_console_width, vh);
    gotoxy(0, CONTENT_Y);
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("=== %s (分析报告) ===   [↑↓]滚动 [PgUp/Dn]翻页 [Home/End]首尾 [Esc]返回", g_viewer_title);
    set_color(COLOR_WHITE, COLOR_BLACK);
    int visible = vh - 2;
    int total = g_viewer_line_count;
    if (g_viewer_scroll > total - visible) g_viewer_scroll = total - visible;
    if (g_viewer_scroll < 0) g_viewer_scroll = 0;
    for (int i = 0; i < visible; i++) {
        int li = g_viewer_scroll + i;
        if (li >= total) break;
        gotoxy(0, CONTENT_Y + 1 + i);
        char disp[118];
        strncpy(disp, g_viewer_lines[li], 117); disp[117] = '\0';
        printf("%s", disp);
        for (int c = get_display_width(disp); c < g_console_width; c++) printf(" ");
    }
    gotoxy(0, CONTENT_Y + vh - 1);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    if (total == 0) printf("(空报告)  [Esc]返回");
    else {
        int last = g_viewer_scroll + visible - 1;
        if (last > total - 1) last = total - 1;
        ui_number_printf("行 %d-%d / %d   (Esc 返回)", g_viewer_scroll, last, total);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* ---------- 对局回放渲染（模态） ---------- */
void render_replay(void) {
    draw_tab_bar();
    draw_help_bar();
    int x = 2, y = CONTENT_Y;
    int w = 60, h = 20;
    char title[96];
    snprintf(title, sizeof(title), " REPLAY: %s ", tui_replay_meta());
    draw_border(x, y, w, h, title);

    /* 重建棋盘 */
    char board[REPLAY_BOARD_SIZE][REPLAY_BOARD_SIZE];
    memset(board, '.', sizeof(board));
    int n = tui_replay_count();
    for (int i = 0; i <= g_replay_cur && i < n; i++) {
        int pos = tui_replay_steps()[i].action;
        if (pos >= 0 && pos < REPLAY_BOARD_SIZE * REPLAY_BOARD_SIZE)
            board[pos / REPLAY_BOARD_SIZE][pos % REPLAY_BOARD_SIZE] =
                (tui_replay_steps()[i].player == 0) ? 'X' : 'O';
    }
    /* 列标 + 棋盘 */
    gotoxy(x + 6, y + 2);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    for (int c = 0; c < REPLAY_BOARD_SIZE; c++) printf(" %d", c);
    set_color(COLOR_WHITE, COLOR_BLACK);
    for (int r = 0; r < REPLAY_BOARD_SIZE; r++) {
        gotoxy(x + 4, y + 3 + r);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("%d", r);
        set_color(COLOR_WHITE, COLOR_BLACK);
        gotoxy(x + 6, y + 3 + r);
        for (int c = 0; c < REPLAY_BOARD_SIZE; c++) {
            if (board[r][c] == 'X') set_color(COLOR_CYAN, COLOR_BLACK);
            else if (board[r][c] == 'O') set_color(COLOR_RED, COLOR_BLACK);
            else set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            printf(" %c", board[r][c]);
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 当前步信息 */
    if (n > 0 && g_replay_cur >= 0 && g_replay_cur < n) {
        GameStep st = tui_replay_steps()[g_replay_cur];
        gotoxy(x + 2, y + 10);
        ui_number_printf("第 %d/%d 步  %s 落子(%d,%d) 奖励 %.3f 终局:%d  ",
            g_replay_cur + 1, n,
            st.player == 0 ? "黑(X)" : "白(O)",
            st.action / REPLAY_BOARD_SIZE, st.action % REPLAY_BOARD_SIZE,
            st.reward, st.done);
    }
    /* 终局胜负 */
    gotoxy(x + 2, y + 12);
    int winner = tui_replay_winner();
    if (winner == 1) printf("对局结束: 黑方(X)获胜  ");
    else if (winner == 2) printf("对局结束: 白方(O)获胜  ");
    else if (winner == 0) printf("对局结束: 平局  ");
    else printf("对局进行中...  ");
    /* 操作提示 */
    gotoxy(x + 2, y + 14);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("[Space/→]下一步 [←]上一步 [Home/End]首尾 [Esc]退出");
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 命令面板 g/m → 加载棋谱并打开回放模态 */
void start_replay(const char* cmd) {
    int n = tui_replay_load(cmd);
    if (n <= 0) {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "%s", tui_replay_meta());
        return;   /* 加载失败，面板保持打开显示原因 */
    }
    g_palette_open = 0;
    g_replay_open = 1;
    g_replay_cur = 0;
    clear_screen();
    need_tab = 0; need_content = 0; need_help = 0;
}

/* ---------- PLAYGROUND 人机对战 ---------- */
/* 本地判胜（5x5 四连，1/2 为 X/O 玩家） */
static int play_win_check(const int board[25], int p) {
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 2; c++)
            if (board[r*5+c] == p && board[r*5+c+1] == p && board[r*5+c+2] == p && board[r*5+c+3] == p) return 1;
    for (int c = 0; c < 5; c++)
        for (int r = 0; r < 2; r++)
            if (board[r*5+c] == p && board[(r+1)*5+c] == p && board[(r+2)*5+c] == p && board[(r+3)*5+c] == p) return 1;
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            if (board[r*5+c] == p && board[(r+1)*5+c+1] == p && board[(r+2)*5+c+2] == p && board[(r+3)*5+c+3] == p) return 1;
    for (int r = 0; r < 2; r++)
        for (int c = 3; c < 5; c++)
            if (board[r*5+c] == p && board[(r+1)*5+c-1] == p && board[(r+2)*5+c-2] == p && board[(r+3)*5+c-3] == p) return 1;
    return 0;
}

/* PLAYGROUND 面板布局：大屏 4 列主区(MODEL|BOARD|AI|INFO) + 3 列下区；中屏 3+3；小屏 BOARD+AI 主区 + 左侧侧边栏 */
void compute_playground_layout(PlayPanel* p) {
    int avail_w = g_console_width - 4;
    int top = CONTENT_Y + 3;   /* 大框顶+标题行+横线 后内容区起点 */
    int content_bot = g_console_height - 10;   /* 内容区底（下方留状态+2 / 横线 / 操作两行+4,+5 / 底边+6） */
    int avail_h = content_bot - top + 1;
    if (avail_h < 12) avail_h = 12;
    int cols = g_console_width >= 180 ? 4 : (g_console_width >= 150 ? 3 : 0);
    if (cols > 0) {
        int cw = avail_w / cols;
        int main_h = avail_h * 55 / 100;
        int low_h = avail_h - main_h - 1;
        int x0 = 2 + (avail_w - cols * cw) / 2;   /* 整体居中 */
        if (x0 < 2) x0 = 2;
        if (cols == 4) {
            int info_h = main_h / 2;
            int eval_h = main_h - info_h - 1;
            int rcw = avail_w - 3*cw;
            p[PLY_MODEL]   = { x0,                 top, cw, main_h };
            p[PLY_BOARD]   = { x0 + cw,            top, cw, main_h };
            p[PLY_AI]      = { x0 + 2*cw,          top, cw, main_h };
            p[PLY_INFO]    = { x0 + 3*cw,          top, rcw, info_h };
            p[PLY_EVAL]    = { x0 + 3*cw,          top + info_h + 1, rcw, eval_h };
            p[PLY_SETTINGS]= { x0,                 top + main_h + 1, cw, low_h };
            p[PLY_HIST]    = { x0 + cw,            top + main_h + 1, cw, low_h };
            p[PLY_TEST]    = { x0 + 2*cw,          top + main_h + 1, 2*cw, low_h };
        } else {
            int info_h = main_h / 2;
            int eval_h = main_h - info_h - 1;
            int rcw = avail_w - 2*cw;
            p[PLY_MODEL]   = { x0,                 top, cw, main_h };
            p[PLY_BOARD]   = { x0 + cw,            top, cw, main_h };
            p[PLY_AI]      = { x0 + 2*cw,          top, rcw, main_h };
            p[PLY_INFO]    = { x0 + 2*cw,          top, rcw, info_h };
            p[PLY_EVAL]    = { x0 + 2*cw,          top + info_h + 1, rcw, eval_h };
            p[PLY_SETTINGS]= { x0,                 top + main_h + 1, cw, low_h };
            p[PLY_HIST]    = { x0 + cw,            top + main_h + 1, cw, low_h };
            p[PLY_TEST]    = { x0 + 2*cw,          top + main_h + 1, rcw, low_h };
        }
    } else {
        /* 小屏：BOARD + AI 主区，MODEL/SETTINGS/HIST/EVAL 收进左侧边栏 */
        int cw = avail_w / 4;
        int main_w = avail_w - cw - 1;
        int colw2 = main_w / 2;
        int x0 = 2;
        int main_x = x0 + cw + 1;
        PlayPanel sb = { x0, top, cw, avail_h };
        p[PLY_MODEL] = sb;
        p[PLY_SETTINGS] = sb;
        p[PLY_HIST] = sb;
        p[PLY_EVAL] = sb;
        p[PLY_TEST] = { main_x, top, main_w, avail_h };
        p[PLY_BOARD] = { main_x, top, colw2, avail_h };
        p[PLY_AI]    = { main_x + colw2, top, main_w - colw2, avail_h };
        p[PLY_INFO]  = { main_x, top + avail_h / 2, main_w, avail_h - avail_h / 2 };   /* 预留，未用 */
    }
}

/* ---- 各面板渲染 ---- */
/* checkpoint 显示短名：checkpoint_phase1_1000.bin → phase1_1000（去前缀/后缀，防窄面板超宽） */
static const char* play_ckpt_disp(void) {
    static char disp[32];
    const char* p = strstr(g_play_ckpt, "checkpoint_");
    if (p) {
        p += 10;   /* 跳过 "checkpoint_" */
        snprintf(disp, sizeof disp, "%s", p);
        char* dot = strstr(disp, ".bin");
        if (dot) *dot = '\0';
        return disp;
    }
    return g_play_ckpt;
}

/* action → 坐标 "C4"（行字母 A-E + 列数字 1-5，与棋盘行标/列标对应） */
static void play_coord(int idx, char* out) {
    out[0] = (char)('A' + (idx / 5));   /* 行 → 字母 */
    out[1] = (char)('1' + (idx % 5));   /* 列 → 数字 */
    out[2] = '\0';
}

static void draw_play_model(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " MODEL SNAPSHOT ");
    char ep[24], prm[24];
    snprintf(ep, sizeof(ep), "%d", g_eval_episode);
    int pc = playground_param_count();
    if (pc >= 1000000) snprintf(prm, sizeof(prm), "%.2fM", pc / 1000000.0);
    else snprintf(prm, sizeof(prm), "%d", pc);
    int mw = p.w - 3;   /* 内容可用列：x+2 .. x+w-2 */
    print_roll_clip(p.x + 2, p.y + 1, mw, "%-10s %s", "Source", "Current Trainer");
    print_roll_clip(p.x + 2, p.y + 2, mw, "%-10s %s", "Episode", ep);
    print_roll_clip(p.x + 2, p.y + 3, mw, "%-10s %s", "Checkpoint", play_ckpt_disp());
    print_roll_clip(p.x + 2, p.y + 4, mw, "%-10s %s / %s", "Backend", g_gpu_device.backend, g_gpu_device.name);
    print_roll_clip(p.x + 2, p.y + 5, mw, "%-10s %s", "Precision", "FP32");
    print_roll_clip(p.x + 2, p.y + 6, mw, "%-10s %s", "Parameters", prm);
    print_roll_clip(p.x + 2, p.y + 7, mw, "%-10s %s", "Created", playground_created_at());
}

static void draw_play_board_panel(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " GAME BOARD ");
    /* 格宽 3（体2+分隔1），串长 16；棋子/列标按步进 3 对齐格子中心 */
    int bx = p.x + (p.w - 16) / 2;
    if (bx < p.x + 2) bx = p.x + 2;
    int by = p.y + 2;
    /* 列标 1-5（对齐格子中心 bx+1+c*3 起点） */
    gotoxy(bx + 1, by);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    for (int c = 0; c < 5; c++) printf(" %d ", c + 1);
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 顶部边框（灰色，与其余边框统一） */
    gotoxy(bx, by + 1);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("┌──┬──┬──┬──┬──┐");
    set_color(COLOR_WHITE, COLOR_BLACK);
    for (int r = 0; r < 5; r++) {
        gotoxy(bx - 1, by + 2 + r * 2);
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        printf("%c", 'A' + r);
        set_color(COLOR_WHITE, COLOR_BLACK);
        for (int c = 0; c < 5; c++) {
            int idx = r * 5 + c;
            gotoxy(bx + 1 + c * 3, by + 2 + r * 2);
            if (idx == g_play_cur && g_play_over < 0) set_color(COLOR_YELLOW, COLOR_BLACK);
            else if (g_play_board[idx] == 1) set_color(COLOR_CYAN, COLOR_BLACK);
            else if (g_play_board[idx] == 2) set_color(COLOR_RED, COLOR_BLACK);
            else set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            printf(" %s ", g_play_board[idx] == 1 ? "X" : g_play_board[idx] == 2 ? "O" : "·");
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
        if (r < 4) {
            gotoxy(bx, by + 3 + r * 2);
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            printf("├──┼──┼──┼──┼──┤");
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    gotoxy(bx, by + 11);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("└──┴──┴──┴──┴──┘");
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 底部状态 */
    gotoxy(p.x + 2, p.y + p.h - 3);
    if (g_play_over < 0) {
        if (g_play_ai_pending && g_play_manual_step) {
            set_color(COLOR_CYAN, COLOR_BLACK);
            printf("AI INFERRED — Check Q table, SPACE to confirm");
            set_color(COLOR_WHITE, COLOR_BLACK);
        } else if (g_play_ai_pending) {
            printf("AI WAITING — Press SPACE");
        } else {
            printf("YOUR TURN — %s", g_play_moves % 2 == 0 ? "WHITE(O)" : "BLACK(X)");
        }
    } else if (g_play_over == 1) printf("AI WINS!");
    else if (g_play_over == 2) printf("YOU WIN!");
    else printf("DRAW");
    /* 手动推进勾选框 */
    gotoxy(p.x + 2, p.y + p.h - 2);
    ui_button(p.x + 2, p.y + p.h - 2, "STEP", {g_play_manual_step != 0, false}, ui_lamp_button());
    set_color(COLOR_WHITE, COLOR_BLACK);
}

static void draw_play_ai(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " AI ANALYSIS ");
    int cx = p.x + 2;
    if (!g_play_analysis_on) { gotoxy(cx, p.y + 1); printf("(分析已关闭, [A] 开启)"); return; }
    if (g_play_cand_n <= 0) { gotoxy(cx, p.y + 1); printf("(等待分析...)"); return; }
    char mc[8];
    play_coord(g_play_cands[0].move, mc);
    gotoxy(cx, p.y + 1); printf("%-17s %s", "Recommended Move", mc);
    gotoxy(cx, p.y + 2); ui_number_printf("%-17s %.4f", "Q Value", g_play_cands[0].q);
    gotoxy(cx, p.y + 3); ui_number_printf("%-17s %.2f%%", "Policy", g_play_cands[0].policy * 100);
    gotoxy(cx, p.y + 4); ui_number_printf("%-17s %.1f%%", "Win Probability", g_play_cands[0].win * 100);
    /* 分隔线 */
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    gotoxy(cx, p.y + 5);
    for (int i = 0; i < p.w - 6 && i < 40; i++) printf("─");
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* TOP CANDIDATE MOVES（保证完整显示，压缩字段宽） */
    gotoxy(cx, p.y + 6); set_color(COLOR_CYAN, COLOR_BLACK);
    printf("TOP CANDIDATE MOVES");
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(cx, p.y + 7); printf("%-5s %-7s %-7s %s", "Move", "Q", "Pol%", "Win%");
    for (int i = 0; i < g_play_cand_n && i < 5; i++) {
        char cc[8];
        play_coord(g_play_cands[i].move, cc);
        gotoxy(cx, p.y + 8 + i);
        printf("%-5s %-7.4f %-7.1f %.1f", cc, g_play_cands[i].q,
            g_play_cands[i].policy * 100, g_play_cands[i].win * 100);
    }
    gotoxy(cx, p.y + p.h - 2);
    ui_number_printf("Inference: %.2f ms", g_play_infer_ms);
}

static void draw_play_settings(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " GAME SETTINGS ");
    int mw = p.w - 3;
    print_clip(p.x + 2, p.y + 1, mw, "%-14s %s", "Your Color", "WHITE");
    print_clip(p.x + 2, p.y + 2, mw, "%-14s %s", "AI Color", "BLACK");
    print_clip(p.x + 2, p.y + 3, mw, "%-14s %s", "Board Size", "5 x 5");
    print_clip(p.x + 2, p.y + 4, mw, "%-14s %s", "Think Time", "2.0 sec");
    print_clip(p.x + 2, p.y + 5, mw, "%-14s %s", "Analysis", g_play_analysis_on ? "ON" : "OFF");
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    control_clip(p.x + 2, p.y + p.h - 2, mw, "[ ENTER ] EDIT SETTINGS", false, false, VK_RETURN);
    set_color(COLOR_WHITE, COLOR_BLACK);
}

static int g_hist_scroll = 0;   /* Move History 滚动偏移 */

static void draw_play_hist(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " MOVE HISTORY ");
    int total = g_play_hist_n;
    int visible = p.h - 3;   /* 可显示行数 */
    if (visible < 1) visible = 1;
    /* 滚轮偏移限幅：从最新往旧滚 */
    int max_scroll = total - visible;
    if (max_scroll < 0) max_scroll = 0;
    if (g_hist_scroll < 0) g_hist_scroll = 0;
    if (g_hist_scroll > max_scroll) g_hist_scroll = max_scroll;
    int mw = p.w - 3;
    int start = total - visible - g_hist_scroll;   /* 从哪条开始显示 */
    if (start < 0) start = 0;
    for (int i = 0; i < visible && start + i < total; i++) {
        int idx = start + i;
        char cc[8];
        int pos = g_play_hist[idx];
        play_coord(pos, cc);
        int player = g_play_board[pos];   /* 1=黑(X)AI, 2=白(O)用户 */
        set_color(player == 1 ? COLOR_CYAN : COLOR_RED, COLOR_BLACK);
        print_clip(p.x + 2, p.y + 1 + i, mw, "%02d  %-3s  ●  %s", idx + 1, cc, player == 1 ? "B" : "W");
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
}

static void draw_play_info(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " GAME INFO ");
    char cc[8];
    play_coord(g_play_cur, cc);
    gotoxy(p.x + 2, p.y + 1); ui_number_printf("%-14s %s", "Column", cc);
    gotoxy(p.x + 2, p.y + 2); ui_number_printf("%-14s %d", "Moves", g_play_moves);
    gotoxy(p.x + 2, p.y + 3); ui_number_printf("%-14s %s", "Your Turn", g_play_moves % 2 == 0 ? "WHITE" : "BLACK");
    gotoxy(p.x + 2, p.y + 5);
    if (g_play_over == 1) printf("Status: YOU WIN");
    else if (g_play_over == 2) printf("Status: AI WINS");
    else if (g_play_over == 0) printf("Status: DRAW");
    else printf("Status: PLAYING");
}

static void draw_play_eval(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " EVALUATION ");
    int mw = p.w - 3;
    print_roll_clip(p.x + 2, p.y + 1, mw, "%-16s %.2f ms", "Inference", g_play_infer_ms);
    print_roll_clip(p.x + 2, p.y + 2, mw, "%-16s %s", "Network", "forward_cpu");
    print_roll_clip(p.x + 2, p.y + 3, mw, "%-16s %.2f ms", "Total Think", g_play_infer_ms);
    print_roll_clip(p.x + 2, p.y + 5, mw, "%-16s %s", "Candidates", g_play_cand_n > 0 ? "5" : "-");
}

/* ========== 技能回归测试面板 ========== */
#define SKILL_MAX_DISPLAY 400
static SkillDisplayRecord g_skill_display[SKILL_MAX_DISPLAY];
static int g_skill_display_count = 0;
static int g_skill_test_scroll = 0;   /* 历史记录滚动偏移（0=最新在顶部） */
static int g_skill_selected_hist = -1; /* 选中的历史记录索引，-1=未选中 */
static DWORD g_skill_last_read = 0;
static char g_snap_test_error[96] = {};

/* 快照测试状态 */
static std::atomic<int> g_snap_test_running{0};
static std::atomic<int> g_snap_test_done{0};
static std::atomic<int> g_snap_test_progress{0};
static std::atomic<int> g_snap_test_phase{0};
static std::atomic<int> g_snap_test_just_done{0};
static DWORD g_snap_test_start_tick = 0;   /* 测试启动时间（超时保护用） */
#define SKILL_TEST_TIMEOUT_MS 120000        /* 2分钟超时自动重置 */
/* DQN_Core skill_test.cpp 导出的实时进度 */
extern volatile int g_skill_test_progress;
extern volatile int g_skill_test_phase;

/* PG 手动 Skill Test 的阶段进度（与训练器共用 stage_progress 模型）。 */
static StageProgress g_snap_stage = {};
static std::mutex g_snap_stage_mutex;

static void pg_stage_cb(const StageProgress* sp, void*) {
    { std::lock_guard<std::mutex> lk(g_snap_stage_mutex); g_snap_stage = *sp; }
    /* 同步旧的 0-100 原子字段，兼容未迁移的显示路径。 */
    if (sp->total > 0 && !sp->indeterminate)
        g_snap_test_progress = (int)(sp->completed * 100.0f / sp->total);
    int ph = 0;
    if (sp->stage == STG_SKILL_EMPTY) ph = 1;
    else if (sp->stage == STG_SKILL_SEEDED) ph = 2;
    else if (sp->stage == STG_SKILL_WIN1) ph = 3;
    else if (sp->stage == STG_SKILL_BLOCK1) ph = 4;
    g_snap_test_phase = ph;
}

static StageProgress read_snap_stage(void) {
    std::lock_guard<std::mutex> lk(g_snap_stage_mutex);
    return g_snap_stage;
}
static DWORD g_run_btn_press = 0;     /* RUN 按钮按下时间（反馈用） */
static DWORD g_copy_btn_press = 0;    /* COPY 按钮按下时间（反馈用） */
static int g_copy_btn_state = 0;      /* 0=空闲, 1=COPIED! */
typedef struct {
    int episode;
    float empty_win_rate;
    float empty_avg_steps;
    float empty_shortest_rate;
    float seeded_win_rate, seeded_avg_steps;
    float win1_fixed;
    float win1_fixed_margin;
    float block1_fixed;
    float block1_fixed_margin;
    /* 详细统计 */
    int win1_correct;
    int win1_total;
    int block1_correct;
    int block1_total;
    int empty_wins;
    int empty_games;
    float win1_avg_q_correct;    /* 选对时的平均 Q 值 */
    float win1_avg_q_wrong;      /* 选错时的平均 Q 值 */
    float block1_avg_q_correct;
    float block1_avg_q_wrong;
    /* min/max 统计 */
    float win1_min_q;            /* 最低 Q 值（所有题目中） */
    float win1_max_q;            /* 最高 Q 值 */
    float win1_min_margin;       /* 最小 margin（最不确定的一题） */
    float win1_max_margin;       /* 最大 margin */
    float block1_min_q;
    float block1_max_q;
    float block1_min_margin;
    float block1_max_margin;
    float empty_min_steps;       /* 最短完成步数 */
    float empty_max_steps;       /* 最长完成步数 */
    /* 64快照统计口径 */
    float empty_stat_mean;       /* 快照平均胜率均值 */
    float empty_stat_std;        /* 标准差 */
    float empty_stat_min;        /* 最低快照胜率 */
    float empty_stat_max;        /* 最高快照胜率 */
    float empty_steps_mean;      /* 快照平均步数均值 */
    float empty_steps_std;       /* 步数标准差 */
    /* 来源信息 */
    int source_phase;            /* 快照来源阶段 */
    int source_episode;          /* 快照来源局数 */
    char source_info[96];        /* 快照来源描述 */
} SnapTestResult;
static SnapTestResult g_snap_result;
static DWORD g_snap_test_time = 0;
static DWORD g_snap_test_start = 0;

static void read_skill_metrics_cached(void) {
    static std::string cached_directory;
    if (cached_directory != g_sampled_directory) {
        g_skill_last_read = 0;
        g_skill_display_count = 0;
        cached_directory = g_sampled_directory;
    }
    DWORD now = GetTickCount();
    if (now - g_skill_last_read < 5000 && g_skill_last_read != 0) return;
    g_skill_last_read = now;
    /* 64 快照只改变统计取样方式，未改变 29 字段记录布局。保持单一 v4
       文件，避免双文件合并、去重及切换边界带来的额外状态。 */
    SkillDisplayRecord incoming[SKILL_MAX_DISPLAY];
    int count = read_skill_metrics_tail(
        tui_metric_log_path(SKILL_METRIC_FILENAME).c_str(), incoming, SKILL_MAX_DISPLAY);
    if (count >= 0) {
        memcpy(g_skill_display, incoming, (size_t)count * sizeof(incoming[0]));
        g_skill_display_count = count;
    }
}

/* Same frozen weights and same scoring code as the trainer. */
static void __cdecl skill_test_thread(void* arg) {
    DQNAgent* frozen = (DQNAgent*)arg;
    SkillTestResult result;
    run_skill_tests(frozen, 0, &result, 0, pg_stage_cb, nullptr);
    independent_training::indep_free(frozen);
    independent_training::indep_free(frozen);
    free(frozen);
    g_snap_result.empty_win_rate = result.empty_win_rate;
    g_snap_result.empty_avg_steps = result.empty_avg_steps;
    g_snap_result.empty_shortest_rate = result.empty_shortest_rate;
    g_snap_result.empty_wins = result.empty_wins;
    g_snap_result.empty_games = result.empty_games;
    g_snap_result.empty_min_steps = result.empty_min_steps;
    g_snap_result.empty_max_steps = result.empty_max_steps;
    g_snap_result.empty_stat_mean = result.empty_stat_mean;
    g_snap_result.empty_stat_std = result.empty_stat_std;
    g_snap_result.empty_stat_min = result.empty_stat_min;
    g_snap_result.empty_stat_max = result.empty_stat_max;
    g_snap_result.empty_steps_mean = result.empty_steps_mean;
    g_snap_result.empty_steps_std = result.empty_steps_std;
    g_snap_result.win1_fixed = result.win1_fixed;
    g_snap_result.win1_fixed_margin = result.win1_fixed_margin;
    g_snap_result.block1_fixed = result.block1_fixed;
    g_snap_result.block1_fixed_margin = result.block1_fixed_margin;
    g_snap_result.win1_total = result.win1_stats.total;
    g_snap_result.win1_correct = result.win1_stats.correct;
    g_snap_result.win1_avg_q_correct = result.win1_stats.q_correct;
    g_snap_result.win1_avg_q_wrong = result.win1_stats.q_wrong;
    g_snap_result.win1_min_q = result.win1_stats.min_q;
    g_snap_result.win1_max_q = result.win1_stats.max_q;
    g_snap_result.win1_min_margin = result.win1_stats.min_margin;
    g_snap_result.win1_max_margin = result.win1_stats.max_margin;
    g_snap_result.block1_total = result.block1_stats.total;
    g_snap_result.block1_correct = result.block1_stats.correct;
    g_snap_result.block1_avg_q_correct = result.block1_stats.q_correct;
    g_snap_result.block1_avg_q_wrong = result.block1_stats.q_wrong;
    g_snap_result.block1_min_q = result.block1_stats.min_q;
    g_snap_result.block1_max_q = result.block1_stats.max_q;
    g_snap_result.block1_min_margin = result.block1_stats.min_margin;
    g_snap_result.block1_max_margin = result.block1_stats.max_margin;
    g_snap_result.seeded_win_rate = result.seeded_win_rate;
    g_snap_result.seeded_avg_steps = result.seeded_avg_steps;
    g_snap_test_progress = 100;
    g_snap_test_phase = 0;
    { std::lock_guard<std::mutex> lk(g_snap_stage_mutex); g_snap_stage.state = TASK_DONE; }
    g_snap_test_time = GetTickCount();
    g_snap_test_done = 1; // release-publish result and timing before exposing completion
    g_snap_test_just_done = 1;
    g_snap_test_running = 0;
}

static void run_snapshot_skill_test(void) {
    if (g_snap_test_running) return;
    g_snap_test_error[0] = 0;
    DQNAgent* frozen = (DQNAgent*)calloc(1, sizeof(DQNAgent));
    if (!frozen || !playground_copy_agent(frozen)) {
        independent_training::indep_free(frozen);
    free(frozen);
        snprintf(g_snap_test_error, sizeof(g_snap_test_error), "No snapshot / insufficient memory");
        return;
    }
    memset(&g_snap_result, 0, sizeof(g_snap_result));
    g_snap_result.source_phase = g_snap_result.source_episode = -1;
    uint32_t hash = 2166136261u;
    const unsigned char* bytes = (const unsigned char*)&frozen->q_net;
    for (size_t i = 0; i < sizeof(frozen->q_net); ++i) hash = (hash ^ bytes[i]) * 16777619u;
    snprintf(g_snap_result.source_info, sizeof(g_snap_result.source_info),
        "Suite v%d / canonical / frozen %08X (episode unknown)", SKILL_SUITE_VERSION, hash);
    g_snap_test_done = 0;
    g_snap_test_progress = 0;
    g_snap_test_phase = 0;
    { std::lock_guard<std::mutex> lk(g_snap_stage_mutex); g_snap_stage = StageProgress{}; }
    g_snap_test_start = GetTickCount();
    g_snap_test_start_tick = GetTickCount();
    g_snap_test_running = 1;
    if (_beginthread(skill_test_thread, 0, frozen) == (uintptr_t)-1L) {
        independent_training::indep_free(frozen);
    free(frozen);
        snprintf(g_snap_test_error, sizeof(g_snap_test_error), "Cannot start test worker");
        g_snap_test_running = 0;
    }
}

/* 按钮点击检测：检查鼠标是否在按钮区域 */
static int skill_test_btn_hit(int mx, int my, int bx, int by, int bw) {
    return (my == by && mx >= bx && mx < bx + bw);
}

/* ========== Neural Probe 面板 ========== */
static volatile int g_probe_auto_refresh = 1;   /* 自动刷新开关 */
static DWORD g_probe_last_refresh = 0;
static int g_probe_conn_err = 0;           /* 最近一次连接错误码 */
static char g_probe_conn_msg[64] = "";     /* 连接状态消息 */
volatile int g_nv_pending_init = 0;        /* 3D 视图待初始化标志 */
volatile int g_nv_3d_ready = 0;            /* 3D 视图状态：0=未初始化, 1=就绪, -1=失败 */
static int g_nv_top_n = 20;               /* 每层显示前 N 条强连接（5~50） */
static int g_nv_threshold_changed = 0;     /* 变化标志 */
typedef enum ProbeLinkMode {
    PROBE_LINK_WEIGHT = 0,
    PROBE_LINK_GRADIENT,
    PROBE_LINK_UPDATE,
    PROBE_LINK_CONTRIBUTION,
    PROBE_LINK_ATTENTION,
    PROBE_LINK_POOLING,
    PROBE_LINK_MODE_COUNT
} ProbeLinkMode;
static int g_probe_link_mode = PROBE_LINK_WEIGHT;
static int g_probe_link_dropdown_open = 0;
static int g_probe_link_drop_x = -1, g_probe_link_drop_y = -1, g_probe_link_drop_w = 23;
static const char* g_probe_link_mode_names[PROBE_LINK_MODE_COUNT] = {
    "参数权重", "参数梯度", "参数更新", "前向贡献", "注意力分数", "池化贡献"
};
static const char* g_probe_link_mode_ui[PROBE_LINK_MODE_COUNT] = {
    "Weight", "EffGrad", "Update", "Contribution", "Attention", "Pooling"
};
static int g_probe_view = 0;               /* 0=Layers, 1=Matrix, 2=Q, 3=Global, 4=3D */
static int g_probe_selected_layer = 0;     /* 选中的层 */
static int g_probe_matrix_scroll = 0;      /* 矩阵视图滚动偏移 */
static int g_nv_lifecycle_initialized = 0;
static int g_nv_3d_error = 0;              /* 最近一次 nv_init/nv_render 阶段错误码 */
/* NVSnapshot is slightly larger than the default 1 MiB Windows thread stack.
   Keep the conversion workspace in static storage: a local instance here was
   the confirmed 0xC00000FD crash when entering View 4. */
static NVSnapshot g_probe_3d_staging;
static uint64_t g_probe_3d_last_snap_id = UINT64_MAX;

static int probe_float_compare(const void* lhs, const void* rhs) {
    const float a = *(const float*)lhs;
    const float b = *(const float*)rhs;
    return (a > b) - (a < b);
}

/* Linear-interpolated P95.  Max-based scaling lets one outlier make an entire
   FC layer look inactive; P95 preserves relative structure while keeping the
   colour scale deterministic and resistant to a few extreme coefficients. */
static float probe_3d_p95(const float* values, uint32_t count) {
    if (!values || !count) return 1.0f;
    float* sorted = (float*)malloc((size_t)count * sizeof(float));
    if (!sorted) return 1.0f;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const float value = values[i];
        if (isfinite(value) && value >= 0.0f) sorted[valid++] = value;
    }
    if (!valid) { free(sorted); return 1.0f; }
    qsort(sorted, valid, sizeof(float), probe_float_compare);
    const double position = 0.95 * (double)(valid - 1u);
    const uint32_t low = (uint32_t)position;
    const uint32_t high = low + 1u < valid ? low + 1u : low;
    const float fraction = (float)(position - (double)low);
    float scale = sorted[low] + (sorted[high] - sorted[low]) * fraction;
    free(sorted);
    if (!isfinite(scale) || scale < 1e-12f) scale = 1.0f;
    return scale;
}

/* P95 is a robust reference value, not a display value by itself.  Dense
   parameter rows commonly occupy a narrow 0.7..1.0 band after division by
   P95; feeding that ratio directly into the renderer turns almost the whole
   layer white.  A cubic perceptual transfer keeps ordering and the P95 anchor
   intact, but reserves white for genuinely top-end rows. */
static float probe_3d_p95_contrast(float value, float scale) {
    if (!isfinite(value) || value <= 0.0f || !isfinite(scale) || scale <= 0.0f)
        return 0.0f;
    float ratio = value / scale;
    if (ratio > 1.0f) ratio = 1.0f;
    return ratio * ratio * ratio;
}

/* TokenEmbedding is a lookup table [3][16], not a three-neuron dense layer.
   The protocol keeps its three vocabulary-row usage counters for diagnostics,
   while the 3D view presents the actual 16 feature channels produced by the
   25 lookups.  This avoids interpreting Empty/Opponent/Self rows as neurons. */
static void probe_3d_fill_token_features(NVLayer* layer,
    const ProbeLayerDesc* desc, const ProbeGlobalState* state,
    const float* weights, const float* gradients) {
    if (!layer || !desc || !weights || desc->weight_count < 3u * TOKEN_EMBED_DIM)
        return;

    const float* matrix = weights;
    float update_scale = 1.0f;
    if (g_probe_link_mode == PROBE_LINK_GRADIENT ||
        g_probe_link_mode == PROBE_LINK_UPDATE) {
        if (!gradients || !state ||
            !(state->diagnostics_flags & PROBE_DIAG_GRADIENT_VALID))
            return;
        matrix = gradients;
        if (g_probe_link_mode == PROBE_LINK_UPDATE && state && state->gradient_span_steps) {
            update_scale = state->learning_rate * (float)state->gradient_span_steps;
            if (!isfinite(update_scale) || update_scale < 0.0f) update_scale = 0.0f;
        }
    }

    float feature_value[TOKEN_EMBED_DIM] = {};
    const int current_input = state &&
        (state->diagnostics_flags & PROBE_DIAG_FORWARD_VALID);
    for (int d = 0; d < TOKEN_EMBED_DIM; ++d) {
        double energy = 0.0;
        uint32_t samples = 0;
        if (current_input && g_probe_link_mode != PROBE_LINK_WEIGHT &&
            g_probe_link_mode != PROBE_LINK_GRADIENT &&
            g_probe_link_mode != PROBE_LINK_UPDATE) {
            /* Current-forward modes: RMS of the 25 embedding lookup results
               for this feature dimension.  Repeated empty cells remain real
               repeated lookups, rather than collapsing into one bright row. */
            for (int cell = 0; cell < CELL_COUNT; ++cell) {
                int token = network_token_index(state->last_input, cell);
                if (token < 0) token = 0;
                if (token > 2) token = 2;
                const double value = matrix[token * TOKEN_EMBED_DIM + d];
                energy += value * value;
                ++samples;
            }
        } else {
            /* Parameter modes: one node per feature channel, aggregating the
               corresponding coefficient from all three vocabulary rows. */
            for (int token = 0; token < 3; ++token) {
                const double value = matrix[token * TOKEN_EMBED_DIM + d] * update_scale;
                energy += value * value;
                ++samples;
            }
        }
        feature_value[d] = samples ? (float)sqrt(energy / (double)samples) : 0.0f;
    }

    const float scale = probe_3d_p95(feature_value, TOKEN_EMBED_DIM);
    for (int d = 0; d < TOKEN_EMBED_DIM && d < (int)layer->neuron_count; ++d) {
        float value = probe_3d_p95_contrast(feature_value[d], scale);
        layer->neurons[d].activation = value;
        if (g_probe_link_mode == PROBE_LINK_GRADIENT)
            layer->neurons[d].gradient = value;
    }
}

/* Select real high-magnitude weights for emphasis, plus a changing sample of
   ordinary weights so consecutive snapshots remain visibly distinguishable.
   Visual node indices may be compressed (attention 32 -> 16), therefore raw
   tensor indices are mapped proportionally onto the displayed grids. */
static void probe_3d_add_tensor_nodes(void) {
    if(g_probe_3d_staging.layer_count!=11)return;
    for(unsigned i=11;i<13;++i) {
        NVLayer& layer=g_probe_3d_staging.layers[i];
        snprintf(layer.name,sizeof(layer.name),"%s",i==11?"EmbeddedTokens":"AttentionMix");
        layer.type=NV_LAYER_UNKNOWN; layer.neuron_count=16;
        // These are computed tensors, not parameter gradients. Neutral nodes.
        for(unsigned n=0;n<16;++n)layer.neurons[n].activation=.05f;
    }
    g_probe_3d_staging.layer_count=13;
}
static void probe_3d_collect_strong_connections(void) {
    g_probe_3d_staging.connection_count = 0;

    /* 注意力是 token→token 的有向关系，不是参数矩阵。把 4 个 head
       的 softmax 概率取均值，并在 PositionEmbedding 的 5×5 平面内画线。 */
    if (g_probe_link_mode == PROBE_LINK_ATTENTION) {
        const ProbeGlobalState* gs = probe_get_global_state();
        if (!gs || !(gs->diagnostics_flags & PROBE_DIAG_FORWARD_VALID) ||
            g_probe_3d_staging.layer_count <= 1 ||
            g_probe_3d_staging.layers[1].neuron_count < CELL_COUNT) return;
        struct AEntry { float value; uint16_t src, dst; } entries[CELL_COUNT * CELL_COUNT];
        uint32_t count = 0;
        float node_peak[CELL_COUNT] = {};
        float global_peak = 1e-12f;
        for (uint16_t src = 0; src < CELL_COUNT; ++src) {
            for (uint16_t dst = 0; dst < CELL_COUNT; ++dst) {
                float value = 0.0f;
                for (int h = 0; h < PROBE_ATTENTION_HEADS; ++h)
                    value += gs->attention_scores[h][src][dst] / PROBE_ATTENTION_HEADS;
                if (isfinite(value) && value > node_peak[src]) node_peak[src] = value;
                if (isfinite(value) && value > global_peak) global_peak = value;
                /* Self-attention is encoded by node brightness; a straight
                   line from a node to itself has zero length and is invisible. */
                if (src != dst && isfinite(value) && value > 0.0f)
                    entries[count++] = { value, src, dst };
            }
        }
        for (uint32_t i = 0; i < CELL_COUNT; ++i)
            g_probe_3d_staging.layers[1].neurons[i].activation = node_peak[i] / global_peak;
        uint32_t top_n = (uint32_t)g_nv_top_n * 4u;
        if (top_n > count) top_n = count;
        if (top_n > NV_MAX_CONNECTIONS) top_n = NV_MAX_CONNECTIONS;
        for (uint32_t i = 0; i < top_n; ++i) {
            uint32_t best = i;
            for (uint32_t j = i + 1; j < count; ++j)
                if (entries[j].value > entries[best].value) best = j;
            AEntry tmp = entries[i]; entries[i] = entries[best]; entries[best] = tmp;
            NVConnection* c = &g_probe_3d_staging.connections[g_probe_3d_staging.connection_count++];
            c->src_layer = c->dst_layer = 1;
            c->src_neuron = entries[i].src;
            c->dst_neuron = entries[i].dst;
            c->weight = entries[i].value;
            c->color = 1;
        }
        return;
    }

    for (uint32_t li = 1; li < g_probe_3d_staging.layer_count; ++li) {
        if(li==1)continue; // position embedding is a lookup/addition, not a dense edge
        const ProbeLayerDesc* ld = probe_get_layer(li);
        const float* weights = probe_get_weights(li);
        const float* gradients = probe_get_gradients(li);
        uint64_t input_count = 0;
        const float* input_activation = probe_get_input_activations(li, &input_count);
        const ProbeGlobalState* gs = probe_get_global_state();
        if (!ld || !weights || !ld->input_size || !ld->output_size || !ld->weight_count) continue;
        if (g_probe_link_mode == PROBE_LINK_GRADIENT &&
            (!gradients || !gs || !(gs->diagnostics_flags & PROBE_DIAG_GRADIENT_VALID))) continue;
        if (g_probe_link_mode == PROBE_LINK_UPDATE &&
            (!gradients || !gs || !(gs->diagnostics_flags & PROBE_DIAG_GRADIENT_VALID))) continue;
        if (g_probe_link_mode == PROBE_LINK_CONTRIBUTION &&
            (!input_activation || input_count < ld->input_size || !gs ||
             !(gs->diagnostics_flags & PROBE_DIAG_FORWARD_VALID))) continue;
        if (g_probe_link_mode == PROBE_LINK_POOLING &&
            (!input_activation || input_count < ld->input_size || !gs ||
             !(gs->diagnostics_flags & PROBE_DIAG_FORWARD_VALID))) continue;
        const uint64_t complete_rows = ld->weight_count / ld->input_size;
        const uint64_t rows = complete_rows < ld->output_size ? complete_rows : ld->output_size;
        const uint64_t usable = rows * ld->input_size;
        if (!usable) continue;

        /* 每两层之间独立：收集权重 → 排序 → 取前 N 条（上限=总线数×10%） */
        const uint32_t source_layer=probe_parameter_source(li);
        NVLayer* src = &g_probe_3d_staging.layers[source_layer];
        NVLayer* dst = &g_probe_3d_staging.layers[li];
        if (!src->neuron_count || !dst->neuron_count) continue;

        /* 收集所有有效权重 */
        struct WEntry { float mag; float value; uint64_t wi; };
        /* Update 模式：|ΔW| = |grad| * lr * span */
        float update_scale = 1.0f;
        if (g_probe_link_mode == PROBE_LINK_UPDATE && gs && gs->gradient_span_steps > 0) {
            update_scale = gs->learning_rate * (float)gs->gradient_span_steps;
            if (update_scale < 1e-12f) update_scale = 1e-12f;
        }
        uint32_t valid_count = 0;
        for (uint64_t wi = 0; wi < usable; ++wi) {
            float value = weights[wi];
            if (g_probe_link_mode == PROBE_LINK_GRADIENT) value = gradients[wi];
            else if (g_probe_link_mode == PROBE_LINK_UPDATE) value = gradients[wi] * update_scale;
            else if (g_probe_link_mode == PROBE_LINK_CONTRIBUTION)
                value *= input_activation[wi % ld->input_size];
            else if (g_probe_link_mode == PROBE_LINK_POOLING)
                value = input_activation[wi % ld->input_size] * fabsf(weights[wi]);
            const float mag = fabsf(value);
            if (isfinite(mag) && mag > 0) valid_count++;
        }
        if (valid_count == 0) continue;

        WEntry* entries = (WEntry*)malloc(valid_count * sizeof(WEntry));
        if (!entries) continue;
        uint32_t ei = 0;
        for (uint64_t wi = 0; wi < usable; ++wi) {
            float value = weights[wi];
            if (g_probe_link_mode == PROBE_LINK_GRADIENT) value = gradients[wi];
            else if (g_probe_link_mode == PROBE_LINK_UPDATE) value = gradients[wi] * update_scale;
            else if (g_probe_link_mode == PROBE_LINK_CONTRIBUTION)
                value *= input_activation[wi % ld->input_size];
            else if (g_probe_link_mode == PROBE_LINK_POOLING)
                value = input_activation[wi % ld->input_size] * fabsf(weights[wi]);
            const float mag = fabsf(value);
            if (isfinite(mag) && mag > 0) {
                entries[ei].mag = mag;
                entries[ei].value = value;
                entries[ei].wi = wi;
                ei++;
            }
        }

        /* 取前 N 条，但不超过总线数的 10% */
        uint32_t requested = (uint32_t)g_nv_top_n;
        uint32_t cap_10pct = valid_count / 10;
        if (cap_10pct < 1) cap_10pct = 1;
        uint32_t top_n = requested < cap_10pct ? requested : cap_10pct;
        if (top_n > valid_count) top_n = valid_count;

        /* 部分排序：只排前 top_n 个最大的 */
        for (uint32_t i = 0; i < top_n; ++i) {
            uint32_t max_idx = i;
            for (uint32_t j = i + 1; j < valid_count; ++j)
                if (entries[j].mag > entries[max_idx].mag) max_idx = j;
            WEntry tmp = entries[i]; entries[i] = entries[max_idx]; entries[max_idx] = tmp;
        }

        /* 取前 top_n 条最强连接 */
        for (uint32_t i = 0; i < top_n &&
             g_probe_3d_staging.connection_count < NV_MAX_CONNECTIONS; ++i) {
            uint64_t wi = entries[i].wi;
            const uint64_t original_src = wi % ld->input_size;
            const uint64_t original_dst = wi / ld->input_size;
            /* Pooling 层：flat_index / 16 = cell_index（0~24），映射到 Attention_O 的 25 个棋盘位置节点。
               同一 cell 的 16 个 feature 连接聚合到同一节点，Top-K 仍在完整 400 维中排序。 */
            uint32_t src_vis = (ld->type == PROBE_LAYER_POOLING)
                ? (uint32_t)(original_src / TOKEN_EMBED_DIM)   /* cell_index = flat_index / 16 */
                : (uint32_t)(original_src * src->neuron_count / ld->input_size);
            uint32_t dst_vis = (uint32_t)(original_dst * dst->neuron_count / ld->output_size);
            if (src_vis >= src->neuron_count) src_vis = src->neuron_count - 1;
            if (dst_vis >= dst->neuron_count) dst_vis = dst->neuron_count - 1;
            bool duplicate = false;
            for (uint32_t ci = 0; ci < g_probe_3d_staging.connection_count; ++ci) {
                const NVConnection* c = &g_probe_3d_staging.connections[ci];
                if (c->src_layer == source_layer && c->dst_layer == li &&
                    c->src_neuron == src_vis && c->dst_neuron == dst_vis) {
                    duplicate = true; break;
                }
            }
            if (duplicate) continue;
            NVConnection* c = &g_probe_3d_staging.connections[g_probe_3d_staging.connection_count++];
            c->src_layer = (uint16_t)source_layer;
            c->dst_layer = (uint16_t)li;
            c->src_neuron = (uint16_t)src_vis;
            c->dst_neuron = (uint16_t)dst_vis;
            c->weight = entries[i].value;
            c->color = 1;
        }
        free(entries);
    }
    // Gray arrows are real dependencies, not measurements or inferred gradients.
    if(g_probe_3d_staging.layer_count==13) {
        const unsigned edges[][2]={{0,11},{1,11},{11,2},{11,3},{11,4},
            {2,12},{3,12},{4,12},{12,5},{11,5},{5,6},{6,7},{7,8},{8,9},{9,10}};
        for(const auto& e:edges)for(unsigned n=0;n<4 && g_probe_3d_staging.connection_count<NV_MAX_CONNECTIONS;++n) {
            const NVLayer& a=g_probe_3d_staging.layers[e[0]];
            const NVLayer& b=g_probe_3d_staging.layers[e[1]];
            if(!a.neuron_count || !b.neuron_count)continue;
            auto& c=g_probe_3d_staging.connections[g_probe_3d_staging.connection_count++];
            c.src_layer=(uint16_t)e[0];c.dst_layer=(uint16_t)e[1];
            c.src_neuron=(uint16_t)(n*a.neuron_count/4);
            c.dst_neuron=(uint16_t)(n*b.neuron_count/4);c.weight=0;c.color=2;
        }
    }
}

static void probe_3d_get_rect(int* out_x, int* out_y, int* out_w, int* out_h) {
    RECT client = {};
    if (!g_hwnd || !GetClientRect(g_hwnd, &client)) {
        *out_x = 3; *out_y = CONTENT_Y + 5; *out_w = 100; *out_h = 100;
        return;
    }
    const int cell_w = g_cell_w > 0 ? g_cell_w : 1;
    const int cell_h = g_cell_h > 0 ? g_cell_h : 1;
    *out_x = 3 * cell_w;
    *out_y = (CONTENT_Y + 5) * cell_h;
    *out_w = client.right - *out_x - 3 * cell_w;
    *out_h = client.bottom - *out_y - 3 * cell_h;
    if (*out_w < 100) *out_w = 100;
    if (*out_h < 100) *out_h = 100;
}

static void probe_3d_set_visible(int visible) {
    HWND child = nv_get_hwnd();
    if (!child || !IsWindow(child)) return;
    if (visible) {
        /* 恢复：移到正确位置 + 显示 */
        int px, py, pw, ph;
        probe_3d_get_rect(&px, &py, &pw, &ph);
        SetWindowPos(child, NULL, px, py, pw, ph, SWP_NOZORDER | SWP_NOACTIVATE);
        if (!IsWindowVisible(child)) ShowWindow(child, SW_SHOWNA);
    } else {
        /* 隐藏：移到屏幕外（避免 D3D swap chain 残影覆盖父窗口 D2D 表面） */
        SetWindowPos(child, NULL, -10000, -10000, 0, 0,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOSIZE);
        if (IsWindowVisible(child)) ShowWindow(child, SW_HIDE);
    }
}

static void probe_3d_sync_window(void) {
    HWND child = nv_get_hwnd();
    if (g_nv_lifecycle_initialized && child && IsWindow(child)) {
        int px, py, pw, ph;
        probe_3d_get_rect(&px, &py, &pw, &ph);
        RECT current = {};
        GetWindowRect(child, &current);
        POINT origin = { current.left, current.top };
        ScreenToClient(g_hwnd, &origin);
        if (origin.x != px || origin.y != py || current.right - current.left != pw || current.bottom - current.top != ph)
            SetWindowPos(child, NULL, px, py, pw, ph, SWP_NOZORDER | SWP_NOACTIVATE);
    }
    /* 延迟显示标志：页面切换动画期间不显示 D3D 窗口 */
    extern int g_delay_3d_show;
    const int show = g_nv_lifecycle_initialized && g_nv_3d_ready == 1 &&
        g_active_tab == 2 && g_probe_view == 4 && probe_client_is_connected() &&
        !g_probe_link_dropdown_open && !g_delay_3d_show &&
        !g_palette_open && !g_viewer_open && !g_replay_open && !g_play_open;
    probe_3d_set_visible(show);
}

static void probe_3d_shutdown(void) {
    nv_shutdown(); /* also cleans a partial failed initialization */
    g_nv_lifecycle_initialized = 0;
    g_nv_pending_init = 0;
    g_nv_3d_ready = 0;
    g_probe_3d_last_snap_id = UINT64_MAX;
}

/* Remove only a genuinely orphaned viewer window.  A live viewer is owned by
   the lifecycle state and must survive the one-second Probe page refresh. */
static void probe_3d_remove_orphan_window(void) {
    if (g_nv_lifecycle_initialized || !g_hwnd || !IsWindow(g_hwnd)) return;
    HWND stray = FindWindowExW(g_hwnd, NULL, L"NeuralViewer3D", NULL);
    if (stray) DestroyWindow(stray);
}

/* The virtual terminal presents the parent HWND at the end of each loop.
   Present D3D only after that parent frame, otherwise Direct2D can overwrite
   the child surface and the network appears for one flash then disappears. */
static void probe_3d_present_after_terminal(void) {
    if (!g_nv_lifecycle_initialized || g_nv_3d_ready != 1 ||
        g_active_tab != 2 || g_probe_view != 4 || !probe_client_is_connected() ||
        g_palette_open || g_viewer_open || g_replay_open || g_play_open ||
        g_delay_3d_show || trans_is_active(&g_transition)) return;
    static DWORD last_present = 0;
    const DWORD now = GetTickCount();
    if (now - last_present < 16) return;
    last_present = now;
    const int render_rc = nv_render();
    if (render_rc != 0) {
        g_nv_3d_error = render_rc;
        probe_3d_shutdown();
        g_nv_3d_ready = -1;
        vt_mark_all_dirty();
    }
}

/* ========== Neural Probe 全屏页面 ========== */
void render_probe(void) {
    draw_help_bar();
    int x = 2, y = CONTENT_Y;
    int h = g_console_height - CONTENT_Y - 4;
    if (h < 10) h = 10;
    clear_rect(0, y, g_console_width, h + 1);
    draw_border(0, y, g_console_width, h + 1, " NEURAL PROBE "); // End above the footer separator.

    /* 非 3D 视图时强制销毁 D3D 子窗口（防止残留覆盖画面） */
    if (g_probe_view != 4 && g_nv_lifecycle_initialized) {
        probe_3d_shutdown();
    }
    /* 兜底只处理生命周期之外的孤儿窗口。旧逻辑每次刷新都无条件
       DestroyWindow，恰好会把刚初始化成功的正常 3D 子窗口销毁，随后
       nv_render() 返回未初始化并被误报为"D3D11 不支持"。 */
    probe_3d_remove_orphan_window();
    char line[256];
    int row = y + 1;
    int mw = g_console_width - 6;

    /* 顶栏：连接按钮 + 自动刷新 + 状态消息 */
    int connected = probe_client_is_connected();
    probe_3d_set_visible(connected && g_probe_view == 4 && g_nv_3d_ready == 1);
    gotoxy(x, row);
    ui_button(x, row, connected ? "CONNECTED" : "CONNECT", {connected != 0, false}, ui_lamp_button());
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(x + 17, row);
    ui_button(x + 17, row, "AUTO", {g_probe_auto_refresh != 0, false}, ui_lamp_button());
    set_color(COLOR_WHITE, COLOR_BLACK);
    if (g_probe_conn_msg[0]) {
        gotoxy(x + 28, row);
        set_color(g_probe_conn_err ? COLOR_RED : COLOR_GREEN, COLOR_BLACK);
        printf("%s", g_probe_conn_msg);
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 这条快照属于哪一方：两方独立网络下，不标棋色的权重无法归因。
       由 Publisher 写进 ProbeLayerDesc._reserved[PROBE_LAYER_SIDE_OFFSET]；
       旧 Publisher 不会写，取值 0 会与「黑方」混淆，因此只在能识别时显示。 */
    {
        const NeuralProbeHeader* ph_side = probe_get_header();
        const ProbeLayerDesc* l0 = (ph_side && ph_side->layer_count > 0) ? probe_get_layer(0) : NULL;
        if (l0) {
            const uint32_t side_tag = (uint32_t)l0->_reserved[PROBE_LAYER_SIDE_OFFSET];
            const char* label = (side_tag == PROBE_LAYER_SIDE_SHARED) ? "SHARED NET"
                : (side_tag == 0) ? "BLACK online" : (side_tag == 1) ? "WHITE online" : "?";
            gotoxy(x + 46, row);
            set_color(side_tag == 1 ? COLOR_CYAN : COLOR_WHITE, COLOR_BLACK);
            printf("WEIGHTS: %s", label);
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    row++;

    /* 未连接提示 */
    if (!connected) {
        row++;
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        print_clip(x, row++, mw, "Trainer not running or probe not enabled.");
        print_clip(x, row++, mw, "Steps: 1. Start Trainer  2. Click CONNECT");
        if (g_probe_conn_err != 0) {
            set_color(COLOR_RED, COLOR_BLACK);
            snprintf(line, sizeof(line), "Error %d: shared memory not found", g_probe_conn_err);
            print_clip(x, row++, mw, "%s", line);
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
        return;
    }

    /* 自动刷新 */
    DWORD now = GetTickCount();
    if (g_probe_auto_refresh && now - g_probe_last_refresh >= PROBE_CLIENT_POLL_MS) {
        int update_rc = probe_client_update();
        if (update_rc > 0) need_content = 1;
        else if (update_rc < 0) {
            probe_3d_shutdown();
            probe_client_disconnect();
            snprintf(g_probe_conn_msg, sizeof(g_probe_conn_msg), "Publisher disconnected (%d)", update_rc);
            g_probe_conn_err = update_rc;
            g_probe_3d_last_snap_id = UINT64_MAX;
            set_color(COLOR_RED, COLOR_BLACK);
            print_clip(x, row++, mw, "%s", g_probe_conn_msg);
            set_color(COLOR_WHITE, COLOR_BLACK);
            return;
        }
        g_probe_last_refresh = now;
    }

    const NeuralProbeHeader* hdr = probe_get_header();
    if (!hdr) { row++; set_color(COLOR_RED, COLOR_BLACK); print_clip(x, row, mw, "Header NULL"); set_color(COLOR_WHITE, COLOR_BLACK); return; }

    /* 状态行 */
    set_color(COLOR_CYAN, COLOR_BLACK);
    const ProbeGlobalState* header_state = probe_get_global_state();
    if (g_probe_view == 4 && header_state &&
        (header_state->diagnostics_flags & PROBE_DIAG_FORWARD_VALID)) {
        unsigned token_count[3] = {};
        for (int cell = 0; cell < CELL_COUNT; ++cell) {
            int token = network_token_index(header_state->last_input, cell);
            if (token >= 0 && token < 3) ++token_count[token];
        }
        snprintf(line, sizeof(line),
            "Snap#%llu  Ep:%llu  Step:%llu  Layers:%u  Buf:%s  Token(O/E/S):%u/%u/%u",
            (unsigned long long)hdr->snapshot_id, (unsigned long long)hdr->episode,
            (unsigned long long)hdr->train_step, hdr->layer_count,
            hdr->active_buffer == 0 ? "A" : "B",
            token_count[0], token_count[1], token_count[2]);
    } else {
        snprintf(line, sizeof(line), "Snap#%llu  Ep:%llu  Step:%llu  Layers:%u  Buf:%s",
            (unsigned long long)hdr->snapshot_id, (unsigned long long)hdr->episode,
            (unsigned long long)hdr->train_step, hdr->layer_count,
            hdr->active_buffer == 0 ? "A" : "B");
    }
    print_clip(x, row++, mw, "%s", line);
    set_color(COLOR_WHITE, COLOR_BLACK);

    /* 子 Tab 栏 */
    const char* sub_tabs[] = { "Layers", "Matrix", "Q-Values", "Global", "3D View" };
    int sub_tab_x = x;
    for (int i = 0; i < 5; i++) {
        gotoxy(sub_tab_x, row);
        char tab_label[32]; snprintf(tab_label, sizeof(tab_label), " %s ", sub_tabs[i]);
        control_here(tab_label, i == g_probe_view, true);
        sub_tab_x += (int)strlen(sub_tabs[i]) + 3;
    }
    g_probe_link_drop_x = -1;
    g_probe_link_drop_y = -1;
    /* 3D 视图：连接数据源下拉框 + 连线数量滑块。 */
    if (g_probe_view == 4 && g_nv_3d_ready == 1) {
        g_probe_link_drop_x = sub_tab_x + 1;
        g_probe_link_drop_y = row;
        gotoxy(g_probe_link_drop_x, row);
        set_color(g_probe_link_dropdown_open ? COLOR_YELLOW : COLOR_CYAN, COLOR_BLACK);
        const char* scale_suffix =
            (g_probe_link_mode == PROBE_LINK_WEIGHT ||
             g_probe_link_mode == PROBE_LINK_GRADIENT ||
             g_probe_link_mode == PROBE_LINK_UPDATE ||
             g_probe_link_mode == PROBE_LINK_POOLING) ? "/P95" : "";
        char mode_label[24] = {};
        snprintf(mode_label, sizeof(mode_label), "%s%s",
            g_probe_link_mode_ui[g_probe_link_mode], scale_suffix);
        char mode_button[40]; snprintf(mode_button, sizeof(mode_button), "Mode:[%-12.12s v]", mode_label);
        control_here(mode_button, g_probe_link_dropdown_open != 0, true);

        int slider_label_x = g_probe_link_drop_x + g_probe_link_drop_w + 3;
        gotoxy(slider_label_x, row);
        set_color(COLOR_CYAN, COLOR_BLACK);
        printf("TopN:[");
        int slider_w = x + mw - slider_label_x - 10; if (slider_w < 8) slider_w = 8;
        int slider_pos = (int)(((float)(g_nv_top_n - 5) / 45.0f) * (slider_w - 1));
        if (slider_pos < 0) slider_pos = 0;
        if (slider_pos > slider_w - 1) slider_pos = slider_w - 1;
        for (int si = 0; si < slider_w; si++) {
            if (si == slider_pos) { set_color(COLOR_YELLOW, COLOR_BLACK); printf("●"); }
            else { set_color(COLOR_CYAN, COLOR_BLACK); printf("─"); }
        }
        printf("]");
        set_color(COLOR_WHITE, COLOR_BLACK);
        ui_number_printf("%d", g_nv_top_n);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    row++;
    if (g_probe_link_dropdown_open && g_probe_link_drop_x >= 0) {
        for (int i = 0; i < PROBE_LINK_MODE_COUNT; ++i) {
            gotoxy(g_probe_link_drop_x, row + i);
            char option[32]; snprintf(option, sizeof(option), "  %-14s", g_probe_link_mode_ui[i]);
            control_here(option, i == g_probe_link_mode, true);
        }
        row += PROBE_LINK_MODE_COUNT;
    }
    /* 分隔线 */
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    for (int c = 0; c < mw && c < 254; c++) line[c] = '-';
    line[mw < 254 ? mw : 254] = '\0';
    print_clip(x, row++, mw, "%s", line);
    set_color(COLOR_WHITE, COLOR_BLACK);

    const char* type_names[] = { "?", "Input", "Embed", "Attn", "Pool", "Linear", "ReLU", "Output" };
    if (g_probe_selected_layer >= (int)hdr->layer_count) g_probe_selected_layer = 0;

    /* ========== View 0: Layer List ========== */
    if (g_probe_view == 0) {
        int col_w = mw / 2 - 1;
        if (col_w < 30) col_w = 30;
        int col2_x = x + col_w + 2;

        /* 左列：层列表（显示正确维度映射） */
        int lr = row;
        for (uint32_t i = 0; i < hdr->layer_count && lr < y + h - 1; i++) {
            const ProbeLayerDesc* layer = probe_get_layer(i);
            if (!layer) continue;
            const char* tn = (layer->type <= PROBE_LAYER_OUTPUT) ? type_names[layer->type] : "?";
            int sel = ((int)i == g_probe_selected_layer);
            float wmn = 0, wmx = 0, wm = 0, ws = 0;
            if (layer->weight_count > 0) probe_get_weight_stats(i, &wmn, &wmx, &wm, &ws);
            set_color(sel ? COLOR_CYAN : COLOR_WHITE, COLOR_BLACK);
            gotoxy(x + (sel ? 0 : 2), lr);
            /* 显示层名 + 类型 + 维度（如 FC1: 128→1024 = W[1024][128]） */
            snprintf(line, sizeof(line), "%s %-10s %-6s %u→%u",
                sel ? ">" : " ", layer->name, tn, layer->input_size, layer->output_size);
            control_clip(x + (sel ? 0 : 2), lr, col_w - (sel ? 0 : 2), line, sel != 0, true);
            if (layer->weight_count > 0 && lr + 1 < y + h - 1) {
                lr++;
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                snprintf(line, sizeof(line), "  W[%llu] %+.4f ~ %+.4f  u=%+.4f s=%.4f",
                    (unsigned long long)layer->weight_count, wmn, wmx, wm, ws);
                print_clip(x + 4, lr, col_w - 4, "%s", line);
            }
            set_color(COLOR_WHITE, COLOR_BLACK);
            lr++;
        }
        /* 右列：选中层详情（按原提示词：Layer, Neuron Index, Activation, Pre-Activation, Gradient, Bias） */
        int rr = row;
        if (g_probe_selected_layer < (int)hdr->layer_count) {
            const ProbeLayerDesc* sl = probe_get_layer(g_probe_selected_layer);
            if (sl) {
                const char* tn = (sl->type <= PROBE_LAYER_OUTPUT) ? type_names[sl->type] : "?";
                set_color(COLOR_CYAN, COLOR_BLACK);
                snprintf(line, sizeof(line), "%s [%s]  W[%u][%u]", sl->name, tn, sl->output_size, sl->input_size);
                print_clip(col2_x, rr++, col_w, "%s", line);
                set_color(COLOR_WHITE, COLOR_BLACK);
                snprintf(line, sizeof(line), "Type: %s (%u)", tn, sl->type);
                print_clip(col2_x, rr++, col_w, "%s", line);
                snprintf(line, sizeof(line), "Input dim:  %u", sl->input_size);
                print_clip(col2_x, rr++, col_w, "%s", line);
                snprintf(line, sizeof(line), "Output dim: %u", sl->output_size);
                print_clip(col2_x, rr++, col_w, "%s", line);
                if (sl->weight_count > 0) {
                    float mn, mx, mean, std;
                    if (probe_get_weight_stats(g_probe_selected_layer, &mn, &mx, &mean, &std) == 0) {
                        rr++;
                        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                        print_clip(col2_x, rr++, col_w, "WEIGHTS");
                        set_color(COLOR_WHITE, COLOR_BLACK);
                        snprintf(line, sizeof(line), "Count:  %llu", (unsigned long long)sl->weight_count);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                        snprintf(line, sizeof(line), "Min:    %+.8f", mn);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                        snprintf(line, sizeof(line), "Max:    %+.8f", mx);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                        snprintf(line, sizeof(line), "Mean:   %+.8f", mean);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                        snprintf(line, sizeof(line), "Std:    %+.8f", std);
                        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                        set_color(COLOR_WHITE, COLOR_BLACK);
                        /* 直方图 */
                        const float* w = probe_get_weights(g_probe_selected_layer);
                        if (w && rr < y + h - 3) {
                            int bins = 20; if (col_w < 30) bins = 10;
                            int hist[20] = {0};
                            float range = mx - mn; if (range < 1e-10f) range = 1.0f;
                            for (uint64_t wi = 0; wi < sl->weight_count; wi++) {
                                int bin = (int)((w[wi] - mn) / range * (bins - 1));
                                if (bin < 0) bin = 0; if (bin >= bins) bin = bins - 1;
                                hist[bin]++;
                            }
                            int max_h = 1; for (int b = 0; b < bins; b++) if (hist[b] > max_h) max_h = hist[b];
                            int bar_max = col_w - 12; if (bar_max < 5) bar_max = 5;
                            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                            for (int b = 0; b < bins && rr < y + h - 1; b++) {
                                int bar_len = (int)((long long)hist[b] * bar_max / max_h);
                                float v = mn + (float)b / (bins - 1) * range;
                                int off = snprintf(line, sizeof(line), "%+6.3f ", v);
                                for (int c = 0; c < bar_len && off < 254; c++) line[off++] = '#';
                                line[off] = '\0';
                                gotoxy(col2_x, rr);
                                print_clip(col2_x, rr, col_w, "%s", line);
                                rr++;
                            }
                        }
                    }
                }
                if (sl->bias_count > 0 && rr < y + h - 1) {
                    float mn, mx, mean, std;
                    if (probe_get_bias_stats(g_probe_selected_layer, &mn, &mx, &mean, &std) == 0) {
                        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                        snprintf(line, sizeof(line), "B[%llu] min:%+.6f max:%+.6f mean:%+.6f",
                            (unsigned long long)sl->bias_count, mn, mx, mean);
                        print_clip(col2_x, rr++, col_w, "%s", line);
                    }
                }
            }
        }
    }
    /* ========== View 1: Weight Matrix (严格按原提示词) ========== */
    else if (g_probe_view == 1) {
        if (g_probe_selected_layer >= (int)hdr->layer_count) g_probe_selected_layer = 0;
        const ProbeLayerDesc* sl = probe_get_layer(g_probe_selected_layer);
        if (!sl || sl->weight_count == 0) {
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "No weights in this layer");
            set_color(COLOR_WHITE, COLOR_BLACK);
        } else {
            const float* w = probe_get_weights(g_probe_selected_layer);
            if (!w) { set_color(COLOR_RED, COLOR_BLACK); print_clip(x, row++, mw, "Weight data NULL"); set_color(COLOR_WHITE, COLOR_BLACK); }
            else {
                /* 权重布局：w[output_size][input_size]（行优先）。旧版 Probe
                   曾把多个张量合并进同一描述符，导致维度乘积与 weight_count
                   不一致。显示端必须以实际元素数为硬边界，绝不能跨参数段读取。 */
                int out_dim = (int)sl->output_size;
                int in_dim = (int)sl->input_size;
                uint64_t complete_rows = in_dim > 0 ? sl->weight_count / (uint64_t)in_dim : 0;
                int matrix_layout_valid = in_dim > 0 && out_dim > 0 &&
                    complete_rows >= (uint64_t)out_dim;
                if (complete_rows < (uint64_t)out_dim) out_dim = (int)complete_rows;
                float mn, mx, mean, std;
                if (probe_get_weight_stats(g_probe_selected_layer, &mn, &mx, &mean, &std) != 0) {
                    mn = mx = mean = std = 0.0f;
                }

                /* 标题行：层名 + 维度 + 统计 */
                snprintf(line, sizeof(line), "%s  W[%d][%d]  range:[%+.6f, %+.6f]  mean:%+.6f  std:%.6f",
                    sl->name, out_dim, in_dim, mn, mx, mean, std);
                set_color(COLOR_CYAN, COLOR_BLACK);
                print_clip(x, row++, mw, "%s", line);
                set_color(COLOR_WHITE, COLOR_BLACK);

                if (!matrix_layout_valid) {
                    set_color(COLOR_YELLOW, COLOR_BLACK);
                    print_clip(x, row++, mw,
                        "Legacy/malformed layout: showing %d complete rows (%llu values)",
                        out_dim, (unsigned long long)sl->weight_count);
                    set_color(COLOR_WHITE, COLOR_BLACK);
                }

                if (out_dim <= 0 || in_dim <= 0) {
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    print_clip(x, row++, mw, "No complete matrix row available");
                    set_color(COLOR_WHITE, COLOR_BLACK);
                    return;
                }

                /* 可用显示区域 */
                int max_display_rows = y + h - row - 3;  /* 留底边给说明 */
                if (max_display_rows < 3) max_display_rows = 3;
                int max_display_cols = mw - 7;  /* 行标占 5 + 分隔 1 + 右边 1 */
                if (max_display_cols < 10) max_display_cols = 10;

                /* 如果矩阵太大，显示可滚动的窗口 */
                int vis_rows = out_dim; if (vis_rows > max_display_rows) vis_rows = max_display_rows;
                int vis_cols = in_dim;  if (vis_cols > max_display_cols) vis_cols = max_display_cols;
                int row_start = g_probe_matrix_scroll;
                if (row_start + vis_rows > out_dim) row_start = out_dim - vis_rows;
                if (row_start < 0) row_start = 0;

                /* 列标（输入维度索引） */
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                gotoxy(x + 5, row);
                printf("│");
                for (int c = 0; c < vis_cols; c++) {
                    int col_idx = c;
                    if (c % 10 == 0) printf("%-4d", col_idx);
                    else if (c % 5 == 0) printf("%-1d", col_idx % 10);
                }
                row++;
                set_color(COLOR_WHITE, COLOR_BLACK);

                /* 权重密度字符：8 级 */
                /* ' ' <0.01  '.' <0.05  ':' <0.1  '+' <0.2  '*' <0.4  '#' <0.7  '@' >=0.7 */
                float abs_max = (mn < 0 ? -mn : mn); if (mx > abs_max) abs_max = mx;
                if (abs_max < 1e-10f) abs_max = 1.0f;

                for (int r = 0; r < vis_rows && row < y + h - 2; r++) {
                    int out_idx = row_start + r;
                    gotoxy(x, row);
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    printf("%4d│", out_idx);
                    set_color(COLOR_WHITE, COLOR_BLACK);
                    int ci = x + 5;
                    for (int c = 0; c < vis_cols && ci < x + mw - 1; c++) {
                        /* w[out][in] = w[out_idx * in_dim + c] */
                        float val = w[out_idx * in_dim + c];
                        float norm = val / abs_max;
                        float abs_n = norm < 0 ? -norm : norm;
                        /* Unicode 横向分数块：8 级密度（▁▂▃▄▅▆▇█），从下往上长 */
                        const char* ch;
                        if      (abs_n < 0.02f) ch = " ";               /* 空 */
                        else if (abs_n < 0.10f) ch = "\xe2\x96\x81";   /* ▁ 1/8 */
                        else if (abs_n < 0.20f) ch = "\xe2\x96\x82";   /* ▂ 2/8 */
                        else if (abs_n < 0.30f) ch = "\xe2\x96\x83";   /* ▃ 3/8 */
                        else if (abs_n < 0.45f) ch = "\xe2\x96\x84";   /* ▄ 4/8 */
                        else if (abs_n < 0.55f) ch = "\xe2\x96\x85";   /* ▅ 5/8 */
                        else if (abs_n < 0.70f) ch = "\xe2\x96\x86";   /* ▆ 6/8 */
                        else if (abs_n < 0.85f) ch = "\xe2\x96\x87";   /* ▇ 7/8 */
                        else                     ch = "\xe2\x96\x88";   /* █ 8/8 */
                        /* 颜色：正=青，负=红，近零=暗灰 */
                        if (norm > 0.01f) set_color(COLOR_CYAN, COLOR_BLACK);
                        else if (norm < -0.01f) set_color(COLOR_RED, COLOR_BLACK);
                        else set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                        gotoxy(ci, row);
                        printf("%s", ch);
                        ci++;
                        ci++;
                    }
                    row++;
                }

                /* 图例（横向分数块） */
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                gotoxy(x, row);
                printf(" ");
                set_color(COLOR_CYAN, COLOR_BLACK);
                printf("\xe2\x96\x81\xe2\x96\x82\xe2\x96\x83\xe2\x96\x84\xe2\x96\x85\xe2\x96\x86\xe2\x96\x87\xe2\x96\x88"); /* ▁▂▃▄▅▆▇█ */
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                printf(" = magnitude  ");
                set_color(COLOR_CYAN, COLOR_BLACK); printf("+");
                set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("/");
                set_color(COLOR_RED, COLOR_BLACK); printf("-");
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                printf(" = sign  scroll:%d/%d", row_start, out_dim - vis_rows);
                set_color(COLOR_WHITE, COLOR_BLACK);
                row++;
            }
        }
    }
    /* ========== View 2: Q-Values（棋盘热力图 + 排名） ========== */
    else if (g_probe_view == 2) {
        const ProbeGlobalState* gs = probe_get_global_state();
        if (!gs) {
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "Global state not available");
            set_color(COLOR_WHITE, COLOR_BLACK);
        } else {
            /* 状态行 */
            snprintf(line, sizeof(line), "Ep:%llu  Step:%llu  eps:%.4f  lr:%.6f  player:%u",
                (unsigned long long)gs->episode, (unsigned long long)gs->train_step,
                gs->epsilon, gs->learning_rate, gs->current_player);
            set_color(COLOR_CYAN, COLOR_BLACK);
            print_clip(x, row++, mw, "%s", line);
            set_color(COLOR_WHITE, COLOR_BLACK);
            row++;

            /* Q 值统计 */
            float q_min = gs->q_values[0], q_max = gs->q_values[0];
            double q_sum = 0;
            for (int i = 0; i < 25; i++) {
                if (gs->q_values[i] < q_min) q_min = gs->q_values[i];
                if (gs->q_values[i] > q_max) q_max = gs->q_values[i];
                q_sum += gs->q_values[i];
            }
            float q_range = q_max - q_min; if (q_range < 1e-10f) q_range = 1.0f;
            float q_mean = (float)(q_sum / 25.0);

            snprintf(line, sizeof(line), "Q range: [%+.4f, %+.4f]  mean:%+.4f  spread:%.4f",
                q_min, q_max, q_mean, q_range);
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "%s", line);
            row++;

            /* 棋盘热力图（ASCII 对齐） */
            /* 列标：每格 9 字符宽（+7.4f = 8 + 空格分隔） */
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            gotoxy(x + 7, row);
            printf("    1         2         3         4         5");
            row++;
            /* 行内容 */
            for (int r = 0; r < 5 && row < y + h - 8; r++) {
                gotoxy(x, row);
                set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                printf("  %c  |", 'A' + r);
                for (int c = 0; c < 5; c++) {
                    float qv = gs->q_values[r * 5 + c];
                    float norm = (qv - q_min) / q_range;
                    if (norm > 0.75f) set_color(COLOR_GREEN, COLOR_BLACK);
                    else if (norm > 0.50f) set_color(COLOR_YELLOW, COLOR_BLACK);
                    else if (norm > 0.25f) set_color(COLOR_RED, COLOR_BLACK);
                    else set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    printf(" %+7.4f ", qv);
                    if (c < 4) { set_color(COLOR_DARK_GRAY, COLOR_BLACK); printf("|"); }
                }
                row++;
                /* 行分隔线 */
                if (r < 4) {
                    gotoxy(x + 7, row);
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    printf("----+---------+---------+---------+---------+");
                    row++;
                }
            }
            set_color(COLOR_WHITE, COLOR_BLACK);
            row++;

            /* Q 值排名（Top 5） */
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "Q Ranking (Top 5):");
            set_color(COLOR_WHITE, COLOR_BLACK);
            /* 简单排序 */
            int idx[25]; for (int i = 0; i < 25; i++) idx[i] = i;
            for (int i = 0; i < 25; i++) for (int j = i + 1; j < 25; j++)
                if (gs->q_values[idx[j]] > gs->q_values[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
            for (int i = 0; i < 5 && row < y + h - 1; i++) {
                int ci = idx[i];
                float qv = gs->q_values[ci];
                float norm = (qv - q_min) / q_range;
                if (norm > 0.75f) set_color(COLOR_GREEN, COLOR_BLACK);
                else if (norm > 0.50f) set_color(COLOR_YELLOW, COLOR_BLACK);
                else set_color(COLOR_RED, COLOR_BLACK);
                snprintf(line, sizeof(line), "  #%d  %c%d  Q=%+8.4f  (%.1f%%)",
                    i + 1, 'A' + ci / 5, ci % 5 + 1, qv, norm * 100);
                print_clip(x, row++, mw, "%s", line);
            }
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    /* ========== View 3: Global State + Network Topology ========== */
    else if (g_probe_view == 3) {
        const ProbeGlobalState* gs = probe_get_global_state();
        if (!gs) {
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "Global state not available");
        } else {
            /* 训练状态 */
            set_color(COLOR_CYAN, COLOR_BLACK);
            print_clip(x, row++, mw, "TRAINING STATE");
            set_color(COLOR_WHITE, COLOR_BLACK);
            snprintf(line, sizeof(line), "  Episode:       %llu", (unsigned long long)gs->episode);
            print_clip(x, row++, mw, "%s", line);
            snprintf(line, sizeof(line), "  Train Step:    %llu", (unsigned long long)gs->train_step);
            print_clip(x, row++, mw, "%s", line);
            snprintf(line, sizeof(line), "  Epsilon:       %.6f", gs->epsilon);
            print_clip(x, row++, mw, "%s", line);
            snprintf(line, sizeof(line), "  Learning Rate: %.8f", gs->learning_rate);
            print_clip(x, row++, mw, "%s", line);
            snprintf(line, sizeof(line), "  Loss:          %.6f", gs->loss);
            print_clip(x, row++, mw, "%s", line);
            snprintf(line, sizeof(line), "  Current Player: %u (%s)", gs->current_player, gs->current_player == 0 ? "BLACK(X)" : "WHITE(O)");
            print_clip(x, row++, mw, "%s", line);
            row++;

            /* 网络拓扑（带数据流箭头） */
            set_color(COLOR_CYAN, COLOR_BLACK);
            print_clip(x, row++, mw, "NETWORK TOPOLOGY");
            set_color(COLOR_WHITE, COLOR_BLACK);

            /* 计算总参数量 */
            uint64_t total_params = 0;
            for (uint32_t i = 0; i < hdr->layer_count; i++) {
                const ProbeLayerDesc* l = probe_get_layer(i);
                if (l) total_params += l->weight_count + l->bias_count;
            }

            for (uint32_t i = 0; i < hdr->layer_count && row < y + h - 3; i++) {
                const ProbeLayerDesc* layer = probe_get_layer(i);
                if (!layer) continue;
                const char* tn = (layer->type <= PROBE_LAYER_OUTPUT) ? type_names[layer->type] : "?";

                /* 层信息 */
                set_color(COLOR_WHITE, COLOR_BLACK);
                gotoxy(x, row);
                snprintf(line, sizeof(line), "  %-12s  %-8s  %u → %u",
                    layer->name, tn, layer->input_size, layer->output_size);
                print_clip(x, row, mw - 20, "%s", line);

                /* 参数占比 */
                if (layer->weight_count > 0) {
                    float pct = (float)(layer->weight_count + layer->bias_count) / (float)total_params * 100.0f;
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    snprintf(line, sizeof(line), "  W:%llu B:%llu (%.1f%%)",
                        (unsigned long long)layer->weight_count, (unsigned long long)layer->bias_count, pct);
                    gotoxy(x + 45, row);
                    print_clip(x + 45, row, mw - 45, "%s", line);
                }
                row++;

                /* 数据流箭头（非最后一层） */
                if (i < hdr->layer_count - 1 && row < y + h - 3) {
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                    gotoxy(x + 8, row);
                    printf("│");
                    gotoxy(x + 8, row + 1);
                    printf("▼");
                    row += 2;
                }
            }

            /* 总参数量 */
            row++;
            set_color(COLOR_CYAN, COLOR_BLACK);
            snprintf(line, sizeof(line), "Total parameters: %llu (%.2f MB FP32)",
                (unsigned long long)total_params, total_params * 4.0f / (1024 * 1024));
            print_clip(x, row++, mw, "%s", line);

            /* 快照信息 */
            row++;
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            snprintf(line, sizeof(line), "Snapshot #%llu  Buffer: %s  Rate: %u Hz",
                (unsigned long long)hdr->snapshot_id,
                hdr->active_buffer == 0 ? "A" : "B",
                hdr->publish_hz);
            print_clip(x, row++, mw, "%s", line);
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    /* ========== View 4: 3D Neural Viewer ========== */
    else if (g_probe_view == 4) {
        if (g_nv_3d_ready == 1) {
            /* 3D 就绪：更新快照 + 渲染 */
            if (connected) {
                const NeuralProbeHeader* ph = probe_get_header();
                if (ph && ph->snapshot_id != g_probe_3d_last_snap_id) {
                    g_probe_3d_last_snap_id = ph->snapshot_id;
                    memset(&g_probe_3d_staging, 0, sizeof(g_probe_3d_staging));
                    g_probe_3d_staging.layer_count =
                        ph->layer_count > NV_MAX_LAYERS ? NV_MAX_LAYERS : ph->layer_count;
                    g_probe_3d_staging.snapshot_id = ph->snapshot_id;
                    g_probe_3d_staging.episode = ph->episode;
                    g_probe_3d_staging.train_step = ph->train_step;
                    for (uint32_t li = 0; li < g_probe_3d_staging.layer_count; li++) {
                        const ProbeLayerDesc* ld = probe_get_layer(li);
                        if (!ld) continue;
                        NVLayer* layer = &g_probe_3d_staging.layers[li];
                        snprintf(layer->name, sizeof(layer->name), "%s", ld->name);
                        layer->input_size = ld->input_size;
                        layer->output_size = ld->output_size;
                        layer->type = ld->type == PROBE_LAYER_INPUT ? NV_LAYER_INPUT :
                            ld->type == PROBE_LAYER_EMBED ? NV_LAYER_EMBED :
                            ld->type == PROBE_LAYER_ATTENTION ? NV_LAYER_ATTENTION :
                            ld->type == PROBE_LAYER_POOLING ? NV_LAYER_POOLING :
                            ld->type == PROBE_LAYER_OUTPUT ? NV_LAYER_OUTPUT : NV_LAYER_LINEAR;
                        /* TokenEmbedding is [3 vocabulary rows][16 features].
                           Show the 16 actual feature channels; the three row
                           hit ratios remain available in the Probe diagnostics.
                           Attention_O is summarized by 25 board positions;
                           Q/K/V retain the established 16-channel view. */
                        uint32_t visual_count = ld->output_size;
                        if (li == 0 && !strcmp(ld->name, "TokenEmbedding")) {
                            visual_count = TOKEN_EMBED_DIM;
                            snprintf(layer->name, sizeof(layer->name), "TokenEmbedding_16D");
                            layer->input_size = 3;
                            layer->output_size = TOKEN_EMBED_DIM;
                        } else if (layer->type == NV_LAYER_ATTENTION && visual_count > 16 &&
                            strstr(layer->name, "_O"))
                            visual_count = CELL_COUNT;   /* 25 个棋盘位置 */
                        else if (layer->type == NV_LAYER_ATTENTION && visual_count > 16)
                            visual_count = 16;
                        layer->neuron_count = visual_count > NV_MAX_NEURONS ? NV_MAX_NEURONS : visual_count;
                        const float* activation_values = probe_get_activations(li);
                        uint64_t input_activation_count = 0;
                        const float* input_activation =
                            probe_get_input_activations(li, &input_activation_count);
                        const bool gradient_metric =
                            g_probe_link_mode == PROBE_LINK_GRADIENT ||
                            g_probe_link_mode == PROBE_LINK_UPDATE;
                        const ProbeGlobalState* global_state = probe_get_global_state();
                        const float* matrix_values = gradient_metric ?
                            probe_get_gradients(li) : probe_get_weights(li);
                        if (gradient_metric && (!global_state ||
                            !(global_state->diagnostics_flags & PROBE_DIAG_GRADIENT_VALID)))
                            matrix_values = NULL;
                        if (li == 0 && !strcmp(ld->name, "TokenEmbedding")) {
                            probe_3d_fill_token_features(layer, ld, probe_get_global_state(),
                                probe_get_weights(li), probe_get_gradients(li));
                        } else if ((g_probe_link_mode == PROBE_LINK_CONTRIBUTION ||
                             g_probe_link_mode == PROBE_LINK_ATTENTION) &&
                            activation_values && ld->activation_count > 0) {
                            float* node_metric = (float*)calloc(layer->neuron_count, sizeof(float));
                            if (!node_metric) continue;
                            for (uint32_t ni = 0; ni < layer->neuron_count; ++ni) {
                                uint64_t begin = (uint64_t)ni * ld->activation_count / layer->neuron_count;
                                uint64_t end = (uint64_t)(ni + 1u) * ld->activation_count / layer->neuron_count;
                                if (begin >= ld->activation_count) begin = ld->activation_count - 1;
                                if (end <= begin) end = begin + 1;
                                if (end > ld->activation_count) end = ld->activation_count;
                                double sum_abs = 0.0;
                                uint64_t count = 0;
                                for (uint64_t source = begin; source < end; ++source) {
                                    const float value = activation_values[source];
                                    if (isfinite(value)) { sum_abs += fabsf(value); ++count; }
                                }
                                node_metric[ni] = count ? (float)(sum_abs / (double)count) : 0.0f;
                            }
                            const float scale = probe_3d_p95(node_metric, layer->neuron_count);
                            for (uint32_t ni = 0; ni < layer->neuron_count; ++ni) {
                                layer->neurons[ni].activation =
                                    probe_3d_p95_contrast(node_metric[ni], scale);
                            }
                            free(node_metric);
                        } else if (matrix_values && ld->weight_count > 0 &&
                            (g_probe_link_mode != PROBE_LINK_POOLING ||
                             (global_state &&
                              (global_state->diagnostics_flags & PROBE_DIAG_FORWARD_VALID) &&
                              input_activation && input_activation_count >= ld->input_size))) {
                            {
                                /* 旧实现把全层 mean 复制给每个节点，导致整层永远
                                   同色，看起来像"有数据"但没有任何节点信息。用每个
                                   输出行的平均绝对权重形成稳定的相对强度，并以
                                   weight_count 限界以兼容旧拓扑。 */
                                uint64_t cols = ld->input_size;
                                uint64_t safe_rows = cols ? ld->weight_count / cols : 0;
                                float* row_metric = (float*)calloc(layer->neuron_count, sizeof(float));
                                if (!row_metric) continue;
                                float update_scale = 1.0f;
                                if (g_probe_link_mode == PROBE_LINK_UPDATE) {
                                    if (!global_state ||
                                        !(global_state->diagnostics_flags & PROBE_DIAG_GRADIENT_VALID)) {
                                        free(row_metric);
                                        continue;
                                    }
                                    update_scale = global_state->learning_rate *
                                        (float)global_state->gradient_span_steps;
                                    if (!isfinite(update_scale) || update_scale < 0.0f)
                                        update_scale = 0.0f;
                                }
                                for (uint32_t ni = 0; ni < layer->neuron_count; ni++) {
                                    if (safe_rows > 0) {
                                        double sum_abs = 0.0;
                                        uint64_t row_begin = (uint64_t)ni * safe_rows / layer->neuron_count;
                                        uint64_t row_end = (uint64_t)(ni + 1u) * safe_rows / layer->neuron_count;
                                        if (row_begin >= safe_rows) row_begin = safe_rows - 1;
                                        if (row_end <= row_begin) row_end = row_begin + 1;
                                        if (row_end > safe_rows) row_end = safe_rows;
                                        for (uint64_t row = row_begin; row < row_end; ++row) {
                                            const uint64_t row_off = row * cols;
                                            for (uint64_t ci = 0; ci < cols; ++ci) {
                                                float magnitude = fabsf(matrix_values[row_off + ci]);
                                                /* Pooling view represents the current forward
                                                   contribution |input * weight|, not another
                                                   copy of the static Weight view. */
                                                if (g_probe_link_mode == PROBE_LINK_POOLING)
                                                    magnitude *= fabsf(input_activation[ci]);
                                                sum_abs += magnitude * update_scale;
                                            }
                                        }
                                        const uint64_t value_count = (row_end - row_begin) * cols;
                                        row_metric[ni] = value_count ?
                                            (float)(sum_abs / (double)value_count) : 0.0f;
                                    }
                                }
                                const float scale = probe_3d_p95(row_metric, layer->neuron_count);
                                for (uint32_t ni = 0; ni < layer->neuron_count; ni++) {
                                    float activation =
                                        probe_3d_p95_contrast(row_metric[ni], scale);
                                    layer->neurons[ni].activation = activation;
                                    if (g_probe_link_mode == PROBE_LINK_GRADIENT)
                                        layer->neurons[ni].gradient = activation;
                                }
                                free(row_metric);
                            }
                        }
                    }
                    probe_3d_add_tensor_nodes();
                    probe_3d_collect_strong_connections();
                    nv_set_snapshot(&g_probe_3d_staging);
                }
                /* 阈值变化时重新收集连线 */
                if (g_nv_threshold_changed) {
                    g_nv_threshold_changed = 0;
                    probe_3d_collect_strong_connections();
                    nv_set_snapshot(&g_probe_3d_staging);
                }
            }
            /* Snapshot conversion happens here; the actual D3D Present is
               deliberately deferred until after vt_present() in main. */
        } else if (g_nv_3d_ready == -1) {
            set_color(COLOR_RED, COLOR_BLACK);
            print_clip(x, row++, mw, "3D Viewer failed (stage code %d). Press 3D View to retry.",
                g_nv_3d_error);
            set_color(COLOR_WHITE, COLOR_BLACK);
        } else {
            /* 未初始化：设置标志让主循环初始化 */
            g_nv_pending_init = 1;
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            print_clip(x, row++, mw, "3D Viewer - Initializing...");
            set_color(COLOR_WHITE, COLOR_BLACK);
        }
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 测试面板所有内容行（用于滚动） */
enum { TLINE_MAX = 128 };
static char  g_tline_text[TLINE_MAX][160];
static int   g_tline_color[TLINE_MAX];
static int   g_tline_hist_idx[TLINE_MAX];  /* 每行对应的历史记录索引，-1=非历史行 */
static int   g_tline_count = 0;

static void tline_begin(void) { g_tline_count = 0; }
static void tline_add(int color, const char* fmt, ...) {
    if (g_tline_count >= TLINE_MAX) return;
    g_tline_color[g_tline_count] = color;
    g_tline_hist_idx[g_tline_count] = -1;
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_tline_text[g_tline_count], 160, fmt, ap);
    va_end(ap);
    g_tline_count++;
}
static void tline_add_history(int color, int hist_idx, const char* fmt, ...) {
    if (g_tline_count >= TLINE_MAX) return;
    g_tline_color[g_tline_count] = color;
    g_tline_hist_idx[g_tline_count] = hist_idx;
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_tline_text[g_tline_count], 160, fmt, ap);
    va_end(ap);
    g_tline_count++;
}

static UiControlRect g_skill_run_hits[3] = {};
static UiControlRect g_skill_copy_hit = {};
static void draw_play_test(PlayPanel p) {
    for (auto& hit : g_skill_run_hits) hit = {};
    g_skill_copy_hit = {};
    read_skill_metrics_cached();
    clear_rect(p.x, p.y, p.w, p.h);
    draw_border(p.x, p.y, p.w, p.h, " SKILL TEST ");
    int mw = p.w - 4;
    if (mw < 10) return;

    DWORD now = GetTickCount();
    int run_btn_highlight = (now - g_run_btn_press < 500) ? 1 : 0;

    /* 超时保护 */
    if (g_snap_test_running &&
        (g_snap_test_start_tick == 0 || now - g_snap_test_start_tick > SKILL_TEST_TIMEOUT_MS)) {
        int was_timeout = (g_snap_test_start_tick != 0);
        g_snap_test_running = 0; g_snap_test_progress = 0; g_snap_test_phase = 0;
        { std::lock_guard<std::mutex> lk(g_snap_stage_mutex); g_snap_stage.state = TASK_TIMEOUT; }
        g_snap_test_start_tick = 0;
        if (was_timeout) snprintf(g_snap_test_error, sizeof(g_snap_test_error), "Test timeout (auto-reset)");
    }

    /* ---- 构建所有内容行 ---- */
    int run_line = -1;
    tline_begin();
    if (g_snap_test_error[0])
        tline_add(COLOR_RED, "%s", g_snap_test_error);

    if (g_snap_test_running) {
        /* 用与训练器一致的阶段进度模型：阶段名 + 已完成/总量 + 单位 + 百分比。 */
        StageProgress st = read_snap_stage();
        const char* name = st.stage != STG_NONE ? stage_progress_name(st.stage) : "准备中";
        if (st.indeterminate || st.total <= 0) {
            tline_add(COLOR_CYAN, "%s...", name);
        } else {
            int spct = (int)(st.completed * 100.0f / st.total);
            tline_add(COLOR_CYAN, "%s  %d/%d %s", name, st.completed, st.total,
                stage_progress_unit(st.unit));
            int bar_w = mw - 6; if (bar_w < 10) bar_w = 10;
            int filled = bar_w * st.completed / st.total;
            char bar[160]; int boff = 0;
            bar[boff++] = '[';
            for (int i = 0; i < bar_w; i++) bar[boff++] = (i < filled) ? '#' : '.';
            bar[boff++] = ']';
            boff += snprintf(bar + boff, 160 - boff, " %d%%", spct);
            bar[boff] = '\0';
            tline_add(COLOR_GREEN, "%s", bar);
        }
    } else if (g_snap_stage.state == TASK_TIMEOUT || g_snap_test_error[0]) {
        /* 失败/取消/超时：显示真实状态，不伪装成完成。 */
        const char* msg = g_snap_test_error[0] ? g_snap_test_error :
            (read_snap_stage().state == TASK_TIMEOUT ? "Test timeout" : "Test failed");
        run_line = g_tline_count;
        tline_add(COLOR_RED, "%s", msg);
        tline_add(COLOR_CYAN, "│ ▶ RUN AGAIN  │");
    } else {
        /* RUN 按钮（3行） */
        run_line = g_tline_count;
        tline_add(run_btn_highlight ? COLOR_CYAN : COLOR_CYAN, "┌──────────────┐");
        if (run_btn_highlight && !g_snap_test_done) tline_add(COLOR_CYAN, "│  ▶ RUNNING!  │");
        else if (g_snap_test_done) tline_add(COLOR_CYAN, "│ ▶ RUN AGAIN  │");
        else tline_add(COLOR_CYAN, "│ ▶ RUN TEST   │");
        tline_add(COLOR_CYAN, "└──────────────┘");
    }

    if (g_snap_test_done) {
        DWORD elapsed = g_snap_test_time - g_snap_test_start;
        SYSTEMTIME st; GetLocalTime(&st);
        tline_add(COLOR_GREEN, "Done %lu.%lus %02d:%02d:%02d  %s",
            elapsed / 1000, (elapsed % 1000) / 100, st.wHour, st.wMinute, st.wSecond,
            g_snap_result.source_info);
        tline_add(COLOR_CYAN, "── SNAPSHOT RESULT ──");

        int ew_pass = (g_snap_result.empty_win_rate >= 0.95f && g_snap_result.empty_avg_steps <= 6.0f);
        int ew_warn = (g_snap_result.empty_win_rate >= 0.85f && g_snap_result.empty_avg_steps <= 10.0f);
        int ew_c = ew_pass ? COLOR_GREEN : (ew_warn ? COLOR_YELLOW : COLOR_RED);
        tline_add(ew_c, "EmptyWin %5.1f%% (%d/%d) %s",
            g_snap_result.empty_win_rate * 100, g_snap_result.empty_wins, g_snap_result.empty_games,
            ew_pass ? "PASS" : (ew_warn ? "WARN" : "FAIL"));
        tline_add(COLOR_DARK_GRAY, "  avg:%.1f min:%.0f max:%.0f",
            g_snap_result.empty_avg_steps, g_snap_result.empty_min_steps, g_snap_result.empty_max_steps);
        /* 64快照统计口径 */
        tline_add(COLOR_DARK_GRAY, "  snap: %.1f%%±%.1f%% [%.1f~%.1f]",
            g_snap_result.empty_stat_mean * 100, g_snap_result.empty_stat_std * 100,
            g_snap_result.empty_stat_min * 100, g_snap_result.empty_stat_max * 100);

        int w1_c = g_snap_result.win1_fixed >= 0.98f ? COLOR_GREEN :
                   (g_snap_result.win1_fixed >= 0.95f ? COLOR_YELLOW : COLOR_RED);
        tline_add(w1_c, "Win1   %5.1f%% (%d/%d) %s",
            g_snap_result.win1_fixed * 100, g_snap_result.win1_correct, g_snap_result.win1_total,
            g_snap_result.win1_fixed >= 0.98f ? "PASS" : (g_snap_result.win1_fixed >= 0.95f ? "WARN" : "FAIL"));
        tline_add(COLOR_DARK_GRAY, "  Q:%+.3f~%+.3f m:%+.3f~%+.3f",
            g_snap_result.win1_min_q, g_snap_result.win1_max_q,
            g_snap_result.win1_min_margin, g_snap_result.win1_max_margin);
        tline_add(COLOR_DARK_GRAY, "  ok:%+.4f miss:%+.4f",
            g_snap_result.win1_avg_q_correct, g_snap_result.win1_avg_q_wrong);

        int b1_c = g_snap_result.block1_fixed >= 0.95f ? COLOR_GREEN :
                   (g_snap_result.block1_fixed >= 0.80f ? COLOR_YELLOW :
                   (g_snap_result.block1_fixed > 0.01f ? COLOR_RED : COLOR_DARK_GRAY));
        const char* b1s = "LEARNING";
        if (g_snap_result.block1_fixed >= 0.95f) b1s = "PASS";
        else if (g_snap_result.block1_fixed >= 0.80f) b1s = "WARN";
        else if (g_snap_result.block1_fixed > 0.01f) b1s = "LOW";
        tline_add(b1_c, "Block1 %5.1f%% (%d/%d) %s",
            g_snap_result.block1_fixed * 100, g_snap_result.block1_correct, g_snap_result.block1_total, b1s);
        tline_add(COLOR_DARK_GRAY, "  Q:%+.3f~%+.3f m:%+.3f~%+.3f",
            g_snap_result.block1_min_q, g_snap_result.block1_max_q,
            g_snap_result.block1_min_margin, g_snap_result.block1_max_margin);
        tline_add(COLOR_DARK_GRAY, "  ok:%+.4f miss:%+.4f",
            g_snap_result.block1_avg_q_correct, g_snap_result.block1_avg_q_wrong);
    }

    /* 分隔线 */
    char sep[160]; int si = 0;
    for (int c = 0; c < mw && c < 158; c++) sep[si++] = '-';
    sep[si] = '\0';
    tline_add(COLOR_DARK_GRAY, "%s", sep);

    /* 历史数据 */
    tline_add(COLOR_DARK_GRAY, "%-8s %-8s %-6s %-8s %-8s  (近1000局/64快照)",
        "Episode", "Empty", "Steps", "Win1", "Blk1");
    if (g_skill_display_count == 0) {
        tline_add(COLOR_DARK_GRAY, "(无历史数据，等待训练评估)");
    } else {
        for (int i = g_skill_display_count - 1; i >= 0; i--) {
            SkillDisplayRecord* r = &g_skill_display[i];
            int ew_c = r->empty_win >= 0.95f ? COLOR_GREEN : (r->empty_win >= 0.85f ? COLOR_YELLOW : COLOR_RED);
            char hl[160]; int ho = 0;
            ho += snprintf(hl + ho, 160 - ho, "%-8d", r->episode);
            ho += snprintf(hl + ho, 160 - ho, "%.1f%%   ", r->empty_win * 100);
            ho += snprintf(hl + ho, 160 - ho, "%-5.1f ", r->avg_steps);
            ho += snprintf(hl + ho, 160 - ho, "%.1f%%   ", r->win1_fixed * 100);
            if (r->block1_fixed > 0.01f) snprintf(hl + ho, 160 - ho, "%.1f%%", r->block1_fixed * 100);
            else snprintf(hl + ho, 160 - ho, "LEARNING");
            tline_add_history(ew_c, i, "%s", hl);
            /* 展开选中记录的详情（对标手动测试报告） */
            if (i == g_skill_selected_hist) {
                tline_add(COLOR_CYAN, "  ── EP %d DETAIL ──", r->episode);
                /* EmptyWin */
                int ew_pass = (r->empty_win >= 0.95f && r->avg_steps <= 6.0f);
                int ew_warn = (r->empty_win >= 0.85f && r->avg_steps <= 10.0f);
                tline_add(COLOR_WHITE, "  EmptyWin %s", ew_pass ? "PASS" : (ew_warn ? "WARN" : "FAIL"));
                tline_add(COLOR_WHITE, "    Win: %.1f%%  avg:%.1f  min:%.0f max:%.0f",
                    r->empty_win * 100, r->avg_steps, r->empty_min_steps, r->empty_max_steps);
                if (r->empty_stat_std > 0)
                    tline_add(COLOR_DARK_GRAY, "    snap: %.1f%% +/- %.1f%% [%.1f ~ %.1f]",
                        r->empty_win * 100, r->empty_stat_std * 100,
                        r->empty_stat_min * 100, r->empty_stat_max * 100);
                /* Win1 */
                int w1_c = r->win1_fixed >= 0.98f ? COLOR_GREEN :
                           (r->win1_fixed >= 0.95f ? COLOR_YELLOW : COLOR_RED);
                tline_add(COLOR_WHITE, "  Win1  %s", r->win1_fixed >= 0.98f ? "PASS" :
                    (r->win1_fixed >= 0.95f ? "WARN" : "FAIL"));
                tline_add(COLOR_WHITE, "    Acc: %.1f%%  margin: %+.4f",
                    r->win1_fixed * 100, r->win1_fixed_margin);
                tline_add(COLOR_DARK_GRAY, "    Q: %+.4f ~ %+.4f  m: %+.4f ~ %+.4f",
                    r->win1_min_q, r->win1_max_q, r->win1_min_margin, r->win1_max_margin);
                tline_add(COLOR_DARK_GRAY, "    ok: %+.4f  miss: %+.4f",
                    r->win1_avg_q_correct, r->win1_avg_q_wrong);
                if (r->win1_acc_std > 0)
                    tline_add(COLOR_DARK_GRAY, "    std: %.1f%%", r->win1_acc_std * 100);
                /* Block1 */
                int b1_c = r->block1_fixed >= 0.95f ? COLOR_GREEN :
                           (r->block1_fixed >= 0.80f ? COLOR_YELLOW :
                           (r->block1_fixed > 0.01f ? COLOR_RED : COLOR_DARK_GRAY));
                tline_add(COLOR_WHITE, "  Block1  %s", r->block1_fixed >= 0.95f ? "PASS" :
                    (r->block1_fixed >= 0.80f ? "WARN" : "LEARNING"));
                tline_add(COLOR_WHITE, "    Acc: %.1f%%  margin: %+.4f",
                    r->block1_fixed * 100, r->block1_fixed_margin);
                tline_add(COLOR_DARK_GRAY, "    Q: %+.4f ~ %+.4f  m: %+.4f ~ %+.4f",
                    r->block1_min_q, r->block1_max_q, r->block1_min_margin, r->block1_max_margin);
                tline_add(COLOR_DARK_GRAY, "    ok: %+.4f  miss: %+.4f",
                    r->block1_avg_q_correct, r->block1_avg_q_wrong);
                if (r->block1_acc_std > 0)
                    tline_add(COLOR_DARK_GRAY, "    std: %.1f%%", r->block1_acc_std * 100);
                tline_add(COLOR_DARK_GRAY, "  [click to collapse]");
            }
        }
    }

    /* ---- 滚动渲染 ---- */
    int avail = p.h - 2;   /* 可用行数（去掉边框） */
    if (g_snap_test_done) avail--;   /* COPY 按钮占1行 */
    int total = g_tline_count;
    int max_scroll = total - avail;
    if (max_scroll < 0) max_scroll = 0;
    if (g_skill_test_scroll > max_scroll) g_skill_test_scroll = max_scroll;
    if (g_skill_test_scroll < 0) g_skill_test_scroll = 0;

    int start = g_skill_test_scroll;
    int end = start + avail;
    if (end > total) end = total;

    /* COPY 按钮始终固定在右下角 */
    if (g_snap_test_done) {
        int copy_btn_x = p.x + p.w - 18;
        int copy_btn_y = p.y + p.h - 2;
        int copy_highlight = (now - g_copy_btn_press < 1000) ? 1 : 0;
        g_copy_btn_state = copy_highlight;
        gotoxy(copy_btn_x, copy_btn_y);
        g_skill_copy_hit = control_clip(copy_btn_x, copy_btn_y, 16,
            copy_highlight ? "[   COPIED!   ]" : "[ COPY REPORT ]", false, false);
        set_color(COLOR_WHITE, COLOR_BLACK);
        /* COPY 按钮占用的行不画内容 */
    }

    for (int i = start; i < end; i++) {
        int screen_row = p.y + 1 + (i - start);
        if (g_snap_test_done && screen_row == p.y + p.h - 2) continue; /* COPY 按钮行 */
        gotoxy(p.x + 2, screen_row);
        set_color(g_tline_color[i], COLOR_BLACK);
        if (run_line >= 0 && i >= run_line && i < run_line + 3) {
            g_skill_run_hits[i - run_line] = control_clip(p.x + 2, screen_row, mw, g_tline_text[i], false, false);
        } else if (g_snap_test_running && i < 3) {
            print_roll_clip(p.x + 2, screen_row, mw, "%s", g_tline_text[i]);
        } else {
            print_clip(p.x + 2, screen_row, mw, "%s", g_tline_text[i]);
        }
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 复制报告到剪贴板 */
static void copy_skill_report_to_clipboard(void) {
    char buf[8192];
    int off = 0;
    DWORD elapsed = g_snap_test_time - g_snap_test_start;
    SYSTEMTIME st;
    GetLocalTime(&st);
    off += snprintf(buf + off, sizeof(buf) - off, "========================================\r\n");
    off += snprintf(buf + off, sizeof(buf) - off, "         SKILL TEST REPORT v%d\r\n", SKILL_SUITE_VERSION);
    off += snprintf(buf + off, sizeof(buf) - off, "========================================\r\n");
    off += snprintf(buf + off, sizeof(buf) - off, "Generated:  %04d-%02d-%02d %02d:%02d:%02d\r\n",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    off += snprintf(buf + off, sizeof(buf) - off, "Source:     %s\r\n", g_snap_result.source_info);
    off += snprintf(buf + off, sizeof(buf) - off, "Model:      frozen CPU weights, canonical input\r\n");
    off += snprintf(buf + off, sizeof(buf) - off, "Duration:   %lu.%lus\r\n", elapsed / 1000, (elapsed % 1000) / 100);
    off += snprintf(buf + off, sizeof(buf) - off, "----------------------------------------\r\n");

    /* EmptyWin */
    off += snprintf(buf + off, sizeof(buf) - off, "\r\n[EmptyWin]  %s\r\n",
        g_snap_result.empty_win_rate >= 0.95f && g_snap_result.empty_avg_steps <= 6.0f ? "PASS" : "WARN");
    off += snprintf(buf + off, sizeof(buf) - off, "  Win rate:    %.1f%%  (%d/%d)\r\n",
        g_snap_result.empty_win_rate * 100, g_snap_result.empty_wins, g_snap_result.empty_games);
    off += snprintf(buf + off, sizeof(buf) - off, "  Avg steps:   %.1f\r\n", g_snap_result.empty_avg_steps);
    off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max:     %.0f / %.0f\r\n",
        g_snap_result.empty_min_steps, g_snap_result.empty_max_steps);

    /* Win1 */
    off += snprintf(buf + off, sizeof(buf) - off, "\r\n[Win-in-1]  %s\r\n",
        g_snap_result.win1_fixed >= 0.98f ? "PASS" : (g_snap_result.win1_fixed >= 0.95f ? "WARN" : "FAIL"));
    off += snprintf(buf + off, sizeof(buf) - off, "  Accuracy:    %.1f%%  (%d/%d)\r\n",
        g_snap_result.win1_fixed * 100, g_snap_result.win1_correct, g_snap_result.win1_total);
    off += snprintf(buf + off, sizeof(buf) - off, "  Avg margin:  %+.4f\r\n", g_snap_result.win1_fixed_margin);
    off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max Q:   %+.4f / %+.4f\r\n",
        g_snap_result.win1_min_q, g_snap_result.win1_max_q);
    off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max margin: %+.4f / %+.4f\r\n",
        g_snap_result.win1_min_margin, g_snap_result.win1_max_margin);
    off += snprintf(buf + off, sizeof(buf) - off, "  Q correct avg: %+.4f\r\n", g_snap_result.win1_avg_q_correct);
    off += snprintf(buf + off, sizeof(buf) - off, "  Q wrong avg:   %+.4f\r\n", g_snap_result.win1_avg_q_wrong);

    /* Block1 */
    off += snprintf(buf + off, sizeof(buf) - off, "\r\n[Block-in-1]  %s\r\n",
        g_snap_result.block1_fixed >= 0.95f ? "PASS" : (g_snap_result.block1_fixed >= 0.80f ? "WARN" : "LEARNING"));
    off += snprintf(buf + off, sizeof(buf) - off, "  Accuracy:    %.1f%%  (%d/%d)\r\n",
        g_snap_result.block1_fixed * 100, g_snap_result.block1_correct, g_snap_result.block1_total);
    off += snprintf(buf + off, sizeof(buf) - off, "  Avg margin:  %+.4f\r\n", g_snap_result.block1_fixed_margin);
    off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max Q:   %+.4f / %+.4f\r\n",
        g_snap_result.block1_min_q, g_snap_result.block1_max_q);
    off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max margin: %+.4f / %+.4f\r\n",
        g_snap_result.block1_min_margin, g_snap_result.block1_max_margin);
    off += snprintf(buf + off, sizeof(buf) - off, "  Q correct avg: %+.4f\r\n", g_snap_result.block1_avg_q_correct);
    off += snprintf(buf + off, sizeof(buf) - off, "  Q wrong avg:   %+.4f\r\n", g_snap_result.block1_avg_q_wrong);

    off += snprintf(buf + off, sizeof(buf) - off, "\r\n========================================\r\n");
    off += snprintf(buf + off, sizeof(buf) - off, "Puzzles: Win1=%d Blk1=%d SoloEmpty=%d\r\n",
        g_snap_result.win1_total, g_snap_result.block1_total, g_snap_result.empty_games);
    off += snprintf(buf + off, sizeof(buf) - off, "Forced-first (25 starts): %.1f%% avg %.2f total stones\r\n",
        g_snap_result.seeded_win_rate * 100, g_snap_result.seeded_avg_steps);
    off += snprintf(buf + off, sizeof(buf) - off, "Margin = best correct Q - best incorrect Q. GPU parity: not run (CPU snapshot).\r\n");

    /* 历史记录（每1000局详细数据，对标手动测试报告） */
    if (g_skill_display_count > 0) {
        off += snprintf(buf + off, sizeof(buf) - off, "\r\n========================================\r\n");
        off += snprintf(buf + off, sizeof(buf) - off, "         HISTORY (%d records)\r\n", g_skill_display_count);
        off += snprintf(buf + off, sizeof(buf) - off, "========================================\r\n");
        for (int i = g_skill_display_count - 1; i >= 0 && off < (int)sizeof(buf) - 400; i--) {
            SkillDisplayRecord* r = &g_skill_display[i];
            off += snprintf(buf + off, sizeof(buf) - off, "\r\n--- Episode %d ---\r\n", r->episode);
            /* EmptyWin */
            off += snprintf(buf + off, sizeof(buf) - off, "[EmptyWin] %s\r\n",
                (r->empty_win >= 0.95f && r->avg_steps <= 6.0f) ? "PASS" : "WARN");
            off += snprintf(buf + off, sizeof(buf) - off, "  Win rate: %.1f%%  avg:%.1f  min:%.0f max:%.0f\r\n",
                r->empty_win * 100, r->avg_steps, r->empty_min_steps, r->empty_max_steps);
            if (r->empty_stat_std > 0)
                off += snprintf(buf + off, sizeof(buf) - off, "  Snap: %.1f%% +/- %.1f%% [%.1f ~ %.1f]\r\n",
                    r->empty_win * 100, r->empty_stat_std * 100,
                    r->empty_stat_min * 100, r->empty_stat_max * 100);
            /* Win1 */
            off += snprintf(buf + off, sizeof(buf) - off, "[Win-in-1] %s\r\n",
                r->win1_fixed >= 0.98f ? "PASS" : (r->win1_fixed >= 0.95f ? "WARN" : "FAIL"));
            off += snprintf(buf + off, sizeof(buf) - off, "  Accuracy: %.1f%%  margin: %+.4f\r\n",
                r->win1_fixed * 100, r->win1_fixed_margin);
            off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max Q: %+.4f / %+.4f\r\n",
                r->win1_min_q, r->win1_max_q);
            off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max margin: %+.4f / %+.4f\r\n",
                r->win1_min_margin, r->win1_max_margin);
            off += snprintf(buf + off, sizeof(buf) - off, "  Q correct avg: %+.4f  Q wrong avg: %+.4f\r\n",
                r->win1_avg_q_correct, r->win1_avg_q_wrong);
            if (r->win1_acc_std > 0)
                off += snprintf(buf + off, sizeof(buf) - off, "  Std: %.1f%%\r\n", r->win1_acc_std * 100);
            /* Block1 */
            off += snprintf(buf + off, sizeof(buf) - off, "[Block-in-1] %s\r\n",
                r->block1_fixed >= 0.95f ? "PASS" : (r->block1_fixed >= 0.80f ? "WARN" : "LEARNING"));
            off += snprintf(buf + off, sizeof(buf) - off, "  Accuracy: %.1f%%  margin: %+.4f\r\n",
                r->block1_fixed * 100, r->block1_fixed_margin);
            off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max Q: %+.4f / %+.4f\r\n",
                r->block1_min_q, r->block1_max_q);
            off += snprintf(buf + off, sizeof(buf) - off, "  Min/Max margin: %+.4f / %+.4f\r\n",
                r->block1_min_margin, r->block1_max_margin);
            off += snprintf(buf + off, sizeof(buf) - off, "  Q correct avg: %+.4f  Q wrong avg: %+.4f\r\n",
                r->block1_avg_q_correct, r->block1_avg_q_wrong);
            if (r->block1_acc_std > 0)
                off += snprintf(buf + off, sizeof(buf) - off, "  Std: %.1f%%\r\n", r->block1_acc_std * 100);
        }
    }

    if (OpenClipboard(NULL)) {
        EmptyClipboard();
        HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, off + 1);
        if (h) {
            char* p = (char*)GlobalLock(h);
            memcpy(p, buf, off + 1);
            GlobalUnlock(h);
            SetClipboardData(CF_TEXT, h);
        }
        CloseClipboard();
    }
}

/* 处理 SKILL TEST 面板的鼠标点击 */
static int handle_play_test_wheel(int mx, int my, int delta) {
    if (g_active_tab != 1 || delta == 0) return 0;
    if (g_console_width < 150 && g_play_sidebar_tab != 4) return 0;
    PlayPanel p[PLY_COUNT];
    compute_playground_layout(p);
    PlayPanel t = p[PLY_TEST];
    if (mx < t.x || mx >= t.x + t.w || my < t.y || my >= t.y + t.h) return 0;
    /* 归一化：向下滚看更早的记录（offset 增大） */
    int step = (delta < 0) ? 1 : -1;
    g_skill_test_scroll += step;
    if (g_skill_test_scroll < 0) g_skill_test_scroll = 0;
    int avail = t.h - 2;
    int max_scroll = g_tline_count - avail + 8;   /* +8 余量，确保最旧记录和展开详情可见 */
    if (max_scroll < 0) max_scroll = 0;
    if (g_skill_test_scroll > max_scroll) g_skill_test_scroll = max_scroll;
    render_playground();
    return 1;
}

/* Move History 面板滚轮 */
static int handle_hist_wheel(int mx, int my, int delta) {
    if (g_active_tab != 1 || delta == 0) return 0;
    if (g_console_width >= 150) {
        /* 大屏：检查 PLY_HIST 区域 */
        PlayPanel p[PLY_COUNT];
        compute_playground_layout(p);
        PlayPanel h = p[PLY_HIST];
        if (mx < h.x || mx >= h.x + h.w || my < h.y || my >= h.y + h.h) return 0;
    } else {
        /* 小屏：检查侧边栏 History tab 区域 */
        if (g_play_sidebar_tab != 2) return 0;
        PlayPanel p[PLY_COUNT];
        compute_playground_layout(p);
        PlayPanel h = p[PLY_MODEL]; /* 侧边栏位置 */
        if (mx < h.x || mx >= h.x + h.w || my < h.y || my >= h.y + h.h) return 0;
    }
    g_hist_scroll += (delta > 0) ? 3 : -3;
    if (g_hist_scroll < 0) g_hist_scroll = 0;
    render_playground();
    return 1;
}

static void handle_play_test_click(int mx, int my, PlayPanel p) {
    /* RUN TEST 按钮：p.y+1 到 p.y+3，p.x+2 到 p.x+17 */
    bool hit_run = false;
    for (const auto& hit : g_skill_run_hits) if (hit.contains(mx, my)) hit_run = true;
    if (hit_run) {
        if (!g_snap_test_running) {
            g_run_btn_press = GetTickCount();
            run_snapshot_skill_test();
        }
        return;
    }
    /* COPY REPORT 按钮：右下角 */
    if (g_snap_test_done) {
        int copy_btn_x = p.x + p.w - 18;
        int copy_btn_y = p.y + p.h - 2;
        if (g_skill_copy_hit.contains(mx, my)) {
            copy_skill_report_to_clipboard();
            g_copy_btn_press = GetTickCount();
            g_copy_btn_state = 1;
            return;
        }
    }
    /* 历史记录点击：展开/收起详情 */
    int click_row = my - (p.y + 1);  /* 面板内相对行号 */
    int line_idx = g_skill_test_scroll + click_row;
    if (line_idx >= 0 && line_idx < g_tline_count && g_tline_hist_idx[line_idx] >= 0) {
        int hist_idx = g_tline_hist_idx[line_idx];
        g_skill_selected_hist = (g_skill_selected_hist == hist_idx) ? -1 : hist_idx;
        render_playground();
    }
}

/* 小屏侧边栏（左侧）：竖线分隔，数字键 1-4 切换展开项 */
static void draw_play_sidebar(PlayPanel p) {
    clear_rect(p.x, p.y, p.w, p.h);
    /* 右侧竖线分隔 */
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    for (int r = 0; r < p.h; r++) { gotoxy(p.x + p.w - 1, p.y + r); printf("│"); }
    set_color(COLOR_WHITE, COLOR_BLACK);
    gotoxy(p.x, p.y);
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("SIDEBAR");
    set_color(COLOR_WHITE, COLOR_BLACK);
    static const char* items[5] = { "1 Model", "2 Settings", "3 History", "4 Eval", "5 Skill Test" };
    for (int i = 0; i < 5; i++) {
        gotoxy(p.x, p.y + 1 + i);
        control_here(items[i], i == g_play_sidebar_tab, true);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    PlayPanel sub = { p.x, p.y + 7, p.w - 1, p.h - 7 };
    switch (g_play_sidebar_tab) {
    case 0: draw_play_model(sub); break;
    case 1: draw_play_settings(sub); break;
    case 2: draw_play_hist(sub); break;
    case 3: draw_play_eval(sub); break;
    }
}

static void refresh_play_analysis(void);

/* 落子（模型执黑 X 先手，用户执白 O 后应） */
/* AI 只推理不落子（更新 Q 表供观察） */
static void play_ai_infer_only(void) {
    if (g_play_over >= 0 || g_play_moves >= 25) return;
    refresh_play_analysis();   /* 重新推理，Q 表会更新 */
    g_play_ai_pending = 1;     /* 标记：AI 已推理，等待确认落子 */
}

/* AI 确认落子 */
static void play_ai_confirm_move(void) {
    if (g_play_over >= 0 || g_play_moves >= 25 || !g_play_ai_pending) return;
    int am = playground_ai_move(g_play_board, 0);   /* AI 执黑 X */
    if (am >= 0 && g_play_board[am] == 0) {
        g_play_board[am] = 1;
        g_play_hist[g_play_hist_n++] = am;
        g_play_moves++;
        if (play_win_check(g_play_board, 1)) g_play_over = 1;
        else if (g_play_moves >= 25) g_play_over = 0;
    }
    g_play_ai_pending = 0;
    refresh_play_analysis();
}

/* AI 自动执行一步（自动模式用） */
static void play_ai_step(void) {
    if (g_play_over >= 0 || g_play_moves >= 25) return;
    int am = playground_ai_move(g_play_board, 0);
    if (am >= 0 && g_play_board[am] == 0) {
        g_play_board[am] = 1;
        g_play_hist[g_play_hist_n++] = am;
        g_play_moves++;
        if (play_win_check(g_play_board, 1)) g_play_over = 1;
        else if (g_play_moves >= 25) g_play_over = 0;
    }
    g_play_ai_pending = 0;
    refresh_play_analysis();
}

static void play_do_move(int idx) {
    if (g_play_over < 0 && g_play_board[idx] == 0) {
        g_play_board[idx] = 2;   /* 用户执白 O */
        g_play_hist[g_play_hist_n++] = idx;
        g_play_moves++;
        if (play_win_check(g_play_board, 2)) g_play_over = 2;   /* 用户白胜 */
        else if (g_play_moves >= 25) g_play_over = 0;
        else if (!g_play_manual_step) {
            play_ai_step();   /* 自动模式：AI 立即跟 */
        } else {
            play_ai_infer_only();   /* 手动模式：只推理，不落子 */
        }
        refresh_play_analysis();
    }
}
static void play_undo(void) {
    if (g_play_hist_n >= 2) {
        g_play_board[g_play_hist[--g_play_hist_n]] = 0;
        g_play_board[g_play_hist[--g_play_hist_n]] = 0;
        g_play_moves -= 2;
        g_play_over = -1;
        refresh_play_analysis();
    }
}
static void play_restart(void) {
    memset(g_play_board, 0, sizeof(g_play_board));
    g_play_cur = 12; g_play_moves = 0; g_play_hist_n = 0; g_play_over = -1;
    /* 模型先手：重置后 AI 走黑 X 第一步 */
    int am = playground_ai_move(g_play_board, 0);
    if (am >= 0 && g_play_board[am] == 0) {
        g_play_board[am] = 1;
        g_play_hist[g_play_hist_n++] = am;
        g_play_moves++;
    }
    refresh_play_analysis();
}

static int snapshot_file_changed(const WIN32_FILE_ATTRIBUTE_DATA* before, int had_before) {
    WIN32_FILE_ATTRIBUTE_DATA current = {};
    if (!GetFileAttributesExA("current_model_package.bin", GetFileExInfoStandard, &current)) return 0;
    return current.nFileSizeHigh == 0 && current.nFileSizeLow > 0 &&
        (!had_before || CompareFileTime(&current.ftLastWriteTime, &before->ftLastWriteTime) != 0 ||
         current.nFileSizeHigh != before->nFileSizeHigh || current.nFileSizeLow != before->nFileSizeLow);
}

/* 已经运行的旧 Trainer 可能只给当前 TUI 读权限。断开只读连接后写入
   一个固定用途的请求文件，由同会话 Watchdog 临时连接旧管道并转发。
   该代理只接受 SAVE_SNAPSHOT，不开放任意 Trainer 命令。 */
static int request_snapshot_via_watchdog(const WIN32_FILE_ATTRIBUTE_DATA* before, int had_before) {
    if (g_pipe != INVALID_HANDLE_VALUE) CloseHandle(g_pipe);
    g_pipe = INVALID_HANDLE_VALUE;
    g_pipe_readonly = 0;
    remove("wd_snapshot_result.txt");
    remove("wd_snapshot_request.txt");
    FILE* f = fopen("wd_snapshot_request.tmp", "wb");
    if (!f) return 0;
    fputs("SAVE_SNAPSHOT\n", f);
    fclose(f);
    if (!MoveFileExA("wd_snapshot_request.tmp", "wd_snapshot_request.txt",
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return 0;

    int updated = 0;
    for (int wait = 0; wait < 200; ++wait) {
        if (snapshot_file_changed(before, had_before)) { updated = 1; break; }
        FILE* result = fopen("wd_snapshot_result.txt", "rb");
        if (result) {
            char line[96] = "";
            fgets(line, sizeof(line), result);
            fclose(result);
            if (strncmp(line, "ERROR", 5) == 0) break;
        }
        win_pump_messages();
        Sleep(50);
    }
    remove("wd_snapshot_result.txt");
    pipe_connect();
    return updated;
}

/* 重拉即时快照：先重连 pipe（训练可能重启过，旧句柄失效），再请求保存并重新加载 */
static void play_snapshot(void) {
    /* A running skill suite must use one immutable network+memory generation. */
    if (g_snap_test_running) {
        snprintf(g_play_ckpt, sizeof(g_play_ckpt), "skill test busy; retry later");
        return;
    }
    /* 记录快照前的局号 */
    int snap_ep = status_lines[0][0] ? atoi(status_lines[0]) : 0;
    WIN32_FILE_ATTRIBUTE_DATA before = {};
    int had_before = GetFileAttributesExA("current_model_package.bin", GetFileExInfoStandard, &before) != 0;
    if (g_pipe == INVALID_HANDLE_VALUE) pipe_connect();
    int updated = 0;
    if (g_pipe != INVALID_HANDLE_VALUE && !g_pipe_readonly && pipe_send("SAVE_SNAPSHOT")) {
        /* 旧文件一直存在，不能把"能打开"当成新快照完成。
           等待修改时间/尺寸真正改变；Trainer 使用临时文件原子替换。 */
        for (int w = 0; w < 200; w++) {
            if (snapshot_file_changed(&before, had_before)) { updated = 1; break; }
            pipe_poll();   /* 同时消费 Trainer 响应，防止管道回压 */
            win_pump_messages();
            Sleep(50);
        }
    } else {
        snprintf(g_play_ckpt, sizeof(g_play_ckpt), "snapshot via watchdog...");
        updated = request_snapshot_via_watchdog(&before, had_before);
    }
    if (!updated) {
        snprintf(g_play_ckpt, sizeof(g_play_ckpt), "snapshot failed (old kept)");
        return;   /* 绝不能把陈旧文件伪装成刚生成的快照并重开棋局。 */
    }
    if (playground_load_snapshot() == 0) {
        /* 标注快照来源局号 */
        if (snap_ep > 0)
            snprintf(g_play_ckpt, sizeof(g_play_ckpt), "EP %d", snap_ep);
        else
            snprintf(g_play_ckpt, sizeof(g_play_ckpt), "snapshot updated");
        play_restart();
    } else {
        snprintf(g_play_ckpt, sizeof(g_play_ckpt), "FAIL");
    }
}

/* 鼠标点击：底部操作栏按钮 / 棋盘落子 / 小屏侧边栏切换 */
static void handle_play_mouse(int mx, int my) {
    int cb = g_console_height - 10;   /* content_bot */
    g_play_btn = -1;
    /* 操作栏行2 (cb+5): [R] Restart(2-12) [S] Snapshot(16-27) [ESC] Back(31-40) [AI Step](40-49) */
    if (my == cb + 5) {
        if (mx >= 2 && mx < 2 + 12) { g_play_btn = 4; play_restart(); return; }
        if (mx >= 16 && mx < 16 + 12) { g_play_btn = 5; play_snapshot(); return; }
        if (mx >= 31 && mx < 31 + 10) { g_play_btn = 6; g_active_tab = 0; need_tab = 1; need_content = 1; return; }
        if (mx >= 40 && mx < 40 + 10 && g_play_over < 0) { g_play_btn = 7; play_ai_step(); return; }
    }
    /* 操作栏行1 (cb+4): [ENTER] Place(22-33) [U] Undo(37-44) [A] Analysis(48-59) */
    if (my == cb + 4) {
        if (mx >= 22 && mx < 22 + 12) { g_play_btn = 1; play_do_move(g_play_cur); return; }
        if (mx >= 37 && mx < 37 + 10) { g_play_btn = 2; play_undo(); return; }
        if (mx >= 48 && mx < 48 + 12) { g_play_btn = 3; g_play_analysis_on = 1 - g_play_analysis_on; refresh_play_analysis(); return; }
    }
    PlayPanel p[PLY_COUNT];
    compute_playground_layout(p);
    if (g_console_width < 150 && g_play_sidebar_tab == 4 &&
        mx >= p[PLY_TEST].x && mx < p[PLY_TEST].x+p[PLY_TEST].w &&
        my >= p[PLY_TEST].y && my < p[PLY_TEST].y+p[PLY_TEST].h) {
        handle_play_test_click(mx, my, p[PLY_TEST]);
        return;
    }

    PlayPanel bp = p[PLY_BOARD];
    int bx = bp.x + (bp.w - 16) / 2;
    if (bx < bp.x + 2) bx = bp.x + 2;
    int by = bp.y + 2;
    /* 棋盘区域点击 → 映射格子 */
    if (mx >= bx - 1 && mx <= bx + 16 && my >= by + 1 && my <= by + 11) {
        int c = (mx - bx) / 3;
        int r = (my - by - 2) / 2;
        if (c < 0) c = 0; if (c > 4) c = 4;
        if (r < 0) r = 0; if (r > 4) r = 4;
        int idx = r * 5 + c;
        g_play_cur = idx;
        play_do_move(idx);
        return;
    }
    /* STEP 勾选框：bp.y + bp.h - 2 */
    int cb_y = bp.y + bp.h - 2;
    if (my == cb_y && mx >= bp.x + 2 && mx < bp.x + 12) {
        g_play_manual_step = 1 - g_play_manual_step;
        return;
    }
    /* 小屏侧边栏点击 */
    if (g_console_width < 150) {
        int top = CONTENT_Y + 3;
        int content_bot = g_console_height - 10;
        int cw = (g_console_width - 4) / 4;
        int x0 = 2;
        if (mx >= x0 && mx < x0 + cw && my >= top && my < top + (content_bot - top + 1)) {
            for (int i = 0; i < 5; i++)
                if (my == top + 1 + i) { g_play_sidebar_tab = i; return; }
        }
    }
    /* SKILL TEST 面板点击 */
    if (g_console_width >= 150) {
        handle_play_test_click(mx, my, p[PLY_TEST]);
    }
}

/* 刷新当前局面的 AI 分析（top 候选 + 推理耗时） */
static void refresh_play_analysis(void) {
    g_play_cand_n = 0;
    if (!g_play_analysis_on) return;
    int turn = g_play_moves % 2;   /* 0=黑(用户)回合, 1=白(AI)回合 */
    float qvals[25];
    g_play_cand_n = playground_analyze(g_play_board, turn, qvals,
        g_play_cands, 5, &g_play_cand_n, &g_play_infer_ms);
}

/* 大框内横线（├────┤） */
static void play_hline(int y) {
    gotoxy(0, y);
    set_color(COLOR_WHITE, COLOR_BLACK);
    printf("├");
    for (int i = 1; i < g_console_width - 1; i++) printf("─");
    printf("┤");
}

void render_playground(void) {
    /* tab 首次进入：加载即时快照 + 初始化棋盘（此后保持状态，不重载） */
    if (!g_play_init) {
        g_play_init = 1;
        play_snapshot();   /* 等待新快照真正落盘，再让模型先手 */
        g_play_ckpt[0] = '\0';
        WIN32_FIND_DATAA fd;
        HANDLE hc = FindFirstFileA(CKPT_DIR "checkpoint_*.bin", &fd);
        if (hc != INVALID_HANDLE_VALUE) {
            FILETIME best = fd.ftLastWriteTime;
            snprintf(g_play_ckpt, sizeof(g_play_ckpt), "%s", fd.cFileName);
            while (FindNextFileA(hc, &fd)) {
                if (CompareFileTime(&fd.ftLastWriteTime, &best) > 0) {
                    best = fd.ftLastWriteTime;
                    snprintf(g_play_ckpt, sizeof(g_play_ckpt), "%s", fd.cFileName);
                }
            }
            FindClose(hc);
        }
        if (!g_play_ckpt[0]) snprintf(g_play_ckpt, sizeof(g_play_ckpt), "CP_0015");
    }
    draw_help_bar();   /* 始终保留底部 help 栏，退出后横线不被占据 */
    PlayPanel p[PLY_COUNT];
    compute_playground_layout(p);
    int btop = CONTENT_Y;
    int bbot = g_console_height - 4;   /* 大框底（help 横线 H-3 上方） */
    int content_bot = g_console_height - 10;

    /* 静态部分（大框/标题/横线/信息面板）只在首次画，动态（棋盘/AI/历史/状态）每帧重绘 */
    int first = !g_page_drawn[1];
    if (first) {
        g_page_drawn[1] = 1;
        /* 大外框 */
        clear_rect(0, btop, g_console_width, bbot - btop + 1);
        draw_border(0, btop, g_console_width, bbot - btop + 1, NULL);
        /* 标题行 */
        gotoxy(2, btop + 1);
        set_color(COLOR_CYAN, COLOR_BLACK);
        printf("RL TRAINER   PLAYGROUND   MODEL SNAPSHOT");
        /* 标题下横线 */
        play_hline(btop + 2);
        /* 静态信息面板 */
        draw_play_model(p[PLY_MODEL]);
        if (g_console_width >= 180) draw_play_info(p[PLY_INFO]);
        if (g_console_width >= 150) {
            draw_play_settings(p[PLY_SETTINGS]);
            draw_play_eval(p[PLY_EVAL]);
        }
    }
    /* 动态面板（每帧重绘，落子后只刷新这里） */
    if (g_console_width < 150) {
        clear_rect(p[PLY_TEST].x, p[PLY_TEST].y, p[PLY_TEST].w, p[PLY_TEST].h);
    }
    if (g_console_width < 150 && g_play_sidebar_tab == 4) {
        draw_play_test(p[PLY_TEST]);
    } else {
        draw_play_board_panel(p[PLY_BOARD]);
        draw_play_ai(p[PLY_AI]);
    }
    if (g_console_width >= 150) {
        draw_play_hist(p[PLY_HIST]);
        draw_play_test(p[PLY_TEST]);
    }
    else draw_play_sidebar(p[PLY_MODEL]);   /* 小屏左侧边栏 */
    /* 内容区下横线 → 状态行 → 横线 → 操作行（状态每帧，横线/操作行仅首次） */
    if (first) play_hline(content_bot + 1);
    clear_rect(2, content_bot + 2, g_console_width - 4, 1);  /* 清残影 */
    gotoxy(2, content_bot + 2);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    {
        char ep_str[16];
        snprintf(ep_str, sizeof(ep_str), "%d", g_eval_episode);
        print_roll_clip(2, content_bot + 2, g_console_width - 4,
            "MODEL: V7.3.2   EP:%s   CP:%s   MOVES: %d   %s   INFERENCE: %.2fms",
            g_eval_episode > 0 ? ep_str : "--",
            play_ckpt_disp(), g_play_moves,
            g_play_over < 0 ? (g_play_moves % 2 == 0 ? "YOUR TURN" : "AI TURN") :
            (g_play_over == 1 ? "AI WINS" : g_play_over == 2 ? "YOU WIN" : "DRAW"),
            g_play_infer_ms);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    if (first) play_hline(content_bot + 3);
    /* 操作栏按钮（动态：平时灰，按下青色高亮，画完复位）分两层显示 */
    gotoxy(2, content_bot + 4);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    printf("[↑↓←→] Move Cursor");
    gotoxy(2 + 18, content_bot + 4);
    control_here("[ENTER] Place", false, false, VK_RETURN);
    gotoxy(2 + 33, content_bot + 4);
    control_here("[U] Undo", false, false, 'U');
    gotoxy(2 + 44, content_bot + 4);
    ui_button(2 + 44, content_bot + 4, "[A] Analysis", {g_play_analysis_on != 0, false}, ui_lamp_button());
    gotoxy(2, content_bot + 5);
    control_here("[R] Restart", false, false, 'R');
    gotoxy(2 + 13, content_bot + 5);
    control_here("[S] Snapshot", false, false, 'S');
    gotoxy(2 + 27, content_bot + 5);
    control_here("[ESC] Back", false, false, VK_ESCAPE);
    gotoxy(2 + 40, content_bot + 5);
    control_here("[AI Step]", false, false, VK_SPACE);
    if (g_console_width < 150) gotoxy(2, content_bot + 6);
    set_color(COLOR_WHITE, COLOR_BLACK);
    g_play_btn = -1;   /* 高亮只保持一帧 */
    /* 状态行跑马灯刷新：每帧重画右上角 TRAINING 状态（跑马灯动画需持续更新） */
    {
        int btop2 = CONTENT_Y;
        gotoxy(g_console_width - 22, btop2 + 1);
        if (process_running(&g_trainer)) {
            vt_set_truecolor_fg(0x39, 0xC5, 0xBB);
            printf("●");
            set_color(COLOR_WHITE, COLOR_BLACK);
            printf(" TRAINING ");
            const char* label = "RUNNING";
            int llen = 7;
            const DWORD HLM = 2000, HLR = 800, HLP = HLM + HLR;
            DWORD tick = GetTickCount() % HLP;
            int show = (tick < HLM);
            int hl_pos = show ? (int)(tick * (unsigned)(llen - 2) / HLM) : 0;
            for (int ci = 0; ci < llen; ci++) {
                int off = ci - hl_pos;
                if (show && off >= 0 && off < 3) {
                    uint32_t hc = (off == 0 || off == 2) ? 0x6BE2DD : 0x9EFFFF;
                    vt_set_truecolor_fg((hc >> 16) & 255, (hc >> 8) & 255, hc & 255);
                } else {
                    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                }
                printf("%c", label[ci]);
            }
        } else {
            set_color(COLOR_DARK_GRAY, COLOR_BLACK);
            printf("● TRAINING STOPPED");
        }
        set_color(COLOR_WHITE, COLOR_BLACK);
    }
    /* 大框底边由 draw_border 画出 */
}

void start_play(void) {
    g_palette_open = 0;
    g_play_init = 0;   /* 下次渲染 render_playground 时重新加载快照 + 初始化 */
    g_active_tab = 1;
    g_page_drawn[1] = 0;
    g_last_tab = -1;
    need_tab = 1; need_content = 1; need_help = 1;
    full_refresh();   /* 清屏 + 全量恢复，render_content 切到 PLAYGROUND 并初始化 */
}

void render_content(void) {
    vt_curve_clear();   /* 每次内容渲染前清曲线 overlay，重新注册（避免跨页残留） */
    vt_crosshair_clear();
    vt_mask_clear();
    /* 切页：清内容区 + 该页重画边框 */
    int switched = (g_active_tab != g_last_tab);
    int real_tab_switch = switched && g_last_tab >= 0;  /* 排除 full_refresh 导致的 g_last_tab=-1 */
    int probe_3d_enter = (real_tab_switch && g_active_tab == 2 && g_nv_3d_ready == 1 && g_probe_view == 4);
    if (probe_3d_enter) {
        probe_3d_set_visible(0);
        g_delay_3d_show = 1;
    }
    if (switched) {
        if (g_last_tab == 2) {
            probe_3d_set_visible(0);
            // Preserve the outgoing particle state when leaving Probe.
            vt_mark_all_dirty();
            switched = 1;
            g_delay_3d_show = 0;
        }
        if (real_tab_switch)
            trans_start(&g_transition, CONTENT_Y, g_console_height - 3, g_console_width, g_console_height);
        /* 在 trans_start 之后设置 skip_pause（trans_start 会重置它） */
        if (probe_3d_enter)
            g_transition.skip_pause = 1;
        g_last_tab = g_active_tab;
        for (int r = CONTENT_Y; r < g_console_height - 3; r++) {
            gotoxy(0, r);
            for (int c = 0; c < g_console_width; c++) printf(" ");
        }
        draw_help_bar(); // Clear footer cells that an outgoing page may have touched.
        g_page_drawn[g_active_tab] = 0;
    }
    /* 切页渲染所有页；tab 0=Training, 1=PLAYGROUND，其余为合并页（多功能面板网格） */
    if (switched) {
        switch (g_active_tab) {
        case 0: render_training(); break;
        case 1: render_playground(); break;
        case 2: render_probe(); break;
        case 3: render_gpu_analyze(); break;
        default: render_merged(g_active_tab - 4); break;
        }
        /* 设置动画目标（仅在动画激活时） */
        if (trans_is_active(&g_transition))
            trans_set_target(&g_transition);
    } else if (g_active_tab == 0) {
        render_training();
    } else if (g_active_tab == 3) {
        render_gpu_analyze();  /* 每帧刷新遥测数据 */
    } else if (g_active_tab >= 4 && g_active_tab < g_total_tabs) {
        render_merged(g_active_tab - 4);
    }
    /* PLAYGROUND(1) 是静态页：切页时已画；2s 轮询不重绘，输入驱动时由键盘/鼠标处理直接重绘 */
}

/* ============================================================================
   命令面板
   ============================================================================ */
/* 尺寸变化式全量刷新（清屏 + 重置页面状态 + 全量重绘） */
void full_refresh(void) {
    trans_cancel(&g_transition);   /* 窗口缩放/设备丢失时取消动画 */
    g_delay_3d_show = 0;
    clear_screen();
    memset(g_page_drawn, 0, sizeof(g_page_drawn));
    g_last_tab = -1;
    need_tab = 1; need_content = 1; need_help = 1;
}

void toggle_palette(void) {
    g_palette_open = !g_palette_open;
    if (g_palette_open) {
        while (vt_key_poll() != -1);   /* 打开时清空积压键，避免误触发 */
        memset(g_palette_buf, 0, sizeof(g_palette_buf));
        g_palette_msg[0] = '\0';
        clear_screen();   /* 先清空背景；面板由主循环 palette 分支每帧重绘 */
    } else {
        full_refresh();   /* 退回时尺寸变化式刷新，完整恢复界面 */
    }
}

void render_palette(void) {
    int w = g_console_width - 4; if (w > 104) w = 104; if (w < 60) w = 60;   /* 加宽：命令教程句子完整显示 */
    int h = g_console_height - 4; if (h > 26) h = 26; if (h < 16) h = 16;
    int x = (g_console_width - w) / 2;
    int y = (g_console_height - h) / 2;
    if (x < 0) x = 0; if (y < 0) y = 0;
    draw_border(x, y, w, h, " Command ");
    /* 输入行 */
    gotoxy(x + 2, y + 1);
    set_color(COLOR_CYAN, COLOR_BLACK);
    printf("> %s", g_palette_buf);
    for (int i = 2 + (int)strlen(g_palette_buf); i < w - 2; i++) printf(" ");   /* 清残留，退格/清空生效 */
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 分隔横线 */
    gotoxy(x + 1, y + 2);
    set_color(COLOR_DARK_GRAY, COLOR_BLACK);
    for (int i = 1; i < w - 1; i++) printf("─");
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 命令教程区（可滚轮滚动） */
    int doc_count = (int)(sizeof(g_palette_docs) / sizeof(g_palette_docs[0]));
    int t_h = h - 5; if (t_h < 3) t_h = 3;
    int max_scroll = doc_count - t_h; if (max_scroll < 0) max_scroll = 0;
    if (g_palette_doc_scroll > max_scroll) g_palette_doc_scroll = max_scroll;
    if (g_palette_doc_scroll < 0) g_palette_doc_scroll = 0;
    clear_rect(x + 1, y + 3, w - 2, t_h);   /* 清教程区，防滚动后残留 */
    for (int i = 0; i < t_h; i++) {
        int di = g_palette_doc_scroll + i;
        if (di >= doc_count) break;
        gotoxy(x + 2, y + 3 + i);
        set_color(COLOR_CYAN, COLOR_BLACK);
        char command_label[128]; snprintf(command_label, sizeof(command_label), "%-30s", g_palette_docs[di].cmd);
        control_here(command_label, false, false);
        set_color(COLOR_LIGHT_GRAY, COLOR_BLACK);
        print_clip(x + 2 + 31, y + 3 + i, w - 34, "%s", g_palette_docs[di].desc);
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
    /* 底部消息 / 提示行（先清整行，防中文行尾残留） */
    clear_rect(x + 2, y + h - 2, w - 4, 1);
    gotoxy(x + 2, y + h - 2);
    if (g_palette_msg[0]) {
        set_color(COLOR_YELLOW, COLOR_BLACK);
        printf("%s", g_palette_msg);
    } else {
        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
        if (doc_count > t_h)
            ui_number_printf("命令 %d-%d / %d  [滚轮]滚动  ", g_palette_doc_scroll + 1,
                g_palette_doc_scroll + t_h > doc_count ? doc_count : g_palette_doc_scroll + t_h, doc_count);
        printf("[Enter] 执行  [Esc] 关闭");
    }
    set_color(COLOR_WHITE, COLOR_BLACK);
}

/* 执行分析命令（logreader 移植）：生成报告文本 → 打开查看器 */
/* analyze/report 原生输出：临时禁用 TUI 渲染（VT），在控制台直接输出报告（可滚动/复制），按任意键返回 */
static void copy_report_to_clipboard(void) {
    const char* text = tui_report_get_text();
    if (!text) return;
    int len = (int)strlen(text);
    int wlen = MultiByteToWideChar(CP_UTF8, 0, text, len, NULL, 0);
    if (wlen <= 0) return;
    wchar_t* wbuf = (wchar_t*)malloc((size_t)(wlen + 1) * sizeof(wchar_t));
    if (!wbuf) return;
    MultiByteToWideChar(CP_UTF8, 0, text, len, wbuf, wlen);
    wbuf[wlen] = L'\0';
    if (OpenClipboard(g_hwnd)) {
        EmptyClipboard();
        HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, (size_t)(wlen + 1) * sizeof(wchar_t));
        if (h) {
            wchar_t* dst = (wchar_t*)GlobalLock(h);
            if (dst) { memcpy(dst, wbuf, (size_t)(wlen + 1) * sizeof(wchar_t)); GlobalUnlock(h); }
            SetClipboardData(CF_UNICODETEXT, h);
        }
        CloseClipboard();
    }
    free(wbuf);
}

static unsigned __stdcall report_worker(void*) {
    tui_report_exec(g_report_command);
    InterlockedExchange(&g_report_ready, 1);
    return 0;
}

void run_report_native(const char* cmd) {
    if (InterlockedCompareExchange(&g_report_running, 1, 0) != 0) {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "分析仍在运行，请稍候");
        return;
    }
    snprintf(g_report_command, sizeof(g_report_command), "%s", cmd);
    tui_report_prepare();
    InterlockedExchange(&g_report_ready, 0);
    g_palette_open = 0;
    viewer_load("正在读取日志并生成报告……\n界面仍可响应，完成后会自动显示结果。\n");
    g_viewer_title = "分析中";
    g_viewer_open = 1;
    uintptr_t th = _beginthreadex(NULL, 0, report_worker, NULL, 0, NULL);
    if (!th) {
        InterlockedExchange(&g_report_running, 0);
        viewer_load("无法创建分析线程，请重试。\n");
        return;
    }
    g_report_thread = (HANDLE)th;
    need_tab = 1; need_content = 1; need_help = 1;
}

void run_palette_command(const char* cmd) {
    /* 归一化到小写副本（大小写不敏感），并去掉首尾空格 */
    char c[256]; int cl = 0;
    for (const char* p = cmd; *p && cl < 255; p++) c[cl++] = (char)tolower((unsigned char)*p);
    c[cl] = '\0';
    char* s = c; while (*s == ' ') s++;
    char* e = s + strlen(s); while (e > s && e[-1] == ' ') e--;
    *e = '\0';

    if (s[0] == '\0') {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "输入命令后按 Enter");
        return;   /* 面板保持打开 */
    }
    if(!strcmp(s,"http status") || !strcmp(s,"http on") || !strcmp(s,"http off")){
        if(!strcmp(s,"http on"))http_start();if(!strcmp(s,"http off"))g_http_server.stop();
        snprintf(g_palette_msg,sizeof(g_palette_msg),"%s",g_http_server.status().c_str());need_content=1;return;
    }
    if(!strcmp(s,"course status")){
        std::string report;for(const auto& line:course_saved_lines())report+=line+"\n";
        viewer_load(report.c_str());g_palette_open=0;return;
    }

    /* 彩蛋：整词匹配，避免含 "ba" 的命令误触发 */
    if (strcmp(s, "ba") == 0 || strcmp(s, "badapple") == 0) {
        g_palette_open = 0;
        play_bav();
        full_refresh();   /* 动画结束清屏 + 全量恢复界面 */
        return;
    }
    g_palette_msg[0] = '\0';

    if (!strncmp(s, "gpu profile", 11)) {
        if (!strcmp(s, "gpu profile off")) {
            g_gpu_profile_arm_batches = 0;
            snprintf(g_palette_msg, sizeof(g_palette_msg), "已取消下一次 Analyzer 采集；当前训练不变");
            need_content = 1;
            return;
        }
        if (!strcmp(s, "gpu profile status")) {
            if (g_gpu_profile_session_batches > 0)
                snprintf(g_palette_msg, sizeof(g_palette_msg), "本会话%s采集：%d批；目录见GPU Analyze",
                    gpu_profile_mode_name(g_gpu_profile_session_mode), g_gpu_profile_session_batches);
            else if (g_gpu_profile_arm_batches > 0)
                snprintf(g_palette_msg, sizeof(g_palette_msg), "已武装下次训练：前%d批 %s",
                    g_gpu_profile_arm_batches, gpu_profile_mode_name(g_gpu_profile_arm_mode));
            else if (g_gpu_profile_last_dir[0])
                snprintf(g_palette_msg, sizeof(g_palette_msg), "上次采集已归档：%d个批次文件；目录见GPU Analyze", g_gpu_profile_last_files);
            else
                snprintf(g_palette_msg, sizeof(g_palette_msg), "GPU采集未启用");
            return;
        }
        int batches = 0; char trailing = 0;
        const bool module_mode = sscanf(s, "gpu profile modules %d %c", &batches, &trailing) == 1;
        if (!module_mode) batches = 0;
        const bool event_mode = !module_mode && sscanf(s, "gpu profile events %d %c", &batches, &trailing) == 1;
        if ((module_mode || event_mode) && batches >= 1 && batches <= 8) {
            g_gpu_profile_arm_batches = batches;
            g_gpu_profile_arm_mode = event_mode ? 2 : 1;
            snprintf(g_palette_msg, sizeof(g_palette_msg), "下次训练采集前%d批 %s；当前训练不变",
                batches, gpu_profile_mode_name(g_gpu_profile_arm_mode));
            need_content = 1;
            return;
        }
        snprintf(g_palette_msg, sizeof(g_palette_msg), "用法: gpu profile modules/events 1-8 | status | off");
        return;
    }
    if (!strcmp(s, "gpu report copy")) {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "%s",
            copy_gpu_analyze_report_to_clipboard() ? "GPU Analyze 报告已复制" : "复制失败：无法访问系统剪贴板");
        need_content = 1;
        return;
    }

    /* 分析命令：原生控制台输出（退出 TUI 渲染，方便滚动/复制） */
    if (strncmp(s, "report", 6) == 0) { run_report_native("report"); return; }
    if (strncmp(s, "analyze", 7) == 0) { run_report_native(s); return; }   /* 需转发参数：analyze length budget=N 等 */
    if (strncmp(s, "trainstat", 9) == 0) { run_report_native(s); return; }
    if (strncmp(s, "eval", 4) == 0) { run_report_native(s); return; }
    /* 训练轨迹分析（重构版）：分析全部历史，输出紧凑文本 + JSON + 明细文件 */
    if (strncmp(s, "trajectory", 10) == 0) { run_report_native(s); return; }
    if (strncmp(s, "traj", 4) == 0) { run_report_native(s); return; }
    /* 回放命令走回放模态 */
    if (strncmp(s, "g ", 2) == 0 || strncmp(s, "m ", 2) == 0) { start_replay(s); return; }
    /* PLAYGROUND 人机对战 */
    if (strcmp(s, "play") == 0 || strcmp(s, "p") == 0) { start_play(); return; }
    /* 折线图主题：theme block|round|right */
    if (strncmp(s, "theme", 5) == 0) {
        const char* arg = s + 5;
        while (*arg == ' ') arg++;
        if (strncmp(arg, "block", 5) == 0) g_line_theme = 0;
        else if (strncmp(arg, "round", 5) == 0) g_line_theme = 1;
        else if (strncmp(arg, "right", 5) == 0) g_line_theme = 2;
        else if (strncmp(arg, "curve", 5) == 0) g_line_theme = 3;
        else {
            snprintf(g_palette_msg, sizeof(g_palette_msg), "theme block|round|right|curve (当前: %d)", g_line_theme);
            return;
        }
        g_palette_open = 0;
        full_refresh();
        return;
    }
    /* 训练/watchdog 自有窗口显示/隐藏切换（立即生效：隐藏或重新唤起，不影响主程序） */
    if (!strcmp(s,"gpu start") || !strcmp(s,"gpu stop") ||
        !strcmp(s,"gpu show") || !strcmp(s,"gpu hide")) {
        bool ok = analyzer_control(s + 4);
        snprintf(g_palette_msg, sizeof(g_palette_msg), "%s", ok ? "GPU Analyzer 指令已发送" : "GPU Analyzer 不可用/启动失败");
        return;
    }
    if (strcmp(s, "console") == 0) {
        if (toggle_console_now())
            snprintf(g_palette_msg, sizeof(g_palette_msg), "训练窗口: %s", g_trainer_console ? "显示" : "隐藏");
        else
            snprintf(g_palette_msg, sizeof(g_palette_msg), "窗口开关: %s (训练未运行，下次启动生效)", g_trainer_console ? "显示" : "隐藏");
        return;
    }
    if (strcmp(s, "resume training") == 0 || strcmp(s, "resume") == 0) {
        if (pipe_send("RESUME_TRAINING"))
            snprintf(g_palette_msg, sizeof(g_palette_msg), "已请求解除哨兵安全暂停");
        else
            snprintf(g_palette_msg, sizeof(g_palette_msg), "Trainer 未连接，无法恢复");
        return;
    }
    if (strcmp(s, "stop watchdog") == 0) {
        snprintf(g_palette_msg, sizeof(g_palette_msg), "%s",
            stop_watchdog_gracefully() ? "Watchdog 已优雅关闭，Trainer 保持运行" :
            "Watchdog 关闭超时，未强制终止");
        return;
    }

    // GPT6-Astra: check cancellation before the broader "close at" command.
    if (strcmp(s, "close off") == 0 || strcmp(s, "close at off") == 0) {
        g_close_at = -1; g_close_time = 0;
        snprintf(g_palette_msg, sizeof(g_palette_msg), "已取消自动关闭");
        return;
    }
    if (strncmp(s, "close at ", 9) == 0 || strcmp(s, "close at") == 0) {
        const char* arg = s + 8;
        while (*arg == ' ') ++arg;
        if (arg[0] == 't' && arg[1] == ' ') {
            arg += 2; while (*arg == ' ') ++arg;
            time_t deadline;
            if (!close_time(arg, time(nullptr), deadline)) {
                snprintf(g_palette_msg, sizeof(g_palette_msg), "用法: close at t 0730 (0000~2359)");
                return;
            }
            g_close_time = deadline; g_close_at = -1;
            char stamp[32];
            strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M", localtime(&deadline));
            snprintf(g_palette_msg, sizeof(g_palette_msg), "%s 保存退出；保持TUI开启", stamp);
        } else {
            if (arg[0] == 'e' && arg[1] == ' ') { arg += 2; while (*arg == ' ') ++arg; }
            int n;
            if (!close_episode(arg, n)) {
                snprintf(g_palette_msg, sizeof(g_palette_msg), "用法: close at e <局数> / close at t 0730 / close off");
                return;
            }
            g_close_at = n; g_close_time = 0;
            snprintf(g_palette_msg, sizeof(g_palette_msg), "将在第 %d 局关闭训练+看门狗", n);
        }
        return;
    }
    /* 会话控制（状态机 guard 时面板保持打开并给出反馈） */
    if (strcmp(s, "start session") == 0 || strcmp(s, "start") == 0) {
        if (g_state != MASTER_IDLE && g_state != MASTER_SHUTDOWN) {
            snprintf(g_palette_msg, sizeof(g_palette_msg), "训练已在运行 (ep %s)", status_lines[0]);
            return;
        }
        start_session();
        if (g_state == MASTER_IDLE) { snprintf(g_palette_msg, sizeof(g_palette_msg), "启动失败"); return; }
        g_palette_open = 0; full_refresh(); return;
    }
    if (strcmp(s, "shutdown session") == 0 || strcmp(s, "shutdown") == 0) {
        if (g_state == MASTER_IDLE || g_state == MASTER_SHUTDOWN) {
            snprintf(g_palette_msg, sizeof(g_palette_msg), "训练未在运行");
            return;
        }
        shutdown_session();
        g_palette_open = 0; full_refresh(); return;
    }
    /* 切页（兼容 "tab N" 和 "switch tab N"） */
    if (strncmp(s, "switch tab", 10) == 0) { s += 10; }
    else if (strncmp(s, "tab", 3) == 0) { s += 3; }
    else { snprintf(g_palette_msg, sizeof(g_palette_msg), "未知命令: %s", cmd); return; }
    {
        int t = atoi(s);
        if (t >= 1 && t <= g_total_tabs) {
            g_active_tab = t - 1;
            g_palette_open = 0; full_refresh(); return;
        }
        snprintf(g_palette_msg, sizeof(g_palette_msg), "tab 1-%d", g_total_tabs);
        return;
    }

    snprintf(g_palette_msg, sizeof(g_palette_msg), "未知命令: %s", cmd);
}

/* ============================================================================
   输入（鼠标 / 热键，复用 Controller 机制）
   ============================================================================ */
int window_focused(void) {
    /* 自有窗口：直接比较前台窗口是否本窗口 */
    return (g_hwnd && GetForegroundWindow() == g_hwnd) ? 1 : 0;
}
/* 按窗口标题子串找顶层窗口（训练/watchdog 自有终端窗口） */
struct TFind { const wchar_t* s; HWND h; };
static BOOL CALLBACK find_title_enum(HWND h, LPARAM lp) {
    struct TFind* p = (struct TFind*)lp;
    wchar_t title[128];
    if (GetWindowTextW(h, title, 128) && wcsstr(title, p->s)) { p->h = h; return FALSE; }
    return TRUE;
}
static HWND find_window_by_title(const wchar_t* sub) {
    struct TFind tf = { sub, NULL };
    EnumWindows(find_title_enum, (LPARAM)&tf);
    return tf.h;
}
/* TUI 启动时：识别训练/watchdog 窗口可见性，同步指示灯状态（任一可见 → 灯亮） */
static void sync_console_indicator(void) {
    HWND ht = find_window_by_title(L"DQN Trainer");
    HWND hw = find_window_by_title(L"DQN Watchdog");
    g_trainer_console = ((ht && IsWindowVisible(ht)) || (hw && IsWindowVisible(hw))) ? 1 : 0;
}
/* GetAsyncKeyState 按下边沿检测：绕过 IME（不读控制台输入缓冲，不触发合成）；每键独立去抖 */
static int key_pressed(int vk) {
    static int last[256] = { 0 };
    int down = (GetAsyncKeyState(vk) & 0x8000) ? 1 : 0;
    int r = down && !last[vk];
    last[vk] = down;
    return r;
}
/* Training 图格 i 的滑块几何（与 draw_line_series 一致：axis_w=9，滑块行 cy+ch-2）。
   轨道右侧留 6 字符给复位按钮；box_y/box_h 可带回格框范围（供滚轮命中判定） */
static int training_slider_geom(int i, int* sl_x0, int* sl_w, int* sl_y, int* box_y, int* box_h) {
    int grid_top = CONTENT_Y + 1;
    int avail_w = g_console_width - 4;
    int pbar_h = 3, mon_h = 3;
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
    int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
    int cols, rows, cw, ch;
    compute_chart_layout(avail_w, avail_h, &cols, &rows, &cw, &ch);
    int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
    if (i < 0 || i >= N) return 0;
    int rem_w = avail_w - cols * cw; if (rem_w < 0) rem_w = 0;
    int c = i % cols;
    int cx = 2 + c * cw + (c < rem_w ? c : rem_w);
    int cy = grid_top + (i / cols) * ch;
    int w_i = cw + (c < rem_w ? 1 : 0);
    *sl_x0 = cx + 1 + 9;
    *sl_w = w_i - 9 - 2; if (*sl_w < 4) return 0;   /* 滑块宽度 <4 时直接不显示 */
    *sl_y = cy + ch - 2;
    if (box_y) *box_y = cy;
    if (box_h) *box_h = ch;
    return 1;
}

static void reset_all_chart_views(void) {
    for (int i = 0; i < MAX_CELLS; i++) {
        int mi = g_cell_metric[i];
        int count = (mi >= 0 && mi < METRIC_COUNT) ? g_hist[mi].n : 0;
        int recent = count < 100 ? count : 100;
        if (recent < 1) recent = 1;
        g_cell_view_start[i] = recent;
        g_cell_view_end[i] = 0;
        g_cell_global_axis[i] = 1;
        g_cell_dirty[i] = 1;
    }
    g_cell_reset_press = -2;   /* -2 = 全局按钮短暂高亮 */
    g_reset_press_time = GetTickCount();
    need_content = 1;
    vt_mark_all_dirty();
}

/* 滑块点击：命中滑块则把窗口中心移到点击处（保持窗口大小），标记 dirty 只重绘该格。
   轨道右侧 [R]复位 恢复全量。返回 1 已处理。窗口大小由鼠标滚轮控制。 */
/* Training 进度条框"控制台开关"按钮几何（渲染与点击共用：progress 右侧隔 1 列）。
   py_pbar 与 render_training 一致（含矮窗口 clamp），保证点击/渲染坐标相同。 */
static void progress_btn_geom(int* x0, int* x1, int* y) {
    int pbar_h = 3;
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    int grid_top = CONTENT_Y + 1;
    if (py_pbar < grid_top + 3 + 3) py_pbar = grid_top + 3 + 3;   /* 与 render_training 的 mon_h=3 clamp 一致 */
    *y = py_pbar + 1;
    int w = g_console_width - 4;
    *x0 = 2 + w - 14;
    *x1 = *x0 + 8;   /* "● CONSOLE" 9 列 */
}
/* BUG-060: 控制台开关按钮点击：切换训练/watchdog 控制台窗口，按下帧高亮反馈；无窗口则提示重启训练 */
static int progress_btn_click(int mx, int my) {
    if (g_active_tab != 0) return 0;
    if (!status_lines[0][0]) return 0;   /* 无会话：按钮未显示 */
    int x0, x1, y;
    progress_btn_geom(&x0, &x1, &y);
    if (my == y && mx >= x0 && mx <= x1) {
        if (!toggle_console_now()) g_progress_btn_note_time = GetTickCount();   /* 无窗口/未运行 → 提示 */
        g_progress_btn_press = 1;
        g_progress_btn_time = GetTickCount();
        need_content = 1;
        return 1;
    }
    return 0;
}

static int slider_click(int mx, int my) {
    if (g_active_tab != 0) return 0;
    if (g_console_width >= 110 && my == CONTENT_Y &&
        mx >= g_console_width - 18 && mx < g_console_width - 2) {
        reset_all_chart_views();
        return 1;
    }
    for (int i = 0; i < MAX_CELLS; i++) {
        int sl_x0, sl_w, sl_y;
        if (!training_slider_geom(i, &sl_x0, &sl_w, &sl_y, NULL, NULL)) continue;
        int mi = g_cell_metric[i];
        if (mi < 0 || mi >= METRIC_COUNT) continue;
        int count = g_hist[mi].n;
        if (count <= 0) continue;
        /* 复位按钮在 y 轴数字标号下方（x 轴行左端 = sl_y-1 行，sl_x0-9..sl_x0-3）：
           默认窗口（最近 2/3 数据），右端到头且实时滚动新数据 */
        if (my == sl_y - 1 && mx >= sl_x0 - 9 && mx <= sl_x0 - 3) {
            g_cell_focus = i;
            int def_win = count < 100 ? count : 100;   /* 固定最近 100 点，不随数据增长 */
            if (def_win < 1) def_win = 1;
            g_cell_view_start[i] = def_win;   /* 窗口右端到最新（ve=0），大小固定 */
            g_cell_view_end[i] = 0;
            g_cell_global_axis[i] = 0;
            g_cell_reset_press = i;             /* 点击反馈：短暂黄色高亮 */
            g_reset_press_time = GetTickCount();
            g_cell_dirty[i] = 1;
            return 1;
        }
        if (my != sl_y || mx < sl_x0 || mx >= sl_x0 + sl_w) continue;
        g_cell_focus = i;
        int idx = (int)((long long)(mx - sl_x0) * count / sl_w);
        if (idx < 0) idx = 0; if (idx > count - 1) idx = count - 1;
        int off;
        int thr = (sl_w / 4) / 2; if (thr < 1) thr = 1;   /* 两端阈值区 = 滑块长度一半 */
        if (mx < sl_x0 + thr) {
            off = count - 1;                             /* 左端区：滑块移到最左（最老数据）并固定 */
        } else if (mx >= sl_x0 + sl_w - thr) {
            off = 0;                                     /* 右端区：移到最右（最新数据）并固定 */
        } else {
            off = count - 1 - idx;                       /* 中间：精确到点击位置 */
        }
        int vs = g_cell_view_start[i], ve = g_cell_view_end[i];
        if (vs < 0) { vs = count - 1; ve = 0; } /* 从全量开始 */
        int win = vs - ve; if (win < 1) win = 1;
        int half = win / 2;                     /* 窗口中心对准点击处 */
        vs = off + half; ve = off - (win - half);
        /* 不 clamp：允许期望中心到数据边界外，绘制时按数据范围钳制显示，滑块位置精确到两端 */
        g_cell_view_start[i] = vs;
        g_cell_view_end[i] = ve;
        g_cell_global_axis[i] = 0;
        g_cell_dirty[i] = 1;
        return 1;
    }
    return 0;
}

// GPT6-Astra: one wheel notch scrolls one control row (two screen lines).
static int config_wheel(int mx, int my, int delta) {
    if (!current_tab_has_fun(FUN_CONFIG) || delta == 0 ||
        mx <= g_config_x || mx >= g_config_x + g_config_w - 1 ||
        my <= g_config_y || my >= g_config_y + g_config_h - 1) return 0;
    int steps = abs(delta) / WHEEL_DELTA;
    if (steps < 1) steps = 1;
    g_config_scroll += delta > 0 ? -steps : steps;
    const int limit = g_config_rows > g_config_visible ? g_config_rows - g_config_visible : 0;
    if (g_config_scroll < 0) g_config_scroll = 0;
    if (g_config_scroll > limit) g_config_scroll = limit;
    // Invalidate old hit boxes until the scrolled controls have been redrawn.
    for (int i = 0; i < (int)CONFIG_SLIDER_COUNT; ++i) g_config_sliders[i].y = -1000;
    return 1;
}

/* 鼠标滚轮：以鼠标位置对应数据点为锚缩放窗口大小（滚轮向上=放大查看/窗口缩小，向下=缩小比例/窗口扩大）。
   不依赖滑块几何（小窗口下滑块可能不显示，但滚轮缩放仍应生效）。 */
static int slider_wheel(int mx, int my, int delta) {
    if (g_active_tab != 0 || delta == 0) return 0;
    int grid_top = CONTENT_Y + 1;
    int avail_w = g_console_width - 4;
    int pbar_h = 3, mon_h = 3;
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
    int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
    int cols, rows, cw, ch;
    compute_chart_layout(avail_w, avail_h, &cols, &rows, &cw, &ch);
    int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
    int rem_w = avail_w - cols * cw; if (rem_w < 0) rem_w = 0;
    for (int i = 0; i < N; i++) {
        int r = i / cols, c = i % cols;
        int cx = 2 + c * cw + (c < rem_w ? c : rem_w);
        int cy = grid_top + r * ch;
        int w_i = cw + (c < rem_w ? 1 : 0);
        /* 鼠标在该图表框内即生效（不依赖滑块是否显示） */
        if (mx < cx || mx >= cx + w_i || my < cy || my >= cy + ch) continue;
        int mi = g_cell_metric[i];
        if (mi < 0 || mi >= METRIC_COUNT) continue;
        int count = g_hist[mi].n;
        if (count <= 0) continue;
        g_cell_focus = i;
        int vs = g_cell_view_start[i], ve = g_cell_view_end[i];
        if (vs < 0) { vs = count - 1; ve = 0; }
        int win = vs - ve; if (win < 1) win = 1;
        int ctr = (vs + ve) / 2;
        int nwin = (delta > 0) ? (int)(win * 0.8f) : (int)(win * 1.25f + 0.5f);
        if (delta > 0 && nwin < 1) nwin = 1;
        if (delta < 0 && nwin <= win) nwin = win + 1;
        if (nwin > count) nwin = count;
        vs = ctr + nwin / 2;
        ve = vs - nwin;
        g_cell_view_start[i] = vs;
        g_cell_view_end[i] = ve;
        g_cell_global_axis[i] = 0;
        g_cell_dirty[i] = 1;
        return 1;
    }
    return 0;
}

/* 指标下拉菜单滚轮：仅鼠标位于菜单内时消费事件，防止穿透到图表缩放。 */
static int metric_menu_wheel(int mx_mouse, int my_mouse, int delta) {
    if (g_active_tab != 0 || g_cell_dropdown < 0 || delta == 0) return 0;
    int grid_top = CONTENT_Y + 1;
    int avail_w = g_console_width - 4;
    int pbar_h = 3, mon_h = 3;
    int py_pbar = g_console_height - 4 - pbar_h + 1;
    if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
    int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
    int cols, rows, cw, ch;
    compute_chart_layout(avail_w, avail_h, &cols, &rows, &cw, &ch);
    int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
    if (g_cell_dropdown >= N) return 0;
    int rem_w = avail_w - cols * cw; if (rem_w < 0) rem_w = 0;
    int dr = g_cell_dropdown / cols, dc = g_cell_dropdown % cols;
    int dcx = 2 + dc * cw + (dc < rem_w ? dc : rem_w);
    int dcy = grid_top + dr * ch;
    int visible = (METRIC_COUNT < METRIC_MENU_VISIBLE) ? METRIC_COUNT : METRIC_MENU_VISIBLE;
    int mw = 22, mh = visible + 2;
    int panel_x = dcx + 2, panel_y = dcy;
    if (panel_x + mw > 2 + avail_w) panel_x = 2 + avail_w - mw;
    if (panel_y < grid_top) panel_y = grid_top;
    if (panel_y + mh > g_console_height - 3) panel_y = g_console_height - 3 - mh;
    if (mx_mouse < panel_x || mx_mouse >= panel_x + mw ||
        my_mouse < panel_y || my_mouse >= panel_y + mh) return 0;
    int steps = abs(delta) / WHEEL_DELTA; if (steps < 1) steps = 1;
    g_cell_drop_cur += (delta > 0) ? -steps : steps;
    if (g_cell_drop_cur < 0) g_cell_drop_cur = 0;
    if (g_cell_drop_cur >= METRIC_COUNT) g_cell_drop_cur = METRIC_COUNT - 1;
    return 1;
}

void handle_mouse_click(int x, int y) {
    const char* profile_button = g_gpu_profile_arm_batches > 0 ? "[Cancel]" : "[Modules 8]";
    const int profile_button_width = ui_button_width(profile_button, ui_push_button());
    if (g_active_tab == 3 && y == CONTENT_Y + 2 && x >= 4 && x < 4 + profile_button_width) {
        if (g_gpu_profile_arm_batches > 0) {
            g_gpu_profile_arm_batches = 0;
            snprintf(g_palette_msg, sizeof(g_palette_msg), "已取消下一次 GPU 采集；当前训练不变");
        } else {
            g_gpu_profile_arm_batches = 8;
            g_gpu_profile_arm_mode = 1;
            snprintf(g_palette_msg, sizeof(g_palette_msg), "已预约下一训练会话前 8 批低开销模块采集");
        }
        need_content = 1;
        return;
    }
    if (g_active_tab == 3 && y == CONTENT_Y + 1) {
        if (x >= 60 && x < 75) {
            if (!copy_gpu_analyze_report_to_clipboard())
                MessageBoxW(g_hwnd, L"无法写入剪贴板，请重试。", L"GPU Analyze", MB_OK | MB_ICONWARNING);
            need_content = 1; return;
        }
        const char* action = nullptr;
        if (x >= 20 && x < 27) action = "start";
        else if (x >= 30 && x < 36) action = "stop";
        else if (x >= 40 && x < 46) action = "show";
        else if (x >= 50 && x < 56) action = "hide";
        if (action) {
            if (!analyzer_control(action)) MessageBoxW(g_hwnd,
                L"GPU Analyzer 操作失败，请确认程序已构建或正在运行。", L"GPU Analyzer", MB_OK | MB_ICONWARNING);
            need_content = 1; return;
        }
    }

    if (current_tab_has_fun(FUN_DASHBOARD) && g_worker_hit_cols == g_console_width &&
        g_worker_hit_rows == g_console_height) {
        for (int id = 0; id < DASH_WORKER_COUNT; ++id) {
            if (g_worker_hits[id].contains(x, y)) { dashboard_worker_click(id); return; }
        }
    }
    /* 导航栏点击（两排动态，用 draw_tab_bar 记录的位置） */
    int row1 = (g_total_tabs > 5) ? 5 : g_total_tabs;
    if (y == TAB_BAR_Y) {
        int top_x = g_console_width - 14;
        if (x >= top_x && x < top_x + 10) { toggle_ontop(); need_tab = 1; return; }
        for (int i = 0; i < row1; i++) {
            if (x >= g_tab_x[i] && x < g_tab_x[i] + g_tab_w[i]) { g_active_tab = i; return; }
        }
        return;
    }
    if (y == TAB_BAR_Y2) {
        for (int i = row1; i < g_total_tabs; i++) {
            if (x >= g_tab_x[i] && x < g_tab_x[i] + g_tab_w[i]) { g_active_tab = i; return; }
        }
        return;
    }
    /* Training 页：点框标题开下拉、点下拉选项选择、点框设焦点 */
    if (g_active_tab == 0) {
        int grid_top = CONTENT_Y + 1;
        int avail_w = g_console_width - 4;
        /* 布局参数与 render_training 完全一致（贴底进度条 + 3行监控值区） */
        int pbar_h = 3, mon_h = 3;
        int py_pbar = g_console_height - 4 - pbar_h + 1;
        if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
        int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
        int cols, rows, cw, ch;
        compute_chart_layout(avail_w, avail_h, &cols, &rows, &cw, &ch);
        int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
        int rem_w = avail_w - cols * cw; if (rem_w < 0) rem_w = 0;
        /* 下拉面板点击 */
        if (g_cell_dropdown >= 0 && g_cell_dropdown < N) {
            int dr = g_cell_dropdown / cols, dc = g_cell_dropdown % cols;
            int dcx = 2 + dc * cw + (dc < rem_w ? dc : rem_w);
            int dcy = grid_top + dr * ch;
            int visible = (METRIC_COUNT < METRIC_MENU_VISIBLE) ? METRIC_COUNT : METRIC_MENU_VISIBLE;
            int menu_start = g_cell_drop_cur - visible / 2;
            if (menu_start < 0) menu_start = 0;
            if (menu_start > METRIC_COUNT - visible) menu_start = METRIC_COUNT - visible;
            int mw = 22, mh = visible + 2;
            int mx = dcx + 2, my = dcy;
            if (mx + mw > 2 + avail_w) mx = 2 + avail_w - mw;
            if (my < grid_top) my = grid_top;
            if (my + mh > g_console_height - 3) my = g_console_height - 3 - mh;
            if (x >= mx && x < mx + mw && y >= my && y < my + mh) {
                int opt = menu_start + y - (my + 1);
                if (opt >= 0 && opt < METRIC_COUNT) {
                    g_cell_checks[g_cell_dropdown][opt] = 1 - g_cell_checks[g_cell_dropdown][opt];   /* 勾选式切换（不关闭，可连续勾选多个，允许全部不勾） */
                    g_cell_metric[g_cell_dropdown] = -1;   /* 先清空，再找第一个勾选的；全不勾时 -1 直接黑框 */
                    for (int k = 0; k < METRIC_COUNT; k++) if (g_cell_checks[g_cell_dropdown][k]) { g_cell_metric[g_cell_dropdown] = k; break; }
                    need_content = 1;
                }
                return;
            }
            g_cell_dropdown = -1;   /* 点击面板外：关闭下拉 */
            g_force_redraw = 1;
            vt_mark_all_dirty();
        }
        /* 框标题 / 框内点击 */
        for (int i = 0; i < N; i++) {
            int r = i / cols, c = i % cols;
            int w_i = cw + (c < rem_w ? 1 : 0);
            int cx = 2 + c * cw + (c < rem_w ? c : rem_w);
            int cy = grid_top + r * ch;
            if (x >= cx && x < cx + w_i && y >= cy && y < cy + ch) {
                g_cell_focus = i;
                if (y == cy) {   /* 标题行 → 打开下拉 */
                    g_cell_dropdown = i;
                    g_cell_drop_cur = g_cell_metric[i];
                    if (g_cell_drop_cur < 0) g_cell_drop_cur = 0;
                }
                need_content = 1;
                return;
            }
        }
    }
    /* PLAYGROUND 页（tab 1）：棋盘落子 / 侧边栏切换，事件驱动重绘 */
    if (g_active_tab == 1) {
        handle_play_mouse(x, y);
        render_playground();
        return;
    }
    /* PROBE 页（tab 2）：连接按钮 + 自动刷新 + 子Tab + 层选择 */
    if (g_active_tab == 2) {
        if (g_probe_link_dropdown_open) {
            if (x >= g_probe_link_drop_x && x < g_probe_link_drop_x + g_probe_link_drop_w &&
                y > g_probe_link_drop_y && y <= g_probe_link_drop_y + PROBE_LINK_MODE_COUNT) {
                const int requested_mode = y - g_probe_link_drop_y - 1;
                const NeuralProbeHeader* mode_header = probe_get_header();
                if (requested_mode != PROBE_LINK_WEIGHT && mode_header && mode_header->version < 3) {
                    g_probe_link_dropdown_open = 0;
                    snprintf(g_probe_conn_msg, sizeof(g_probe_conn_msg),
                        "Diagnostics need Trainer restart (Probe v3)");
                    g_probe_conn_err = 0;
                    vt_mark_all_dirty();
                    render_probe();
                    return;
                }
                g_probe_link_mode = requested_mode;
                g_probe_link_dropdown_open = 0;
                g_nv_threshold_changed = 1;
                g_probe_3d_last_snap_id = UINT64_MAX;
                nv_set_display_mode(g_probe_link_mode == PROBE_LINK_GRADIENT ? NV_MODE_GRADIENT :
                    g_probe_link_mode == PROBE_LINK_CONTRIBUTION ? NV_MODE_CONTRIBUTION :
                    g_probe_link_mode == PROBE_LINK_ATTENTION ? NV_MODE_ATTENTION : NV_MODE_WEIGHT);
                vt_mark_all_dirty();
                render_probe();
                return;
            }
            if (!(y == g_probe_link_drop_y && x >= g_probe_link_drop_x &&
                x < g_probe_link_drop_x + g_probe_link_drop_w)) {
                g_probe_link_dropdown_open = 0;
                vt_mark_all_dirty();
                render_probe();
                return;
            }
        }
        if (g_probe_view == 4 && g_nv_3d_ready == 1 &&
            y == g_probe_link_drop_y && x >= g_probe_link_drop_x &&
            x < g_probe_link_drop_x + g_probe_link_drop_w) {
            g_probe_link_dropdown_open = !g_probe_link_dropdown_open;
            vt_mark_all_dirty();
            render_probe();
            return;
        }
        /* CONNECT 按钮 */
        if (y == CONTENT_Y + 1 && x >= 2 && x < 17) {
            if (probe_client_is_connected()) {
                probe_client_disconnect();
                g_probe_3d_last_snap_id = UINT64_MAX;
                snprintf(g_probe_conn_msg, sizeof(g_probe_conn_msg), "Disconnected");
                g_probe_conn_err = 0;
            } else {
                int err = probe_client_connect();
                g_probe_3d_last_snap_id = UINT64_MAX;
                g_probe_conn_err = err;
                if (err == 0) snprintf(g_probe_conn_msg, sizeof(g_probe_conn_msg), "Connected OK!");
                else snprintf(g_probe_conn_msg, sizeof(g_probe_conn_msg), "Failed: %d", err);
            }
            render_probe();
            return;
        }
        /* AUTO 勾选 */
        if (y == CONTENT_Y + 1 && x >= 19 && x < 27) {
            g_probe_auto_refresh = 1 - g_probe_auto_refresh;
            render_probe();
            return;
        }
        /* 子 Tab 点击 + 阈值滑块点击：CONTENT_Y + 3 行 */
        int sub_row = CONTENT_Y + 3;
        if (y == sub_row) {
            int sx = 2;
            const char* sub_tabs[] = { "Layers", "Matrix", "Q-Values", "Global", "3D View" };
            for (int i = 0; i < 5; i++) {
                int sw = (int)strlen(sub_tabs[i]) + 3;
                if (x >= sx && x < sx + sw) {
                    if (i == 4 && g_nv_3d_ready == -1) {
                        probe_3d_shutdown();
                        g_nv_3d_error = 0;
                    }
                    g_probe_view = i;
                    render_probe();
                    return;
                }
                sx += sw;
            }
            /* 连线数量滑块点击 */
            if (g_probe_view == 4 && g_nv_3d_ready == 1) {
                int slider_label_x = g_probe_link_drop_x + g_probe_link_drop_w + 3;
                int slider_x0 = slider_label_x + 6;   /* "TopN:[" 之后 */
                int slider_w = g_console_width - slider_x0 - 10; if (slider_w < 8) slider_w = 8;
                if (x >= slider_x0 && x < slider_x0 + slider_w + 3) {  /* 右扩3像素方便到达50 */
                    float ratio = (float)(x - slider_x0) / (float)(slider_w - 1);
                    g_nv_top_n = 5 + (int)(ratio * 45.0f + 0.5f);   /* 5~50，四舍五入 */
                    if (g_nv_top_n < 5) g_nv_top_n = 5;
                    if (g_nv_top_n > 50) g_nv_top_n = 50;
                    g_nv_threshold_changed = 1;
                    render_probe();
                    return;
                }
            }
        }
        /* 层列表点击（View 0）：选择层 */
        if (g_probe_view == 0 && y > CONTENT_Y + 4) {
            int layer_row = CONTENT_Y + 4;
            for (uint32_t i = 0; i < probe_get_layer_count(); i++) {
                if (y == layer_row) { g_probe_selected_layer = (int)i; render_probe(); return; }
                layer_row++;
                const ProbeLayerDesc* l = probe_get_layer(i);
                if (l && l->weight_count > 0) layer_row++;  /* min/max 行 */
            }
        }
        return;
    }
    /* Config 页滑块（鼠标点击滑杆调整，滑杆起点 s->x+14 与 draw_slider 对齐；合并页内也生效） */
    if (current_tab_has_fun(FUN_CONFIG)) {
        for (int i = 0; i < CONFIG_SLIDER_COUNT; i++) {
            Slider* s = &g_config_sliders[i];
            int sx = s->x + 14;
            int ex = sx + s->width + 2;
            if (x >= sx && x <= ex && y == s->y) {
                g_config_focus = i;
                int rel = x - sx;
                if (rel < 0) rel = 0; if (rel > s->width) rel = s->width;
                float ratio = (float)rel / s->width;
                s->value = s->min_val + ratio * (s->max_val - s->min_val);
                int steps = (int)((s->value - s->min_val) / s->step + 0.5f);
                s->value = s->min_val + steps * s->step;
                if (s->value > s->max_val) s->value = s->max_val;
                if (strcmp(s->key, "learning_rate") == 0) g_learning_rate_auto = 0;
                g_slider_dirty[i] = 1;   /* IND-01：用户已修改该字段 */
                apply_slider(i);
                return;
            }
        }
    }
    /* Config 页按钮（点击设焦点 -> 高亮） */
    if (current_tab_has_fun(FUN_CONFIG)) {
        for (int i = 0; i < CONFIG_BTN_COUNT; i++) {
            if (button_hit(&g_config_buttons[i], x, y)) {
                g_config_focus = CONFIG_SLIDER_COUNT + i;   /* 高亮选中 */
                if (strcmp(g_config_buttons[i].command, "APPLY") == 0) {
                    apply_dirty_sliders();   /* IND-01：只提交修改过的字段 */
                } else if (strcmp(g_config_buttons[i].command, "MEMORY_TOGGLE") == 0) {
                    g_memory_enabled = 1 - g_memory_enabled;
                    char cmd[64];
                    snprintf(cmd, sizeof(cmd), "SET memory_enabled %d", g_memory_enabled);
                    pipe_send(cmd);
                } else if (strcmp(g_config_buttons[i].command, "RESET") == 0) {
                    reset_sliders_to_default();
                }
                return;
            }
        }
    }
    /* 走到这里 = 点击了空白处（无任何按钮/滑块/格子命中）：取消所有选中 */
    if (g_active_tab == 0) {
        if (g_cell_dropdown >= 0) { g_cell_dropdown = -1; g_force_redraw = 1; vt_mark_all_dirty(); }
        if (g_cell_focus >= 0) g_cell_focus = -1;         /* 清除焦点格 */
    }
    if (current_tab_has_fun(FUN_CONFIG)) {
        if (g_config_focus >= 0) g_config_focus = -1;     /* 清除滑块/按钮选中 */
    }
}

/* ============================================================================
   主循环
   ============================================================================ */
/* 动画期间跳过场景淡入淡出（防止旧页面残影叠加在黑幕上） */
int g_skip_scene_crossfade = 0;

static void present_ui_scene(void) {
    int modal = g_palette_open ? 1 : g_viewer_open ? 2 : g_replay_open ? 3 : g_play_open ? 4 : 0;
    int scene = g_active_tab + 32 * modal + 256 * (g_active_tab == 2 ? g_probe_view : 0);
    static int last_modal = 0;
    if (last_modal != modal) { trans_cancel(&g_transition); g_delay_3d_show = 0; }
    last_modal = modal;
    /* 先设标志再判断，确保 d2d_end 在动画期间跳过淡入淡出 */
    g_skip_scene_crossfade = trans_is_active(&g_transition) ? 1 : 0;
    if (g_skip_scene_crossfade) {
        if (!trans_update(&g_transition)) {
            trans_destroy(&g_transition);
            win_set_scene(scene); // Consume scene change without a second crossfade.
            g_skip_scene_crossfade = 0;
            vt_mark_all_dirty();
            vt_present();
            if (g_delay_3d_show) {
                g_delay_3d_show = 0;
                extern void nv_force_intro_animation(void);
                nv_force_intro_animation();
                probe_3d_set_visible(1);
            }
        } else {
            trans_render_overlay(&g_transition);
            if (!trans_is_active(&g_transition)) {
                g_skip_scene_crossfade = 0;
                g_delay_3d_show = 0;
                vt_mark_all_dirty(); vt_present();
            }
        }
    } else {
        win_set_scene(scene);
        vt_present();
    }
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR lpCmd, int nShow) {
    (void)hInst; (void)hPrev; (void)lpCmd; (void)nShow;
    init_metric_colors();
    /* 工作目录固定为 exe 所在目录（启动训练/watchdog 时传递正确工作目录，checkpoints/logs 相对路径一致） */
    {
        /* BUG-065: 使用 32768 缓冲区避免长路径截断（MAX_PATH=260 不够） */
        wchar_t exePath[32768];
        DWORD len = GetModuleFileNameW(NULL, exePath, 32768);
        if (len > 0 && len < 32768) {
            wchar_t* slash = wcsrchr(exePath, L'\\');
            if (slash) { *slash = L'\0'; SetCurrentDirectoryW(exePath); }
        }
    }
    /* The TUI is also an inference host (Skill Test and Playground).  Load the
       same policy parameters and persistent bank before either feature runs. */
    load_config();
    memory_load_deployed();
    detect_gpu_device();
    /* PerMonitor DPI 感知：按物理像素渲染，消除 DIP 缩放导致的分数像素模糊毛边 */
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    /* 自有窗口下禁用中文输入法（TSF + IMM，避免热键唤出候选栏） */
    ImmDisableIME(0);
    ImmDisableTextFrameService(TRUE);
    /* 创建自有窗口（不再依赖 Windows Terminal 宿主） */
    if (!win_create()) return 1;
    ImmAssociateContext(g_hwnd, NULL);   /* 窗口不关联输入法上下文 */
    sync_console_indicator();   /* 识别训练/watchdog 窗口可见性，自动调整指示灯状态 */
    win_register_renderer();
    RECT rc;
    GetClientRect(g_hwnd, &rc);
    int cols = rc.right / g_cell_w;
    int rows = rc.bottom / g_cell_h;
    if (cols < 40) cols = 40;
    if (rows < 15) rows = 15;
    vt_init(cols, rows);
    g_console_width = get_console_width();
    g_console_height = get_console_height();
    compute_merge_layout();   /* 根据窗口算合并 tab 布局 */
    clear_screen();
    reset_all_chart_views();  /* 启动即采用全局复位视图：普通 200 局，Win 20 检查点 */

    static int key_s = 0, key_x = 0, key_q = 0, key_r = 0;
    static int key_ctrlp = 0;
    static int key_up = 0, key_down = 0, key_left = 0, key_right = 0, key_enter = 0, key_tab = 0;
    static int key_esc = 0;
    static int key_pgup = 0, key_pgdn = 0;
    static int key_u = 0, key_a = 0;
    DWORD last_read = 0;
    DWORD last_gpu_detect = GetTickCount();

    /* 启动遥测后台线程（每 ~1s 采样缓存，UI 只读，避免 Sleep 阻塞渲染） */
    start_telemetry();
    http_start();
    /* 每次打开自动拉起训练会话 */
    {char autostart[8]={};GetEnvironmentVariableA("DQN_TUI_AUTOSTART",autostart,sizeof(autostart));if(strcmp(autostart,"0"))start_session();}

    while (g_running) {
        const ULONGLONG frame_started = GetTickCount64();
        auto wait_for_frame = [frame_started](DWORD budget) {
            ULONGLONG elapsed = GetTickCount64() - frame_started;
            DWORD remaining = elapsed < budget ? budget - (DWORD)elapsed : 0;
            MsgWaitForMultipleObjects(0, NULL, FALSE, remaining, QS_ALLINPUT);
        };
        win_pump_messages();   /* 消息泵（WM_PAINT/SIZE/输入） */
        if(g_running)http_tick();
        if (!g_running) break; /* WM_CLOSE 后不得再执行采样、IPC 或渲染一整轮。 */
        if (InterlockedExchange(&g_report_ready, 0) != 0) {
            if (g_report_thread) {
                WaitForSingleObject(g_report_thread, 0);
                CloseHandle(g_report_thread);
                g_report_thread = NULL;
            }
            copy_report_to_clipboard();
            viewer_load(tui_report_get_text());
            g_viewer_title = "分析报告";
            g_viewer_open = 1;
            InterlockedExchange(&g_report_running, 0);
            need_tab = 1; need_content = 1; need_help = 1;
        }
        DWORD now = GetTickCount();

        /* 设备热插拔/驱动重置立即刷新；另外周期复核，覆盖未发 WM_DEVICECHANGE 的后端变化。 */
        if (g_display_config_changed || now - last_gpu_detect >= 10000) {
            g_display_config_changed = 0;
            last_gpu_detect = now;
            GpuDeviceInfo old = g_gpu_device;
            detect_gpu_device();
            if (memcmp(&old, &g_gpu_device, sizeof(old)) != 0) {
                memset(g_page_drawn, 0, sizeof(g_page_drawn));
                need_content = 1;
                g_force_redraw = 1;
                vt_mark_all_dirty();
            }
        }

        /* 尺寸变化：重置渲染状态强制全量重画 */
        int w = get_console_width();
        int h = get_console_height();
        if (w != g_console_width || h != g_console_height) {
            g_console_width = w; g_console_height = h;
            compute_merge_layout();   /* 重新计算合并布局 */
            clear_screen();
            memset(g_page_drawn, 0, sizeof(g_page_drawn));
            g_last_tab = -1;
            need_tab = 1; need_content = 1; need_help = 1;
            probe_3d_sync_window();
        }

        /* 状态机推进 */
        if (g_state == MASTER_STARTING) {
            if (pipe_connect()) {
                state_transition(MASTER_IPC_READY);
                if (g_start_watchdog_for_session && g_dashboard_workers.watchdog_enabled &&
                    GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE) == INVALID_FILE_ATTRIBUTES && start_process(&g_watchdog)) {
                    state_transition(MASTER_WATCHDOG_READY);
                    state_transition(MASTER_RUNNING);
                } else {
                    state_transition(MASTER_RUNNING);
                }
            }
        }
        /* BUG-003: 训练退出后回到 MASTER_IDLE（不卡在 MASTER_IPC_READY 等 pipe_connect 阻塞） */
        if ((g_state == MASTER_RUNNING || g_state == MASTER_STARTING) && !process_running(&g_trainer)) {
            if (g_pipe != INVALID_HANDLE_VALUE) { CloseHandle(g_pipe); g_pipe = INVALID_HANDLE_VALUE; }
            state_transition(MASTER_IDLE);
            if (g_trainer_exit_code != 0 && g_trainer_shutdown_state != 4)
                MessageBoxW(NULL, L"训练发生错误并已停止，本次未保存的进度可能丢失。请检查训练错误日志；旧存档已保留。",
                    L"训练异常停止", MB_OK | MB_ICONERROR);
        }
        /* BUG-003: MASTER_IDLE 时检测 Trainer 是否已被 Watchdog 重启，自动重连 IPC */
        if (g_state == MASTER_IDLE && process_running(&g_trainer)) {
            if (g_guard_user_enabled && !process_running(&g_vindicator)) start_process(&g_vindicator);
            static DWORD s_reconnect_tick = 0;
            if (GetTickCount() - s_reconnect_tick > 2000) {   /* 每 2s 尝试一次 */
                s_reconnect_tick = GetTickCount();
                if (pipe_connect()) {
                    state_transition(MASTER_IPC_READY);
                    state_transition(MASTER_RUNNING);
                    /* 重置采样偏移，避免与旧数据错位 */
                    g_train_offset = -1;
                    g_tg_offset = -1;
                    for (int i = 0; i < BR_MAX_FILES; i++) g_bin_first_read_done[i] = 0;
                    printf("[TUI] 已重连训练进程 IPC\n");
                }
            }
        }
        for (int id = 0; id < DASH_WORKER_COUNT; ++id) {
            if (g_dashboard_workers.stopping[id] && !process_running(dashboard_worker_process(id))) {
                g_dashboard_workers.stopping[id] = false;
                need_content = 1;
            }
        }
        // GPT6-Astra: both modes use the existing save-and-shutdown path.
        if (g_state == MASTER_RUNNING) {
            const int cur_ep = atoi(status_lines[0][0] ? status_lines[0] : "0");
            if ((g_close_at >= 0 && cur_ep >= g_close_at) ||
                (g_close_time != 0 && time(nullptr) >= g_close_time)) {
                g_close_at = -1; g_close_time = 0;
                shutdown_session();
            }
        }

        /* 管道 + 定期读 */
        pipe_poll();
        /* Phase 0 预训练期间部分正式指标文件为空。进入正式阶段后必须重新执行
           首次尾部扫描，否则空文件的 reader 状态会让这些图只等增量、历史看似消失。 */
        {
            static int s_last_chart_phase = -1;
            static int s_last_pretrain_state = -1;
            int current_chart_phase = -1;
            if (sscanf(status_lines[1], "Phase %d", &current_chart_phase) == 1) {
                int phase_changed = s_last_chart_phase >= 0 && current_chart_phase != s_last_chart_phase;
                int pretrain_finished = s_last_pretrain_state == 1 && g_pretrain_active == 0;
                if (phase_changed || pretrain_finished) {
                    memset(g_hist, 0, sizeof(g_hist));
                    for (int bi = 0; bi < BR_MAX_FILES; bi++) g_bin_first_read_done[bi] = 0;
                    g_train_offset = -1;
                    g_tg_offset = -1;
                    g_last_ep_time_episode = -1;
                    reset_all_chart_views();
                    printf("[TUI] 训练阶段切换（Phase %d -> %d, Pretrain %d -> %d），已重新扫描图表历史\n",
                        s_last_chart_phase, current_chart_phase,
                        s_last_pretrain_state, g_pretrain_active);
                }
                s_last_chart_phase = current_chart_phase;
                s_last_pretrain_state = g_pretrain_active;
            }
        }
        if (now - last_read > 2000) {
            last_read = now;
            sample_bin_metrics();   /* 优先读 bin 文件（自动发现 metric_*.bin） */
            read_eval();            /* eval.dat 文本回退（bin 无 WIN 数据时用） */
            read_train_logs();
            sample_train_metrics(); /* train.dat 文本回退（bin 无 Loss/Q/Eps 时用） */
            sample_train_steps();   /* game.dat 文本回退（bin 无 GAME_LEN 时用） */
            need_content = 1;
        }

        /* 命令面板输入（键队列：WM_CHAR/WM_KEYDOWN 喂入） */
        if (g_palette_open) {
            vt_curve_clear();   /* 模态时清曲线 overlay */
            vt_crosshair_clear();
            vt_mark_all_dirty();   /* 强制重画覆盖 D2D 背缓冲中残留的旧曲线像素 */
            render_palette();
            if (g_mouse_wheel_ready) {   /* 滚轮滚动命令教程 */
                g_mouse_wheel_ready = 0;
                if (g_mouse_wheel_delta > 0) g_palette_doc_scroll--;
                else g_palette_doc_scroll++;
            }
            int ch;
            while ((ch = vt_key_poll()) != -1) {
                if (ch == 13) {
                    run_palette_command(g_palette_buf);
                    g_palette_buf[0] = '\0';   /* 执行后清空输入区（失败时便于重新输入） */
                    need_content = 1;
                } else if (ch == 27) {
                    toggle_palette();
                    need_content = 1;
                } else if (ch == 8) {
                    int l = (int)strlen(g_palette_buf);
                    if (l > 0) g_palette_buf[l - 1] = '\0';
                    need_content = 1;
                } else if (ch >= 32 && ch < 127) {
                    int l = (int)strlen(g_palette_buf);
                    if (l < 63) { g_palette_buf[l] = (char)ch; g_palette_buf[l + 1] = '\0'; }
                    need_content = 1;
                }
            }
            present_ui_scene();
            wait_for_frame(16);
            continue;
        }

        /* 报告查看器输入（模态，Esc 返回；GetAsyncKeyState 绕过 IME） */
        if (g_viewer_open) {
            vt_curve_clear();   /* 模态时清曲线 overlay */
            vt_crosshair_clear();
            vt_mark_all_dirty();   /* 强制重画覆盖 D2D 背缓冲中残留的旧曲线像素 */
            render_viewer();
            int vh = g_console_height - CONTENT_Y - 3;
            if (vh < 4) vh = 4;
            int visible = vh - 2;
            if (g_mouse_wheel_ready) {   /* 鼠标滚轮滚动报告 */
                g_mouse_wheel_ready = 0;
                if (g_mouse_wheel_delta > 0) g_viewer_scroll--;
                else g_viewer_scroll++;
            }
            if (key_pressed(VK_PRIOR)) g_viewer_scroll -= visible;   /* PgUp */
            if (key_pressed(VK_NEXT))  g_viewer_scroll += visible;   /* PgDn */
            if (key_pressed(VK_UP))    g_viewer_scroll--;            /* ↑ */
            if (key_pressed(VK_DOWN))  g_viewer_scroll++;            /* ↓ */
            if (key_pressed(VK_HOME))  g_viewer_scroll = 0;          /* Home */
            if (key_pressed(VK_END))   g_viewer_scroll = 1 << 30;    /* End */
            if (key_pressed(VK_ESCAPE)) { g_viewer_open = 0; full_refresh(); }
            present_ui_scene();   /* 模态渲染后立即上屏，否则黑屏 */
            wait_for_frame(16);
            continue;
        }

        /* 对局回放输入（模态，Space/→ 前进，Esc 退出；GetAsyncKeyState 绕过 IME） */
        if (g_replay_open) {
            vt_curve_clear();   /* 模态时清曲线 overlay */
            vt_crosshair_clear();
            vt_mark_all_dirty();   /* 强制重画覆盖 D2D 背缓冲中残留的旧曲线像素 */
            render_replay();
            int n = tui_replay_count();
            if (key_pressed(VK_RIGHT)) g_replay_cur++;               /* → */
            if (key_pressed(VK_LEFT))  g_replay_cur--;               /* ← */
            if (key_pressed(VK_HOME))  g_replay_cur = 0;             /* Home */
            if (key_pressed(VK_END))   g_replay_cur = n - 1;         /* End */
            if (key_pressed(VK_SPACE) || key_pressed(VK_RETURN)) g_replay_cur++;
            if (key_pressed(VK_ESCAPE) || key_pressed('Q')) { g_replay_open = 0; full_refresh(); }
            if (g_replay_cur >= n) g_replay_cur = (n > 0) ? n - 1 : 0;
            if (g_replay_cur < 0) g_replay_cur = 0;
            present_ui_scene();   /* 模态渲染后立即上屏，否则黑屏 */
            wait_for_frame(16);
            continue;
        }

        /* PLAYGROUND 人机对战输入（模态，事件驱动：只在有输入事件时重绘，空闲不刷新） */
        if (g_play_open) {
            int has_input = 0;
            /* 鼠标点击（侧边栏 / 棋盘落子） */
            HANDLE hInP = GetStdHandle(STD_INPUT_HANDLE);
            INPUT_RECORD irP;
            DWORD evtP;
            if (PeekConsoleInput(hInP, &irP, 1, &evtP) && evtP > 0 && irP.EventType == MOUSE_EVENT) {
                ReadConsoleInput(hInP, &irP, 1, &evtP);
                MOUSE_EVENT_RECORD* m = &irP.Event.MouseEvent;
                if (m->dwEventFlags == 0 && (m->dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED)) {
                    handle_play_mouse(m->dwMousePosition.X, m->dwMousePosition.Y);
                    has_input = 1;
                }
            }
            /* 键盘（GetAsyncKeyState 绕过 IME，不唤出候选栏） */
            if (key_pressed(VK_UP))    { g_play_cur -= 5; has_input = 1; }
            if (key_pressed(VK_DOWN))  { g_play_cur += 5; has_input = 1; }
            if (key_pressed(VK_LEFT))  { g_play_cur--; has_input = 1; }
            if (key_pressed(VK_RIGHT)) { g_play_cur++; has_input = 1; }
            if (g_play_cur < 0) g_play_cur = 0;
            if (g_play_cur > 24) g_play_cur = 24;
            if (key_pressed(VK_RETURN)) {
                if (g_play_over < 0 && g_play_board[g_play_cur] == 0) {
                    g_play_board[g_play_cur] = 1;     /* 用户执黑 */
                    g_play_hist[g_play_hist_n++] = g_play_cur;
                    g_play_moves++;
                    if (play_win_check(g_play_board, 1)) g_play_over = 1;
                    else if (g_play_moves >= 25) g_play_over = 0;
                    else {
                        int am = playground_ai_move(g_play_board, 1);   /* AI 执白 */
                        if (am >= 0 && g_play_board[am] == 0) {
                            g_play_board[am] = 2;
                            g_play_hist[g_play_hist_n++] = am;
                            g_play_moves++;
                            if (play_win_check(g_play_board, 2)) g_play_over = 2;
                            else if (g_play_moves >= 25) g_play_over = 0;
                        }
                    }
                }
                refresh_play_analysis();
                has_input = 1;
            }
            if (key_pressed('U')) {
                if (g_play_hist_n >= 2) {   /* 撤销 AI+用户各一步 */
                    g_play_board[g_play_hist[--g_play_hist_n]] = 0;
                    g_play_board[g_play_hist[--g_play_hist_n]] = 0;
                    g_play_moves -= 2;
                    g_play_over = -1;
                }
                refresh_play_analysis();
                has_input = 1;
            }
            if (key_pressed('R')) {
                memset(g_play_board, 0, sizeof(g_play_board));
                g_play_cur = 12; g_play_moves = 0; g_play_hist_n = 0; g_play_over = -1;
                refresh_play_analysis();
                has_input = 1;
            }
            if (key_pressed('A')) {
                g_play_analysis_on = 1 - g_play_analysis_on;   /* 切换分析 */
                refresh_play_analysis();
                has_input = 1;
            }
            if (key_pressed('S')) {
                /* Keyboard and mouse share the same generation-aware wait path. */
                play_snapshot();
                has_input = 1;
            }
            if (key_pressed(VK_TAB)) {
                if (g_console_width < 150) g_play_sidebar_tab = (g_play_sidebar_tab + 1) % 5;
                has_input = 1;
            }
            for (int d = 0; d < 5; d++)
                if (key_pressed('1' + d)) { if (g_console_width < 150) g_play_sidebar_tab = d; has_input = 1; }
            if (key_pressed(VK_ESCAPE) || key_pressed('Q')) { g_play_open = 0; full_refresh(); }
            if (has_input) render_playground();   /* 事件驱动：有输入才重绘 */
            present_ui_scene(); // Continue an interrupted color transition even without new input.
            wait_for_frame(16);
            continue;
        }

        /* 键盘（GetAsyncKeyState + 前台检测） */
        int focused = window_focused();
        if (!focused) { key_s = key_x = key_q = key_ctrlp = 0; }
#define KEY_DOWN(k) (GetAsyncKeyState(k) & 0x8000)
        if (focused) {
            /* 1-9 切页（按当前 tab 数） */
            for (int i = 0; i < g_total_tabs; i++) {
                if (KEY_DOWN('1' + i)) { g_active_tab = i; need_tab = 1; need_content = 1; }
            }
            /* Ctrl+P 命令面板 */
            if (KEY_DOWN(VK_CONTROL) && KEY_DOWN('P')) {
                if (!key_ctrlp) { key_ctrlp = 1; toggle_palette(); need_tab = 1; need_content = 1; }
            } else { key_ctrlp = 0; }
            /* PROBE 页（tab 2）定时刷新 + 3D 初始化 + WASD 摄像机 */
            if (g_active_tab == 2) {
                /* WASD + Space/Shift 控制 3D 摄像机（仅在3D视图激活时） */
                if (g_nv_3d_ready == 1 && g_probe_view == 4) {
                    if (KEY_DOWN('W') || KEY_DOWN('w')) { nv_on_key('W', 1); render_probe(); }
                    if (KEY_DOWN('S') && !(GetAsyncKeyState(VK_CONTROL) & 0x8000)) { nv_on_key('S', 1); render_probe(); }
                    if (KEY_DOWN('A') || KEY_DOWN('a')) { nv_on_key('A', 1); render_probe(); }
                    if (KEY_DOWN('D') || KEY_DOWN('d')) { nv_on_key('D', 1); render_probe(); }
                    if (KEY_DOWN(VK_SPACE))  { nv_on_key(VK_SPACE, 1); render_probe(); }
                    if (KEY_DOWN(VK_LSHIFT)) { nv_on_key(VK_LSHIFT, 1); render_probe(); }
                }
                static DWORD s_probe_tick = 0;
                DWORD now3 = GetTickCount();
                /* 3D 视图初始化（在主循环做，不在渲染函数里） */
                if (g_nv_pending_init) {
                    g_nv_pending_init = 0;
                    if (g_nv_lifecycle_initialized) probe_3d_shutdown();
                    int nv_x, nv_y, nv_w, nv_h;
                    probe_3d_get_rect(&nv_x, &nv_y, &nv_w, &nv_h);
                    int ret = nv_init(g_hwnd, nv_x, nv_y, nv_w, nv_h);
                    if (ret == 0) {
                        g_nv_lifecycle_initialized = 1;
                        g_nv_3d_ready = 1;
                        g_nv_3d_error = 0;
                        probe_3d_sync_window();
                    } else {
                        g_nv_3d_error = ret;
                        g_nv_3d_ready = -1;  /* 初始化失败 */
                    }
                }
                if (now3 - s_probe_tick >= PROBE_CLIENT_POLL_MS) { s_probe_tick = now3; render_probe(); }
            }
            /* PLAYGROUND 页（tab 1）输入：方向键/Enter/U/R/A/S，事件驱动重绘 */
            if (g_active_tab == 1) {
                /* 技能测试完成时强制刷新一帧 */
                if (g_snap_test_just_done) { g_snap_test_just_done = 0; render_playground(); }
                if (g_snap_test_running) {
                    /* 同步 Core 的实时进度 */
                    g_snap_test_progress = g_skill_test_progress;
                    g_snap_test_phase = g_skill_test_phase;
                    static DWORD s_last_prog_tick = 0;
                    DWORD now2 = GetTickCount();
                    if (now2 - s_last_prog_tick > 100) { s_last_prog_tick = now2; render_playground(); }
                }
                int pr = 0;
                if (KEY_DOWN(VK_UP))    { if (!key_up)   { key_up=1;   g_play_cur -= 5; pr=1; } } else key_up=0;
                if (KEY_DOWN(VK_DOWN))  { if (!key_down) { key_down=1; g_play_cur += 5; pr=1; } } else key_down=0;
                if (KEY_DOWN(VK_LEFT))  { if (!key_left) { key_left=1; g_play_cur--; pr=1; } } else key_left=0;
                if (KEY_DOWN(VK_RIGHT)) { if (!key_right){ key_right=1; g_play_cur++; pr=1; } } else key_right=0;
                if (g_play_cur < 0) g_play_cur = 0;
                if (g_play_cur > 24) g_play_cur = 24;
                if (KEY_DOWN(VK_RETURN)) { if (!key_enter) { key_enter=1; play_do_move(g_play_cur); pr=1; } } else key_enter=0;
                if (KEY_DOWN('U')) { if (!key_u) { key_u=1; play_undo(); pr=1; } } else key_u=0;
                if (KEY_DOWN('R')) { if (!key_r) { key_r=1; play_restart(); pr=1; } } else key_r=0;
                if (KEY_DOWN('A')) { if (!key_a) { key_a=1; g_play_analysis_on = 1-g_play_analysis_on; refresh_play_analysis(); pr=1; } } else key_a=0;
                if (KEY_DOWN('S')) { if (!key_s) { key_s=1; play_snapshot(); pr=1; } } else key_s=0;
                if (KEY_DOWN(VK_TAB)) { if (!key_tab) { key_tab=1; if (g_console_width < 150) g_play_sidebar_tab = (g_play_sidebar_tab+1)%5; pr=1; } } else key_tab=0;
                /* 空格：AI 落一子（不管当前是谁的回合） */
                { static int key_space=0;
                  if (KEY_DOWN(VK_SPACE)) {
                      if (!key_space && g_play_over < 0) {
                          key_space = 1;
                          play_ai_step();
                          pr = 1;
                      }
                  } else key_space = 0;
                }
                if (pr) render_playground();
            }
            /* S / X / Q（PLAYGROUND 页 S 已被占用为快照） */
            if (KEY_DOWN('S') && g_active_tab != 1) { if (!key_s) { key_s = 1; start_session(); need_tab = 1; need_content = 1; } }
            else key_s = 0;
            if (KEY_DOWN('Q')) { if (!key_q) { key_q = 1; g_running = 0; } }
            else key_q = 0;
            if (g_active_tab == 3) {
                static int prev=0;
                int keys=(KEY_DOWN(VK_PRIOR)?1:0)|(KEY_DOWN(VK_NEXT)?2:0);
                if((keys&1)&&!(prev&1)){g_analyze_scroll-=g_analyze_view_rows;need_content=1;}
                if((keys&2)&&!(prev&2)){g_analyze_scroll+=g_analyze_view_rows;need_content=1;}
                if(g_analyze_scroll<0)g_analyze_scroll=0;
                if(g_analyze_scroll>g_analyze_scroll_max)g_analyze_scroll=g_analyze_scroll_max;
                prev=keys;
            }
            /* Config 滑块 + 按钮控制（独立 tab 或合并页内均生效） */
            if (current_tab_has_fun(FUN_CONFIG)) {
                if (KEY_DOWN(VK_TAB)) {
                    if (!key_tab) { key_tab = 1; g_config_focus = (g_config_focus < CONFIG_SLIDER_COUNT) ? CONFIG_SLIDER_COUNT : 0; need_content = 1; }
                } else key_tab = 0;
                if (KEY_DOWN(VK_UP)) {
                    if (!key_up) {
                        key_up = 1;
                        if (g_config_focus >= CONFIG_SLIDER_COUNT) { g_config_focus--; if (g_config_focus < CONFIG_SLIDER_COUNT) g_config_focus = CONFIG_SLIDER_COUNT + CONFIG_BTN_COUNT - 1; }
                        else { g_config_focus--; if (g_config_focus < 0) g_config_focus = CONFIG_SLIDER_COUNT - 1; }
                        need_content = 1;
                    }
                } else key_up = 0;
                if (KEY_DOWN(VK_DOWN)) {
                    if (!key_down) {
                        key_down = 1;
                        if (g_config_focus < CONFIG_SLIDER_COUNT) { g_config_focus++; if (g_config_focus >= CONFIG_SLIDER_COUNT) g_config_focus = 0; }
                        else { g_config_focus++; if (g_config_focus >= CONFIG_SLIDER_COUNT + CONFIG_BTN_COUNT) g_config_focus = CONFIG_SLIDER_COUNT; }
                        need_content = 1;
                    }
                } else key_down = 0;
                if (KEY_DOWN(VK_LEFT)) {
                    if (!key_left) {
                        key_left = 1;
                        if (g_config_focus < CONFIG_SLIDER_COUNT) {
                            Slider* s = &g_config_sliders[g_config_focus];
                            s->value -= s->step;
                            if (s->value < s->min_val) s->value = s->min_val;
                            if (strcmp(s->key, "learning_rate") == 0) g_learning_rate_auto = 0;
                            g_slider_dirty[g_config_focus] = 1;
                            need_content = 1;
                        }
                    }
                } else key_left = 0;
                if (KEY_DOWN(VK_RIGHT)) {
                    if (!key_right) {
                        key_right = 1;
                        if (g_config_focus < CONFIG_SLIDER_COUNT) {
                            Slider* s = &g_config_sliders[g_config_focus];
                            s->value += s->step;
                            if (s->value > s->max_val) s->value = s->max_val;
                            if (strcmp(s->key, "learning_rate") == 0) g_learning_rate_auto = 0;
                            g_slider_dirty[g_config_focus] = 1;
                            need_content = 1;
                        }
                    }
                } else key_right = 0;
                if (KEY_DOWN(VK_RETURN)) {
                    if (!key_enter) {
                        key_enter = 1;
                        if (g_config_focus < CONFIG_SLIDER_COUNT) {
                            apply_slider(g_config_focus);
                        } else {
                            int bi = g_config_focus - CONFIG_SLIDER_COUNT;
                            if (bi >= 0 && bi < CONFIG_BTN_COUNT) {
                                if (strcmp(g_config_buttons[bi].command, "APPLY") == 0) { apply_dirty_sliders(); }
                                else if (strcmp(g_config_buttons[bi].command, "MEMORY_TOGGLE") == 0) {
                                    g_memory_enabled = 1 - g_memory_enabled;
                                    char cmd[64]; snprintf(cmd, sizeof(cmd), "SET memory_enabled %d", g_memory_enabled);
                                    pipe_send(cmd);
                                } else if (strcmp(g_config_buttons[bi].command, "RESET") == 0) {
                                    reset_sliders_to_default();
                                }
                            }
                        }
                        need_content = 1;
                    }
                } else key_enter = 0;
                if (KEY_DOWN('R')) {
                    if (!key_r) {
                        key_r = 1;
                        reset_sliders_to_default();
                        need_content = 1;
                    }
                } else key_r = 0;
            }
            /* Checkpoint 页（合并页内）：↑↓/PgUp/PgDn/Home/End 滚动列表（与 Config 方向键互斥） */
            if (current_tab_has_fun(FUN_CHECKPOINT) && !current_tab_has_fun(FUN_CONFIG)) {
                if (KEY_DOWN(VK_UP)) { if (!key_up) { key_up = 1; g_ckpt_scroll--; need_content = 1; } } else key_up = 0;
                if (KEY_DOWN(VK_DOWN)) { if (!key_down) { key_down = 1; g_ckpt_scroll++; need_content = 1; } } else key_down = 0;
                if (KEY_DOWN(VK_PRIOR)) { if (!key_pgup) { key_pgup = 1; g_ckpt_scroll -= 10; need_content = 1; } } else key_pgup = 0;
                if (KEY_DOWN(VK_NEXT)) { if (!key_pgdn) { key_pgdn = 1; g_ckpt_scroll += 10; need_content = 1; } } else key_pgdn = 0;
                if (KEY_DOWN(VK_HOME)) { g_ckpt_scroll = 0; need_content = 1; }
                if (KEY_DOWN(VK_END)) { g_ckpt_scroll = 1 << 30; need_content = 1; }
            }
            /* Training 页：Tab 切框焦点，Enter 弹该框下拉选择，←→ 也可直接轮换 */
            if (g_active_tab == 0) {
                int grid_top = CONTENT_Y + 1;
                int avail_h = (g_console_height - 4) - grid_top - 3;
                int cols, rows, cw, ch;
                compute_chart_layout(g_console_width - 4, avail_h, &cols, &rows, &cw, &ch);
                int N = cols * rows; if (N > MAX_CELLS) N = MAX_CELLS;
                if (g_cell_focus >= N) g_cell_focus = N - 1;
                if (g_cell_dropdown >= 0) {
                    /* 下拉菜单输入 */
                    if (KEY_DOWN(VK_UP)) {
                        if (!key_up) { key_up = 1; g_cell_drop_cur--; if (g_cell_drop_cur < 0) g_cell_drop_cur = METRIC_COUNT - 1; need_content = 1; }
                    } else key_up = 0;
                    if (KEY_DOWN(VK_DOWN)) {
                        if (!key_down) { key_down = 1; g_cell_drop_cur++; if (g_cell_drop_cur >= METRIC_COUNT) g_cell_drop_cur = 0; need_content = 1; }
                    } else key_down = 0;
                    if (KEY_DOWN(VK_RETURN)) {
                        if (!key_enter) {
                            key_enter = 1;
                            g_cell_checks[g_cell_dropdown][g_cell_drop_cur] = 1 - g_cell_checks[g_cell_dropdown][g_cell_drop_cur];   /* 不关闭，可连续勾选，允许全部不勾 */
                            g_cell_metric[g_cell_dropdown] = -1;   /* 先清空，再找第一个勾选的；全不勾时 -1 直接黑框 */
                            for (int k = 0; k < METRIC_COUNT; k++) if (g_cell_checks[g_cell_dropdown][k]) { g_cell_metric[g_cell_dropdown] = k; break; }
                            need_content = 1;
                        }
                    } else key_enter = 0;
                    if (KEY_DOWN(VK_ESCAPE)) {
                        if (!key_esc) { key_esc = 1; g_cell_dropdown = -1; g_force_redraw = 1; vt_mark_all_dirty(); need_content = 1; }
                    } else key_esc = 0;
                } else {
                    if (KEY_DOWN(VK_TAB)) {
                        if (!key_tab) { key_tab = 1; g_cell_focus = (g_cell_focus + 1) % N; need_content = 1; }
                    } else key_tab = 0;
                    if (KEY_DOWN(VK_RETURN)) {
                        if (!key_enter) {
                            key_enter = 1;
                            g_cell_dropdown = g_cell_focus;
                            g_cell_drop_cur = g_cell_metric[g_cell_focus];
                            if (g_cell_drop_cur < 0) g_cell_drop_cur = 0;
                            need_content = 1;
                        }
                    } else key_enter = 0;
                    if (KEY_DOWN(VK_LEFT)) {
                        if (!key_left) { key_left = 1; cycle_cell_metric(-1); need_content = 1; }
                    } else key_left = 0;
                    if (KEY_DOWN(VK_RIGHT)) {
                        if (!key_right) { key_right = 1; cycle_cell_metric(+1); need_content = 1; }
                    } else key_right = 0;
                }
            }
        }

        // Redraw on press/release even when telemetry has not changed.
        {
            static int previous_buttons = 0;
            int buttons = ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) ? 1 : 0) |
                          ((GetAsyncKeyState(VK_RETURN) & 0x8000) ? 2 : 0);
            const int button_keys[] = {'R', 'U', 'S', 'A', VK_SPACE, VK_ESCAPE};
            for (int i = 0; i < 6; ++i) if (GetAsyncKeyState(button_keys[i]) & 0x8000) buttons |= (4 << i);
            if (buttons != previous_buttons) need_content = 1;
            previous_buttons = buttons;
        }
        /* 鼠标（WM_LBUTTONDOWN/WM_MOUSEWHEEL 喂入的全局字符坐标） */
        if (g_mouse_click_ready) {
            g_mouse_click_ready = 0;
            int mx = g_mouse_click_x, my = g_mouse_click_y;
            if (!slider_click(mx, my) && !progress_btn_click(mx, my)) handle_mouse_click(mx, my);
            need_tab = 1; need_content = 1;
        }
        if (g_mouse_wheel_ready) {
            g_mouse_wheel_ready = 0;
            if (metric_menu_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                need_tab = need_content = 1;
            else if (analyzer_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                need_content = 1;
            else if (config_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                need_content = 1;
            else if (slider_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                need_tab = need_content = 1;
            else if (handle_play_test_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                { /* test panel scrolled, already rendered */ }
            else if (handle_hist_wheel(g_mouse_wheel_x, g_mouse_wheel_y, g_mouse_wheel_delta))
                { /* history scrolled, already rendered */ }
            else if (current_tab_has_fun(FUN_CHECKPOINT)) {   /* Checkpoint 页滚轮滚动列表 */
                if (g_mouse_wheel_delta > 0) g_ckpt_scroll--;
                else g_ckpt_scroll++;
                need_tab = need_content = 1;
            }
        }
        /* 十字轴：Training 页 hover 曲线图显示准线（不触发全量重绘，overlay 强制上屏） */
        if (g_mouse_move_ready) {
            g_mouse_move_ready = 0;
            vt_crosshair_clear();
            if (g_active_tab == 0) {
                int grid_top = CONTENT_Y + 1;
                int avail_w = g_console_width - 4;
                int pbar_h = 3, mon_h = 3;
                int py_pbar = g_console_height - 4 - pbar_h + 1;
                if (py_pbar < grid_top + mon_h + 3) py_pbar = grid_top + mon_h + 3;
                int avail_h = py_pbar - grid_top - mon_h - 2; if (avail_h < 8) avail_h = 8;
                int c2, r2, w2, h2;
                compute_chart_layout(avail_w, avail_h, &c2, &r2, &w2, &h2);
                int N2 = c2 * r2; if (N2 > MAX_CELLS) N2 = MAX_CELLS;
                int rem2 = avail_w - c2 * w2; if (rem2 < 0) rem2 = 0;
                int mxx = g_mouse_move_x, myy = g_mouse_move_y;
                for (int i = 0; i < N2; i++) {
                    int cc = i % c2;
                    int cx = 2 + cc * w2 + (cc < rem2 ? cc : rem2);
                    int cy = grid_top + (i / c2) * h2;
                    if (g_cell_metric[i] < 0 || g_cell_metric[i] >= METRIC_COUNT) continue;
                    int px = cx + 1 + 9;                /* plot 区起点（axis_w=9） */
                    int py0 = cy + 1;
                    int pw3 = (w2 + (cc < rem2 ? 1 : 0)) - 9 - 3;
                    int ph3 = h2 - 5;                   /* 与 draw_line_series 的曲线 ph(=h-5) 一致，保证交点过滤能匹配到曲线 */
                    if (pw3 < 2 || ph3 < 2) continue;
                    if (mxx >= px && mxx < px + pw3 && myy >= py0 && myy < py0 + ph3) {
                        VTCrosshair chh;
                        chh.x = px; chh.y = py0; chh.w = pw3; chh.h = ph3;
                        chh.x_ratio = (float)(mxx - px) / (pw3 > 1 ? pw3 - 1 : 1);
                        chh.y_ratio = (float)(myy - py0) / (ph3 > 1 ? ph3 - 1 : 1);
                        chh.color = vt_palette_rgb(COLOR_DARK_GRAY);
                        chh.line_count = 0;
                        /* 数值标签：鼠标 x 对应各勾选指标的值。
                           视口用"距最新点的偏移"表示（vs=0=最新，ve=0=最新），
                           每个指标按自己的 count 映射到实际数据索引。 */
                        {
                            int vs2 = g_cell_view_start[i], ve2 = g_cell_view_end[i];
                            float gx0 = 0, gx1 = 0;
                            if (cell_global_episode_range(i, &gx0, &gx1)) {
                                float target_ep = gx0 + (gx1 - gx0) * chh.x_ratio;
                                swprintf(chh.lines[chh.line_count++], 48, L"EP %.0f", target_ep);
                                for (int k = 0; k < METRIC_COUNT && chh.line_count < VT_CROSSHAIR_LINES; k++) {
                                    if (!g_cell_checks[i][k] || g_hist[k].n <= 0) continue;
                                    int best = -1;
                                    if (k >= METRIC_WIN0 && k <= METRIC_WIN2) {
                                        /* Win 是阶梯量：显示目标局之前最近一次评估。 */
                                        for (int j = 0; j < g_hist[k].n; j++)
                                            if (g_hist[k].ep[j] > 0 && g_hist[k].ep[j] <= target_ep) best = j;
                                    } else {
                                        float bd = 1e30f;
                                        for (int j = 0; j < g_hist[k].n; j++) if (g_hist[k].ep[j] > 0) {
                                            float d = fabsf(g_hist[k].ep[j] - target_ep);
                                            if (d < bd) { bd = d; best = j; }
                                        }
                                    }
                                    if (best < 0) continue;
                                    wchar_t wlab[24];
                                    MultiByteToWideChar(CP_UTF8, 0, g_metric_labels[k], -1, wlab, 24);
                                    if (k == METRIC_GAME_LEN)
                                        swprintf(chh.lines[chh.line_count], 48, L"%s %.0f", wlab, g_hist[k].v[best]);
                                    else
                                        swprintf(chh.lines[chh.line_count], 48, L"%s %.4f", wlab, g_hist[k].v[best]);
                                    chh.line_count++;
                                }
                            } else {
                            /* vs2/ve2 是"距最新点的偏移"，对任何 count 的指标都有效：
                               ds = count-1-vs2（含 vs2=-1 全量），de = count-1-ve2 */
                            int ep_metric = g_cell_metric[i];
                            if (ep_metric < 0 || ep_metric >= METRIC_COUNT || g_hist[ep_metric].n <= 0) {
                                ep_metric = -1;
                                for (int k = 0; k < METRIC_COUNT; ++k)
                                    if (g_cell_checks[i][k] && g_hist[k].n > 0) { ep_metric = k; break; }
                            }
                            int main_cnt = ep_metric >= 0 ? g_hist[ep_metric].n : 0;
                            int main_ds = 0, main_de = main_cnt - 1;
                            if (vs2 >= 0 && main_cnt > 0) {
                                main_ds = main_cnt - 1 - vs2; if (main_ds < 0) main_ds = 0;
                                main_de = main_cnt - 1 - ve2; if (main_de < main_ds) main_de = main_ds;
                                if (main_de > main_cnt - 1) main_de = main_cnt - 1;
                            }
                            int main_vn = main_de - main_ds + 1;
                            /* 主指标的 idx（用于边界检查） */
                            int main_idx = main_ds + (int)(chh.x_ratio * (main_vn > 1 ? main_vn - 1 : 0));
                            if (main_idx > main_de) main_idx = main_de;
                            /* 非全局轴过去没有添加 EP 行，导致同一个跟随框会随
                               图表组合时有时有 EP、有时没有。始终以实际选中的
                               主曲线点为准；旧文本点才按当前局和索引距离回推。 */
                            int hover_ep = 0;
                            if (ep_metric >= 0 && main_idx >= 0 && main_idx < main_cnt) {
                                if (g_hist[ep_metric].ep[main_idx] > 0)
                                    hover_ep = (int)lroundf(g_hist[ep_metric].ep[main_idx]);
                                else {
                                    int current_ep = atoi(status_lines[0]);
                                    if (current_ep > 0) hover_ep = current_ep - (main_cnt - 1 - main_idx);
                                    if (hover_ep <= 0) hover_ep = g_hist[ep_metric].base + main_idx + 1;
                                }
                            }
                            swprintf(chh.lines[chh.line_count++], 48, L"EP %d", hover_ep);
                            for (int k = 0; k < METRIC_COUNT && chh.line_count < VT_CROSSHAIR_LINES; k++) {
                                if (!g_cell_checks[i][k]) continue;
                                int cnt_k = g_hist[k].n;
                                if (cnt_k <= 0) continue;
                                /* 每个指标用 vs2/ve2（距最新偏移）映射到自己的 count */
                                int ds_k = 0, de_k = cnt_k - 1;
                                if (vs2 >= 0) {
                                    ds_k = cnt_k - 1 - vs2; if (ds_k < 0) ds_k = 0;
                                    de_k = cnt_k - 1 - ve2; if (de_k < ds_k) de_k = ds_k;
                                    if (de_k > cnt_k - 1) de_k = cnt_k - 1;
                                }
                                int vn_k = de_k - ds_k + 1;
                                int idx_k = ds_k + (int)(chh.x_ratio * (vn_k > 1 ? vn_k - 1 : 0));
                                if (idx_k > de_k) idx_k = de_k;
                                if (idx_k < 0 || idx_k >= cnt_k) continue;
                                wchar_t wlab[24];
                                MultiByteToWideChar(CP_UTF8, 0, g_metric_labels[k], -1, wlab, 24);
                                if (k == METRIC_GAME_LEN)
                                    swprintf(chh.lines[chh.line_count], 48, L"%s %.0f", wlab, g_hist[k].v[idx_k]);
                                else
                                    swprintf(chh.lines[chh.line_count], 48, L"%s %.4f", wlab, g_hist[k].v[idx_k]);
                                chh.line_count++;
                            }
                            }
                        }
                        vt_crosshair_add(&chh);
                        break;
                    }
                }
            }
            vt_mark_all_dirty();   /* 重画 cell 覆盖旧十字轴，消除残影 */
        }

        /* 3D 是原生子窗口；切走 Probe/断开连接/打开模态页时必须隐藏，
           否则它会覆盖在新的 cell UI 上方。 */
        probe_3d_sync_window();
        /* 渲染（细粒度刷新）——命令面板/查看器/回放是模态界面，打开时只渲染对应模态 */
        if (!g_palette_open && !g_viewer_open && !g_replay_open && !g_play_open) {
            if (need_tab) { draw_tab_bar(); need_tab = 0; }
            if (need_content) { render_content(); need_content = 0; }
            if (need_help) { draw_help_bar(); need_help = 0; }
            /* 最小窗口警告 */
            if (g_console_width < 85) {
                gotoxy(0, g_console_height - 4);
                set_color(COLOR_RED, COLOR_BLACK);
                int msg_len = printf("警告: 窗口过窄 (%d 列), 建议拉宽到 150+ 列（图表需要 120+, 完整布局 180+）", g_console_width);
                for (int i = msg_len; i < g_console_width; i++) printf(" ");
                set_color(COLOR_WHITE, COLOR_BLACK);
            }
        }
        /* 复位按钮点击反馈：短暂高亮后清除 */
        if (g_cell_reset_press != -1 && GetTickCount() - g_reset_press_time > 250) {
            g_cell_reset_press = -1;
            need_content = 1;
        }
        /* 跑马灯动效：Training 页刷新进度条行；PLAYGROUND 页刷新状态行（不整页渲染） */
        {
            static DWORD last_marquee = 0;
            DWORD nm2 = GetTickCount();
            if (nm2 - last_marquee >= 16) {
                last_marquee = nm2;
                if (g_active_tab == 0 && !g_palette_open && !g_viewer_open && !g_replay_open && !g_play_open)
                    draw_progress_bar(2, g_console_width - 4);
                /* Checkpoint 页面定时刷新（每 2 秒重绘一次，列出新存档） */
                static DWORD last_ckpt_refresh = 0;
                if (nm2 - last_ckpt_refresh >= 2000) {
                    last_ckpt_refresh = nm2;
                    if (current_tab_has_fun(FUN_CHECKPOINT)) {
                        g_page_drawn[g_active_tab] = 0;   /* 清标记 → 下一帧重绘 checkpoint 列表 */
                        need_content = 1;
                    }
                }
                if (g_active_tab == 1 && !g_palette_open && !g_viewer_open && !g_replay_open && !g_play_open) {
                    /* PLAYGROUND 状态行跑马灯：只重画右上角状态区 */
                    int btop3 = CONTENT_Y;
                    gotoxy(g_console_width - 22, btop3 + 1);
                    if (process_running(&g_trainer)) {
                        vt_set_truecolor_fg(0x39, 0xC5, 0xBB); printf("●");
                        set_color(COLOR_WHITE, COLOR_BLACK); printf(" TRAINING ");
                        const char* lb = "RUNNING"; int ll = 7;
                        /* 3 格高光条从左侧滑入、最后一个字符滑出右边界后消失，滑完等 1.5 秒再滑下一次
                           右边缘从 0 滑到 ll+1（共 ll+2=9 步），滑动 0.8s + 等待 1.5s = 总周期 2.3s */
                        const DWORD SLIDE_MS = 800;     /* 滑动耗时 0.8 秒 */
                        const DWORD WAIT_MS = 1500;     /* 滑完等待 1.5 秒 */
                        const DWORD HP2 = SLIDE_MS + WAIT_MS;
                        DWORD tk = GetTickCount() % HP2;
                        float center = -2.0f + (ll + 4.0f) * (float)tk / SLIDE_MS;
                        for (int c = 0; c < ll; c++) {
                            float distance = c - center;
                            if (distance < 0) distance = -distance;
                            float light = tk < SLIDE_MS ? 1.0f - distance / 2.0f : 0.0f;
                            if (light < 0) light = 0;
                            light = light * light * (3.0f - 2.0f * light);
                            vt_set_truecolor_fg((int)(0x76 + (0x39 - 0x76) * light),
                                (int)(0x76 + (0xC5 - 0x76) * light),
                                (int)(0x76 + (0xBB - 0x76) * light));
                            printf("%c", lb[c]);
                        }
                    } else {
                        set_color(COLOR_DARK_GRAY, COLOR_BLACK);
                        printf("● TRAINING STOPPED");
                    }
                    set_color(COLOR_WHITE, COLOR_BLACK);
                }
            }
        }
        present_ui_scene();   /* 把 cell buffer diff 刷到窗口 */
        probe_3d_present_after_terminal(); /* child surface must be last */
        /* 空闲时消息等待：有输入立即醒（响应快），超时 ≤50ms 保证采样/渲染节奏 */
        {
            DWORD n2 = GetTickCount();
            int busy = trans_is_active(&g_transition) || need_tab || need_content || need_help || vt_has_dirty() || g_mouse_move_ready;
            if (!busy) {
                DWORD wait = (last_read + 2000 > n2) ? (last_read + 2000 - n2) : 0;
                DWORD frame_wait = (g_active_tab <= 2) ? 16 : 50;
                if (wait > frame_wait) wait = frame_wait;
                wait_for_frame(wait);
            } else {
                wait_for_frame(16);
            }
        }
    }

    /* 退出不杀后台 */
    if (g_report_thread) {
        tui_report_request_cancel();
        /* Report scans cooperate with cancellation; join before globals and
           CRT state disappear so a closed window cannot leave a hidden TUI. */
        WaitForSingleObject(g_report_thread, INFINITE);
        CloseHandle(g_report_thread);
        g_report_thread = NULL;
    }
    g_http_server.stop();
    stop_telemetry();
    probe_3d_shutdown();
    probe_client_disconnect();
    if (g_gpu_factory) { g_gpu_factory->Release(); g_gpu_factory = NULL; }
    if (g_pipe != INVALID_HANDLE_VALUE) CloseHandle(g_pipe);
    win_shutdown();
    return 0;
}
