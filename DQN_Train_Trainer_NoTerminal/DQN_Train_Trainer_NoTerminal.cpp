#define _CRT_SECURE_NO_WARNINGS
#pragma comment(linker, "/STACK:16777216")
#include "core.h"
#include "terminal_outcome_credit.h"
#include "outcome_metrics.h"
#include "skill_test.h"
#include "competency_training.h"
#include "action_policy.h"
#include "training_task_boundary.h"
#include "training_resume.h"
#include "training_log_rollback.h"
#include "training_log_timeline.h"
#include "metric_log_writer.h"
#include "neural_probe_protocol.h"
#include "neural_probe_schedule.h"
#include "shutdown_save.h"
#include "memory_weight_scheduler.h"
#include "snapshot_collector.h"
#include "sentinel_guard.h"
#include "independent_networks.h"
#include <omp.h>
#include <signal.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <share.h>
#include <process.h>
#include <windows.h>
#include <memory>
#include <new>
#include <stdarg.h>

// GPU 初始化函数（定义在 gpu_neural_net.hip，用 hipcc 编译）
extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);

/* 方案 (b) 的切换开销测量：显式按方绑定是「设备↔影子」的 2×3.4MB 往返，
   必须量化真实代价，不能只凭提交次数认定性能可接受。
   吞吐用**相邻两次采样点的差值**算区间速率：垫档恢复时 g_start_time 为 0，
   用它算出的 eps/s 是从 1970 年起除，得到无效数字。 */
static unsigned long long g_indep_bind_count = 0;
static double g_indep_bind_seconds = 0.0;
static int g_indep_perf_prev_ep = 0;
static double g_indep_perf_prev_t = 0.0;
static double now_seconds(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* 保存/导出/评测前的统一同步点：四句柄驻留模式下把**两方四个句柄**都回读到各自
   CPU 影子，保证后续读到的两方来自同一逻辑时点；非独立（共享网络）模式退化为
   共享那一对，逐位保持历史行为。返回 0 表示可用。 */
static int sync_gpu_before_persist(DQNAgent* agent) {
    if (!agent || !gpu_net_active()) return 1;
    if (independent_training::indep_active(agent))
        return gpu_net_download_sides_pair(agent) == 0;
    return gpu_net_download_agent(agent) == 0;
}

// ============================================================================
// Training program specific globals
// ============================================================================
// Note: g_ctx, g_learning_rate, g_wins, g_start_time, g_hot_config,
//       g_current_agent, g_pipe_thread, g_pipe_running are defined in core library
// ============================================================================

static volatile sig_atomic_t g_exit_requested = 0; /* 所有关闭入口只置位，由训练循环同步 GPU 后存档 */
static volatile sig_atomic_t g_pending_exit_signal = 0;

/* Probe 发布策略与训练课程解耦：预训练的局数增长极快，按墙钟节流；
   正式对局较慢，按真实环境步计数。发布只发生在一次训练更新完成后的
   安全边界，并且 neural_probe_enabled() 会过滤掉没有 Viewer 的情况。 */
static ProbePublishSchedule g_probe_publish_schedule = {};

/* Memory Weight Scheduler：根据多深度评估动态调节 Pretrain Memory 采样比例 */
static MemoryWeightScheduler g_mws = {};
static int g_mws_initialized = 0;

/* 多快照采集器：每1000局间隔内采集16个快照 */
static SnapshotCollector g_snap_collector = {};

/* -------------------------------------------------------------------------
   Side-balance feedback

   One trainable q_net is shared by both colours. The online net must actually
   play both sides before a side win rate is meaningful. We keep a rolling
   score window (win=1, draw=0.5, loss=0), compare online-net results on black
   and white, and make only a small, slow, mean-preserving adjustment to
   terminal win rewards. Hysteresis filters ordinary match noise.

   完整控制器状态（窗口 + 计数 + 倍率）现在随 checkpoint 持久化（V11），
   保存在 agent->side_balance，避免重启后 scale 重置为 1.0 与 replay 存量
   样本标尺不一致。结构定义与常量 SIDE_BALANCE_WINDOW 在 core.h。 */
enum {
    SIDE_BALANCE_UPDATE_INTERVAL = 100,
    SIDE_BALANCE_MIN_SAMPLES = 100
};

static float clamp_float(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static void side_balance_apply_scales(DQNAgent* agent) {
    set_terminal_win_reward_scales(agent->side_balance.reward_scale[0],
        agent->side_balance.reward_scale[1]);
}

static void side_balance_record(DQNAgent* agent, int online_side, int winner) {
    if (online_side < 0 || online_side > 1) return;
    SideBalanceController& sb = agent->side_balance;

    float score = 0.5f;
    if (winner == online_side + 1) score = 1.0f;
    else if (winner != 0) score = 0.0f;

    if (sb.size == SIDE_BALANCE_WINDOW) {
        const SideBalanceSample& old = sb.samples[sb.write_pos];
        if (old.online_side < 2) {
            sb.samples_per_side[old.online_side]--;
            sb.score_sum[old.online_side] -= old.score;
        }
    }
    else {
        sb.size++;
    }

    sb.samples[sb.write_pos] = {
        (unsigned char)online_side, score
    };
    sb.write_pos = (sb.write_pos + 1) % SIDE_BALANCE_WINDOW;
    sb.samples_per_side[online_side]++;
    sb.score_sum[online_side] += score;
    sb.since_update++;

    if (sb.since_update < SIDE_BALANCE_UPDATE_INTERVAL ||
        sb.samples_per_side[0] < SIDE_BALANCE_MIN_SAMPLES ||
        sb.samples_per_side[1] < SIDE_BALANCE_MIN_SAMPLES) return;
    sb.since_update = 0;

    const float black_rate = sb.score_sum[0] /
        (float)sb.samples_per_side[0];
    const float white_rate = sb.score_sum[1] /
        (float)sb.samples_per_side[1];
    const float difference = black_rate - white_rate;
    const float deadband = 0.05f;
    const float max_adjustment = 0.15f;
    const float controller_gain = 0.60f;
    const float smoothing = 0.20f;

    float desired_black = 1.0f;
    float desired_white = 1.0f;
    if (fabsf(difference) > deadband) {
        const float adjustment = clamp_float(
            controller_gain * (fabsf(difference) - deadband), 0.0f, max_adjustment);
        if (difference > 0.0f) {
            desired_black -= adjustment;
            desired_white += adjustment;
        }
        else {
            desired_black += adjustment;
            desired_white -= adjustment;
        }
    }

    /* At most three percentage points per update. Inside the deadband this
       same low-pass filter automatically restores both factors to 1.0. */
    sb.reward_scale[0] += smoothing *
        (desired_black - sb.reward_scale[0]);
    sb.reward_scale[1] += smoothing *
        (desired_white - sb.reward_scale[1]);
    sb.reward_scale[0] = clamp_float(sb.reward_scale[0], 0.85f, 1.15f);
    sb.reward_scale[1] = clamp_float(sb.reward_scale[1], 0.85f, 1.15f);
    if (fabsf(sb.reward_scale[0] - 1.0f) < 0.001f)
        sb.reward_scale[0] = 1.0f;
    if (fabsf(sb.reward_scale[1] - 1.0f) < 0.001f)
        sb.reward_scale[1] = 1.0f;
    side_balance_apply_scales(agent);

    printf("[SIDE_BALANCE] online score B=%.1f%%(%d) W=%.1f%%(%d), win reward x%.3f/x%.3f\n",
        black_rate * 100.0f, sb.samples_per_side[0],
        white_rate * 100.0f, sb.samples_per_side[1],
        sb.reward_scale[0], sb.reward_scale[1]);
}

static void maybe_publish_probe_pretrain(DQNAgent* agent, uint64_t episode) {
    if (probe_schedule_pretrain(&g_probe_publish_schedule,
        neural_probe_enabled(), GetTickCount64())) neural_probe_publish(agent, episode);
}

static void maybe_publish_probe_formal(DQNAgent* agent, uint64_t episode) {
    if (probe_schedule_formal_step(&g_probe_publish_schedule,
        neural_probe_enabled())) neural_probe_publish(agent, episode);
}

#define INTENTIONAL_SHUTDOWN_FILE "intentional_shutdown.flag"
#define TRAINING_FAULT_FILE "training_fault.flag"

static void mark_intentional_shutdown(void) {
    FILE* f = fopen("intentional_shutdown.tmp", "wb");
    if (!f) return;
    fputs("trainer graceful shutdown\n", f);
    fflush(f);
    fclose(f);
    MoveFileExA("intentional_shutdown.tmp", INTENTIONAL_SHUTDOWN_FILE,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

static void mark_training_fault_shutdown(void) {
    FILE* f = fopen("training_fault.tmp", "wb");
    if (!f) return;
    fprintf(f, "trainer fault reason=%ld ep=%d step=%lld\n",
        (long)InterlockedCompareExchange(&g_training_fault, 0, 0),
        g_ctx.total_games_played,
        (long long)(g_current_agent ? g_current_agent->global_step : -1));
    fflush(f);
    _commit(_fileno(f));
    fclose(f);
    MoveFileExA("training_fault.tmp", TRAINING_FAULT_FILE,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

// Signals may interrupt an update or CRT file operation. Never enter stdio,
// configuration or mutable memory caches here; the training owner handles them.
static void signal_handler(int sig) {
    g_pending_exit_signal = sig;
    g_exit_requested = 1;
}

// ===================== 自有终端窗口（NoTerminal：训练输出显示，效果同控制台） =====================
#define TERM_LINES 200
#define TERM_COLS  160
static HWND g_hwnd = NULL;
static HFONT g_term_font = NULL;               /* 窗口线程创建，WM_PAINT 复用（不每帧创建） */
static void request_graceful_exit(void) {
    if (InterlockedCompareExchange(&g_training_fault, 0, 0) != 0) {
        /* Numeric/device/I/O failures are not user-requested stops.  Keep a
           separate latch so Watchdog reports the fault instead of silently
           treating it as a graceful shutdown or entering a restart loop. */
        mark_training_fault_shutdown();
        InterlockedExchange(&g_shutdown_state, 5);
    }
    else {
        mark_intentional_shutdown();
        InterlockedCompareExchange(&g_shutdown_state, 1, 3);
    }
    g_exit_requested = 1;
}
// Call only on the training owner, at a safe boundary or while awaiting a save
// retry. The intentional-stop marker must exist before a successful exit so
// Watchdog does not interpret SIGINT/SIGTERM as a crash and restart training.
static void service_signal_exit_request(void) {
    if (!g_pending_exit_signal) return;
    g_pending_exit_signal = 0;
    request_graceful_exit();
}
static char g_term[TERM_LINES][TERM_COLS + 1];
static int g_term_n = 0;                        /* 已存逻辑行数 */
static int g_term_head = 0;                     /* 环形缓冲写头 */
static CRITICAL_SECTION g_term_lock;



static FILE* text_log = NULL;
static FILE* reward_bin = NULL;
static FILE* epsilon_bin = NULL;
static FILE* time_bin = NULL;
static FILE* eval_win_bin = NULL;
static FILE* eval_detail_bin = NULL;

static bool truncate_stream(FILE* stream, int64_t size) {
    if (!stream || size < 0 || fflush(stream) != 0 ||
        _chsize_s(_fileno(stream), size) != 0 || _fseeki64(stream, 0, SEEK_END) != 0) return false;
    clearerr(stream);
    return true;
}

static bool append_bytes_transaction(FILE* stream, const void* bytes, size_t size) {
    if (!stream || (!bytes && size)) return false;
    if (fflush(stream) != 0 || _fseeki64(stream, 0, SEEK_END) != 0) return false;
    const int64_t start = _ftelli64(stream);
    if (start < 0) return false;
    const bool ok = (size == 0 || fwrite(bytes, 1, size, stream) == size) && fflush(stream) == 0;
    if (ok) return true;
    truncate_stream(stream, start);
    return false;
}

static bool append_file_transaction(FILE* destination, FILE* staged) {
    if (!destination || !staged || fflush(staged) != 0 || ferror(staged) ||
        _fseeki64(staged, 0, SEEK_SET) != 0 || fflush(destination) != 0 ||
        _fseeki64(destination, 0, SEEK_END) != 0) return false;
    const int64_t start = _ftelli64(destination);
    if (start < 0) return false;
    char buffer[64 * 1024];
    bool ok = true;
    for (;;) {
        const size_t got = fread(buffer, 1, sizeof(buffer), staged);
        if (got && fwrite(buffer, 1, got, destination) != got) { ok = false; break; }
        if (got < sizeof(buffer)) { if (ferror(staged)) ok = false; break; }
    }
    if (ok && fflush(destination) == 0) return true;
    truncate_stream(destination, start);
    return false;
}

struct EvalWinRecord { int32_t ep, step; float fields[4]; };
struct EvalDetailRecord { int32_t ep, step; float fields[7]; };
static bool append_eval_metric_transaction(FILE* eval_text, const char* text, size_t text_size,
    const EvalWinRecord win_rows[3], const EvalDetailRecord detail_rows[3]) {
    if (!eval_win_bin)
        eval_win_bin = metric_log_open_append(dqn_log_path("metric_win.bin").c_str(), 4);
    if (!eval_detail_bin)
        eval_detail_bin = metric_log_open_append(dqn_log_path("metric_eval.bin").c_str(), 7);
    if (!eval_win_bin || !eval_detail_bin || (eval_text && fflush(eval_text) != 0) ||
        fflush(eval_win_bin) != 0 || fflush(eval_detail_bin) != 0 ||
        (eval_text && _fseeki64(eval_text, 0, SEEK_END) != 0) || _fseeki64(eval_win_bin, 0, SEEK_END) != 0 ||
        _fseeki64(eval_detail_bin, 0, SEEK_END) != 0) return false;
    const int64_t text_start = eval_text ? _ftelli64(eval_text) : 0;
    const int64_t win_start = _ftelli64(eval_win_bin);
    const int64_t detail_start = _ftelli64(eval_detail_bin);
    if (text_start < 0 || win_start < 0 || detail_start < 0) return false;
    const bool text_ok = !eval_text || fwrite(text, 1, text_size, eval_text) == text_size;
    const bool ok = text_ok && fwrite(win_rows, sizeof(EvalWinRecord), 3, eval_win_bin) == 3 &&
        fwrite(detail_rows, sizeof(EvalDetailRecord), 3, eval_detail_bin) == 3 &&
        (!eval_text || fflush(eval_text) == 0) && fflush(eval_win_bin) == 0 && fflush(eval_detail_bin) == 0;
    if (ok) return true;
    const bool rolled_text = !eval_text || truncate_stream(eval_text, text_start);
    const bool rolled_win = truncate_stream(eval_win_bin, win_start);
    const bool rolled_detail = truncate_stream(eval_detail_bin, detail_start);
    (void)rolled_text; (void)rolled_win; (void)rolled_detail;
    return false;
}
static bool close_training_logs(DQNAgent* agent) {
    FILE** handles[] = { &agent->log_game, &agent->log_train, &agent->log_steps_bin,
        &agent->log_game_bin, &agent->log_loss_bin, &agent->log_eps_bin,
        &text_log, &reward_bin, &epsilon_bin, &time_bin,
        &eval_win_bin, &eval_detail_bin };
    bool ok = true;
    for (FILE** handle : handles) if (*handle) {
        if (fclose(*handle) != 0) ok = false;
        *handle = NULL;
    }
    return ok;
}
static void log_episode_rewards(int episode, float epsilon, const float raw[7], const float weights[7]) {
    if (!text_log) text_log = fopen(dqn_log_path("reward_episode.dat").c_str(), "a");
    if (!reward_bin)
        reward_bin = metric_log_open_append(dqn_log_path("metric_rewards.bin").c_str(), 8);
    if (!epsilon_bin)
        epsilon_bin = metric_log_open_append(dqn_log_path("metric_epsilon_episode.bin").c_str(), 1);
    float actual[7], total = 0.0f;
    for (int i = 0; i < 7; i++) { actual[i] = raw[i] * (i == 3 ? 1.0f : weights[i]); total += actual[i]; }
    char line[2048];
    int used = snprintf(line, sizeof(line), "%d %.6f", episode, (double)epsilon);
    for (int i = 0; i < 7 && used > 0 && used < (int)sizeof(line); i++)
        used += snprintf(line + used, sizeof(line) - used, " %.6f", (double)raw[i]);
    for (int i = 0; i < 7 && used > 0 && used < (int)sizeof(line); i++)
        used += snprintf(line + used, sizeof(line) - used, " %.6f", (double)(i == 3 ? 1.0f : weights[i]));
    for (int i = 0; i < 7 && used > 0 && used < (int)sizeof(line); i++)
        used += snprintf(line + used, sizeof(line) - used, " %.6f", (double)actual[i]);
    if (used > 0 && used < (int)sizeof(line))
        used += snprintf(line + used, sizeof(line) - used, " %.6f\n", (double)total);
    struct RewardRow { int32_t ep, step; float fields[8]; } reward_row = { (int32_t)episode, 0, {0} };
    struct EpsilonRow { int32_t ep, step; float value; } epsilon_row = { (int32_t)episode, 0, epsilon };
    for (int i = 0; i < 7; ++i) reward_row.fields[i] = actual[i];
    reward_row.fields[7] = total;
    bool ok = text_log && reward_bin && epsilon_bin && used > 0 && used < (int)sizeof(line) &&
        fflush(text_log) == 0 && fflush(reward_bin) == 0 && fflush(epsilon_bin) == 0 &&
        _fseeki64(text_log, 0, SEEK_END) == 0 && _fseeki64(reward_bin, 0, SEEK_END) == 0 &&
        _fseeki64(epsilon_bin, 0, SEEK_END) == 0;
    const int64_t text_start = ok ? _ftelli64(text_log) : -1;
    const int64_t reward_start = ok ? _ftelli64(reward_bin) : -1;
    const int64_t epsilon_start = ok ? _ftelli64(epsilon_bin) : -1;
    ok = ok && text_start >= 0 && reward_start >= 0 && epsilon_start >= 0 &&
        fwrite(line, 1, (size_t)used, text_log) == (size_t)used &&
        fwrite(&reward_row, sizeof(reward_row), 1, reward_bin) == 1 &&
        fwrite(&epsilon_row, sizeof(epsilon_row), 1, epsilon_bin) == 1 &&
        fflush(text_log) == 0 && fflush(reward_bin) == 0 && fflush(epsilon_bin) == 0;
    if (!ok && text_start >= 0 && reward_start >= 0 && epsilon_start >= 0) {
        truncate_stream(text_log, text_start);
        truncate_stream(reward_bin, reward_start);
        truncate_stream(epsilon_bin, epsilon_start);
    }
    if (!ok) { printf("[LOG] Episode reward transaction failed; training stopped.\n"); training_fail(3); }
}

/* 每局真实墙钟耗时。独立 schema 避免修改已有 rewards.bin 的记录宽度；
   TUI 启动时可直接恢复最近 200 局，而不是依赖在线 IPC 抽样。 */
static void log_episode_time(int episode, float seconds) {
    if (!time_bin)
        time_bin = metric_log_open_append(dqn_log_path("metric_episode_time.bin").c_str(), 1);
    if (!isfinite(seconds) || seconds < 0.0f) return;
    if (!time_bin) { training_fail(3); return; }
    struct { int32_t ep, step; float value; } rec = { (int32_t)episode, 0, seconds };
    if (!append_bytes_transaction(time_bin, &rec, sizeof(rec))) {
        printf("[LOG] Episode-time transaction failed; training stopped.\n");
        training_fail(3);
    }
}
static HANDLE g_pipe_read = NULL, g_pipe_write = NULL;

/* 追加一行到窗口文本缓冲（线程安全），并请求重绘 */
static void win_show_line(const char* s) {
    EnterCriticalSection(&g_term_lock);
    strncpy(g_term[g_term_head], s, TERM_COLS);
    g_term[g_term_head][TERM_COLS] = '\0';
    g_term_head = (g_term_head + 1) % TERM_LINES;
    if (g_term_n < TERM_LINES) g_term_n++;
    LeaveCriticalSection(&g_term_lock);
    if (g_hwnd) InvalidateRect(g_hwnd, NULL, FALSE);
}
/* 覆盖最后一行（终端 \r 效果：进度条原地刷新不滚动） */
static void win_overwrite_line(const char* s) {
    EnterCriticalSection(&g_term_lock);
    if (g_term_n > 0) {
        int last = (g_term_head - 1 + TERM_LINES) % TERM_LINES;
        strncpy(g_term[last], s, TERM_COLS);
        g_term[last][TERM_COLS] = '\0';
    } else {
        strncpy(g_term[g_term_head], s, TERM_COLS);
        g_term[g_term_head][TERM_COLS] = '\0';
        g_term_head = (g_term_head + 1) % TERM_LINES;
        g_term_n = 1;
    }
    LeaveCriticalSection(&g_term_lock);
    if (g_hwnd) InvalidateRect(g_hwnd, NULL, FALSE);
}
/* 消息泵（训练循环/主循环周期调用） */
static void win_pump_messages(void) {
    service_signal_exit_request();
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}
/* 供 IPC CONSOLE_SHOW/HIDE 控制前台显示 */
void win_show_window(int show) {
    if (g_hwnd) ShowWindow(g_hwnd, show ? SW_SHOW : SW_HIDE);
}

static LRESULT CALLBACK term_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        RECT rc; GetClientRect(hwnd, &rc);
        HBRUSH bg = CreateSolidBrush(RGB(12, 12, 12));
        FillRect(dc, &rc, bg);
        DeleteObject(bg);
        HFONT old = (HFONT)SelectObject(dc, g_term_font);
        SetTextColor(dc, RGB(0x39, 0xC5, 0xBB));
        SetBkMode(dc, TRANSPARENT);
        EnterCriticalSection(&g_term_lock);
        int total = g_term_n;
        int start = (total > 40) ? (total - 40) : 0;   /* 显示最近 40 行 */
        int cy = 4;
        wchar_t wbuf[TERM_COLS * 2 + 4];
        for (int i = start; i < total && cy < rc.bottom - 4; i++) {
            int idx = (g_term_head - total + i + TERM_LINES * 2) % TERM_LINES;
            int wlen = MultiByteToWideChar(CP_UTF8, 0, g_term[idx], -1, wbuf, TERM_COLS * 2);
            if (wlen > 1) TextOutW(dc, 4, cy, wbuf, wlen - 1);   /* UTF-8 → UTF-16，中文清晰无毛边 */
            cy += 18;
        }
        LeaveCriticalSection(&g_term_lock);
        SelectObject(dc, old);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_CLOSE:
        mark_intentional_shutdown();
        g_exit_requested = 1;   /* 训练循环检测到后保存 checkpoint 再退出 */
        /* 关闭显示窗口不等于强杀进程；训练主线程会在本局结束后同步 GPU、
           原子保存并自然退出。不要在 UI 线程阻塞 30 秒。 */
        DestroyWindow(hwnd);
        PostQuitMessage(0);     /* 结束窗口线程消息循环 */
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

/* 窗口线程：创建窗口 + 独立消息循环（窗口始终响应，刷新不卡顿、不改变大小） */
static void win_thread(void*) {
    WNDCLASSW wc = { 0 };
    wc.lpfnWndProc = term_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.lpszClassName = L"DQNTrainerTermWindow";
    /* BUG-066: 检查初始化 API 返回值，失败时输出诊断信息 */
    if (!RegisterClassW(&wc)) { printf("[WIN] RegisterClassW 失败: %lu\n", GetLastError()); return; }
    g_term_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, FIXED_PITCH, L"Cascadia Mono");
    if (!g_term_font) { printf("[WIN] CreateFontW 失败，回退系统字体\n"); g_term_font = (HFONT)GetStockObject(SYSTEM_FIXED_FONT); }
    g_hwnd = CreateWindowW(L"DQNTrainerTermWindow", L"DQN Trainer  (NoTerminal)",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) { printf("[WIN] CreateWindowW 失败: %lu\n", GetLastError()); return; }
    ShowWindow(g_hwnd, SW_SHOW);   /* 默认显示；TUI 发 CONSOLE_HIDE 可隐藏 */
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    /* 窗口关闭后线程退出；训练主线程检测 g_exit_requested 保存退出 */
}
static void win_init(void) {
    InitializeCriticalSection(&g_term_lock);
    _beginthread(win_thread, 0, NULL);
}

/* stdout/stderr → 匿名 pipe → 窗口读线程（只显示到窗口，不落盘日志文件）。
   解析 \r（覆盖最后一行，进度原地刷新）与 \n（追加行）。 */
static void stdout_thread(void*) {
    char buf[2048];
    DWORD n;
    char line[TERM_COLS + 1]; int li = 0; int overwrite = 0;
    while (ReadFile(g_pipe_read, buf, sizeof buf - 1, &n, NULL) && n > 0) {
        buf[n] = '\0';
        for (char* p = buf; *p; p++) {
            if (*p == '\n') {
                line[li] = '\0';
                if (overwrite) win_overwrite_line(line); else win_show_line(line);
                li = 0; overwrite = 0;
            } else if (*p == '\r') {
                overwrite = 1; li = 0;   /* 回车：覆盖当前行 */
            } else if (li < TERM_COLS) {
                line[li++] = *p;
            }
        }
    }
}

/* 创建匿名 pipe 并把 stdout/stderr 重定向到写端（读端由 stdout_thread 消费）。
   Win32 无控制台时 CRT stdout 初始无效，先建隐藏控制台 + freopen CONOUT$ 使其有效，再 _dup2 到 pipe。 */
static void redirect_stdout_to_pipe(void) {
    if (!GetConsoleWindow()) {
        AllocConsole();
        ShowWindow(GetConsoleWindow(), SW_HIDE);   /* 隐藏控制台窗口（仅用于让 stdout 有效） */
        FILE* fo = NULL;
        freopen_s(&fo, "CONOUT$", "w", stdout);
        freopen_s(&fo, "CONOUT$", "w", stderr);
    }
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    if (!CreatePipe(&g_pipe_read, &g_pipe_write, &sa, 0)) return;
    SetHandleInformation(g_pipe_write, HANDLE_FLAG_INHERIT, 0);
    _beginthread(stdout_thread, 0, NULL);
    int fd = _open_osfhandle((intptr_t)g_pipe_write, _O_WRONLY);
    if (fd >= 0) {
        _dup2(fd, _fileno(stdout));
        _dup2(fd, _fileno(stderr));
        _close(fd);
    }
}

// Windows 崩溃处理：写 crash.log + minidump
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
static LONG WINAPI crash_handler(EXCEPTION_POINTERS* exInfo) {
    /* 1. 文本日志 */
    FILE* f = fopen(LOG_DIR "crash.log", "a");
    if (f) {
        fprintf(f, "\n=== CRASH ===\n");
        fprintf(f, "Exception code: 0x%08X\n", exInfo->ExceptionRecord->ExceptionCode);
        fprintf(f, "Exception address: %p\n", exInfo->ExceptionRecord->ExceptionAddress);
        fprintf(f, "RIP: 0x%llX\n", (unsigned long long)exInfo->ContextRecord->Rip);
        fprintf(f, "RSP: 0x%llX\n", (unsigned long long)exInfo->ContextRecord->Rsp);
        fprintf(f, "RBP: 0x%llX\n", (unsigned long long)exInfo->ContextRecord->Rbp);
        fprintf(f, "Total games: %d\n", g_ctx.total_games_played);
        fflush(f);
        fclose(f);
    }
    /* 2. Minidump（完整调栈 + 局部变量） */
    HANDLE hFile = CreateFileA(LOG_DIR "crash.dmp", GENERIC_WRITE, 0,
        NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mdei = {};
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = exInfo;
        mdei.ClientPointers = FALSE;
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
            hFile, MiniDumpWithFullMemory, &mdei, NULL, NULL);
        CloseHandle(hFile);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// 种群训练多进度条
static volatile int g_progress_start_line = 0;

// 训练时间（供 IPC 状态推送，TUI 显示真实训练进度）
float g_ep_time = 0, g_elapsed = 0, g_remaining = 0;

/* 训练分阶段累计计时 (微秒/步, 每局累加, IPC 推送后重置) */
volatile double g_phase_forward_us = 0, g_phase_backward_us = 0, g_phase_target_us = 0;
volatile int g_phase_steps = 0;
/* 按方分阶段计时 (Black=0, White=1) */
volatile double g_side_forward_us[2] = {0, 0};
volatile double g_side_backward_us[2] = {0, 0};
volatile int g_side_steps[2] = {0, 0};
typedef struct {
    volatile int local_ep;
    volatile int save_interval;
    volatile double episode_time;
    volatile int active;  // 0=未启动, 1=运行中, 2=已完成
} AgentProgress;
static AgentProgress g_agent_progress[8];

// ============================================================================
// 种群训练进度持久化（训练程序特有）
// ============================================================================

void save_population_progress(const PopulationProgress* p) {
    const char* tmp = "population_progress.bin.tmp";
    FILE* f = fopen(tmp, "wb");
    if (!f) return;
    int ok = fwrite(p, sizeof(PopulationProgress), 1, f) == 1 && fflush(f) == 0;
    if (ok) ok = (_commit(_fileno(f)) == 0);
    if (fclose(f) != 0) ok = 0;
    if (!ok || !MoveFileExA(tmp, "population_progress.bin", MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) remove(tmp);
}

int load_population_progress(PopulationProgress* p) {
    FILE* f = fopen("population_progress.bin", "rb");
    if (!f) {
        p->round = 0; p->agent_idx = -1; p->completed = 0;
        return 0;
    }
    if (fread(p, sizeof(PopulationProgress), 1, f) != 1) {
        fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

void scan_population_progress(PopulationProgress* p) {
    p->round = 0;
    p->agent_idx = -1;
    p->completed = 0;

    int max_ep_per_agent[POP_SIZE];
    int min_ep = 1000000;
    int any_agent_has_checkpoint = 0;

    for (int i = 0; i < POP_SIZE; i++) {
        max_ep_per_agent[i] = -1;
        char pattern[64];
        snprintf(pattern, sizeof(pattern), "checkpoint_seed%d_*.bin", i);
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA(pattern, &findData);
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                const char* last_underscore = strrchr(findData.cFileName, '_');
                if (last_underscore) {
                    int ep;
                    if (sscanf(last_underscore + 1, "%d", &ep) == 1) {
                        if (ep > max_ep_per_agent[i]) {
                            max_ep_per_agent[i] = ep;
                        }
                    }
                }
            } while (FindNextFileA(hFind, &findData));
            FindClose(hFind);
            if (max_ep_per_agent[i] >= 0) {
                any_agent_has_checkpoint = 1;
                if (max_ep_per_agent[i] < min_ep) {
                    min_ep = max_ep_per_agent[i];
                }
            }
        }
    }

    if (!any_agent_has_checkpoint) return;

    int completed_rounds = min_ep / ROUND_EPISODES;
    p->round = completed_rounds;

    int target_ep_for_round = (completed_rounds + 1) * ROUND_EPISODES;
    int first_incomplete_agent = -1;
    int all_completed_current_round = 1;

    for (int i = 0; i < POP_SIZE; i++) {
        if (max_ep_per_agent[i] < target_ep_for_round) {
            all_completed_current_round = 0;
            if (first_incomplete_agent == -1) {
                first_incomplete_agent = i;
            }
        }
    }

    if (all_completed_current_round) {
        p->round++;
        if (p->round >= TOTAL_ROUNDS) {
            p->completed = 1;
            return;
        }
        p->agent_idx = -1;
    }
    else {
        p->agent_idx = first_incomplete_agent - 1;
    }
}

int find_latest_seed_checkpoint(int agent_idx, char* filename, size_t size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "checkpoint_seed%d_*.bin", agent_idx);
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(pattern, &findData);
    if (hFind == INVALID_HANDLE_VALUE) return 0;

    int best_ep = -1;
    char best[256] = "";
    do {
        const char* last_underscore = strrchr(findData.cFileName, '_');
        if (last_underscore) {
            int ep;
            if (sscanf(last_underscore + 1, "%d", &ep) == 1 && ep > best_ep) {
                best_ep = ep;
                snprintf(best, sizeof(best), "%s", findData.cFileName);
            }
        }
    } while (FindNextFileA(hFind, &findData));

    FindClose(hFind);
    if (best_ep >= 0) {
        strncpy(filename, best, size - 1);
        filename[size - 1] = '\0';
        return 1;
    }
    return 0;
}

// ============================================================================
// 线程私有变量（OpenMP 种群训练用）
// ============================================================================
static int g_thread_local_wins[3] = { 0, 0, 0 };
#pragma omp threadprivate(g_thread_local_wins)

static int g_thread_hard_counter = 0;
#pragma omp threadprivate(g_thread_hard_counter)

// ============================================================================
// 辅助函数（训练程序特有）
// ============================================================================

// 种群训练：绘制一帧（单次 fwrite，调用者负责互斥）
static void draw_progress_frame(int num_agents) {
    if (g_progress_start_line <= 0) return;

    const int BAR_WIDTH = 30;
    char buf[8 * 160 + 64];
    int off = 0;

    for (int i = 0; i < num_agents; i++) {
        AgentProgress* prog = &g_agent_progress[i];
        int line = g_progress_start_line + i;
        off += snprintf(buf + off, sizeof(buf) - off, "\033[%d;1H\033[2K", line);

        if (prog->active == 0) {
            off += snprintf(buf + off, sizeof(buf) - off, "  Agent %d: [%-30s] 等待中...", i, "");
        }
        else if (prog->active == 2) {
            off += snprintf(buf + off, sizeof(buf) - off, "  Agent %d: [##############################] \033[32m完成\033[0m", i);
        }
        else {
            int le = prog->local_ep;
            int si = prog->save_interval;
            float pct = (si > 0) ? (float)le / si * 100.0f : 0.0f;
            int filled = (int)((float)le / si * BAR_WIDTH);
            if (filled > BAR_WIDTH) filled = BAR_WIDTH;

            char bar[40];
            memset(bar, '#', filled);
            memset(bar + filled, '-', BAR_WIDTH - filled);
            bar[BAR_WIDTH] = '\0';

            off += snprintf(buf + off, sizeof(buf) - off, "  Agent %d: [%s] %5.1f%% (%d/%d) %.3fs/局",
                i, bar, pct, le, si, prog->episode_time);
        }
    }
    off += snprintf(buf + off, sizeof(buf) - off, "\033[%d;1H", g_progress_start_line + num_agents);

    fwrite(buf, 1, off, stdout);
    fflush(stdout);
}

// 种群训练：初始化进度条区域
static void init_population_progress(int num_agents, int round) {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }

    printf("\n");
    printf("  \033[36m═══════ 种群训练 Round %d ═══════\033[0m\n", round);

    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hOut, &csbi)) {
        g_progress_start_line = csbi.dwCursorPosition.Y + 1;
    }

    for (int i = 0; i < num_agents; i++) {
        memset(&g_agent_progress[i], 0, sizeof(AgentProgress));
    }

    draw_progress_frame(num_agents);
}
static int select_near_existing(const GameState* g, int moves[], int cnt) {
    int best_idx = 0; float best_dist = 1e9f;
    for (int i = 0; i < cnt; i++) {
        int pos = moves[i], r = pos / BOARD_SIZE, c = pos % BOARD_SIZE;
        float min_dist = 1e9f;
        for (int rr = 0; rr < BOARD_SIZE; rr++)
            for (int cc = 0; cc < BOARD_SIZE; cc++) {
                if (g->board[rr * BOARD_SIZE + cc] != 0) {
                    float d = sqrtf((float)((rr - r) * (rr - r) + (cc - c) * (cc - c)));
                    if (d < min_dist) min_dist = d;
                }
            }
        if (min_dist == 1e9f) return moves[rand() % cnt];
        if (min_dist < best_dist) { best_dist = min_dist; best_idx = i; }
    }
    return moves[best_idx];
}

static int select_far_existing(const GameState* g, int moves[], int cnt) {
    int best_idx = 0; float best_dist = -1e9f;
    for (int i = 0; i < cnt; i++) {
        int pos = moves[i], r = pos / BOARD_SIZE, c = pos % BOARD_SIZE;
        float min_dist = 1e9f;
        for (int rr = 0; rr < BOARD_SIZE; rr++)
            for (int cc = 0; cc < BOARD_SIZE; cc++) {
                if (g->board[rr * BOARD_SIZE + cc] != 0) {
                    float d = sqrtf((float)((rr - r) * (rr - r) + (cc - c) * (cc - c)));
                    if (d < min_dist) min_dist = d;
                }
            }
        if (min_dist == 1e9f) return moves[rand() % cnt];
        if (min_dist > best_dist) { best_dist = min_dist; best_idx = i; }
    }
    return moves[best_idx];
}

void evaluate_against_minimax_eps(DQNAgent* agent, int games, int depth, int dq_color,
    float eps, int* wins, int* draws, int* losses) {
    *wins = *draws = *losses = 0;
    float orig = agent->epsilon;
    agent->epsilon = eps;
    for (int g = 0; g < games; g++) {
        GameState game; reset_game(&game);
        int completed = 0;
        for (int step = 0; step < CELL_COUNT; ++step) {
            int turn = game.turn, action = -1;
            if (turn == dq_color)
                action = select_action(agent, &game, 0, eps, &agent->eval_rng, 0);
            else
                action = minimax_best_move(&game, depth, 0.0f, &agent->eval_rng);
            if (g_training_fault || !is_legal_move(&game, action)) {
                if (!g_training_fault) training_fail(2);
                agent->epsilon = orig;
                return;
            }
            make_move(&game, action);
            int winner = check_winner(&game);
            if (winner) {
                if ((winner == 1 && dq_color == 0) || (winner == 2 && dq_color == 1))
                    (*wins)++;
                else (*losses)++;
                completed = 1;
                break;
            }
            if (is_full(&game)) { (*draws)++; completed = 1; break; }
        }
        if (!completed) {
            training_fail(2);
            agent->epsilon = orig;
            return;
        }
    }
    agent->epsilon = orig;
}

// ============================================================================
// 单局训练 play_episode
// ============================================================================
static void play_episode(DQNAgent* agent, int ep, int global_ep, int phase,
    int online_player, float minimax_prob, float pool_prob, int train_enabled,
    int target_mode, int first_move_enabled,
    int use_separate_buffer, int total_episodes,
    TrainingContext* ctx, int is_population) {
    const CompetencyConfig course=g_hot_config.competency;
    const unsigned stage=agent->competency.stage;
    bool minimax_game=course.enabled && stage>0 && stage<3 &&
        prng_float(&agent->train_rng)<course.minimax_ratio[stage];
    if(course.enabled && !is_population && stage<3 && !minimax_game){
        const int side=ep%2;
        int rc=competency_skill_batch(agent,side,0,g_learning_rate);
        if(rc==0){training_fail_detail(2,"competency_solo",side);return;}
        if(rc==1){agent->global_step+=BATCH_SIZE;if(update_target_side(agent,side,1)!=0)training_fail(1);}
        return;
    }
    GameState game;
    reset_game(&game);
    StepOutcomeScope outcome_scope(g_hot_config.step_outcome);
    TerminalOutcomeCredit outcome_credit;
    int step = 0;
    float first_state[CELL_COUNT + 1];
    int first_action = -1;
    int first_player = -1;

    const int MAX_STEPS = 50;
    int step_counter = 0;

    // Memory system buffers for both perspectives
    int x_boards[MAX_EPISODE_STEPS][CELL_COUNT];
    int x_piece_counts[MAX_EPISODE_STEPS];
    int x_steps[MAX_EPISODE_STEPS];
    int x_step_count = 0;

    int o_boards[MAX_EPISODE_STEPS][CELL_COUNT];
    int o_piece_counts[MAX_EPISODE_STEPS];
    int o_steps[MAX_EPISODE_STEPS];
    int o_step_count = 0;

    // 种群训练时使用线程私有 wins 统计
    if (is_population && ep == 0) {
        g_thread_local_wins[0] = g_thread_local_wins[1] = g_thread_local_wins[2] = 0;
        g_thread_hard_counter = 0;
    }

    /* 训练分阶段计时 (QPC 微秒精度, 每局累加, 非 static 以便 IPC 读取) */
    LARGE_INTEGER pf; QueryPerformanceFrequency(&pf);

    while (step_counter < MAX_STEPS && !g_training_fault) {
        service_snapshot_requests(agent);
        int player = game.turn;
        int action = -1;
        LARGE_INTEGER t_fwd_start; QueryPerformanceCounter(&t_fwd_start);

        /* ============================================================================
           自博弈已收敛为「两个独立玩家」：轮到谁就用**谁自己的 online** 选招。
           ----------------------------------------------------------------------------
           旧实现在这里分叉成「本人分支 / 对手分支」，对手分支再选 minimax / 历史池 /
           opponent_net。那条路径使「两网络 vs 一网络」的效果与「自博弈 vs 对手」的效果
           纠缠在一起，无法归因。现在：
             * 按方路由生效（本次测试的机制）⇒ 一条分支，轮到谁用谁的 online；
             * 未生效（共享网络基线）⇒ 保留历史的分支结构，保持基线可跑且逐位不变。
           minimax 保留为**独立评测**用途（evaluate_* 那条路径），不再是训练对手。
           ============================================================================ */
        if(course.enabled && !is_population && player!=online_player && minimax_game){
            action=minimax_best_move(&game,1,0.f,&agent->train_rng);
        }else if(course.enabled && !is_population && player!=online_player && (stage==3 || stage==4)){
            if(!agent->competency_opponents || !agent->competency.snapshots_ready){training_fail_detail(2,"competency_opponent_missing",player);return;}
            NeuralNet* opponent=stage==3?&agent->competency_opponents->frozen[player]:&agent->competency_opponents->lagged[player];
            action=select_action_with_net(agent,&game,opponent,0.f,&agent->train_rng);
        }else if (independent_training::indep_active(agent)) {
            NeuralNet* actor = independent_training::agent_online_net(agent,
                independent_training::side_from_physical(player));
            if (!actor) { training_fail(2); return; }
            action = select_action_with_net(agent, &game, actor, -1.0f, &agent->train_rng);
        }
        else if (player == online_player) {
            action = select_action(agent, &game, 0, -1.0f, &agent->train_rng, 1);
        }
        else {
            int use_minimax = prng_float(&agent->train_rng) < minimax_prob;
            if (use_minimax) {
                action = minimax_best_move(&game, global_minimax_depth, 0.0f,
                    &agent->train_rng);
            }
            else {
                /* D3（受限版）：独立模式下历史池**单槽且被 checkpoint 的 pool[0]
                   钉死**（size==1，提升它会让既有存档的 pool 段校验全部失败），
                   因此池里只可能有一条、且来源曾是恒定的共享 target_net。这里
                   改为使用**按方冻结对手**：它是该物理棋色自己的副本、且由上面
                   的周期刷新推进，来源明确（HANDOFF:72 要的就是这个）。真正的
                   「双方各自保留多条历史池」需要独立的格式迁移步骤，仍记为未闭合。 */
                int use_pool = prng_float(&agent->train_rng) < pool_prob;
                if (independent_training::indep_active(agent)) {
                    NeuralNet* opp = independent_training::agent_opponent_net(agent,
                        independent_training::side_from_physical(player));
                    if (opp) {
                        action = select_action_with_net(agent, &game, opp, -1.0f,
                            &agent->train_rng);
                    }
                }
                else if (use_pool && agent->opponent_pool_count > 0) {
                    float use_latest = prng_float(&agent->train_rng);
                    NeuralNet* pool_net;
                    if (use_latest < 0.3f && agent->opponent_pool_count > 1) {
                        int latest_idx = opponent_pool_latest_index(agent->opponent_pool_next,
                            agent->opponent_pool_size);
                        pool_net = &agent->opponent_pool[latest_idx];
                    }
                    else {
                        pool_net = select_opponent_from_pool(agent, &agent->train_rng);
                    }
                    action = select_action_with_net(agent, &game, pool_net, -1.0f,
                        &agent->train_rng);
                }
                else {
                    if (get_curriculum_phase(global_ep) == 0) {
                        int moves[CELL_COUNT];
                        int cnt = get_legal_moves(&game, moves);
                        if (cnt == 0) break;
                        action = moves[prng_int(&agent->train_rng, cnt)];
                    }
                    else {
                        /* D2：对手落子也必须走**该物理棋色自己的**冻结对手，
                           不能再用共享 opponent_net（独立模式下它是种子副本）。 */
                        NeuralNet* opp = independent_training::agent_opponent_net(agent,
                            independent_training::side_from_physical(player));
                        action = select_action_with_net(agent, &game,
                            opp ? opp : &agent->opponent_net, -1.0f, &agent->train_rng);
                    }
                }
            }
        }

        /* Forward 计时结束 (按当前玩家归因) */
        LARGE_INTEGER t_fwd_end; QueryPerformanceCounter(&t_fwd_end);
        double fwd_us = (double)(t_fwd_end.QuadPart - t_fwd_start.QuadPart) / (double)pf.QuadPart * 1e6;
        g_phase_forward_us += fwd_us;
        g_side_forward_us[player] += fwd_us;

        if (g_training_fault) return;
        if (!is_legal_move(&game, action)) {
            printf("[TRAIN] Invalid action %d at phase=%d episode=%d step=%d\n",
                action, phase, global_ep, step);
            training_fail(2);
            return; // never commit a partial episode to replay/statistics/logs
        }
        if (first_action == -1) {
            state_to_input(&game, first_state);
            first_action = action;
            first_player = player;
        }

        GameState next_game = game;
        make_move(&next_game, action);

        int curriculum_phase = get_curriculum_phase(global_ep);
        float reward = compute_reward(agent, &game, &next_game, action, player,
            curriculum_phase, ep, step);

        int done = 0;
        if (check_winner(&next_game) != 0 || is_full(&next_game)) done = 1;
        if (!done && is_full(&next_game)) done = 1;

        log_game_step(agent, global_ep, step, player, action, reward, done);

        // Record board state for both perspectives
        if (g_hot_config.memory_enabled) {
            int pc = 0;
            for (int j = 0; j < CELL_COUNT; j++)
                if (next_game.board[j] != 0) pc++;

            // X's perspective
            if (x_step_count < MAX_EPISODE_STEPS) {
                GameState x_game;
                normalize_state_to_self(&next_game, &x_game, 0);
                for (int j = 0; j < CELL_COUNT; j++)
                    x_boards[x_step_count][j] = x_game.board[j];
                x_piece_counts[x_step_count] = pc;
                x_steps[x_step_count] = step;
                x_step_count++;
            }
            // O's perspective
            if (o_step_count < MAX_EPISODE_STEPS) {
                GameState o_game;
                normalize_state_to_self(&next_game, &o_game, 1);
                for (int j = 0; j < CELL_COUNT; j++)
                    o_boards[o_step_count][j] = o_game.board[j];
                o_piece_counts[o_step_count] = pc;
                o_steps[o_step_count] = step;
                o_step_count++;
            }
        }

        // At end of game, update memory for both perspectives
        if (g_hot_config.memory_enabled && done) {
            int winner = check_winner(&next_game);
            int is_x_win = winner == 0 ? -1 : (winner == 1);
            int is_o_win = winner == 0 ? -1 : (winner == 2);
            // X's memory
            for (int s = 0; s < x_step_count; s++) {
                memory_cache_update(x_boards[s], is_x_win, global_ep,
                    x_steps[s], x_piece_counts[s], 0);
            }
            // O's memory
            for (int s = 0; s < o_step_count; s++) {
                memory_cache_update(o_boards[s], is_o_win, global_ep,
                    o_steps[s], o_piece_counts[s], 1);
            }
            // 每10局晋升一次缓存库到正式库
            if (global_ep % 10 == 0) {
                memory_promote_to_kb(global_ep);
                memory_prune(global_ep);
            }
        }

        float input[CELL_COUNT + 1], next_input[CELL_COUNT + 1];
        state_to_input(&game, input);
        state_to_input(&next_game, next_input);

        /* 第 4 步：独立网络模式下经验**必须**走两 ply 生产路径
           （本方决策 -> 对手真实回应 -> 本方下一个决策），否则会落进下面的
           一 ply 直写分支，收益传播契约立刻破损。因此门槛是
           「独立网络开启 或 (step_outcome_enabled && phase>=1)」。 */
        if (independent_training::indep_active(agent) ||
            (outcome_scope.config.enabled && curriculum_phase >= 1)) {
            outcome_credit.observe(agent,game,next_game,action,reward,curriculum_phase,ep);
        }
        else if (use_separate_buffer) {
            store_experience(agent, input, action, reward, next_input, done, player);
        }
        else {
            Experience* buf = (player == 0) ? agent->replay_black : agent->replay_white;
            int* idx = (player == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
            int* cnt = (player == 0) ? &agent->replay_black_count : &agent->replay_white_count;
            Experience* exp = &buf[*idx];
            memcpy(exp->state, input, sizeof(float) * (CELL_COUNT + 1));
            exp->action = action;
            exp->reward = reward;
            memcpy(exp->next_state, next_input, sizeof(float) * (CELL_COUNT + 1));
            exp->done = done;
            *idx = (*idx + 1) % REPLAY_CAPACITY;
            if (*cnt < REPLAY_CAPACITY) (*cnt)++;
        }

        game = next_game;

        /* TRAIN_STEP_EACH 原先只定义未使用，导致每一步都做两组 batch。
           按配置间隔训练，减少重复 GPU 调度且保持预期训练频率。 */
        int trained_this_step = train_enabled && (agent->step_counter % TRAIN_STEP_EACH == 0);
        LARGE_INTEGER t_bwd_start;
        if (trained_this_step) {
            QueryPerformanceCounter(&t_bwd_start);
            int repeat = g_hot_config.train_repeat_per_step;
            if (repeat < 1) repeat = 1;
            if (repeat > 20) repeat = 20;
            for (int r = 0; r < repeat && !g_training_fault; r++) {
                /* 方案 (b)：双方都训练，但**每个批次显式绑定它自己的物理棋色**。
                   绝不按「本局在线方」推断绑定——每步黑白两个缓冲都可能攒够
                   BATCH_SIZE，只有一个绑定必然让另一方命中 indep_side_not_bound
                   把训练闩死（实测证据 build_verify/indep_trainer_short/run3/）。
                   绑定失败立即放弃本次更新，绝不用旧绑定继续训练另一方。
                   切换本身负责把上一方的设备状态写回其影子，因此双方的
                   online/target/按方状态在切换过程中都保持完整。 */
                for (int side_id = 0; side_id < 2 && !g_training_fault; ++side_id) {
                    if(course.enabled && (stage==3 || stage==4) && side_id!=online_player)continue;
                    Experience* buf = (side_id == 0) ? agent->replay_black : agent->replay_white;
                    int* cnt = (side_id == 0) ? &agent->replay_black_count : &agent->replay_white_count;
                    int* idx = (side_id == 0) ? &agent->replay_black_idx : &agent->replay_white_idx;
                    if (*cnt < BATCH_SIZE) continue;
                    /* 按方 backward 计时 */
                    LARGE_INTEGER t_side_bwd; QueryPerformanceCounter(&t_side_bwd);
                    if (independent_training::indep_active(agent)) {
                        /* 四句柄同时驻留：本批不需要任何绑定/切换，句柄按 (side, which)
                           直取。这里保留一次 gpu_net_bind_side 调用只是为了延续
                           indep_bound 这条兼容记录与开销取样；它现在退化为
                           「校验该方可用 + 影子与句柄对齐」，不再搬运权重。 */
                        if (gpu_net_active()) {
                            LARGE_INTEGER bs, be, bf;
                            QueryPerformanceFrequency(&bf);
                            QueryPerformanceCounter(&bs);
                            const int bind_rc = gpu_net_bind_side(agent, side_id);
                            QueryPerformanceCounter(&be);
                            g_indep_bind_count++;
                            g_indep_bind_seconds +=
                                (double)(be.QuadPart - bs.QuadPart) / (double)bf.QuadPart;
                            if (bind_rc != 0) {
                                printf("[INDEP] 批次前校验 %s 失败；放弃本批更新。\n",
                                    side_id == 0 ? "black" : "white");
                                training_fail_detail(1, "gpu_bind_side_batch", side_id);
                                break;
                            }
                        }
                        agent->indep->set_bound(independent_training::side_from_physical(side_id));
                    }
                    replay_from_buffer(agent, buf, *cnt, idx);
                    /* 按方 backward 计时结束 */
                    LARGE_INTEGER t_side_bwd_end; QueryPerformanceCounter(&t_side_bwd_end);
                    g_side_backward_us[side_id] += (double)(t_side_bwd_end.QuadPart - t_side_bwd.QuadPart) / (double)pf.QuadPart * 1e6;
                    g_side_steps[side_id]++;
                }
            }
            /* Backward 计时结束 */
            LARGE_INTEGER t_bwd_end; QueryPerformanceCounter(&t_bwd_end);
            g_phase_backward_us += (double)(t_bwd_end.QuadPart - t_bwd_start.QuadPart) / (double)pf.QuadPart * 1e6;
        }

        agent->step_counter++;

        LARGE_INTEGER t_tgt_start; QueryPerformanceCounter(&t_tgt_start);
        if (target_mode == TARGET_UPDATE_HARD) {
            if (++g_thread_hard_counter % 1000 == 0) {
                if (independent_training::indep_active(agent)) {
                    /* 双方都训练，因此目标同步也按方做。四句柄驻留下
                       update_target_side 直接取该方自己的两个句柄，
                       不需要任何绑定/切换。 */
                    for (int side_id = 0; side_id < 2; ++side_id) {
                        const int rc = update_target_side(agent, side_id, 0);
                        if (rc != 0)
                            printf("[TARGET] hard update 失败（side=%d, rc=%d），该方同步已跳过\n",
                                side_id, rc);
                    }
                } else if (gpu_net_active()) {
                    if (gpu_net_hard_update() != 0)
                        printf("[TARGET] GPU hard update 失败，本次目标同步已跳过\n");
                } else {
                    copy_net(&agent->target_net, &agent->q_net);
                }
            }
            agent->target_hard_counter = g_thread_hard_counter;   /* IND-05：随 checkpoint 持久化 */
        }
        else {
            if (trained_this_step) {
                if (independent_training::indep_active(agent)) {
                    /* 软更新同样按方：每方只把**自己的** online 混入自己的 target，
                       直接取该方句柄，不需要绑定。 */
                    for (int side_id = 0; side_id < 2; ++side_id) {
                        if (update_target_side(agent, side_id, 1) != 0)
                            printf("[TARGET] CPU soft update 失败（side=%d），该方同步已跳过\n",
                                side_id);
                    }
                }
                else {
                    soft_update_target(agent);
                }
            }
        }
        /* Target sync 计时结束 */
        LARGE_INTEGER t_tgt_end; QueryPerformanceCounter(&t_tgt_end);
        g_phase_target_us += (double)(t_tgt_end.QuadPart - t_tgt_start.QuadPart) / (double)pf.QuadPart * 1e6;
        g_phase_steps++;

        if (!is_population)
            maybe_publish_probe_formal(agent, (uint64_t)global_ep + 1u);

        step++;
        step_counter++;
        if (done) break;
    }

    if (step_counter >= MAX_STEPS) {
        // 达到最大步数，强制结束（静默处理）
    }

    if (g_training_fault) return; // no win/reward statistics for an interrupted game
    /* 首步反思 */
    if (first_move_enabled && first_player == 0 && first_player == online_player) {
        int winner = check_winner(&game);
        float target_val = 0.0f;
        if (winner == 1) target_val = FIRST_MOVE_WIN_BONUS;
        else if (winner == 0) target_val = FIRST_MOVE_DRAW_BONUS;
        else target_val = -FIRST_MOVE_WIN_BONUS;
        if (target_val != 0.0f) {
            int idx = agent->first_move_buf_idx % FIRST_MOVE_BATCH_SIZE;
            memcpy(agent->first_move_buf[idx].state, first_state, sizeof(first_state));
            agent->first_move_buf[idx].action = first_action;
            agent->first_move_buf[idx].target = target_val;
            agent->first_move_buf_idx++;
            if (agent->first_move_buf_idx % FIRST_MOVE_BATCH_SIZE == 0) {
                float fm_states[FIRST_MOVE_BATCH_SIZE][CELL_COUNT + 1];
                int fm_actions[FIRST_MOVE_BATCH_SIZE];
                float fm_targets[FIRST_MOVE_BATCH_SIZE];
                for (int i = 0; i < FIRST_MOVE_BATCH_SIZE; i++) {
                    memcpy(fm_states[i], agent->first_move_buf[i].state, sizeof(fm_states[i]));
                    fm_actions[i] = agent->first_move_buf[i].action;
                    fm_targets[i] = agent->first_move_buf[i].target;
                }
                /* 首步反思的物理棋色就是 first_player。方案 (b) 要求**显式绑定**
                   该棋色后再提交，不按在线方推断。 */
                const float fm_lr = g_learning_rate * 0.1f;
                float fm_loss = 0.0f;
                int trained = 0;
                if (independent_training::indep_active(agent)) {
                    bool bind_ok = true;
                    if (gpu_net_active() && gpu_net_bind_side(agent, first_player) != 0) {
                        printf("[FIRST_MOVE] 绑定 side=%d 失败，放弃本批反思\n", first_player);
                        bind_ok = false;
                    } else {
                        agent->indep->set_bound(
                            independent_training::side_from_physical(first_player));
                    }
                    if (bind_ok) {
                        trained = train_agent_batch_side(agent, first_player, (const float*)fm_states,
                            fm_actions, fm_targets, FIRST_MOVE_BATCH_SIZE, fm_lr, &fm_loss);
                    }
                } else {
                    trained = train_agent_batch(agent, (const float*)fm_states,
                        fm_actions, fm_targets, FIRST_MOVE_BATCH_SIZE, fm_lr, &fm_loss);
                }
                if (trained == 1) agent->global_step += FIRST_MOVE_BATCH_SIZE;
                else if (trained == 0) printf("[FIRST_MOVE] GPU/CPU 回退均失败，本批反思已跳过\n");
                /* trained == 2 安全拒绝：不推进 global_step，静默跳过 */
            }
        }
    }

    int win = check_winner(&game);
    if(!is_population && !log_outcome_metrics(agent,game,global_ep,online_player,
        get_curriculum_phase(global_ep),outcome_scope.config))
        printf("[OUTCOME] Outcome metrics write failed at episode %d\n",global_ep);
    if (is_population) {
        /* Population workers retain their historic physical X/O counters. */
        if (win == 1) g_thread_local_wins[0]++;
        else if (win == 2) g_thread_local_wins[1]++;
        else g_thread_local_wins[2]++;
    }
    else {
        /* Main status W/L/D describes the online learner, whose physical side
           is randomized.  Counting X/O here made a strong white opponent look
           like a rapidly improving learner. */
        if (win == 0) g_wins[2]++;
        else if (win == online_player + 1) g_wins[0]++;
        else g_wins[1]++;
    }

    // Population workers do not mutate the process-wide side controller.
    if (!is_population) {
        side_balance_record(agent, online_player, win);
    }

    if ((ep + 1) % 100 == 0) {
        int snapshot_ok = 1;
        if (!sync_gpu_before_persist(agent)) {
            snapshot_ok = 0;
            printf("[OPPONENT] GPU 参数同步失败，保留上一版对手\n");
        }
        if (snapshot_ok) {
            if (independent_training::indep_active(agent)) {
                /* D2：独立模式下「以某方出场的冻结对手」= 该方自己的 target 网络
                   （`agent_opponent_net`），它由按方目标同步连续推进，因此这里
                   不需要再拷一份副本、也不再读共享 `q_net`（后者从播种起就不再
                   更新，等于永远对着预训练种子下棋）。
                   历史池**不在这里写入**：共享 `target_net` 在独立模式下是陈旧的
                   种子权重，把它塞进池里只会制造误导性条目。池的多方保留需要
                   独立的格式迁移，仍记为未闭合。 */
                printf("[OPPONENT] 独立模式：按方冻结对手 = 该方 target（随按方目标同步推进）；"
                    "共享 opponent_net / 历史池不参与\n");
            } else {
                copy_net(&agent->opponent_net, &agent->q_net);
                add_to_opponent_pool(agent);
            }
        }
    }
}

// ============================================================================
// 评估与保存检查点
// ============================================================================
static int evaluation_cancelled(void*) {
    win_pump_messages(); // UI and IPC requests must remain serviceable during evaluation.
    service_signal_exit_request();
    return g_exit_requested != 0;
}

static void evaluation_progressed(int completed, int total, void*) {
    InterlockedExchange(&g_eval_completed, completed);
    InterlockedExchange(&g_eval_total, total);
    /* IPC clients read a coherent status snapshot.  Publishing once per
       completed game is cheap compared with Minimax search and gives the TUI
       a smooth, truthful progress bar without touching evaluation semantics. */
    publish_training_status(g_current_agent);
}

/* ===== Minimax 后的阶段进度（技能测试 / 多快照 / 保存） =====
   这些阶段在 g_eval_active 清零后同步执行，此前无进度显示会让用户误以为卡住。 */
static DWORD g_stage_last_publish = 0;
static int g_stage_last_id = -1;

static void trainer_stage_cb(const StageProgress* sp, void* user) {
    DQNAgent* agent = (DQNAgent*)user;
    publish_training_stage(sp);   /* 立即更新内部快照（廉价、整体一致） */
    DWORD now = GetTickCount();
    int stage_changed = (sp->stage != g_stage_last_id);
    /* 阶段切换立即推送；同一阶段内限频，避免逐题/逐局刷 IPC。 */
    if (stage_changed || now - g_stage_last_publish >= 100 || g_stage_last_publish == 0) {
        g_stage_last_publish = now;
        g_stage_last_id = sp->stage;
        publish_training_status(agent);
    }
}

/* 未知总量阶段（GPU 同步 / 保存）：使用不定进度，不虚构百分比。 */
static void trainer_stage_indeterminate(DQNAgent* agent, int stage) {
    StageProgress sp = {};
    sp.stage = stage;
    sp.indeterminate = 1;
    sp.state = TASK_RUNNING;
    publish_training_stage(&sp);
    g_stage_last_id = stage;
    publish_training_status(agent);
}

/* 后评估任务结束：清除阶段，回到普通训练进度。 */
static void trainer_stage_idle(DQNAgent* agent) {
    StageProgress sp = {};
    sp.stage = STG_NONE;
    sp.state = TASK_IDLE;
    publish_training_stage(&sp);
    g_stage_last_id = -1;
    publish_training_status(agent);
}

/* Display-only latest per-side Minimax result.  metric_win.bin intentionally
   stores the black/white average used by existing charts, so keep this small
   atomic snapshot separate instead of changing a stable metric format. */
static void save_latest_side_win(int episode, const int bw[3], const int ww[3]) {
    const std::string path = dqn_log_path("latest_side_win.txt");
    const std::string temp = path + ".tmp";
    FILE* f = fopen(temp.c_str(), "wb");
    if (!f) return;
    const int wrote = fprintf(f, "side_win_v1 %d %.3f %.3f %.3f %.3f %.3f %.3f\n",
        episode,
        bw[0] * 0.5f, bw[1] * 0.5f, bw[2] * 0.5f,
        ww[0] * 0.5f, ww[1] * 0.5f, ww[2] * 0.5f);
    int ok = wrote > 0 && fflush(f) == 0 && _commit(_fileno(f)) == 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok || !MoveFileExA(temp.c_str(), path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        remove(temp.c_str());
}

static void evaluate_and_save(DQNAgent* agent, int phase, int ep, int total_episodes,
    int save_interval, FILE* eval_log, FILE* minimax_log,
    const char* ckpt_prefix, int seed_id) {
    int is_quiet = (seed_id >= 0); // 种群模式静默
    const int metric_episode = g_ctx.global_episode_offset + ep + 1;

    if (!is_quiet) {
        printf("\n");   /* NoTerminal 版：不再清行覆盖，输出普通换行 */
    }

    int bw[3], bd[3], bl[3], ww[3], wd[3], wl[3];
    float avg_win[3], avg_draw[3], avg_lose[3];

    const int evaluated_depth = global_minimax_depth;
    FILE* staged_minimax = minimax_log ? tmpfile() : NULL;
    if (minimax_log && !staged_minimax) {
        printf("[EVAL] Cannot create staging stream; evaluation not started.\n");
        training_fail(3);
        return;
    }
    InterlockedExchange(&g_eval_completed, 0);
    InterlockedExchange(&g_eval_total, 3 * 2 * 200);
    InterlockedExchange(&g_eval_active, 1);
    publish_training_status(agent);
    EvaluationStatus evaluation = record_minimax_games_for_episode(agent, metric_episode, evaluated_depth,
        bw, bd, bl, ww, wd, wl,
        avg_win, avg_draw, avg_lose,
        staged_minimax, evaluation_cancelled, nullptr, evaluation_progressed);
    if (evaluation == EVALUATION_COMPLETE)
        InterlockedExchange(&g_eval_completed, InterlockedCompareExchange(&g_eval_total, 0, 0));
    InterlockedExchange(&g_eval_active, 0);
    /* Minimax 结束后进入「评估后处理 → 技能测试 → 保存 → 恢复训练」阶段链，
       各阶段之间不再退回普通训练进度。 */
    trainer_stage_indeterminate(agent, STG_POST_EVAL);
    publish_training_status(agent);
    if (evaluation != EVALUATION_COMPLETE) {
        if (staged_minimax) fclose(staged_minimax);
        printf("[EVAL] %s; no scores or difficulty changes committed.\n",
            evaluation == EVALUATION_CANCELLED ? "Cancelled" : "Failed");
        trainer_stage_idle(agent);
        return;
    }
    save_latest_side_win(metric_episode, bw, ww);
    if (staged_minimax) {
        const bool committed = append_file_transaction(minimax_log, staged_minimax);
        fclose(staged_minimax);
        if (!committed) {
            printf("[EVAL] Game-log transaction failed; scores not committed.\n");
            training_fail(3);
            return;
        }
    }

    char eval_lines[1024];
    int eval_used = 0;
    EvalWinRecord win_rows[3] = {};
    EvalDetailRecord detail_rows[3] = {};
    for (int m = 0; m < 3; ++m) {
        const int wrote = snprintf(eval_lines + eval_used, sizeof(eval_lines) - eval_used,
            "%d %d %d %.4f %d %.1f %.1f %.1f\n", metric_episode, phase,
            evaluated_depth, agent->epsilon, m, avg_win[m] * 100.0f,
            avg_draw[m] * 100.0f, avg_lose[m] * 100.0f);
        if (wrote <= 0 || wrote >= (int)sizeof(eval_lines) - eval_used) {
            training_fail(3); return;
        }
        eval_used += wrote;
        win_rows[m] = { metric_episode, 0, {(float)m, avg_win[m] * 100.0f,
            avg_draw[m] * 100.0f, avg_lose[m] * 100.0f} };
        detail_rows[m] = { metric_episode, 0, {(float)phase, (float)evaluated_depth,
            agent->epsilon, (float)m, avg_win[m] * 100.0f,
            avg_draw[m] * 100.0f, avg_lose[m] * 100.0f} };
    }
    if (!append_eval_metric_transaction(eval_log, eval_lines, (size_t)eval_used,
        win_rows, detail_rows)) {
        printf("[EVAL] Metric transaction failed; scores not committed.\n");
        training_fail(3);
        return;
    }

    /* ========== Memory Weight Scheduler：复用主评估结果（mode=0 Greedy） ========== */
    if (!g_mws_initialized) {
        mws_init(&g_mws);
        // Formal replay intentionally excludes the solitaire task. Until a
        // separate retention sampler exists, do not report a fictitious ratio.
        g_mws.enabled = false;
        g_mws.current_weight = 0.0f;
        g_mws_initialized = 1;
        if (!is_quiet)
            printf("[MEMORY] Pretrain retention sampler is not connected; actual sampling ratio is 0%%.\n");
    }
    if (!g_mws.enabled && !is_quiet) {
        g_mws.last_snapshot = {};
        g_mws.last_snapshot.episode = (uint64_t)metric_episode;
        mws_write_metric(&g_mws, LOG_DIR);
    }
    if (g_mws.enabled && mws_should_eval(&g_mws, (uint64_t)metric_episode) && !is_quiet) {
        DepthEvalResult mws_result = {};
        mws_result.depth = evaluated_depth;
        mws_result.wins = bw[0] + ww[0];
        mws_result.draws = bd[0] + wd[0];
        mws_result.losses = bl[0] + wl[0];
        mws_result.win_rate = avg_win[0];
        mws_result.draw_rate = avg_draw[0];
        mws_result.loss_rate = avg_lose[0];
        mws_update(&g_mws, &mws_result, 1, (uint64_t)metric_episode);
        mws_write_metric(&g_mws, LOG_DIR);
        mws_print_summary(&g_mws);
    }

    const bool has_previous_score = agent->ema_winrate != 0 || agent->prev_performance != 0;
    float raw_winrate = avg_win[0];
    if (agent->ema_winrate == 0.0f) {
        agent->ema_winrate = raw_winrate;
    }
    else {
        agent->ema_winrate = agent->ema_alpha * agent->ema_winrate +
            (1.0f - agent->ema_alpha) * raw_winrate;
    }
    float smoothed = agent->ema_winrate;
    int should_up = (smoothed > 0.65f);
    int should_down = (smoothed < 0.25f);

    if (should_up) {
        agent->depth_adjust_counter = (agent->depth_adjust_counter > 0) ?
            agent->depth_adjust_counter + 1 : 1;
    }
    else if (should_down) {
        agent->depth_adjust_counter = (agent->depth_adjust_counter < 0) ?
            agent->depth_adjust_counter - 1 : -1;
    }
    else {
        agent->depth_adjust_counter = 0;
    }

    if (!is_quiet) {
        printf("[DIFFICULTY] Minimax depth=%d raw=%.2f%% EMA=%.2f%% streak=%d/%d (up>65%%, down<25%%)\n",
            global_minimax_depth, raw_winrate * 100.0f, smoothed * 100.0f,
            agent->depth_adjust_counter, agent->depth_adjust_threshold);
    }

    if (agent->depth_adjust_counter >= agent->depth_adjust_threshold &&
        global_minimax_depth < 6) {
        global_minimax_depth++;
        agent->depth_adjust_counter = 0;
        if (!is_quiet)
            printf("Minimax 深度提升至 %d (EMA胜率 %.2f%%, 原始 %.2f%%)\n",
                global_minimax_depth, smoothed * 100, raw_winrate * 100);
    }
    else if (agent->depth_adjust_counter <= -agent->depth_adjust_threshold &&
        global_minimax_depth > 2) {
        global_minimax_depth--;
        agent->depth_adjust_counter = 0;
        if (!is_quiet)
            printf("Minimax 深度降至 %d (EMA胜率 %.2f%%, 原始 %.2f%%)\n",
                global_minimax_depth, smoothed * 100, raw_winrate * 100);
    }

    float current_performance = avg_win[0] + 0.5f * avg_draw[0];
    float delta = current_performance - agent->prev_performance;
    if (has_previous_score && delta > 0.02f)
        agent->concentration_factor = fminf(1.0f, agent->concentration_factor + 0.05f);
    else if (has_previous_score && delta < -0.02f)
        agent->concentration_factor = fmaxf(0.0f, agent->concentration_factor - 0.05f);
    agent->prev_performance = current_performance;

    if (!is_quiet) {
        printf("Ep %d [%.1f%%], Black wins: %d, White wins: %d, Draws: %d, "
            "ε=%.4f, vs Minimax (depth=%d), lr=%.6f\n",
            ep + 1, 100.0 * (ep + 1) / total_episodes,
            g_wins[0], g_wins[1], g_wins[2], agent->epsilon,
            evaluated_depth, g_learning_rate);

        const char* mode_names[] = { "Greedy (ε=0)", "Stable (ε=0.01)", "Robust (ε=0.02)" };
        for (int m = 0; m < 3; m++) {
            printf("  %s:\n", mode_names[m]);
            printf("    Black: Win %d (%.1f%%)  Draw %d (%.1f%%)  Lose %d (%.1f%%)\n",
                bw[m], 100.0 * bw[m] / 200, bd[m], 100.0 * bd[m] / 200,
                bl[m], 100.0 * bl[m] / 200);
            printf("    White: Win %d (%.1f%%)  Draw %d (%.1f%%)  Lose %d (%.1f%%)\n",
                ww[m], 100.0 * ww[m] / 200, wd[m], 100.0 * wd[m] / 200,
                wl[m], 100.0 * wl[m] / 200);
            printf("    Avg Win %.1f%%  Avg Draw %.1f%%  Avg Lose %.1f%%\n",
                avg_win[m] * 100, avg_draw[m] * 100, avg_lose[m] * 100);
        }
        printf("    聚集度因子: %.2f\n", agent->concentration_factor);
        printf("    奖励权重: [%.2f %.2f %.2f %.2f %.2f %.2f %.2f]\n",
            agent->reward_weights[0], agent->reward_weights[1], agent->reward_weights[2],
            agent->reward_weights[3], agent->reward_weights[4], agent->reward_weights[5],
            agent->reward_weights[6]);
    }

    // 只有主训练才更新 best_winrate
    if (seed_id == -1 && avg_win[0] > g_ctx.best_winrate) {
        char best_name[256];
        snprintf(best_name, sizeof(best_name), CKPT_DIR "dqn_model_best_phase%d.bin", phase);
        /* BUG-069: 保存最佳模型前先从 GPU 同步最新参数 */
        if (!sync_gpu_before_persist(agent)) {
            printf("[BEST] GPU 参数同步失败，本次最佳模型未保存\n");
        } else {
            if (save_model_pair(agent, best_name)) {
                g_ctx.best_winrate = avg_win[0];
                printf("[BEST] 新最佳模型已保存 (胜率 %.2f%%)\n", g_ctx.best_winrate * 100);
            } else {
                printf("[BEST] 模型文件写入失败，本次最佳成绩未提交\n");
            }
        }
    }

    // Scores above belong to evaluated_depth; checkpoint configuration below
    // belongs to the NEXT training difficulty. Do not carry its EMA/best across.
    if (evaluated_depth != global_minimax_depth) {
        agent->ema_winrate = 0;
        agent->prev_performance = 0;
        agent->depth_adjust_counter = 0;
        g_ctx.best_winrate = -1;
    }

    char ckpt_name[256];
    const char* prefix = ckpt_prefix ? ckpt_prefix : CKPT_DIR "checkpoint_phase";   /* 正式档 → checkpoints\ */
    snprintf(ckpt_name, sizeof(ckpt_name), "%s%d_%d.bin", prefix, phase, ep + 1);

    // 保存前从 GPU 下载当前参数到 CPU（GPU 驻留时参数在 GPU 上）
    if (gpu_net_active()) {
        trainer_stage_indeterminate(agent, STG_GPU_SYNC);
        if (!sync_gpu_before_persist(agent)) {
            printf("[SAVE] GPU 参数同步失败，取消本次 checkpoint 保存\n");
            trainer_stage_idle(agent);
            return;
        }
    }

    /* ========== 技能回归测试（每 1000 局执行一次） ========== */
    {
        if (agent->skill_tracker.initialized != SKILL_SUITE_VERSION) skill_tracker_init(&agent->skill_tracker);

        /* 每 1000 局跑一次技能测试 */
        if ((ep + 1) % 1000 == 0) {
            SkillTestResult skill_result;
            int use_gpu = gpu_net_active();
            run_skill_tests(agent, metric_episode, &skill_result, use_gpu, trainer_stage_cb, agent);

            /* 写入 metric_skill.bin */
            SkillMetricRecord rec;
            rec.episode = metric_episode;
            /* EmptyWin */
            rec.empty_win = skill_result.empty_win_rate;
            rec.avg_steps = skill_result.empty_avg_steps;
            rec.shortest = skill_result.empty_shortest_rate;
            rec.empty_min_steps = skill_result.empty_min_steps;
            rec.empty_max_steps = skill_result.empty_max_steps;
            /* Win1 */
            rec.win1_fixed = skill_result.win1_fixed;
            rec.win1_fixed_margin = skill_result.win1_fixed_margin;
            rec.win1_random = skill_result.win1_random;
            rec.win1_avg_q_correct = skill_result.win1_stats.q_correct;
            rec.win1_avg_q_wrong = skill_result.win1_stats.q_wrong;
            rec.win1_min_q = skill_result.win1_stats.min_q;
            rec.win1_max_q = skill_result.win1_stats.max_q;
            rec.win1_min_margin = skill_result.win1_stats.min_margin;
            rec.win1_max_margin = skill_result.win1_stats.max_margin;
            /* Block1 */
            rec.block1_fixed = skill_result.block1_fixed;
            rec.block1_fixed_margin = skill_result.block1_fixed_margin;
            rec.block1_random = skill_result.block1_random;
            rec.block1_avg_q_correct = skill_result.block1_stats.q_correct;
            rec.block1_avg_q_wrong = skill_result.block1_stats.q_wrong;
            rec.block1_min_q = skill_result.block1_stats.min_q;
            rec.block1_max_q = skill_result.block1_stats.max_q;
            rec.block1_min_margin = skill_result.block1_stats.min_margin;
            rec.block1_max_margin = skill_result.block1_stats.max_margin;
            /* 多快照统计 */
            rec.empty_stat_std = skill_result.empty_stat_std;
            rec.empty_stat_min = skill_result.empty_stat_min;
            rec.empty_stat_max = skill_result.empty_stat_max;
            rec.win1_acc_std = 0;  /* 多快照测试后更新 */
            rec.block1_acc_std = 0;
            rec.mastered_puzzle_count = 0;

            /* 追踪技能状态，检查回归 */
            int alert = skill_tracker_update(&agent->skill_tracker, &skill_result);
            if (alert) {
                /* 保存异常 checkpoint */
                char alert_name[256];
                snprintf(alert_name, sizeof(alert_name), CKPT_DIR "checkpoint_alert_phase%d_ep%d.bin", phase, metric_episode);
                if (!save_checkpoint(agent, phase, ep + 1, global_minimax_depth, alert_name)) {
                    printf("[SKILL] regression alert checkpoint failed; training stopped.\n");
                    training_fail(3);
                    trainer_stage_idle(agent);
                    return;
                }
                printf("[SKILL] ⚠ 回归报警！已保存异常 checkpoint: %s\n", alert_name);
            } else {
                /* 完整 Skill Test 通过。原先这里调用 sentinel_set_healthy_snapshot
                   保存「共享 q_net」作为最远回滚点；该 API 已删除：按方路由下消费者
                   读的是 online(side)，共享快照既回滚不了任何一方，在 GPU 四句柄
                   驻留下也不保证是评测真实用过的网络。逐方回滚由
                   train_agent_batch_side 内的 before 副本承担（已有回归覆盖）。
                   这里只保留一条显式记录，便于事后把「通过了 Skill Test」与
                   「当时的按方提交/拒绝状态」对齐，不再假装有一个共享回滚点。 */
                const int bound_side = independent_training::side_to_physical(
                    independent_training::indep_bound(agent));
                printf("[SKILL] health snapshot removed (shared q_net cannot roll back either "
                       "side); per-side rollback is the per-batch before-copy. bound_side=%d "
                       "committed black=%llu white=%llu sentinel_rejects=%llu\n",
                    bound_side, agent->indep ? agent->indep->updates(
                        independent_training::Side::Black).committed_batches : 0ull,
                    agent->indep ? agent->indep->updates(
                        independent_training::Side::White).committed_batches : 0ull,
                    sentinel_reject_count());
            }

            /* ========== 过去约1000局的64快照技能历史 ========== */
            int history_ready = 0;
            if (snapshot_collector_count(&g_snap_collector) == SNAPSHOT_COUNT) {
                int snap_cnt = snapshot_collector_count(&g_snap_collector);
                const NeuralNet* snaps[SNAPSHOT_COUNT];
                for (int i = 0; i < snap_cnt; i++)
                    snaps[i] = snapshot_collector_get(&g_snap_collector, i);

                MultiSnapshotResult win1_result, block1_result;
                SkillTestResult empty_result;
                int prog = 0, ph = 0;

                printf("[MULTI-SNAP] EmptyWin: testing %d snapshots x 2 games...\n", snap_cnt);
                int empty_ok = run_multi_snapshot_empty(snaps, snap_cnt, &empty_result, &prog, &ph, trainer_stage_cb, agent);
                printf("[MULTI-SNAP] Win1: testing %d snapshots...\n", snap_cnt);
                run_multi_snapshot_win1(snaps, snap_cnt, &win1_result, &prog, &ph, trainer_stage_cb, agent);
                printf("[MULTI-SNAP] Win1: acc=%.1f%% ± %.1f%% [%.1f~%.1f] current=%.1f%%\n",
                    win1_result.acc_mean * 100, win1_result.acc_std * 100,
                    win1_result.acc_min * 100, win1_result.acc_max * 100,
                    win1_result.current_accuracy * 100);
                printf("[MULTI-SNAP] Win1 puzzles: mastered=%d borderline=%d unknown=%d\n",
                    win1_result.mastered_count, win1_result.borderline_count, win1_result.unknown_count);

                printf("[MULTI-SNAP] Block1: testing %d snapshots...\n", snap_cnt);
                run_multi_snapshot_block1(snaps, snap_cnt, &block1_result, &prog, &ph, trainer_stage_cb, agent);
                printf("[MULTI-SNAP] Block1: acc=%.1f%% ± %.1f%% [%.1f~%.1f] current=%.1f%%\n",
                    block1_result.acc_mean * 100, block1_result.acc_std * 100,
                    block1_result.acc_min * 100, block1_result.acc_max * 100,
                    block1_result.current_accuracy * 100);
                printf("[MULTI-SNAP] Block1 puzzles: mastered=%d borderline=%d unknown=%d\n",
                    block1_result.mastered_count, block1_result.borderline_count, block1_result.unknown_count);

                /* 历史表的主值也是跨快照均值；当前模型的单点结果仍由
                   skill_result 用于遗忘报警，二者不能混成同一统计口径。 */
                if (empty_ok) {
                    rec.empty_win = empty_result.empty_win_rate;
                    rec.avg_steps = empty_result.empty_avg_steps;
                    rec.shortest = empty_result.empty_shortest_rate;
                    rec.empty_min_steps = empty_result.empty_min_steps;
                    rec.empty_max_steps = empty_result.empty_max_steps;
                    rec.empty_stat_std = empty_result.empty_stat_std;
                    rec.empty_stat_min = empty_result.empty_stat_min;
                    rec.empty_stat_max = empty_result.empty_stat_max;
                    rec.win1_fixed = win1_result.acc_mean;
                    rec.win1_fixed_margin = win1_result.margin_mean;
                    rec.block1_fixed = block1_result.acc_mean;
                    rec.block1_fixed_margin = block1_result.margin_mean;
                    rec.win1_acc_std = win1_result.acc_std;
                    rec.block1_acc_std = block1_result.acc_std;
                    rec.mastered_puzzle_count = win1_result.mastered_count + block1_result.mastered_count;
                    history_ready = 1;
                } else {
                    printf("[MULTI-SNAP] EmptyWin allocation/evaluation failed; history row skipped.\n");
                }

                /* 重置采集器，开始下一个1000局间隔 */
                snapshot_collector_init(&g_snap_collector, SKILL_WINDOW_INTERVAL);
            } else {
                printf("[MULTI-SNAP] History deferred: only %d/%d snapshots collected.\n",
                    snapshot_collector_count(&g_snap_collector), SNAPSHOT_COUNT);
            }

            /* 所有三项跨快照统计完成后只写一次，避免旧代码先写全0扩展字段。 */
            if (history_ready &&
                !write_skill_metric(dqn_log_path(SKILL_METRIC_FILENAME).c_str(), &rec)) {
                printf("[SKILL] metric write failed; forgetting history skipped for ep=%d.\n",
                    metric_episode);
            }
        }
    }

    /* 只有检查点完整原子落盘后才发布 latest_eval。旧顺序先发布评估，
       GPU 下载或磁盘写入一旦失败，看门狗就会看到一个不存在的恢复点。 */
    trainer_stage_indeterminate(agent, STG_SAVE);
    if (save_checkpoint(agent, phase, ep + 1, global_minimax_depth, ckpt_name)) {
        FILE* f_eval = fopen("latest_eval.txt.tmp", "w");
        if (f_eval) {
            /* 维护器需要真实败率；1-win 会把平局错误算成失败。 */
            float lose_rate = avg_lose[0];
            fprintf(f_eval, "%d %.6f %s %d %d\n",
                ep + 1, lose_rate, ckpt_name, evaluated_depth, phase);
            int eval_ok = fflush(f_eval) == 0 && _commit(_fileno(f_eval)) == 0;
            if (fclose(f_eval) != 0) eval_ok = 0;
            if (!eval_ok || !MoveFileExA("latest_eval.txt.tmp", "latest_eval.txt",
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) remove("latest_eval.txt.tmp");
        }
    } else {
        printf("[SAVE] Required checkpoint failed: %s\n", ckpt_name);
        training_fail_detail(3, "evaluation_checkpoint", metric_episode);
    }
    trainer_stage_idle(agent);

}

// ============================================================================
// 正式训练前：单人四连课程预训练
// ============================================================================
#define PRETRAIN_MIN_EPISODES       5000
#define PRETRAIN_EVAL_INTERVAL       500
#define PRETRAIN_EVAL_GAMES          256
#define PRETRAIN_REQUIRED_PASSES       3
#define PRETRAIN_SUCCESS_TARGET     0.99f
#define PRETRAIN_AVG_STEPS_TARGET   4.20f
#define PRETRAIN_CONSOLIDATE_BLOCK  10000
#define PRETRAIN_CONSOLIDATE_LR_SCALE 0.20f
#define PRETRAIN_CONSOLIDATE_EPS      0.01f
#define PRETRAIN_CANDIDATE_MODEL CKPT_DIR "dqn_model_pretrain_candidate.bin"

/* 只统计最近一段单人训练胜局，避免某个固定四连位置长期垄断策略。
   5x5 连四实际只有 28 个长度为 4 的线段；这里保留通用上限，便于棋盘参数调整。 */
#define PRETRAIN_LINE_MAX          (4 * CELL_COUNT)
#define PRETRAIN_LINE_HISTORY             256
#define PRETRAIN_LINE_MIN_OBS              64
#define PRETRAIN_LINE_SHARE_LIMIT       0.15f
#define PRETRAIN_LINE_MAX_PENALTY       0.40f
/* 暂停胜利线位置去重惩罚：预训练胜局统一获得完整终局奖励。
   保留统计实现，之后需要做对照实验时可直接重新启用。 */
#define PRETRAIN_LINE_DIVERSITY_ENABLED    0

static int g_pretrain_line_ring[PRETRAIN_LINE_HISTORY];
static int g_pretrain_line_counts[PRETRAIN_LINE_MAX];
static int g_pretrain_line_ring_pos = 0;
static int g_pretrain_line_ring_count = 0;
static int g_pretrain_last_line = -1;
static float g_pretrain_last_diversity_penalty = 0.0f;

static void pretrain_reset_line_history(void) {
    memset(g_pretrain_line_ring, 0xff, sizeof(g_pretrain_line_ring));
    memset(g_pretrain_line_counts, 0, sizeof(g_pretrain_line_counts));
    g_pretrain_line_ring_pos = 0;
    g_pretrain_line_ring_count = 0;
    g_pretrain_last_line = -1;
    g_pretrain_last_diversity_penalty = 0.0f;
}

/* 在所有由最后一步完成的四连线段中，优先把胜局记到近期最少出现的线段。
   这样十字形等一次完成多条线的局面不会被枚举顺序误判成总在同一位置。 */
static int pretrain_choose_winning_line(const GameState* game, int action) {
    static const int dirs[4][2] = { {0,1}, {1,0}, {1,1}, {1,-1} };
    int action_r = action / BOARD_SIZE, action_c = action % BOARD_SIZE;
    int best_line = -1, best_count = 0x7fffffff, line_id = 0;
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0], dc = dirs[d][1];
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                int end_r = r + (WIN_STREAK - 1) * dr;
                int end_c = c + (WIN_STREAK - 1) * dc;
                if (end_r < 0 || end_r >= BOARD_SIZE || end_c < 0 || end_c >= BOARD_SIZE)
                    continue;
                int contains_action = 0, complete = 1;
                for (int k = 0; k < WIN_STREAK; k++) {
                    int rr = r + k * dr, cc = c + k * dc;
                    if (rr == action_r && cc == action_c) contains_action = 1;
                    if (game->board[rr * BOARD_SIZE + cc] != 1) complete = 0;
                }
                if (contains_action && complete && line_id < PRETRAIN_LINE_MAX &&
                    g_pretrain_line_counts[line_id] < best_count) {
                    best_line = line_id;
                    best_count = g_pretrain_line_counts[line_id];
                }
                line_id++;
            }
        }
    }
    return best_line;
}

static float pretrain_apply_line_diversity(const GameState* game, int action) {
    int line = pretrain_choose_winning_line(game, action);
    float penalty = 0.0f;
    if (line >= 0 && line < PRETRAIN_LINE_MAX) {
        if (g_pretrain_line_ring_count >= PRETRAIN_LINE_MIN_OBS) {
            float share = (float)g_pretrain_line_counts[line] /
                (float)g_pretrain_line_ring_count;
            if (share > PRETRAIN_LINE_SHARE_LIMIT) {
                float excess = (share - PRETRAIN_LINE_SHARE_LIMIT) /
                    (1.0f - PRETRAIN_LINE_SHARE_LIMIT);
                penalty = excess * PRETRAIN_LINE_MAX_PENALTY;
                if (penalty > PRETRAIN_LINE_MAX_PENALTY)
                    penalty = PRETRAIN_LINE_MAX_PENALTY;
            }
        }

        if (g_pretrain_line_ring_count == PRETRAIN_LINE_HISTORY) {
            int old_line = g_pretrain_line_ring[g_pretrain_line_ring_pos];
            if (old_line >= 0 && old_line < PRETRAIN_LINE_MAX &&
                g_pretrain_line_counts[old_line] > 0)
                g_pretrain_line_counts[old_line]--;
        }
        else {
            g_pretrain_line_ring_count++;
        }
        g_pretrain_line_ring[g_pretrain_line_ring_pos] = line;
        g_pretrain_line_counts[line]++;
        g_pretrain_line_ring_pos =
            (g_pretrain_line_ring_pos + 1) % PRETRAIN_LINE_HISTORY;
    }
    g_pretrain_last_line = line;
    g_pretrain_last_diversity_penalty = penalty;
    return 1.0f - penalty;
}

static void pretrain_store(DQNAgent* agent, const float* state, int action,
    float reward, const float* next_state, int done) {
    /* 不调用 store_experience：单人课程只需要黑棋池，不生成对局用的
       颜色互换副本，也不占用白棋池。 */
    int idx = agent->replay_black_idx;
    Experience* exp = &agent->replay_black[idx];
    memcpy(exp->state, state, sizeof(float) * (CELL_COUNT + 1));
    exp->action = action;
    exp->reward = reward;
    memcpy(exp->next_state, next_state, sizeof(float) * (CELL_COUNT + 1));
    exp->done = done;
    agent->replay_black_idx = (idx + 1) % REPLAY_CAPACITY;
    if (agent->replay_black_count < REPLAY_CAPACITY) agent->replay_black_count++;
}

static int pretrain_play(DQNAgent* agent, float epsilon, int train_enabled) {
    GameState game; reset_game(&game);
    int steps = 0;
    if (train_enabled) {
        g_pretrain_last_line = -1;
        g_pretrain_last_diversity_penalty = 0.0f;
    }
    while (steps < CELL_COUNT && !g_training_fault) {
        game.turn = 0; /* 单人阶段始终落 X；make_move 后会翻 turn，下一步再归零。 */
        int action;
        if (!train_enabled && agent != g_current_agent) {
            // Offline candidates must not latch a live training fault if their
            // finite weights overflow during inference. Invalid scores fail the
            // candidate instead; do not touch GPU or the live RNG.
            float input[CELL_COUNT + 1], q[CELL_COUNT];
            state_to_input(&game, input);
            forward_cpu_only(&agent->q_net, input, q);
            memory_apply_model_scores(&game, q);
            action = greedy_legal_action(&game, q, &agent->eval_rng);
        } else {
            action = select_action(agent, &game, 0, epsilon,
                train_enabled ? &agent->train_rng : &agent->eval_rng, 1);
        }
        if (g_training_fault) return -CELL_COUNT;
        if (!is_legal_move(&game, action)) {
            // Offline candidates are untrusted inputs: reject the candidate,
            // but do not latch a fault into the live Trainer.
            if (train_enabled || agent == g_current_agent) training_fail(2);
            return -CELL_COUNT;
        }

        float state[CELL_COUNT + 1], next_state[CELL_COUNT + 1];
        state_to_input(&game, state);
        GameState next = game;
        make_move(&next, action);
        next.turn = 0;
        steps++;

        int won = (check_winner(&next) == 1);
        int done = won || is_full(&next);
        float reward = won
            ? (train_enabled && PRETRAIN_LINE_DIVERSITY_ENABLED
                ? pretrain_apply_line_diversity(&next, action) : 1.0f)
            : (done ? -1.0f : -0.01f);
        state_to_input(&next, next_state);

        if (train_enabled) {
            pretrain_store(agent, state, action, reward, next_state, done);
            int repeat = g_hot_config.train_repeat_per_step;
            if (repeat < 1) repeat = 1; if (repeat > 20) repeat = 20;
            if (agent->replay_black_count >= BATCH_SIZE &&
                agent->step_counter % TRAIN_STEP_EACH == 0) {
                for (int r = 0; r < repeat && !g_training_fault; r++)
                    replay_from_buffer(agent, agent->replay_black,
                        agent->replay_black_count, &agent->replay_black_idx);
                soft_update_target(agent);
            }
            agent->step_counter++;
        }
        game = next;
        if (done) return won ? steps : -steps;
    }
    return -steps;
}

static int pretrain_evaluate(DQNAgent* agent, float* success_rate, float* avg_steps) {
    int success = 0, step_sum = 0;
    for (int i = 0; i < PRETRAIN_EVAL_GAMES; i++) {
        if (evaluation_cancelled(nullptr) || g_training_fault) return 0;
        int result = pretrain_play(agent, 0.0f, 0);
        if (g_training_fault) return 0;
        if (result > 0) { success++; step_sum += result; }
        else step_sum += -result;
    }
    *success_rate = (float)success / PRETRAIN_EVAL_GAMES;
    *avg_steps = (float)step_sum / PRETRAIN_EVAL_GAMES;
    return 1;
}

// A candidate is weights only. Loading/testing it cannot roll back the live
// agent's replay, LR, RNG, progress, or resident GPU buffers.
static std::unique_ptr<DQNAgent> assess_pretrain_candidate(const char* path,
    bool checkpoint, float* success_rate, float* avg_steps) {
    auto candidate = std::unique_ptr<DQNAgent>(new (std::nothrow) DQNAgent);
    if (!candidate) return nullptr;
    memset(candidate.get(), 0, sizeof(DQNAgent));
    candidate->seed_id = 0; // no memory observer side effects for an offline candidate
    prng_init(&candidate->eval_rng, 223456789u);
    if (checkpoint ? !load_checkpoint_model(&candidate->q_net, path) :
        !load_model(candidate.get(), path)) return nullptr;
    if (!pretrain_evaluate(candidate.get(), success_rate, avg_steps) ||
        *success_rate < PRETRAIN_SUCCESS_TARGET || *avg_steps > PRETRAIN_AVG_STEPS_TARGET)
        return nullptr;
    SkillTestResult result;
    run_skill_tests(candidate.get(), 0, &result, 0);
    skill_tracker_init(&candidate->skill_tracker);
    if (skill_tracker_update(&candidate->skill_tracker, &result) ||
        evaluation_cancelled(nullptr) || g_training_fault) return nullptr;
    return candidate;
}

static int commit_pretrain_candidate(DQNAgent* live, const DQNAgent* candidate) {
    if (evaluation_cancelled(nullptr) || g_training_fault) return 0;
    copy_net(&live->q_net, &candidate->q_net);
    copy_net(&live->target_net, &candidate->q_net);
    copy_net(&live->opponent_net, &candidate->q_net);
    if (gpu_net_active() && gpu_net_upload_agent(live) != 0) {
        training_fail(1); // partial device upload must never resume training/save
        return 0;
    }
    return 1;
}

// New task, same learned policy. Apply BEFORE saving Phase1_0, and also when
// resuming an old Phase1_0 file. The replay is not an auxiliary retention loss.
static int prepare_formal_task(DQNAgent* agent) {
    copy_net(&agent->target_net, &agent->q_net);
    copy_net(&agent->opponent_net, &agent->q_net);
    if (gpu_net_active() && gpu_net_upload_agent(agent) != 0) {
        training_fail(1); return 0;
    }
    agent->replay_black_idx = agent->replay_white_idx = 0;
    agent->replay_black_count = agent->replay_white_count = 0;
    if (agent->replay_black) memset(agent->replay_black, 0, sizeof(Experience) * REPLAY_CAPACITY);
    if (agent->replay_white) memset(agent->replay_white, 0, sizeof(Experience) * REPLAY_CAPACITY);
    agent->first_move_buf_idx = agent->step_counter = 0;
    memset(agent->first_move_buf, 0, sizeof(agent->first_move_buf));
    agent->reward_episode = -1;
    memset(agent->reward_stats, 0, sizeof(agent->reward_stats));
    memset(agent->reward_usage, 0, sizeof(agent->reward_usage));
    agent->ema_winrate = agent->prev_performance = 0;
    agent->depth_adjust_counter = 0;
    agent->epsilon = EPS_MAX;
    if (agent->opponent_pool && agent->opponent_pool_size > 0) {
        copy_net(&agent->opponent_pool[0], &agent->q_net);
        agent->opponent_pool_count = 1;
        agent->opponent_pool_next = 1 % agent->opponent_pool_size;
    }
    g_ctx.best_winrate = -1;
    g_wins[0] = g_wins[1] = g_wins[2] = 0;
    g_start_time = time(nullptr);
    return 1;
}

/* Phase 0 → Phase 1 是最重要的能力基线。毕业瞬间固定跑一次完整题库，
   用原子覆盖的独立报告供 TUI/人工查看。正式 skill metric 只接收完整
   1000局快照窗口，避免两种统计口径混在一张历史表。测试只推理。 */
static int run_phase0_exit_skill_test(DQNAgent* agent, int pretrain_episode) {
    SkillTestResult result;
    run_skill_tests(agent, pretrain_episode, &result, agent == g_current_agent && gpu_net_active());
    skill_tracker_init(&agent->skill_tracker);
    if (skill_tracker_update(&agent->skill_tracker, &result)) {
        printf("[SKILL] Graduation test invalid; baseline was NOT established.\n");
        return 0;
    }
    SYSTEMTIME now;
    GetLocalTime(&now);
    FILE* f = fopen(LOG_DIR "phase0_skill_test.tmp", "wb");
    if (f) {
        fprintf(f, "=== PHASE 0 EXIT SKILL TEST v%d (shared stochastic greedy policy) ===\r\n", SKILL_SUITE_VERSION);
        fprintf(f, "Generated: %04u-%02u-%02u %02u:%02u:%02u\r\n",
            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond);
        fprintf(f, "Pretrain episode: %d\r\n", pretrain_episode);
        fprintf(f, "Model: checkpoint_phase1_0.bin (before formal training)\r\n\r\n");
        fprintf(f, "[EmptyWin]     %6.2f%%  avg_steps %.3f  shortest %.2f%%\r\n",
            result.empty_win_rate * 100.0f, result.empty_avg_steps,
            result.empty_shortest_rate * 100.0f);
        fprintf(f, "[Win1-Fixed]   %6.2f%%  margin %+.6f\r\n",
            result.win1_fixed * 100.0f, result.win1_fixed_margin);
        fprintf(f, "[Win1-Random]  %6.2f%%  margin %+.6f\r\n",
            result.win1_random * 100.0f, result.win1_random_margin);
        fprintf(f, "[Block1-Fixed] %6.2f%%  margin %+.6f\r\n",
            result.block1_fixed * 100.0f, result.block1_fixed_margin);
        fprintf(f, "[Block1-Random]%6.2f%%  margin %+.6f\r\n",
            result.block1_random * 100.0f, result.block1_random_margin);
        fprintf(f, "[Forced-first] %.2f%% avg_steps %.3f (25 starts, includes preset stone)\r\n",
            result.seeded_win_rate * 100, result.seeded_avg_steps);
        /* 黑白分色：两方独立网络下不分色就无法判断「哪一方学会了」。
           固定题库全是黑方走，所以固定题正确率标为 black；白方分色用白先对局。 */
        if (result.side_split_valid) {
            fprintf(f, "[Side-Split] valid=1 black_to_move_win %.2f%% avg_steps %.3f (n=%d) | "
                "white_to_move_win %.2f%% avg_steps %.3f (n=%d)\r\n",
                result.black_win_rate * 100, result.black_avg_steps, result.black_games,
                result.white_win_rate * 100, result.white_avg_steps, result.white_games);
            fprintf(f, "[Side-Split-Fixed] win1_black_to_move %.2f%% block1_black_to_move %.2f%% "
                "(fixed banks are generated with player=1, i.e. black to move)\r\n",
                result.win1_fixed_black * 100, result.block1_fixed_black * 100);
        } else {
            fprintf(f, "[Side-Split] valid=0 (not collected on this path)\r\n");
        }
        fprintf(f, "[Parity] status=%d max_diff=%g argmax_match=%g\r\n",
            result.parity_status, result.parity_max_diff, result.parity_argmax_match);
        int ok = fflush(f) == 0 && _commit(_fileno(f)) == 0;
        if (fclose(f) != 0) ok = 0;
        if (!ok || !MoveFileExA(LOG_DIR "phase0_skill_test.tmp",
            LOG_DIR "phase0_skill_test.txt",
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            remove(LOG_DIR "phase0_skill_test.tmp");
    }

    FILE* history = fopen(LOG_DIR "phase0_skill_test_history_v3.dat", "a");
    if (history) {
        fprintf(history, "%04u-%02u-%02uT%02u:%02u:%02u %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
            pretrain_episode, result.empty_win_rate, result.empty_avg_steps,
            result.empty_shortest_rate, result.win1_fixed, result.win1_random,
            result.block1_fixed, result.block1_random);
        fflush(history);
        fclose(history);
    }
    printf("[PRETRAIN TEST] Phase 0 毕业基线已写入 %s\n",
        LOG_DIR "phase0_skill_test.txt");
    return 1;
}

/* 已经完成过巩固区间但最终策略振荡时，从日志找回巩固期间最后一个
   “连续稳定且恰好有 checkpoint”的候选。这样异常退出/旧版本无限训练后
   也不用从零开始。 */
static int pretrain_find_recovery_checkpoint(int start_episode, char* out, size_t out_size,
    int* out_candidate_ep, int* out_target) {
    FILE* f = fopen(LOG_DIR "pretrain.dat", "r");
    if (!f) return 0;
    char line[256];
    int trigger = 0, target = 0, candidate = 0;
    while (fgets(line, sizeof(line), f)) {
        int a = 0, b = 0;
        float sr = 0.0f, avg = 0.0f;
        int stable = 0;
        if (sscanf(line, "CONSOLIDATE %d %d", &a, &b) == 2) {
            trigger = a; target = b; candidate = 0;
        }
        else if (trigger > 0 && sscanf(line, "EVAL %d %f %f %d", &a, &sr, &avg, &stable) == 4) {
            if (a >= trigger && a <= target && a % 1000 == 0 &&
                sr >= PRETRAIN_SUCCESS_TARGET && avg <= PRETRAIN_AVG_STEPS_TARGET &&
                stable >= PRETRAIN_REQUIRED_PASSES) {
                char path[256];
                snprintf(path, sizeof(path), CKPT_DIR "checkpoint_phase0_%d.bin", a);
                DWORD attr = GetFileAttributesA(path);
                if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY))
                    candidate = a;
            }
        }
    }
    fclose(f);
    if (target <= 0 || start_episode < target || candidate <= 0) return 0;
    snprintf(out, out_size, CKPT_DIR "checkpoint_phase0_%d.bin", candidate);
    if (out_candidate_ep) *out_candidate_ep = candidate;
    if (out_target) *out_target = target;
    return 1;
}

static bool write_pretrain_log(FILE* log, const char* format, ...) {
    if (!log || !format) return false;
    va_list args;
    va_start(args, format);
    const int wrote = vfprintf(log, format, args);
    va_end(args);
    if (wrote < 0 || fflush(log) != 0) {
        printf("[PRETRAIN] progress log write failed; stopping at the last committed checkpoint.\n");
        training_fail(3);
        return false;
    }
    return true;
}

/* 返回 1=毕业，0=收到退出请求。phase0 checkpoint 可在下次启动续训。 */
static int train_single_player_pretrain(DQNAgent* agent, int start_episode) {
    // GPT6-Astra: formal outcome settings must not alter solo-course anchors.
    StepOutcomeConfig pretrain_outcome=g_hot_config.step_outcome;
    pretrain_outcome.enabled=0;
    StepOutcomeScope pretrain_outcome_scope(pretrain_outcome);
    service_signal_exit_request();
    g_current_agent = agent;
    neural_probe_init();
    int stable_passes = agent->pretrain_stable_passes;
    int consolidation_target = agent->pretrain_consolidation_target;
    if (agent->checkpoint_version < 6)
        restore_pretrain_progress(LOG_DIR "pretrain.dat", start_episode, &consolidation_target, &stable_passes);
    agent->pretrain_consolidation_target = consolidation_target;
    agent->pretrain_stable_passes = stable_passes;
    int candidate_ready = (GetFileAttributesA(PRETRAIN_CANDIDATE_MODEL) != INVALID_FILE_ATTRIBUTES);
    int recent_steps[100] = { 0 }, recent_success[100] = { 0 };
    int recent_count = 0, recent_pos = 0;
    FILE* log = fopen(LOG_DIR "pretrain.dat", "a");
    if (!log) {
        printf("[PRETRAIN] Cannot open progress log; training was not started.\n");
        training_fail(3);
        return 0;
    }
    printf("\n===== Phase 0：单人四连课程预训练 =====\n");
    printf("目标：连续 %d 次评估成功率 >= %.1f%% 且平均步数 <= %.2f（至少 %d 局）\n",
        PRETRAIN_REQUIRED_PASSES, PRETRAIN_SUCCESS_TARGET * 100.0f,
        PRETRAIN_AVG_STEPS_TARGET, PRETRAIN_MIN_EPISODES);
    g_ctx.global_episode_offset = 0;
    g_ctx.total_games_played = start_episode;
    g_start_time = time(NULL);
    g_pretrain_active = 1;
    g_pretrain_steps = 0;
    g_pretrain_avg_steps = 0.0f;
    g_pretrain_success_rate = 0.0f;
    g_pretrain_stable_passes = 0;
    pretrain_reset_line_history();

    /* 兼容本次已经发生的振荡：52k checkpoint 曾在 60k 巩固目标前稳定
       达标，而旧程序继续训练到了目标之后。启动新版时自动验证并救回它。 */
    {
        char recovery[256]; int recovery_ep = 0, old_target = 0;
        if (pretrain_find_recovery_checkpoint(start_episode, recovery, sizeof(recovery),
            &recovery_ep, &old_target)) {
            printf("[PRETRAIN] 检测到巩固后振荡，验证稳定候选 %s\n", recovery);
            float sr = 0.0f, avg = 0.0f;
            auto candidate = assess_pretrain_candidate(recovery, true, &sr, &avg);
            if (candidate) {
                if (run_phase0_exit_skill_test(candidate.get(), recovery_ep) &&
                    commit_pretrain_candidate(agent, candidate.get()) && prepare_formal_task(agent)) {
                    agent->skill_tracker = candidate->skill_tracker;
                    if (!save_before_exit([&]() {
                        return save_model_pair(agent, CKPT_DIR "dqn_model_pretrain_final.bin") &&
                            save_checkpoint(agent, 1, 0, global_minimax_depth, CKPT_DIR "checkpoint_phase1_0.bin");
                        }, []() { win_pump_messages(); Sleep(50); }) || g_exit_requested) {
                        if (log) fclose(log);
                        g_pretrain_active = 0;
                        service_signal_exit_request(); // Signal may have arrived during the save.
                        return 0;
                    }
                    InterlockedExchange(&g_shutdown_state, 0);
                    printf("[PRETRAIN] 已从 ep=%d 稳定候选恢复（success=%.2f%% avg=%.3f），转入 Phase 1。\n",
                        recovery_ep, sr * 100.0f, avg);
                    if (!write_pretrain_log(log, "RECOVER %d %d %.6f %.6f\n",
                        start_episode, recovery_ep, sr, avg)) {
                        fclose(log); g_pretrain_active = 0; return 0;
                    }
                    fclose(log);
                    g_pretrain_active = 0;
                    return 1;
                }
            }
            printf("[PRETRAIN] 候选复验未通过，继续低强度巩固训练。\n");
            // A rejected candidate does not mutate current consolidation progress.
        }
    }

    for (int ep = start_episode; !g_exit_requested; ep++) {
        while (sentinel_guard_paused() && !g_exit_requested) {
            service_snapshot_requests(agent);
            apply_pending_ipc_config(agent);
            publish_training_status(agent);
            win_pump_messages();
            Sleep(100);
        }
        service_snapshot_requests(agent);
        apply_pending_ipc_config(agent);
        /* 约 2 万局降到 0.05；达到下限后继续训练直到毕业。 */
        float epsilon = 1.0f - 0.95f * ((float)ep / 20000.0f);
        if (epsilon < 0.05f) epsilon = 0.05f;
        if (consolidation_target > 0) epsilon = PRETRAIN_CONSOLIDATE_EPS;
        agent->epsilon = epsilon;
        float base_lr = g_learning_rate;
        if (consolidation_target > 0)
            g_learning_rate = base_lr * PRETRAIN_CONSOLIDATE_LR_SCALE;
        int result = pretrain_play(agent, epsilon, 1);
        /* Bug-9 修复：只还原「巩固临时倍率」，保留本局内哨兵/数值重试的安全降速。
           非巩固阶段不做任何还原（原无条件 base_lr 会覆盖安全降速）。 */
        if (consolidation_target > 0) {
            const float expected = base_lr * PRETRAIN_CONSOLIDATE_LR_SCALE;
            if (g_learning_rate >= expected * (1.0f - 1e-6f))
                g_learning_rate = base_lr;
        }
        maybe_publish_probe_pretrain(agent, (uint64_t)ep + 1u);
        if (g_training_fault) break; // never count a partially failed game as completed
        int success = result > 0;
        int steps = success ? result : -result;
        recent_steps[recent_pos] = steps;
        recent_success[recent_pos] = success;
        recent_pos = (recent_pos + 1) % 100;
        if (recent_count < 100) recent_count++;
        int step_sum = 0, success_sum = 0;
        for (int ri = 0; ri < recent_count; ri++) {
            step_sum += recent_steps[ri]; success_sum += recent_success[ri];
        }
        g_pretrain_steps = steps;
        g_pretrain_avg_steps = recent_count ? (float)step_sum / recent_count : 0.0f;
        g_pretrain_success_rate = recent_count ? (float)success_sum / recent_count : 0.0f;
        g_ctx.total_games_played = ep + 1;
        g_wins[0] += success; g_wins[1] += !success;
        publish_training_status(agent);

        if (!write_pretrain_log(log, "%d %d %d %.6f %.6f %d\n", ep + 1,
            success, steps, epsilon, g_pretrain_last_diversity_penalty,
            g_pretrain_last_line)) break;
        if ((ep + 1) % 100 == 0)
            printf("[PRETRAIN] ep=%d success=%d steps=%d eps=%.4f replay=%d line=%d penalty=%.3f\n",
                ep + 1, success, steps, epsilon, agent->replay_black_count,
                g_pretrain_last_line, g_pretrain_last_diversity_penalty);


        if ((ep + 1) >= PRETRAIN_MIN_EPISODES && (ep + 1) % PRETRAIN_EVAL_INTERVAL == 0) {
            float sr = 0, avg = 0;
            if (!pretrain_evaluate(agent, &sr, &avg)) break;
            int passed = sr >= PRETRAIN_SUCCESS_TARGET && avg <= PRETRAIN_AVG_STEPS_TARGET;
            stable_passes = passed ? stable_passes + 1 : 0;
            agent->pretrain_stable_passes = stable_passes;
            g_pretrain_stable_passes = stable_passes;
            printf("[PRETRAIN EVAL] ep=%d success=%.2f%% avg_steps=%.3f stable=%d/%d\n",
                ep + 1, sr * 100.0f, avg, stable_passes, PRETRAIN_REQUIRED_PASSES);
            if (!write_pretrain_log(log, "EVAL %d %.6f %.6f %d\n",
                ep + 1, sr, avg, stable_passes)) break;
            if (stable_passes >= PRETRAIN_REQUIRED_PASSES && consolidation_target == 0) {
                /* 即使恰好落在整万局，也再完成一个完整巩固区间。 */
                consolidation_target =
                    (((ep + 1) / PRETRAIN_CONSOLIDATE_BLOCK) + 1) *
                    PRETRAIN_CONSOLIDATE_BLOCK;
                printf("[PRETRAIN] 首次稳定达标；继续巩固到 ep=%d，届时仍须满足毕业标准。\n",
                    consolidation_target);
                agent->pretrain_consolidation_target = consolidation_target;
                if (!write_pretrain_log(log, "CONSOLIDATE %d %d\n",
                    ep + 1, consolidation_target)) break;
                /* 清掉由早期高探索慢策略占满的回放，只用当前稳定策略附近的
                   新样本做巩固；同时锚定 target，避免旧价值继续把策略拉回去。 */
                agent->replay_black_idx = agent->replay_white_idx = 0;
                agent->replay_black_count = agent->replay_white_count = 0;
                if (gpu_net_active()) {
                    if (sync_gpu_before_persist(agent) &&
                        save_model_pair(agent, PRETRAIN_CANDIDATE_MODEL)) candidate_ready = 1;
                    if (gpu_net_hard_update() != 0)
                        printf("[PRETRAIN] 巩固阶段 target 锚定失败，将继续由 soft update 维护\n");
                } else {
                    copy_net(&agent->target_net, &agent->q_net);
                    if (save_model_pair(agent, PRETRAIN_CANDIDATE_MODEL)) candidate_ready = 1;
                }
            }
            if (consolidation_target > 0 && (ep + 1) >= consolidation_target) {
                int graduate = stable_passes >= PRETRAIN_REQUIRED_PASSES;
                bool candidate_baseline = false;
                if (!graduate && candidate_ready) {
                    /* 额外一万局确实执行，但若巩固把策略冲坏，则采用训练期间
                       已连续三次通过的候选，避免无上限振荡。 */
                    float csr = 0.0f, cavg = 0.0f;
                    auto candidate = assess_pretrain_candidate(PRETRAIN_CANDIDATE_MODEL, false, &csr, &cavg);
                    graduate = candidate && run_phase0_exit_skill_test(candidate.get(), ep + 1) &&
                        commit_pretrain_candidate(agent, candidate.get());
                    if (graduate) {
                        agent->skill_tracker = candidate->skill_tracker;
                        candidate_baseline = true;
                    }
                    printf("[PRETRAIN] 巩固末尾退化，候选复验 success=%.2f%% avg=%.3f\n",
                        csr * 100.0f, cavg);
                }
                if (!graduate) continue;
                if (!sync_gpu_before_persist(agent)) {
                    printf("[PRETRAIN] 毕业时 GPU 参数同步失败，继续训练并重试\n");
                    stable_passes = 0;
                    continue;
                }
                if (!candidate_baseline && !run_phase0_exit_skill_test(agent, ep + 1)) continue;
                if (!prepare_formal_task(agent)) break;
                if (!save_before_exit([&]() {
                    return save_model_pair(agent, CKPT_DIR "dqn_model_pretrain_final.bin") &&
                        save_checkpoint(agent, 1, 0, global_minimax_depth, CKPT_DIR "checkpoint_phase1_0.bin");
                    }, []() { win_pump_messages(); Sleep(50); }) || g_exit_requested) {
                    g_pretrain_active = 0;
                    if (log) fclose(log);
                    service_signal_exit_request(); // Covers successful save + simultaneous signal.
                    return 0;
                }
                InterlockedExchange(&g_shutdown_state, 0);
                printf("[PRETRAIN] 已稳定毕业，权重转入正式 Phase 1。\n");
                g_pretrain_active = 0;
                if (log) fclose(log);
                return 1;
            }
        }
        if ((ep + 1) % 1000 == 0) {
            if (!sync_gpu_before_persist(agent))
                printf("[PRETRAIN] GPU 参数同步失败，本次 checkpoint 跳过\n");
            else {
                char ckpt[256];
                snprintf(ckpt, sizeof(ckpt), CKPT_DIR "checkpoint_phase0_%d.bin", ep + 1);
                if (!save_checkpoint(agent, 0, ep + 1, global_minimax_depth, ckpt)) {
                    printf("[PRETRAIN] Periodic checkpoint failed; training stopped.\n");
                    training_fail(3);
                    break;
                }
            }
        }
    }
    {
        char ckpt[256];
        service_signal_exit_request();
        snprintf(ckpt, sizeof(ckpt), CKPT_DIR "checkpoint_phase0_%d.bin", g_ctx.total_games_played);
        save_before_exit([&]() { return save_checkpoint(agent, 0,
            g_ctx.total_games_played, global_minimax_depth, ckpt) != 0; },
            []() { win_pump_messages(); Sleep(50); });
    }
    if (log) fclose(log);
    g_pretrain_active = 0;
    return 0;
}

// ============================================================================
// 训练主循环 train_dqn（增加 seed_id 参数）
// ============================================================================
void train_dqn(DQNAgent* agent, int total_episodes, int phase,
    float eps_start, float eps_end, int save_interval,
    int clone_opponent, float white_epsilon,
    int use_separate_buffer,
    int start_episode,
    int init_minimax_depth,
    const char* ckpt_prefix,
    int seed_id) {

    int is_population = (seed_id >= 0);
    int interrupted = 0;

    // 种群模式：跳过所有全局状态写入（在并行前已完成）
    if (!is_population) {
        g_current_agent = agent;
        side_balance_apply_scales(agent);

        /* Neural Probe: 训练启动时自动初始化共享内存 */
        neural_probe_init();

        if (save_interval <= 0) save_interval = 1000;

        if (!isfinite(g_learning_rate) || g_learning_rate <= 0.0f || g_learning_rate > 1.0f) {
            printf("[警告] 学习率异常(%.8f)，重置为 0.0005\n", g_learning_rate);
            g_learning_rate = 0.0005f;
        }

        if (ckpt_prefix != NULL) {
            g_ctx.total_games_played = 0;
            g_wins[0] = g_wins[1] = g_wins[2] = 0;
        }

        if (start_episode == 0) {
            global_minimax_depth = init_minimax_depth;
            if (g_start_time == 0) {
                g_wins[0] = g_wins[1] = g_wins[2] = 0;
                g_start_time = time(NULL);
                agent->concentration_factor = 0.5f;
                agent->prev_performance = 0.0f;
                g_ctx.best_winrate = -1.0f;
            }
        }
    }
    else {
        // 种群模式：只做局部初始化，不碰全局状态
        if (save_interval <= 0) save_interval = 1000;
        global_minimax_depth = init_minimax_depth;
    }

    // ========== 日志文件：种群模式共享文件 + seed_id 列 ==========
    if (is_population) {
        agent->seed_id = seed_id;
        if (agent->log_game) { fclose(agent->log_game); agent->log_game = NULL; }
        if (agent->log_train) { fclose(agent->log_train); agent->log_train = NULL; }
        agent->log_game = fopen(dqn_log_path("game.dat").c_str(), "a");    /* 通信兼日志 → logs\ */
        agent->log_train = fopen(dqn_log_path("train.dat").c_str(), "a");
        if (!agent->log_game || !agent->log_train) {
            printf("[警告] 种群训练日志文件创建失败 (seed %d)\n", seed_id);
        }
    }
    // ===============================================

    if (start_episode > 0) {
        if (!is_population) printf("[RESUME] Preserving checkpoint opponent network.\n");
    }
    else if (clone_opponent) {
        copy_net(&agent->opponent_net, &agent->q_net);
        if (!is_population)
            printf("阶段 %d：白棋使用 opponent_net（初始为 q_net 副本）\n", phase);
    }
    else {
        init_net(&agent->opponent_net);
        if (!is_population)
            printf("阶段 %d：白棋使用随机网络\n", phase);
    }

    if (!is_population) {
        if (use_separate_buffer)
            printf("阶段 %d：黑棋和白棋使用独立 Replay Buffer\n", phase);
        else
            printf("阶段 %d：黑棋和白棋共享 Replay Buffer\n", phase);

        printf("===== 阶段 %d 训练 (%d 局，学习率初始 %.6f) =====\n",
            phase, total_episodes, g_learning_rate);
        printf("  多头注意力已启用（4头，嵌入维度 %d）\n", ATTENTION_EMBED);
        printf("  课程学习使用全局偏移 %d，阶段由全局局数决定\n", g_ctx.global_episode_offset);
        printf("  评估/存档间隔：%d 局\n", save_interval);
    }

    const int PROGRESS_WIDTH = 50;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    LARGE_INTEGER start_1000;
    QueryPerformanceCounter(&start_1000);
    double total_elapsed = 0.0;

    // ========== 日志文件 ==========
    FILE* eval_log = NULL;
    FILE* minimax_log = NULL;
    if (is_population) {
        // 种群模式：共享文件，evaluate_and_save 已跳过，不需要单独打开
        eval_log = NULL;
        minimax_log = NULL;
    }
    else {
        eval_log = fopen(dqn_log_path("eval.dat").c_str(), "a");            /* 通信兼日志 → logs\ */
        minimax_log = fopen(dqn_log_path("minimax_games.dat").c_str(), "a");
    }
    if (!is_population && (!eval_log || !minimax_log)) {
        printf("[EVAL] Required evaluation logs are unavailable; stage was not started.\n");
        if (eval_log) fclose(eval_log);
        if (minimax_log) fclose(minimax_log);
        training_fail(3);
        return;
    }
    // =====================================================

    for (int ep = start_episode; ep < total_episodes; ep++) {
        while (!is_population && sentinel_guard_paused() && !g_exit_requested) {
            /* 安全暂停不推进 episode，仍服务快照、TUI 与恢复命令。 */
            service_snapshot_requests(agent);
            apply_pending_ipc_config(agent);
            publish_training_status(agent);
            win_pump_messages();
            Sleep(100);
        }
        if (g_exit_requested || g_training_fault) {
            interrupted = 1;
            if (!is_population) {
                service_signal_exit_request();
                char path[256];
                snprintf(path, sizeof(path), CKPT_TEMP_DIR "checkpoint_phase%d_%d.bin", phase, ep);
                save_before_exit([&]() { return save_checkpoint_and_memory([&]() {
                    return save_checkpoint(agent, phase, ep, global_minimax_depth, path) != 0;
                }); }, []() { win_pump_messages(); Sleep(50); });
            }
            break;
        }
        service_snapshot_requests(agent);
        if (!is_population) apply_pending_ipc_config(agent);
        static time_t last_hot_check = 0;
        time_t now = time(NULL);
        if (now - last_hot_check >= 5) {
            last_hot_check = now;
            check_hot_config();
        }

        /* This value used to be write-only while q_net always played black.
           It now directly chooses the online network's colour for this game. */
        agent->current_perspective = prng_int(&agent->train_rng, 2);

        if (agent->reward_episode != ep) {
            memset(agent->reward_stats, 0, sizeof(agent->reward_stats));
            memset(agent->reward_usage, 0, sizeof(agent->reward_usage));
            agent->reward_episode = ep;
        }

        int global_ep = ep + g_ctx.global_episode_offset;

        float epsilon_value = compute_epsilon(ep, global_ep, total_episodes);
        agent->epsilon = epsilon_value;
        if (g_hot_config.epsilon_override >= 0) {
            agent->epsilon = g_hot_config.epsilon_override;
        }

        float shaping_scale = compute_shaping_scale(ep, global_ep);
        agent->current_shaping_scale = shaping_scale;

        float pool_prob = compute_pool_prob(global_ep);
        float minimax_prob = compute_minimax_prob(global_ep);
        /* 正式训练从第 0 局开始，每一局都允许 Replay + 反向传播。
           不再让旧 curriculum 的 Phase0 充当 2 万局纯采样期。 */
        int train_enabled = 1;
        int target_mode = compute_target_mode(global_ep);
        int first_move_enabled = compute_first_move_enabled(global_ep);

        float episode_reward_weights[7];
        memcpy(episode_reward_weights, agent->reward_weights, sizeof(episode_reward_weights));

        /* 本局在线方的物理棋色一旦确定（上一行 current_perspective），就把 GPU 会话
           绑定到该方：设备上是这一方的 online/target，forward 与 train 都作用于它。
           切换绑定会先把上一方的设备状态写回其 CPU 影子，绝不串线。
           未开启独立网络时 indep_active()==0，本块是空操作。 */
        if (independent_training::indep_active(agent)) {
            const int side = agent->current_perspective;
            if (gpu_net_active() && gpu_net_bind_side(agent, side) != 0) {
                printf("[INDEP] 本局 GPU 绑定到 %s 失败；训练停止，避免串线。\n",
                    side == 0 ? "black" : "white");
                training_fail_detail(1, "gpu_bind_side_episode", side);
                interrupted = 1;
                InterlockedExchange(&g_shutdown_state, 5);
                break;
            }
            agent->indep->set_bound(independent_training::side_from_physical(side));
        }

        LARGE_INTEGER t_start, t_end;
        QueryPerformanceCounter(&t_start);
        play_episode(agent, ep, global_ep, phase, agent->current_perspective,
            minimax_prob, pool_prob,
            train_enabled, target_mode, first_move_enabled,
            use_separate_buffer, total_episodes, &g_ctx, is_population);
        if(!is_population && !g_training_fault && !competency_after_game(agent,[](){return g_exit_requested!=0;}))
            training_fail_detail(2,"competency_runtime",0);
        if (g_training_fault) {
            interrupted = 1;
            InterlockedExchange(&g_shutdown_state, 5);
            break;
        }
        if (!is_population) {
            log_episode_rewards(global_ep + 1, agent->epsilon,
                agent->reward_stats, episode_reward_weights);
        }
        QueryPerformanceCounter(&t_end);
        double episode_time = (double)(t_end.QuadPart - t_start.QuadPart) / freq.QuadPart;
        if (!is_population)
            log_episode_time(global_ep + 1, (float)episode_time);

        // 只有主训练才更新已完成总局数和学习率衰减
        if (!is_population) {
            g_ctx.total_games_played = global_ep + 1; /* global_ep 为 0-based，状态为完成局数 */

            /* 多快照采集：先判断是否到期，避免过去每局下载 Q+Target 共约 7MB，
               然后才丢弃 61/62 次结果。诊断快照失败不是训练状态失败。 */
            const uint64_t snapshot_episode = (uint64_t)global_ep + 1;
            if (snapshot_collector_due(&g_snap_collector, snapshot_episode)) {
                if (gpu_net_active()) {
                    std::unique_ptr<NeuralNet> frozen(new (std::nothrow) NeuralNet);
                    /* 快照必须明确是哪一方的权重：按方路由下用该方在线网络。
                       方案 (b) 的批次循环会把绑定留在「最后训练的那一方」，
                       所以这里必须**先显式绑定 current_perspective** 再取快照，
                       否则黑方在线的局会被 gpu_net_snapshot_side 的 bound==side
                       守卫拒绝并静默跳过，冻结对手/历史池的来源就少了一半。 */
                    int snap_ok = -1;
                    if (frozen && independent_training::indep_active(agent)) {
                        const int snap_side = agent->current_perspective;
                        if (gpu_net_bind_side(agent, snap_side) != 0) {
                            printf("[SNAPSHOT] 绑定 side=%d 失败，本局技能快照跳过\n", snap_side);
                        } else {
                            agent->indep->set_bound(
                                independent_training::side_from_physical(snap_side));
                            snap_ok = gpu_net_snapshot_side(agent, snap_side, frozen.get());
                        }
                    } else if (frozen) {
                        snap_ok = gpu_net_snapshot_q(frozen.get());
                    }
                    if (snap_ok == 0)
                        snapshot_collector_push(&g_snap_collector, frozen.get(), snapshot_episode);
                    else if (independent_training::indep_active(agent) && snap_ok != -1)
                        printf("[SNAPSHOT] Optional skill snapshot skipped; training continues.\n");
                }
                else {
                    snapshot_collector_push(&g_snap_collector,
                        independent_training::agent_online_net(agent,
                            independent_training::side_from_physical(agent->current_perspective)),
                        snapshot_episode);
                }
            }

            if (g_ctx.total_games_played % LR_DECAY_INTERVAL == 0) {
                g_learning_rate *= LR_DECAY_FACTOR;
                /* 按方路由下每方各自持有学习率（安全回退只改一方）。因此全局衰减
                   必须显式下发到两方，否则两方会永远停留在首批提交时的取值，
                   LR 衰减对独立网络静默失效。 */
                if (independent_training::indep_active(agent)) {
                    agent->indep->set_learning_rate(independent_training::Side::Black, g_learning_rate);
                    agent->indep->set_learning_rate(independent_training::Side::White, g_learning_rate);
                }
                printf("学习率已衰减至 %.6f\n", g_learning_rate);   /* NoTerminal 版：去 \r */
            }

            // Log stats every 100 episodes
            if (g_ctx.total_games_played % 100 == 0) {
                // (GPU stats logging will be added when GPU training is active)
            }
            /* 切换开销与按方吞吐：每 100 局打印一次，写进日志目录以便无终端取证。 */
            if (independent_training::indep_active(agent) && g_ctx.total_games_played % 100 == 0) {
                const int ep_now = g_ctx.total_games_played;
                const double t_now = now_seconds();
                if (g_indep_perf_prev_t <= 0.0) {
                    g_indep_perf_prev_t = t_now;
                    g_indep_perf_prev_ep = ep_now;
                }
                const double dt = t_now - g_indep_perf_prev_t;
                const int dep = ep_now - g_indep_perf_prev_ep;
                const double interval_eps = (dt > 0.0 && dep > 0) ? (double)dep / dt : 0.0;
                printf("[INDEP] switch cost: binds=%llu total=%.3fs avg=%.3f ms/bind; "
                    "interval throughput: %.3f episodes/s over %d episodes (%.3fs)\n",
                    g_indep_bind_count, g_indep_bind_seconds,
                    g_indep_bind_count ? g_indep_bind_seconds * 1000.0 / (double)g_indep_bind_count : 0.0,
                    interval_eps, dep, dt);
                FILE* perf = fopen(LOG_DIR "indep_switch_cost.log", "a");
                if (perf) {
                    fprintf(perf, "ep=%d binds=%llu total_s=%.6f avg_ms=%.6f "
                        "interval_episodes=%d interval_s=%.6f eps_per_s=%.6f\n",
                        ep_now, g_indep_bind_count, g_indep_bind_seconds,
                        g_indep_bind_count ? g_indep_bind_seconds * 1000.0 / (double)g_indep_bind_count : 0.0,
                        dep, dt, interval_eps);
                    fclose(perf);
                }
                g_indep_perf_prev_t = t_now;
                g_indep_perf_prev_ep = ep_now;
                FILE* state = fopen(LOG_DIR "indep_side_state.log", "a");
                if (state) {
                    const independent_training::SideUpdateState& bs =
                        agent->indep->updates(independent_training::Side::Black);
                    const independent_training::SideUpdateState& ws =
                        agent->indep->updates(independent_training::Side::White);
                    fprintf(state, "ep=%d black_committed=%llu black_rejected=%llu "
                        "black_rejected_delta=%llu black_rejected_loss=%llu black_delta_ema=%.9g "
                        "black_lr=%.9g | white_committed=%llu white_rejected=%llu "
                        "white_rejected_delta=%llu white_rejected_loss=%llu white_delta_ema=%.9g "
                        "white_lr=%.9g\n",
                        g_ctx.total_games_played,
                        bs.committed_batches, bs.rejected_batches, bs.rejected_delta_gate,
                        bs.rejected_loss_batches, bs.delta_ema, bs.learning_rate,
                        ws.committed_batches, ws.rejected_batches, ws.rejected_delta_gate,
                        ws.rejected_loss_batches, ws.delta_ema, ws.learning_rate);
                    fclose(state);
                }
            }
            publish_training_status(agent);
        }

        int local_ep = ep % save_interval;
        /* global_ep 已在循环外定义（行 987），此处直接使用 */
        /* 进度条用全局 EP 计算（不是 local_ep），避免从中间恢复时显示 0% */
        int global_seg = global_ep % save_interval;
        int filled = (int)((float)global_seg / save_interval * PROGRESS_WIDTH);
        char bar[PROGRESS_WIDTH + 2];
        memset(bar, '=', filled);
        if (filled < PROGRESS_WIDTH) {
            bar[filled] = '>';
            bar[filled + 1] = '\0';
        }
        else {
            bar[PROGRESS_WIDTH] = '\0';
        }
        float percent = (float)global_seg / save_interval * 100;

        LARGE_INTEGER now2;
        QueryPerformanceCounter(&now2);
        double elapsed_1000 = (double)(now2.QuadPart - start_1000.QuadPart) / freq.QuadPart;
        double remaining_segment = 0.0;
        if (global_seg > 0) {
            double avg_ep_this_segment = elapsed_1000 / global_seg;
            int remaining_eps_in_segment = save_interval - global_seg;
            remaining_segment = avg_ep_this_segment * remaining_eps_in_segment;
        }
        if (is_population && seed_id >= 0 && seed_id < POP_SIZE) {
            // 种群模式：更新数据 + 每20局刷新一次
            AgentProgress* prog = &g_agent_progress[seed_id];
            prog->local_ep = local_ep;
            prog->save_interval = save_interval;
            prog->episode_time = episode_time;
            prog->active = 1;

            if (local_ep % 20 == 0) {
#pragma omp critical
                draw_progress_frame(POP_SIZE);
            }
        }
        else {
            // Main training mode: progress bar + memory status
            g_ep_time = (float)episode_time;
            g_elapsed = (float)elapsed_1000;
            g_remaining = (float)remaining_segment;
            /* 进度：\r 覆盖窗口最后一行（原地刷新，效果同控制台；窗口终端模拟 \r 覆盖） */
            printf("\rProgress: [%-50s] %5.1f%% (EP:%d  %d/%d)  Time: %.3fs | Elapsed: %.1fs | Remaining: %.1fs | Mem: %s Cache:%d KB:%d\n",
                bar, percent, global_ep, local_ep, save_interval, episode_time, elapsed_1000, remaining_segment,
                g_hot_config.memory_enabled ? "ON" : "OFF", g_cache_count, g_kb_count);
            fflush(stdout);
        }

        /* 窗口消息泵 + 关闭请求检查（WM_CLOSE → 保存 checkpoint 后退出） */
        win_pump_messages();
        if (g_exit_requested) {
            interrupted = 1;
            if (!is_population) {
                printf("\n[EXIT] 关闭请求：同步 GPU 参数并保存当前进度后退出\n");
                char ckpt_name[128];
                snprintf(ckpt_name, sizeof(ckpt_name), CKPT_TEMP_DIR "checkpoint_phase%d_%d.bin", phase, ep + 1);
                {
                    if (save_before_exit([&]() { return save_checkpoint_and_memory([&]() {
                        return save_checkpoint(agent, phase, ep + 1, global_minimax_depth, ckpt_name) != 0;
                    }); },
                        []() { win_pump_messages(); Sleep(50); })) {
                        /* 只有新档已原子提交成功，才能清理旧中间档。保存失败时旧档
                           是唯一可恢复点，绝不能继续删除。 */
                        WIN32_FIND_DATAA tfd;
                        HANDLE th = FindFirstFileA(CKPT_TEMP_DIR "*.bin", &tfd);
                        if (th != INVALID_HANDLE_VALUE) {
                            do {
                                char old[160];
                                snprintf(old, sizeof(old), CKPT_TEMP_DIR "%s", tfd.cFileName);
                                if (_stricmp(old, ckpt_name) != 0) {
                                    DeleteFileA(old);
                                    char old_pool[192];
                                    snprintf(old_pool, sizeof(old_pool), "%s.pool", old);
                                    DeleteFileA(old_pool);
                                }
                            } while (FindNextFileA(th, &tfd));
                            FindClose(th);
                        }
                    }
                    else {
                        printf("[EXIT] 新 checkpoint 保存失败，已保留所有旧中间档。\n");
                    }
                }
                fflush(stdout);
            }
            break;
        }

        /* 保存 checkpoint：用全局 EP 判断（ep + offset），避免从中间恢复后永远不保存 */
        if ((global_ep + 1) % save_interval == 0 && (ep + 1) % save_interval == 0) {
            if (!is_population) {
                evaluate_and_save(agent, phase, ep, total_episodes, save_interval,
                    eval_log, minimax_log, ckpt_prefix, seed_id);
                if (g_exit_requested || g_training_fault) {
                    interrupted = 1;
                    char path[256];
                    snprintf(path, sizeof(path), CKPT_TEMP_DIR "checkpoint_phase%d_%d.bin", phase, ep + 1);
                    save_before_exit([&]() { return save_checkpoint_and_memory([&]() {
                        return save_checkpoint(agent, phase, ep + 1, global_minimax_depth, path) != 0;
                    }); }, []() { win_pump_messages(); Sleep(50); });
                    break; // including the last episode: do not advance the phase on cancellation
                }
                // Save memory system every checkpoint
                if (g_hot_config.memory_enabled) {
                    memory_save_cache_perspective("memory_cache_b.dat", 0);
                    memory_save_cache_perspective("memory_cache_w.dat", 1);
                    memory_save_kb_perspective("memory_kb_b.dat", 0);
                    memory_save_kb_perspective("memory_kb_w.dat", 1);
                }
            }
            QueryPerformanceCounter(&start_1000);
            total_elapsed += elapsed_1000;
        }
    }

    // 种群模式：训练线程不做任何磁盘操作，由主线程统一保存
    if (is_population) {
        if (eval_log) fclose(eval_log);
        if (minimax_log) fclose(minimax_log);
        return;
    }

    /* 中断只代表本阶段暂停，绝不能执行阶段完成逻辑、生成 final/下一阶段档，
       也不能增加完整阶段的 global offset。保留退出标志供 main 停止后续阶段。 */
    if (interrupted) {
        if (eval_log) fclose(eval_log);
        if (minimax_log) fclose(minimax_log);
        return;
    }

    // ===== 以下仅主训练执行 =====
    /* offset 表示当前阶段之前已经完成的总局数，与本次从阶段中何处恢复无关。 */
    g_ctx.global_episode_offset += total_episodes;

    /* 阶段末保存的 model/checkpoint 同样必须包含 GPU 上的最新权重。 */
    if (!sync_gpu_before_persist(agent)) {
        printf("[SAVE] 阶段结束时 GPU 参数同步失败，拒绝写入 final/checkpoint\n");
        training_fail(1);
        InterlockedExchange(&g_shutdown_state, 5);
        if (eval_log) fclose(eval_log);
        if (minimax_log) fclose(minimax_log);
        return;
    }

    char model_fname[256];
    snprintf(model_fname, sizeof(model_fname), CKPT_DIR "dqn_model_phase%d_final.bin", phase);

    printf("阶段 %d 完成。统计：黑胜 %d，白胜 %d，平局 %d\n",
        phase, g_wins[0], g_wins[1], g_wins[2]);

    char ckpt_name[256];
    const char* prefix = ckpt_prefix ? ckpt_prefix : CKPT_DIR "checkpoint_phase";
    snprintf(ckpt_name, sizeof(ckpt_name), "%s%d_%d.bin", prefix, phase, total_episodes);
    bool committed = save_before_exit([&]() {
        if (!save_model_pair(agent, model_fname) ||
            !save_checkpoint(agent, phase, total_episodes, global_minimax_depth, ckpt_name)) return false;
        if (ckpt_prefix == NULL) {
            char end_name[256];
            snprintf(end_name, sizeof(end_name), CKPT_DIR "checkpoint_phase%d_0.bin", phase + 1);
            if (!save_checkpoint(agent, phase + 1, 0, global_minimax_depth, end_name)) return false;
        }
        return true;
    }, []() { win_pump_messages(); Sleep(50); });
    if (committed && !g_exit_requested) InterlockedExchange(&g_shutdown_state, 0);

    // Save memory system at end of phase
    if (!is_population && g_hot_config.memory_enabled) {
        memory_save_cache_perspective("memory_cache_b.dat", 0);
        memory_save_cache_perspective("memory_cache_w.dat", 1);
        memory_save_kb_perspective("memory_kb_b.dat", 0);
        memory_save_kb_perspective("memory_kb_w.dat", 1);
        printf("[MEMORY] Saved B cache(%d) KB(%d) | W cache(%d) KB(%d)\n",
            memory_cache_count_perspective(0), memory_kb_count_perspective(0),
            memory_cache_count_perspective(1), memory_kb_count_perspective(1));
    }

    if (eval_log) fclose(eval_log);
    if (minimax_log) fclose(minimax_log);
}

// ============================================================================
// 交互与评估函数
// ============================================================================
void evaluate_ai(DQNAgent* agent, int games) {
    printf("\n=== AI 自我对弈评估 (%d 局) ===\n", games);
    int wins[3] = { 0, 0, 0 };
    for (int g = 0; g < games; g++) {
        GameState game; reset_game(&game);
        int completed = 0;
        for (int step = 0; step < CELL_COUNT; ++step) {
            int use_opponent = (game.turn == 1);
            int action = select_action(agent, &game, use_opponent, 0.0f, &agent->eval_rng, 0);
            if (g_training_fault || !is_legal_move(&game, action)) {
                if (!g_training_fault) training_fail(2);
                return;
            }
            make_move(&game, action);
            int winner = check_winner(&game);
            if (winner) {
                if (winner == 1) wins[0]++;
                else wins[1]++;
                completed = 1;
                break;
            }
            if (is_full(&game)) { wins[2]++; completed = 1; break; }
        }
        if (!completed) {
            training_fail(2);
            return;
        }
    }
    printf("黑胜: %d (%.1f%%), 白胜: %d (%.1f%%), 平局: %d (%.1f%%)\n",
        wins[0], 100.0 * wins[0] / games,
        wins[1], 100.0 * wins[1] / games,
        wins[2], 100.0 * wins[2] / games);
}

void play_vs_ai(DQNAgent* agent) {
    GameState game; reset_game(&game);
    printf("\n您执 O（后手），AI执 X（先手）。\n");
    printf("输入走法：行列号（如 1 2 表示第1行第2列）\n");
    while (1) {
        printf("  1 2 3 4 5\n");
        for (int r = 0; r < BOARD_SIZE; r++) {
            printf("%d ", r + 1);
            for (int c = 0; c < BOARD_SIZE; c++) {
                int val = game.board[r * BOARD_SIZE + c];
                char ch = (val == 0) ? '.' : (val == 1) ? 'X' : 'O';
                printf("%c ", ch);
            }
            printf("\n");
        }
        if (game.turn == 0) {
            int action = select_action(agent, &game, 0, 0.0f, &agent->eval_rng, 0);
            if (g_training_fault || !is_legal_move(&game, action)) {
                if (!g_training_fault) training_fail(2);
                printf("AI 推理失败，已安全结束本局。\n");
                return;
            }
            make_move(&game, action);
            int r = action / BOARD_SIZE, c = action % BOARD_SIZE;
            printf("AI 下在了 %d %d\n", r + 1, c + 1);
        }
        else {
            int row, col;
            printf("您的走法 (行 列): ");
            if (scanf("%d %d", &row, &col) != 2) break;
            int pos = (row - 1) * BOARD_SIZE + (col - 1);
            if (pos < 0 || pos >= CELL_COUNT || game.board[pos] != 0) {
                printf("非法走法，请重新输入。\n");
                continue;
            }
            make_move(&game, pos);
        }
        int winner = check_winner(&game);
        if (winner != 0) {
            printf("  1 2 3 4 5\n");
            for (int r = 0; r < BOARD_SIZE; r++) {
                printf("%d ", r + 1);
                for (int c = 0; c < BOARD_SIZE; c++) {
                    int val = game.board[r * BOARD_SIZE + c];
                    char ch = (val == 0) ? '.' : (val == 1) ? 'X' : 'O';
                    printf("%c ", ch);
                }
                printf("\n");
            }
            if (winner == 1) printf("AI 获胜！\n");
            else printf("您获胜了！\n");
            break;
        }
        if (is_full(&game)) { printf("平局！\n"); break; }
    }
}

void test_check_winner(void) {
    int failed = 0;
    GameState g;
    /* 测试1：行方向四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 0] = 1;
    g.board[0 * BOARD_SIZE + 1] = 1;
    g.board[0 * BOARD_SIZE + 2] = 1;
    g.board[0 * BOARD_SIZE + 3] = 1;
    if (check_winner(&g) != 1) { printf("[TEST FAIL] 行四连未检测到\n"); failed++; }
    /* 测试2：列方向四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 0] = 1;
    g.board[1 * BOARD_SIZE + 0] = 1;
    g.board[2 * BOARD_SIZE + 0] = 1;
    g.board[3 * BOARD_SIZE + 0] = 1;
    if (check_winner(&g) != 1) { printf("[TEST FAIL] 列四连未检测到\n"); failed++; }
    /* 测试3：主对角线四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 0] = 1;
    g.board[1 * BOARD_SIZE + 1] = 1;
    g.board[2 * BOARD_SIZE + 2] = 1;
    g.board[3 * BOARD_SIZE + 3] = 1;
    if (check_winner(&g) != 1) { printf("[TEST FAIL] 主对角线四连未检测到\n"); failed++; }
    /* 测试4：副对角线四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 3] = 1;
    g.board[1 * BOARD_SIZE + 2] = 1;
    g.board[2 * BOARD_SIZE + 1] = 1;
    g.board[3 * BOARD_SIZE + 0] = 1;
    if (check_winner(&g) != 1) { printf("[TEST FAIL] 副对角线四连未检测到\n"); failed++; }
    /* 测试5：五连 */
    reset_game(&g);
    for (int i = 0; i < 5; i++) g.board[0 * BOARD_SIZE + i] = 1;
    if (check_winner(&g) != 1) { printf("[TEST FAIL] 五连未检测到\n"); failed++; }
    /* 测试6：不足四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 0] = 1;
    g.board[0 * BOARD_SIZE + 1] = 1;
    g.board[0 * BOARD_SIZE + 2] = 1;
    if (check_winner(&g) != 0) { printf("[TEST FAIL] 三连误报为胜利\n"); failed++; }
    /* 测试7：白棋四连 */
    reset_game(&g);
    g.board[0 * BOARD_SIZE + 0] = 2;
    g.board[0 * BOARD_SIZE + 1] = 2;
    g.board[0 * BOARD_SIZE + 2] = 2;
    g.board[0 * BOARD_SIZE + 3] = 2;
    if (check_winner(&g) != 2) { printf("[TEST FAIL] 白棋四连未检测到\n"); failed++; }
    /* 测试8：空棋盘 */
    reset_game(&g);
    if (check_winner(&g) != 0) { printf("[TEST FAIL] 空棋盘误报\n"); failed++; }
    if (failed == 0) printf("[TEST] check_winner 全部通过 (%d 项)\n", 8);
    else printf("[TEST] %d / %d 项失败\n", failed, 8);
}

// ============================================================================
// 主函数（含 OpenMP 并行化种群训练）
// ============================================================================
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow) {
    (void)hInstance; (void)hPrevInstance; (void)lpCmdLine; (void)nCmdShow;
    /* 工作目录固定为 exe 所在目录 */
    {
        wchar_t exePath[32768];
        DWORD len = GetModuleFileNameW(NULL, exePath, 32768);
        if (len > 0 && len < 32768) {
            wchar_t* slash = wcsrchr(exePath, L'\\');
            if (slash) { *slash = L'\0'; SetCurrentDirectoryW(exePath); }
        }
    }

    /* 单实例保护 */
    HANDLE trainer_instance = CreateMutexW(NULL, FALSE, L"Local\\DQN_Trainer_NoTerminal_SingleInstance_v1");
    DWORD trainer_mutex_error = GetLastError();
    if (!trainer_instance || trainer_mutex_error == ERROR_ALREADY_EXISTS) {
        if (trainer_instance) CloseHandle(trainer_instance);
        return 23;
    }
    struct TrainerInstanceGuard {
        HANDLE handle;
        ~TrainerInstanceGuard() { if (handle) CloseHandle(handle); }
    } trainer_instance_guard = { trainer_instance };

    CreateDirectoryA(LOG_DIR, NULL);
    CreateDirectoryA(CKPT_DIR, NULL);
    CreateDirectoryA(CKPT_TEMP_DIR, NULL);
    g_trainer_show_window_cb = win_show_window;
    g_trainer_request_exit_cb = request_graceful_exit;
    win_init();
    redirect_stdout_to_pipe();

    // 崩溃处理
    SetUnhandledExceptionFilter(crash_handler);

    // 控制台输出代码页设为 UTF-8（hipcc/clang 编译的中文按 UTF-8 输出）
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // 增大栈空间：OpenMP 线程 + 主线程
    _putenv_s("OMP_STACKSIZE", "16777216");  // 16MB
    _putenv_s("KMP_STACKSIZE", "16777216");  // Intel OpenMP 兼容

    // Register signal handler for graceful exit
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // ===== GPU 初始化（ROCm/HIP） =====
    printf("========================================\n");
    printf("  DQN Training System - ROCm GPU\n");
    printf("========================================\n");

    // Add ROCm bin to PATH for DLL dependencies
    {
        char path_env[4096];
        GetEnvironmentVariableA("PATH", path_env, sizeof(path_env));
        char new_path[8192];
        snprintf(new_path, sizeof(new_path), "D:\\AMD_ROCm\\bin;%s", path_env);
        SetEnvironmentVariableA("PATH", new_path);
    }

    // Set HIP environment variables
    SetEnvironmentVariableA("HIP_PLATFORM", "amd");
    SetEnvironmentVariableA("ROCM_PATH", "D:\\AMD_ROCm");
    SetEnvironmentVariableA("HIP_PATH", "D:\\AMD_ROCm");

    // 初始化 HIP/ROCm（整个程序由 hipcc 编译，HIP 直接静态链接）
    if (gpu_init() != 0) {
        printf("[ERROR] HIP/ROCm 初始化失败\n");
        return 1;
    }

    /* 窗口化：stdout 已在 wWinMain 开头 redirect 到 pipe（窗口显示 + trainer.log），此处不再重定向 */

    printf("\n========================================\n\n");

    test_check_winner();

    load_hot_config();
    load_config();  // Load saved config (memory settings, etc.)

    // Network weights and the persistent bank form one deployed model.
    memory_load_deployed();
    if (g_cache_count > 0 || g_kb_count > 0) {
        printf("[MEMORY] Loaded cache (%d: B%d/W%d) KB (%d: B%d/W%d)\n",
            g_cache_count, memory_cache_count_perspective(0), memory_cache_count_perspective(1),
            g_kb_count, memory_kb_count_perspective(0), memory_kb_count_perspective(1));
    }

    srand((unsigned)time(NULL));

    TrainingContext* ctx = &g_ctx;
    memset(ctx, 0, sizeof(TrainingContext));
    ctx->learning_rate = 0.0005f;
    ctx->save_interval = 1000;
    ctx->ema_alpha = 0.85f;
    ctx->depth_adjust_threshold = 3;

    load_params();
    if (!isfinite(g_learning_rate) || g_learning_rate <= 0.0f || g_learning_rate > 1.0f) {
        printf("[警告] 学习率异常(%.8f)，重置为默认值 0.0005\n", g_learning_rate);
        g_learning_rate = 0.0005f;
    }

    DQNAgent* agent = NULL;
    int from_population = 0;
    int latest_ep = -1;

    char dummy[256] = { 0 };
    int has_existing_checkpoint = 0;

    // 检查点文件有效性校验
    if (find_latest_checkpoint(dummy, sizeof(dummy), &latest_ep)) {
        struct _stat st;
        if (_stat(dummy, &st) == 0) {
            // 最小合法大小：OldNeuralNet 检查点（版本1/2，约6.4MB）
            long long min_valid_size = 1024;
            if (st.st_size < min_valid_size) {
                printf("[MAIN] 检查点文件 %s 太小 (%lld 字节)，跳过。\n", dummy, (long long)st.st_size);
                has_existing_checkpoint = 0;
            }
            else {
                has_existing_checkpoint = 1;
                printf("[MAIN] 找到检查点：%s (大小 %lld 字节, 支持版本 1-7)\n", dummy, (long long)st.st_size);
            }
        }
        else {
            printf("[MAIN] 无法获取检查点文件 %s 的大小，跳过加载。\n", dummy);
            has_existing_checkpoint = 0;
        }
    }
    else {
        has_existing_checkpoint = 0;
    }

    if (!has_existing_checkpoint) {
        const char* patterns[] = { CKPT_DIR "checkpoint_phase*_*.bin", CKPT_TEMP_DIR "checkpoint_phase*_*.bin" };
        for (const char* pattern : patterns) {
            WIN32_FIND_DATAA data;
            HANDLE search = FindFirstFileA(pattern, &data);
            if (search != INVALID_HANDLE_VALUE) {
                bool has_file = false;
                do { if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) { has_file = true; break; } }
                while (FindNextFileA(search, &data));
                FindClose(search);
                if (has_file) {
                    /* The scanner already tried to quarantine every invalid
                       candidate.  A file can remain only when Windows denied
                       the move (for example antivirus/indexer contention).
                       It is still unusable, so report persistently and start
                       clean instead of exiting in a way that looks like a crash. */
                    CreateDirectoryA(LOG_DIR, NULL);
                    FILE* recovery = fopen(LOG_DIR "checkpoint_recovery.log", "a");
                    if (recovery) {
                        fprintf(recovery, "Unrestorable checkpoint remains in %s; starting fresh.\n", pattern);
                        fclose(recovery);
                    }
                    printf("[MAIN] Invalid checkpoint remains (%s); starting fresh. Original file preserved.\n", pattern);
                }
            }
        }
    }

    if (has_existing_checkpoint) {
        // Load existing checkpoint
        agent = (DQNAgent*)calloc(1, sizeof(DQNAgent));
        if (!agent) {
            printf("[FATAL] Agent allocation failed!\n");
            return 1;
        }
        init_dqn_ex(agent, 0);
        from_population = 0;
    } else {
        // Skip population training - go straight to main training for GPU debugging
        printf("[MAIN] No checkpoint found, skipping population training (GPU debug mode)\n");
        agent = (DQNAgent*)calloc(1, sizeof(DQNAgent));
        if (!agent) {
            printf("[FATAL] Agent allocation failed!\n");
            return 1;
        }
        init_dqn_ex(agent, 0);
        from_population = 0;
    }

    if (!agent->replay_black || !agent->replay_white || !agent->opponent_pool) {
        printf("[FATAL] Agent 内部缓冲区分配失败，停止启动以避免空指针训练。\n");
        free_dqn(agent);
        free(agent);
        return 1;
    }

    // Population training removed for GPU debugging
    // goto skip_population; // removed - not needed

    /* 独立网络容器必须在 load_checkpoint 之前存在：V13 存档要把两方 online/target
       与按方计数精确恢复进该容器。容器创建时只是从共享权重复制一份种子且
       active==0，因此这一步不改变任何既有行为；真正开启按方路由在 pretrain 毕业
       之后（见下方 INDEP 接入点）。 */
    if (!independent_training::indep_ensure(agent)) {
        printf("[FATAL] 独立网络容器分配失败，停止启动以免静默退回共享网络。\n");
        free_dqn(agent);
        free(agent);
        return 1;
    }

    load_params();

    int start_phase = 1, start_episode = 0;

    if (from_population) {
        g_ctx.global_episode_offset = 0;
        g_ctx.minimax_depth = 2;
        printf("[MAIN] 从种群训练继承 Agent，课程阶段从零开始，与主训练阶段配置对齐。\n");
        g_current_agent = agent;
    }
    int loaded_checkpoint = has_existing_checkpoint &&
        load_checkpoint(agent, &start_phase, &start_episode, &g_ctx.minimax_depth, dummy, 1);
    if (loaded_checkpoint) {
        /* IND-05：恢复 HARD 目标同步相位，使保存后继续训练与连续运行等价。 */
        g_thread_hard_counter = agent->target_hard_counter;
        if (start_phase == 1 && start_episode == 0 && !prepare_formal_task(agent)) {
            free_dqn(agent); free(agent); gpu_shutdown(); return 1;
        }
        if (start_phase > 0) {
            /* 独立网络走两 ply 契约，必须用**跨度**谓词清理一 ply 非终局回放。
               绝不能在这里用 discard_solitaire_replay：它的谓词是 turn 相等，
               而两 ply 正式样本同样满足 turn 相等，会把合法的 V13 新回放整批删掉；
               同时对一 ply 正式样本一条都不删，随后第一条非终局样本就命中
               replay_span 把 g_training_fault 闩死（正式阶段中途的 V11/V12 档）。
               未开启按方路由时保持历史行为逐位不变。 */
            int removed = 0;
            if (independent_training::indep_active(agent)) {
                removed = discard_ply1_replay(agent->replay_black,
                    &agent->replay_black_count, &agent->replay_black_idx);
                removed += discard_ply1_replay(agent->replay_white,
                    &agent->replay_white_count, &agent->replay_white_idx);
                if (removed) {
                    printf("[RESUME] Removed %d one-ply replay entries incompatible with the "
                        "2-ply independent-network contract (from_semantics=1 to_semantics=2); "
                        "weights/progress preserved.\n", removed);
                    FILE* mig = fopen(LOG_DIR "replay_semantics_migration.log", "a");
                    if (mig) {
                        SYSTEMTIME now; GetLocalTime(&now);
                        fprintf(mig, "%04u-%02u-%02u %02u:%02u:%02u source=resume_phase%d_ep%d "
                            "from_semantics=1 to_semantics=2 dropped=%d "
                            "action=cleared_ply1_replay_weights_preserved\n",
                            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
                            start_phase, start_episode, removed);
                        fclose(mig);
                    }
                }
            } else {
                removed = discard_solitaire_replay(agent->replay_black,
                    &agent->replay_black_count, &agent->replay_black_idx);
                removed += discard_solitaire_replay(agent->replay_white,
                    &agent->replay_white_count, &agent->replay_white_idx);
                if (removed) printf("[RESUME] Removed %d incompatible solo-task replay entries; weights/progress preserved.\n", removed);
            }
        }
        /* checkpoint episode 是阶段内偏移；课程调度需要补上之前完整阶段。 */
        switch (start_phase) {
        case 0:
        case 1: g_ctx.global_episode_offset = 0; break;
        case 2: g_ctx.global_episode_offset = PHASE1_EPISODES; break;
        case 3: g_ctx.global_episode_offset = PHASE1_EPISODES + PHASE2_EPISODES; break;
        case 4:
            g_ctx.global_episode_offset = PHASE1_EPISODES + PHASE2_EPISODES + PHASE3_EPISODES;
            break;
        default:
            g_ctx.global_episode_offset = PHASE1_EPISODES + PHASE2_EPISODES +
                PHASE3_EPISODES + PHASE4_EPISODES;
            break;
        }
        g_ctx.total_games_played = g_ctx.global_episode_offset + start_episode;
        if (agent->checkpoint_version < 6 && start_phase > 0) {
            g_learning_rate *= powf(LR_DECAY_FACTOR,
                (float)(g_ctx.total_games_played / LR_DECAY_INTERVAL));
            printf("[RESUME] Legacy checkpoint: reconstructed scheduled LR %.8f from params.\n", g_learning_rate);
        }
        g_current_agent = agent;
    }
    else {
        /* Only a genuinely fresh run starts Phase 0. A rejected checkpoint
           must not silently overwrite the previous training timeline. */
        if (has_existing_checkpoint) {
            printf("[MAIN] Checkpoint rejected; refusing to overwrite the existing training timeline.\n");
            mark_intentional_shutdown();
            free_dqn(agent); free(agent); gpu_shutdown(); return 1;
        }
        g_ctx.global_episode_offset = 0;
        start_phase = 0;
        start_episode = 0;
        g_ctx.minimax_depth = 2;
        g_current_agent = agent;
    }

    // All log handles are still closed. The checkpoint is the authority for
    // progress; a watchdog marker can be stale or phase-local.
    if (!select_training_log_directory(LOG_DIR, start_phase == 0) ||
        (loaded_checkpoint && !rollback_training_logs(training_writer_dir().c_str(),
            g_ctx.total_games_played, agent->global_step)) ||
        (!loaded_checkpoint && GetFileAttributesA("rollback_to.txt") != INVALID_FILE_ATTRIBUTES)) {
        printf("[ROLLBACK] Failed/incomplete rollback; original backups retained. Training not started.\n");
        mark_intentional_shutdown();
        free_dqn(agent); free(agent); g_current_agent = NULL; gpu_shutdown(); return 1;
    }
    if (loaded_checkpoint) remove("rollback_to.txt");
    if (apply_recovery_override("recovery_override.txt") < 0 || !open_dqn_logs(agent)) {
        printf("[STARTUP] Invalid recovery request or log writer failure; training not started.\n");
        mark_intentional_shutdown();
        free_dqn(agent); free(agent); g_current_agent = NULL; gpu_shutdown(); return 1;
    }
    start_ipc_server();

    // 激活 GPU 驻留网络：上传 q_net/target_net 到 GPU，推理走 GPU
    if (gpu_net_activate(agent) == 0) {
        printf("[GPU] 神经网络参数已驻留 GPU，前向推理走 GPU\n");
    } else {
        printf("[WARN] GPU 网络激活失败，使用 CPU 推理\n");
    }

    if (loaded_checkpoint && start_phase > 0 && agent->skill_tracker.initialized != SKILL_SUITE_VERSION) {
        printf("[SKILL] Legacy checkpoint has no graduation baseline; measuring resume baseline now.\n");
        SkillTestResult resume_result;
        run_skill_tests(agent, g_ctx.total_games_played, &resume_result, gpu_net_active());
        skill_tracker_init(&agent->skill_tracker);
        skill_tracker_update(&agent->skill_tracker, &resume_result);
    }

    /* 无存档，或上次停在 phase0：先完成单人四连课程。phase1_0 存档代表已毕业。 */
    if(g_hot_config.competency.enabled || (g_hot_config.competency.rehearsal_enabled && loaded_checkpoint && start_phase>0)){
        if(!independent_training::indep_active(agent) && !independent_training::indep_start_from_shared(agent)){
            mark_intentional_shutdown();stop_ipc_server();gpu_net_deactivate();gpu_shutdown();free_dqn(agent);free(agent);return 1;
        }
        // A restored V14 stage is authoritative; missing old state starts at Stage 0.
        if(!agent->competency.initialized){agent->competency=CompetencyState{};if(!g_hot_config.competency.enabled)agent->competency.stage=5;}
        agent->competency.initialized=1;
        const int removed=discard_ply1_replay(agent->replay_black,&agent->replay_black_count,&agent->replay_black_idx)+
            discard_ply1_replay(agent->replay_white,&agent->replay_white_count,&agent->replay_white_idx);
        if(removed)printf("[COURSE] Removed %d incompatible one-ply replay samples; source checkpoint retained.\n",removed);
        if(!competency_refresh_opponents(agent,g_hot_config.competency)){
            mark_intentional_shutdown();stop_ipc_server();gpu_net_deactivate();gpu_shutdown();free_dqn(agent);free(agent);return 1;
        }
        if(start_phase==0){start_phase=1;start_episode=0;}
        competency_event(agent,"runtime_started");competency_write_status(agent);
    }
    if ((!loaded_checkpoint || start_phase == 0) && !g_hot_config.competency.enabled) {
        int pretrain_start = (start_phase == 0) ? start_episode : 0;
        if (!train_single_player_pretrain(agent, pretrain_start)) {
            printf(g_shutdown_state == 4 ? "[EXIT] Phase 0 saved; ready to resume.\n" :
                "[EXIT] Fatal training error; current progress NOT saved. Last checkpoint preserved.\n");
            stop_ipc_server();
            gpu_net_deactivate();
            gpu_shutdown();
            free_dqn(agent);
            free(agent);
            return g_training_fault ? 2 : 0;
        }
        // Graduation checkpoint has already committed.  Always attempt the new
        // timeline even if a Phase-0 fclose reports a delayed flush error.  The
        // old short-circuit expression skipped both directory selection and
        // reopen in that case, making a recoverable log error look like a
        // checkpoint-incompatible crash exactly at graduation.
        const bool phase0_logs_clean = close_training_logs(agent);
        if (!phase0_logs_clean)
            printf("[LOG] Phase 0 writer reported a close/flush error; Phase 1 checkpoint is safe, continuing log handoff.\n");
        const bool formal_timeline_selected = select_training_log_directory(LOG_DIR, false);
        const bool formal_logs_opened = formal_timeline_selected && open_dqn_logs(agent);
        if (!formal_logs_opened) {
            printf("[LOG] Timeline handoff failed; graduation checkpoint preserved.\n");
            mark_intentional_shutdown(); stop_ipc_server(); gpu_net_deactivate();
            gpu_shutdown(); free_dqn(agent); free(agent); return 1;
        }
        start_phase = 1;
        start_episode = 0;
        g_ctx.global_episode_offset = 0;
        g_ctx.total_games_played = 0;
        g_wins[0] = g_wins[1] = g_wins[2] = 0;
        g_start_time = time(NULL);
    }

    /* ===== 黑白独立网络：正式训练路径接入点 =====================================
       位置刻意选在这里，理由是可核验的：
         - phase0 单人预训练已经毕业（或从正式存档恢复），共享预训练权重已经定型；
         - 这是「由共享权重初始化两个独立网络」的唯一时刻，之后绝不再复制；
         - 恢复 V11/V12 旧档时加载出的仍是共享 q_net/target_net，本块把它作为
           迁移种子克隆到黑白两方，并在日志里显式记录这是迁移初始化。
       未接入或初始化失败时保持历史共享语义（indep_active()==0），
       生产路径逐字节不变，不会因为本块失败而假装隔离成功。 =============== */
    {
        IndependentNetworks* nets = independent_training::indep_ensure(agent);
        /* A/B 对照开关：同一份构建、同一套自博弈规则，**只切按方路由**。
             未设置      → 正常（按方启用，本次测试的机制）
             DQN_INDEP_ACTIVE=0 → 共享网络自博弈基线（indep_active 保持 0）
             DQN_INDEP_ACTIVE=1 → 显式要求启用
           这是为了让「两网络 vs 一网络」的差异不被「自博弈 vs minimax/历史池对手」
           之类的差异污染。它只影响是否播种两方，不改变自博弈规则本身。 */
        const char* ab_env = getenv("DQN_INDEP_ACTIVE");
        const int ab_request = (!ab_env || !ab_env[0]) ? 1 : (atoi(ab_env) != 0);
        if (!nets) {
            printf("[INDEP] 独立网络分配失败；本次运行继续使用共享网络。\n");
        } else if (!ab_request && !independent_training::indep_active(agent)) {
            printf("[INDEP] A/B: DQN_INDEP_ACTIVE=0 -> 共享网络自博弈基线"
                   "（indep_active=0，与按方路径同一份构建/同一套自博弈规则）。\n");
        } else if (independent_training::indep_active(agent)) {
            /* V13 存档已经把两方精确恢复；此处绝不重新复制共享权重。 */
            printf("[INDEP] 已从 V13 存档恢复两方独立网络与按方状态；不重新播种。\n");
        } else {
            /* 播种：两方都从当前共享权重出发，只迁移权重，不迁移回放语义。 */
            // GPT6-Astra: graduation/legacy migration must upgrade the already
            // active shared GPU session to four handles, using the device seed.
            if (!independent_training::indep_start_from_shared(agent)) {
                mark_intentional_shutdown(); stop_ipc_server();
                gpu_net_deactivate(); gpu_shutdown();
                free_dqn(agent); free(agent); g_current_agent = NULL;
                return 1;
            }
            /* 旧回放迁移：预训练产物与 V11/V12 存档里的回放属于 PLY1 契约，
               混入 PLY2 契约要么第一条非终局样本就命中 replay_span 把训练闩死，
               要么静默错 bootstrap。按 HANDOFF:58「显式清空不兼容回放并记录原因」
               处理：清空并记录条数，网络权重完全不受影响。
               注意预训练毕业路径已经清过一次（prepare_formal_task），但
               「阶段中途的 V11/V12 存档恢复」没有经过那条路径，所以这里必须做。 */
            {
                const int dropped_black = agent->replay_black_count;
                const int dropped_white = agent->replay_white_count;
                int dropped = 0;
                if (dropped_black > 0 && agent->replay_black) {
                    dropped += discard_ply1_replay(agent->replay_black,
                        &agent->replay_black_count, &agent->replay_black_idx);
                }
                if (dropped_white > 0 && agent->replay_white) {
                    dropped += discard_ply1_replay(agent->replay_white,
                        &agent->replay_white_count, &agent->replay_white_idx);
                }
                printf("[INDEP] 回放语义迁移：由 PLY1 升到 PLY2，丢弃 %d 条一 ply 非终局样本"
                    "（black=%d white=%d 原有），不重新标记、不重算目标；权重不受影响。\n",
                    dropped, dropped_black, dropped_white);
                FILE* mig = fopen(LOG_DIR "replay_semantics_migration.log", "a");
                if (mig) {
                    SYSTEMTIME now; GetLocalTime(&now);
                    fprintf(mig, "%04u-%02u-%02u %02u:%02u:%02u source=indep_seed "
                        "from_semantics=1 to_semantics=2 dropped=%d dropped_black_capacity=%d "
                        "dropped_white_capacity=%d action=cleared_ply1_replay_weights_preserved\n",
                        now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
                        dropped, dropped_black, dropped_white);
                    fclose(mig);
                }
            }
            if (gpu_net_active() && gpu_net_bind_side(agent, 0) != 0) {
                /* 设备状态不可信，不能一边宣称隔离一边让句柄串线。 */
                printf("[INDEP] GPU 按方绑定失败；训练停止，避免两方权重串线。\n");
                mark_intentional_shutdown();
                stop_ipc_server();
                gpu_net_deactivate();
                gpu_shutdown();
                free_dqn(agent);
                free(agent);
                g_current_agent = NULL;
                return 1;
            }
            printf("[INDEP] 黑白独立网络已接入：由共享%s权重播种两方；"
                "之后每局只更新当方 online，回放按物理棋色路由。\n",
                loaded_checkpoint ? "存档" : "预训练");
        }
    }

    printf("===== 5x5 DQN 自博弈 + Minimax 评估 =====\n");
    printf("  棋盘: %dx%d, 胜利条件: %d 连\n", BOARD_SIZE, BOARD_SIZE, WIN_STREAK);
    printf("  网络: 全连接 %d-%d-%d 隐藏 + 多头注意力(%d头, 嵌入%d)\n",
        HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE, ATTENTION_HEADS, ATTENTION_EMBED);
    printf("  训练: Batch=%d, Replay=%d, γ=%.2f, τ=%.3f\n",
        BATCH_SIZE, REPLAY_CAPACITY, GAMMA, TAU);
    printf("  初始学习率: %.6f\n", g_learning_rate);

    struct {
        int phase;
        float eps_start;
        float eps_end;
        int clone;
        float white_eps;
        int sep;
    } stages[] = {
        {1, EPS_MAX, EPS_MIN, 1, -1.0f, 1},
        {2, EPS_MAX, EPS_MIN, 1, -1.0f, 1},
        {3, EPS_MAX, EPS_MIN, 1, -1.0f, 1},
        {4, EPS_MAX, EPS_MIN, 1, 0.0f, 0}
    };
    int episodes[] = {
        PHASE1_EPISODES, PHASE2_EPISODES, PHASE3_EPISODES, PHASE4_EPISODES
    };

    for (int i = start_phase - 1; i < 4; i++) {
        int remaining = episodes[i] - (i == start_phase - 1 ? start_episode : 0);
        if (remaining <= 0) {
            printf("阶段 %d 已完成，跳过。\n", stages[i].phase);
            continue;
        }
        printf("\n准备训练阶段 %d（%d 局，使用 %s 缓冲区）\n",
            stages[i].phase, remaining,
            stages[i].sep ? "独立" : "共享");
        /* 初始化多快照采集器 */
        snapshot_collector_init(&g_snap_collector, SKILL_WINDOW_INTERVAL);
        train_dqn(agent,
            episodes[i],
            stages[i].phase,
            stages[i].eps_start,
            stages[i].eps_end,
            1000,
            stages[i].clone,
            stages[i].white_eps,
            stages[i].sep,
            (i == start_phase - 1) ? start_episode : 0,
            g_ctx.minimax_depth,
            NULL,
            -1);   // seed_id = -1 表示主训练
        if (g_exit_requested) { printf("[EXIT] 停止后续阶段\n"); break; }   /* 窗口关闭：保存后退出 */
        start_episode = 0;
        // train_dqn already committed the next-stage checkpoint before returning.
    }

    if (!g_exit_requested) evaluate_ai(agent, EVAL_GAMES);
    service_signal_exit_request();

    /* NoTerminal 后台版：跳过人机对战 stdin 交互（无终端，直接进入清理退出）
    printf("\n是否进行人机对战？(y/n): ");
    char ch;
    scanf(" %c", &ch);
    if (ch == 'y' || ch == 'Y') play_vs_ai(agent);
    */

    stop_ipc_server();

    // 释放 GPU 驻留网络
    gpu_net_deactivate();

    // 关闭 GPU
    neural_probe_shutdown();
    gpu_shutdown();

    free_dqn(agent);
    free(agent);
    return g_training_fault ? 2 : 0;
}
