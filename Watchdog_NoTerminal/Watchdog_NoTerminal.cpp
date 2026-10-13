#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <tlhelp32.h>
#include <time.h>
#include <wchar.h>
#include <sys/stat.h>
#include <math.h>
#include <process.h>
#include <io.h>

// ========== 配置参数 ==========
#define EVAL_FILE       "latest_eval.txt"    /* 维护器专用：局内 ep/败率/checkpoint/depth/phase */
#define PARAM_FILE      "params.txt"
#define CHECKPOINT_PAT  "checkpoint_*.bin"
#define TRAIN_EXE       L"DQN_Train_Trainer_NoTerminal.exe"
#define PIPE_NAME       "\\\\.\\pipe\\dqn_control_pipe"
#define SNAPSHOT_REQUEST_FILE "wd_snapshot_request.txt"
#define SNAPSHOT_RESULT_FILE  "wd_snapshot_result.txt"
#define TRAINER_COMMAND_FILE   "wd_trainer_command.txt"
#define TRAINER_RESULT_FILE    "wd_trainer_result.txt"
#define INTENTIONAL_SHUTDOWN_FILE "intentional_shutdown.flag"
#define TRAINING_FAULT_FILE "training_fault.flag"
#define WATCHDOG_SHUTDOWN_REQUEST "watchdog_shutdown.request"
#define WATCHDOG_SHUTDOWN_ACK     "watchdog_shutdown.ack"

/* 目录布局：根目录只放程序+dll+通信文件；日志/存档分文件夹 */
#define LOG_DIR       "logs\\"
#define CKPT_DIR      "checkpoints\\"
#define CKPT_TEMP_DIR "checkpoints_temp\\"

#define LOSE_RATE_THRESHOLD  0.90f
#define PATIENCE           5
#define CHECK_INTERVAL_MS  1000    // 主循环 1s（终端命令响应延迟 ≤1s；eval 轮询仍由 POLL_INTERVAL_SEC=20 控制）
#define POLL_INTERVAL_SEC  20            // 20 seconds
#define MAX_RESTARTS       3
#define WARMUP_EPISODES    5000

// ========== 全局变量 ==========
static HANDLE hTrainProcess = NULL;
static DWORD  train_pid = 0;
static int restart_count = 0;
static float lose_history[PATIENCE];
static int history_count = 0;
static int last_known_depth = -1;
static int watch_dog_running = 1;
static int no_data_warned = 0;
static int has_good_performance = 0;      // 是否曾达到良好性能（败率<阈值）

// 上次读取的数据
static int last_episode = -1;
static int last_depth = -1;
static float last_lose_rate = -1.0f;
static int last_eval_phase = 1;
static int collapse_phase = -1;
static time_t last_poll_time = 0;
static void poll_watchdog_shutdown_request(void);

// ========== 自有终端窗口（NoTerminal：显示 watchdog 状态/日志） ==========
static HWND g_hwnd = NULL;
static HFONT g_wterm_font = NULL;   /* 窗口线程创建，WM_PAINT 复用 */
#define WTERM_LINES 200
#define WTERM_COLS  140
static char g_wterm[WTERM_LINES][WTERM_COLS + 1];
static int g_wterm_n = 0;
static int g_wterm_head = 0;
static CRITICAL_SECTION g_wterm_lock;

static void win_show_line(const char* s) {
    EnterCriticalSection(&g_wterm_lock);
    strncpy(g_wterm[g_wterm_head], s, WTERM_COLS);
    g_wterm[g_wterm_head][WTERM_COLS] = '\0';
    g_wterm_head = (g_wterm_head + 1) % WTERM_LINES;
    if (g_wterm_n < WTERM_LINES) g_wterm_n++;
    LeaveCriticalSection(&g_wterm_lock);
    if (g_hwnd) InvalidateRect(g_hwnd, NULL, FALSE);
}
static void win_pump_messages(void) {
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
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
        HFONT old = (HFONT)SelectObject(dc, g_wterm_font);
        SetTextColor(dc, RGB(0x39, 0xC5, 0xBB));
        SetBkMode(dc, TRANSPARENT);
        EnterCriticalSection(&g_wterm_lock);
        int total = g_wterm_n;
        int start = (total > 40) ? (total - 40) : 0;   /* 显示最近 40 行 */
        int cy = 4;
        wchar_t wbuf[WTERM_COLS * 2 + 4];
        for (int i = start; i < total && cy < rc.bottom - 4; i++) {
            int idx = (g_wterm_head - total + i + WTERM_LINES * 2) % WTERM_LINES;
            int wlen = MultiByteToWideChar(CP_UTF8, 0, g_wterm[idx], -1, wbuf, WTERM_COLS * 2);
            if (wlen > 1) TextOutW(dc, 4, cy, wbuf, wlen - 1);   /* UTF-8 → UTF-16，中文清晰 */
            cy += 18;
        }
        LeaveCriticalSection(&g_wterm_lock);
        SelectObject(dc, old);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_CLOSE:
        watch_dog_running = 0;   /* 主循环检测后退出 */
        DestroyWindow(hwnd);
        PostQuitMessage(0);      /* 结束窗口线程消息循环 */
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}
/* 窗口线程：创建窗口 + 独立消息循环（窗口始终响应，刷新不卡顿） */
static void win_thread(void*) {
    WNDCLASSW wc = { 0 };
    wc.lpfnWndProc = term_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.lpszClassName = L"DQNWatchdogTermWindow";
    ATOM atom = RegisterClassW(&wc);
    if (!atom && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        watch_dog_running = 0;
        return;
    }
    g_wterm_font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, FIXED_PITCH, L"Cascadia Mono");
    g_hwnd = CreateWindowW(L"DQNWatchdogTermWindow", L"DQN Watchdog (NoTerminal)",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 640, 420,
        NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) { watch_dog_running = 0; return; }
    ShowWindow(g_hwnd, SW_SHOW);   /* 默认显示；TUI 发 HIDE 隐藏 */
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    /* 窗口关闭后线程退出；主循环检测 watch_dog_running=0 退出 */
}
static void win_init(void) {
    InitializeCriticalSection(&g_wterm_lock);
    _beginthread(win_thread, 0, NULL);
}

// ========== 日志 ==========
void log_message(const char* msg) {
    FILE* log = fopen(LOG_DIR "watchdog.log", "a");
    if (log) {
        time_t now = time(NULL);
        char* ts = ctime(&now);
        if (ts) { size_t len = strlen(ts); if (len > 0 && ts[len-1] == '\n') ts[len-1] = '\0'; }   /* BUG-062: 去掉 ctime 自带换行 */
        fprintf(log, "[%s] %s\n", ts ? ts : "?", msg);
        fclose(log);
    }
    win_show_line(msg);   /* 窗口显示 */
}

// ========== 进程管理 ==========
DWORD find_process_by_name(const wchar_t* process_name) {
    DWORD pid = 0;
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32W pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32W);
    if (!Process32FirstW(hSnapshot, &pe32)) { CloseHandle(hSnapshot); return 0; }
    do {
        if (_wcsicmp(pe32.szExeFile, process_name) == 0) {
            pid = pe32.th32ProcessID;
            break;
        }
    } while (Process32NextW(hSnapshot, &pe32));
    CloseHandle(hSnapshot);
    return pid;
}

int get_train_process_handle(void) {
    if (hTrainProcess) {
        DWORD exitCode;
        if (GetExitCodeProcess(hTrainProcess, &exitCode) && exitCode == STILL_ACTIVE)
            return 1;
        CloseHandle(hTrainProcess);
        hTrainProcess = NULL;
        train_pid = 0;
    }
    DWORD pid = find_process_by_name(TRAIN_EXE);
    if (pid) {
        hTrainProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);   /* BUG-041: 最小权限 */
        if (hTrainProcess) {
            train_pid = pid;
            char msg[256];
            snprintf(msg, sizeof(msg), "[看门狗] 找到训练进程 (PID %lu)", pid);
            log_message(msg);
            return 1;
        }
    }
    return 0;
}

// ========== 读取 eval.dat 最新败率 ==========
int read_latest_eval(float* lose_rate, int* episode, int* depth) {
    FILE* f = fopen(EVAL_FILE, "rb");
    if (!f) {
        if (!no_data_warned) {
            log_message("[看门狗] eval.dat 尚未生成，等待训练首次评估...");
            no_data_warned = 1;
        }
        return 0;
    }

    char checkpoint[256];
    int ep = 0, dep = 0, ph = 1;
    float lr = 0.0f;
    int found = fscanf(f, "%d %f %255s %d %d", &ep, &lr, checkpoint, &dep, &ph) == 5;
    fclose(f);

    if (!found || !isfinite(lr) || lr < 0.0f || lr > 1.0f ||
        ep < 0 || dep < 0 || dep > 25 || ph < 0 || ph > 5) {
        if (!no_data_warned) {
            log_message("[看门狗] latest_eval.txt 格式无效，等待下一次原子更新");
            no_data_warned = 1;
        }
        return 0;
    }
    no_data_warned = 0;
    *lose_rate = lr;
    *episode = ep;
    *depth = dep;
    last_eval_phase = ph;
    return 1;
}

/* IND-03：隔离 checkpoints_temp\ 里更「新」的中间档。与正式目录处理互相独立，
   即使正式目录没有合格条目也必须执行，否则重启时 find_latest_checkpoint 会
   选中刚保存、可能仍处于异常状态的临时档，使回滚失效。
   返回 1 表示临时目录已清空（或本就为空）；返回 0 表示仍有临时档无法隔离，
   调用方应停止恢复流程，避免按「恢复成功」继续启动训练。 */
static int quarantine_temp_checkpoints(void) {
    /* 首次进入（冷启动）时目标目录可能尚不存在，必须自行确保，否则 MoveFileExA
       会因目标目录缺失而失败，异常临时档仍留在恢复目录。 */
    CreateDirectoryA("checkpoints_quarantine", NULL);
    WIN32_FIND_DATAA tfd;
    HANDLE th = FindFirstFileA(CKPT_TEMP_DIR "checkpoint_*.bin", &tfd);
    int temp_quarantined = 0;
    int temp_remaining = 0;
    if (th != INVALID_HANDLE_VALUE) {
        do {
            char src[512], dst[512];
            snprintf(src, sizeof(src), CKPT_TEMP_DIR "%s", tfd.cFileName);
            snprintf(dst, sizeof(dst), "checkpoints_quarantine\\%s", tfd.cFileName);
            if (MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                temp_quarantined++;
                char side_src[520], side_dst[520];
                snprintf(side_src, sizeof(side_src), "%s.pool", src);
                snprintf(side_dst, sizeof(side_dst), "%s.pool", dst);
                MoveFileExA(side_src, side_dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
                snprintf(side_src, sizeof(side_src), "%s.mem", src);
                snprintf(side_dst, sizeof(side_dst), "%s.mem", dst);
                MoveFileExA(side_src, side_dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
            } else {
                temp_remaining++;
                char msg2[256];
                snprintf(msg2, sizeof(msg2), "[看门狗] 临时检查点隔离失败: %s (error %lu)", src, GetLastError());
                log_message(msg2);
            }
        } while (FindNextFileA(th, &tfd));
        FindClose(th);
    }
    if (temp_quarantined) {
        char msg2[256];
        snprintf(msg2, sizeof(msg2), "[看门狗] 已隔离 %d 个临时检查点（避免回滚后重新载入异常状态）", temp_quarantined);
        log_message(msg2);
    }
    return temp_remaining == 0 ? 1 : 0;
}

// ========== 删除最近N个检查点（新命名格式：checkpoint_phase{P}_{EP}.bin） ==========
/* 返回 1 表示临时恢复目录已清空（正式档 + 临时档均无残留）；返回 0 表示仍有
   临时档无法隔离，调用方应停止恢复流程。 */
int delete_bad_checkpoints_extended(int bad_episode, int count) {
    char msg[512];
    snprintf(msg, sizeof(msg), "[看门狗] 删除最近 %d 个检查点（从 episode %d 往前，按 phase/ep 倒序）", count, bad_episode);
    log_message(msg);

    /* BUG-004: 动态分配枚举所有 checkpoint，按 (phase, ep) 倒序排列，删前 count 个 */
    struct CkEntry { int phase, ep; char name[256]; };
    struct CkEntry* entries = NULL;
    int n = 0, cap = 0;

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(CKPT_DIR "checkpoint_*.bin", &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (n >= cap) {
                int next_cap = cap ? cap * 2 : 64;
                void* grown = realloc(entries, (size_t)next_cap * sizeof(struct CkEntry));
                if (!grown) break;
                entries = (struct CkEntry*)grown;
                cap = next_cap;
            }
            int ph = 0, ep = 0;
            if (sscanf(fd.cFileName, "checkpoint_phase%d_%d.bin", &ph, &ep) == 2 ||
                sscanf(fd.cFileName, "checkpoint_%d.bin", &ep) == 1) {
                entries[n].phase = ph;
                entries[n].ep = ep;
                snprintf(entries[n].name, sizeof(entries[n].name), CKPT_DIR "%s", fd.cFileName);
                n++;
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }

    if (!entries || n == 0) {
        /* IND-03（残留1）：正式目录无条目也不能跳过临时档隔离，否则仅剩临时
           恢复点时仍会重新载入本应处置的状态。 */
        free(entries);
        return quarantine_temp_checkpoints();
    }

    /* 按 (phase, ep) 倒序排列 */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (entries[j].phase > entries[i].phase ||
                (entries[j].phase == entries[i].phase && entries[j].ep > entries[i].ep)) {
                struct CkEntry tmp = entries[i]; entries[i] = entries[j]; entries[j] = tmp;
            }

    /* Bug-8 修复：崩坏处置不再删除恢复点，改为隔离到 checkpoints_quarantine。
       FINAL 高风险修复：必须隔离的「前 count 个」正式候选一旦任一移动失败，
       就应让整次处置失败，绝不能跳过失败的新档去移动更老的恢复点凑数。 */
    CreateDirectoryA("checkpoints_quarantine", NULL);
    int deleted = 0;
    int formal_failed = 0;
    for (int i = 0; i < n && i < count; i++) {
        const char* base = strrchr(entries[i].name, '\\');
        base = base ? base + 1 : entries[i].name;
        char dest[512];
        snprintf(dest, sizeof(dest), "checkpoints_quarantine\\%s", base);
        if (MoveFileExA(entries[i].name, dest, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            char sidecar_src[300], sidecar_dst[320];
            snprintf(sidecar_src, sizeof(sidecar_src), "%s.pool", entries[i].name);
            snprintf(sidecar_dst, sizeof(sidecar_dst), "checkpoints_quarantine\\%s.pool", base);
            MoveFileExA(sidecar_src, sidecar_dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
            char msg2[256];
            snprintf(msg2, sizeof(msg2), "[看门狗] 已隔离: %s", entries[i].name);
            log_message(msg2);
            deleted++;
        } else {
            formal_failed++;
            char msg2[256];
            snprintf(msg2, sizeof(msg2), "[看门狗] 正式检查点隔离失败: %s (error %lu)", entries[i].name, GetLastError());
            log_message(msg2);
        }
    }
    {
        char msg2[256];
        snprintf(msg2, sizeof(msg2), "[看门狗] 共隔离 %d 个检查点（保留可恢复）", deleted);
        log_message(msg2);
    }
    /* IND-03：隔离临时目录（与正式目录处理独立，避免只处理一个目录）。 */
    int temp_ok = quarantine_temp_checkpoints();
    free(entries);
    return (formal_failed == 0 && temp_ok) ? 1 : 0;
}

// ========== 回滚标记 ==========
void write_rollback_marker(int target_episode) {
    FILE* f = fopen("rollback_to.txt", "w");
    if (f) {
        fprintf(f, "%d\n", target_episode);
        fclose(f);
        char msg[256];
        snprintf(msg, sizeof(msg), "[看门狗] 写入回滚标记，目标 episode %d", target_episode);
        log_message(msg);
    }
}

// ========== 调整超参数（指数衰减，基于重启次数） ==========
void adjust_parameters(void) {
    // 基础学习率 × 0.5^restart_count
    float base_lr = 0.0004f;
    float lr = base_lr;
    for (int i = 0; i < restart_count; i++) lr *= 0.5f;
    if (lr < 0.00001f) lr = 0.00001f;  // 下限

    FILE* f = fopen(PARAM_FILE, "w");
    if (!f) { log_message("[看门狗] 无法创建参数文件"); return; }
    fprintf(f, "learning_rate = %.6f\n", lr);
    fclose(f);

    // Explicit one-shot override, applied after V6/V7 checkpoint state is restored.
    FILE* override_file = fopen("recovery_override.txt.tmp", "wb");
    if (override_file) {
        int ok = fprintf(override_file, "version=1\nlearning_rate=%.8f\n", lr) > 0;
        if (fflush(override_file) != 0 || _commit(_fileno(override_file)) != 0) ok = 0;
        if (fclose(override_file) != 0) ok = 0;
        if (!ok || !MoveFileExA("recovery_override.txt.tmp", "recovery_override.txt",
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            remove("recovery_override.txt.tmp");
            log_message("[看门狗] 恢复学习率请求写入失败，不能确认降学习率生效");
        }
    }

    char msg[256];
    snprintf(msg, sizeof(msg), "[看门狗] 超参数已更新: learning_rate=%.6f (restart_count=%d)", lr, restart_count);
    log_message(msg);
}

// ========== 进程控制 ==========
int start_train_process(void) {
    if (GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE) != INVALID_FILE_ATTRIBUTES) {
        log_message("[看门狗] 检测到主动停机标记，取消启动训练进程");
        return 0;
    }
    if (GetFileAttributesA(TRAINING_FAULT_FILE) != INVALID_FILE_ATTRIBUTES) {
        log_message("[看门狗] 检测到训练故障标记，取消自动启动，等待人工恢复");
        return 0;
    }
    /* 退避期间可能由用户/TUI手动启动了 Trainer。创建进程前必须重新
       枚举一次，否则会出现两个训练器同时写日志、checkpoint 和 Probe。 */
    if (get_train_process_handle()) {
        log_message("[看门狗] 已存在训练进程，本次启动改为附加");
        return 1;
    }
    if (hTrainProcess) { CloseHandle(hTrainProcess); hTrainProcess = NULL; train_pid = 0; }
    wchar_t cmd_line[512];
    swprintf(cmd_line, 512, L"\"%s\"", TRAIN_EXE);
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    /* 直接创建 Trainer，句柄/PID 才属于真实训练进程。经 cmd /c 启动会
       监控到短命 cmd.exe，造成误判退出和重复拉起。 */
    if (!CreateProcessW(TRAIN_EXE, cmd_line, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        char msg[256];
        snprintf(msg, sizeof(msg), "[看门狗] 启动训练进程失败 (错误 %lu)", GetLastError());
        log_message(msg);
        return 0;
    }
    hTrainProcess = pi.hProcess;
    train_pid = pi.dwProcessId;
    CloseHandle(pi.hThread);
    char msg[256];
    snprintf(msg, sizeof(msg), "[看门狗] 训练进程已启动 (PID %lu)", pi.dwProcessId);
    log_message(msg);
    return 1;
}

void stop_train_process(void) {
    if (!hTrainProcess) return;
    /* Bug-8 修复：崩坏处置先请求优雅保存退出，而非直接 TerminateProcess。
       通过命名管道发送 SAVE_AND_EXIT，Trainer 会同步 GPU 并保存 checkpoint 后退出。 */
    log_message("[看门狗] 请求 Trainer 优雅保存退出（SAVE_AND_EXIT）...");
    HANDLE pipe = CreateFileA(PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, NULL,
        OPEN_EXISTING, 0, NULL);
    if (pipe != INVALID_HANDLE_VALUE) {
        DWORD mode = PIPE_READMODE_MESSAGE;
        SetNamedPipeHandleState(pipe, &mode, NULL, NULL);
        const char command[] = "SAVE_AND_EXIT";
        DWORD written = 0;
        WriteFile(pipe, command, (DWORD)strlen(command), &written, NULL);
        CloseHandle(pipe);
    }
    /* 等待优雅退出（最多 15 秒），超时才回退到强杀。 */
    for (int wait = 0; wait < 150; ++wait) {
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(hTrainProcess, &exit_code) || exit_code != STILL_ACTIVE) {
            CloseHandle(hTrainProcess);
            hTrainProcess = NULL;
            train_pid = 0;
            log_message("[看门狗] Trainer 已优雅保存退出");
            return;
        }
        Sleep(100);
    }
    log_message("[看门狗] Trainer 优雅退出超时，强制终止（最后手段）");
    TerminateProcess(hTrainProcess, 0);
    CloseHandle(hTrainProcess);
    hTrainProcess = NULL;
    train_pid = 0;
}

static ULONGLONG file_write_stamp(const char* path, DWORD* size_low) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &data)) return 0;
    ULARGE_INTEGER stamp;
    stamp.LowPart = data.ftLastWriteTime.dwLowDateTime;
    stamp.HighPart = data.ftLastWriteTime.dwHighDateTime;
    if (size_low) *size_low = data.nFileSizeLow;
    return stamp.QuadPart;
}

static void write_snapshot_result(const char* result) {
    FILE* f = fopen("wd_snapshot_result.tmp", "wb");
    if (!f) return;
    fputs(result, f);
    fclose(f);
    MoveFileExA("wd_snapshot_result.tmp", SNAPSHOT_RESULT_FILE,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

/* 兼容已经运行的旧 Trainer：旧管道 ACL 可能只允许后来启动的 TUI
   读取状态。Watchdog 与 Trainer 由同一会话启动，可在 TUI 主动断开后
   临时接管唯一的旧管道实例，只转发固定的 SAVE_SNAPSHOT 命令。 */
static void poll_snapshot_request(void) {
    FILE* request = fopen(SNAPSHOT_REQUEST_FILE, "rb");
    if (!request) return;
    fclose(request);
    remove(SNAPSHOT_REQUEST_FILE);

    DWORD before_size = 0;
    ULONGLONG before_stamp = file_write_stamp("current_snapshot.bin", &before_size);
    HANDLE pipe = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 50 && watch_dog_running; ++attempt) {
        pipe = CreateFileA(PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, NULL,
            OPEN_EXISTING, 0, NULL);
        if (pipe != INVALID_HANDLE_VALUE) break;
        WaitNamedPipeA(PIPE_NAME, 100);
        Sleep(100);
    }
    if (pipe == INVALID_HANDLE_VALUE) {
        write_snapshot_result("ERROR pipe unavailable");
        log_message("[看门狗] 快照转发失败：无法连接 Trainer 管道");
        return;
    }

    DWORD mode = PIPE_READMODE_MESSAGE;
    SetNamedPipeHandleState(pipe, &mode, NULL, NULL);
    const char command[] = "SAVE_SNAPSHOT";
    DWORD written = 0;
    if (!WriteFile(pipe, command, (DWORD)strlen(command), &written, NULL) || written == 0) {
        CloseHandle(pipe);
        write_snapshot_result("ERROR command write failed");
        log_message("[看门狗] 快照转发失败：命令写入失败");
        return;
    }

    int saved = 0;
    for (int wait = 0; wait < 120 && watch_dog_running; ++wait) {
        DWORD current_size = 0;
        ULONGLONG current_stamp = file_write_stamp("current_snapshot.bin", &current_size);
        if (current_size > 0 && (current_stamp != before_stamp || current_size != before_size)) {
            saved = 1;
            break;
        }
        /* 消费状态推送，避免旧服务端因客户端不读而产生回压。 */
        DWORD available = 0;
        if (PeekNamedPipe(pipe, NULL, 0, NULL, &available, NULL) && available > 0) {
            char discard[1024];
            DWORD got = 0;
            ReadFile(pipe, discard, sizeof(discard), &got, NULL);
        }
        Sleep(50);
    }
    CloseHandle(pipe);
    if (saved) {
        write_snapshot_result("OK snapshot updated");
        log_message("[看门狗] 已为 TUI 转发并完成即时快照");
    } else {
        write_snapshot_result("ERROR snapshot timeout");
        log_message("[看门狗] 快照转发超时：Trainer 未更新文件");
    }
}

static void write_trainer_result(const char* result) {
    FILE* f = fopen("wd_trainer_result.tmp", "wb");
    if (!f) return;
    fputs(result, f);
    fclose(f);
    MoveFileExA("wd_trainer_result.tmp", TRAINER_RESULT_FILE,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

/* Console 按钮兼容代理。严格限制为显示/隐藏两个无损命令，不能通过
   文件通道向 Trainer 注入任意 IPC 指令。 */
static void poll_trainer_control_request(void) {
    FILE* request = fopen(TRAINER_COMMAND_FILE, "rb");
    if (!request) return;
    char command[32] = "";
    fgets(command, sizeof(command), request);
    fclose(request);
    remove(TRAINER_COMMAND_FILE);
    command[strcspn(command, "\r\n")] = '\0';
    if (strcmp(command, "CONSOLE_SHOW") != 0 && strcmp(command, "CONSOLE_HIDE") != 0) {
        write_trainer_result("ERROR command rejected");
        return;
    }

    HANDLE pipe = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 50 && watch_dog_running; ++attempt) {
        pipe = CreateFileA(PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, NULL,
            OPEN_EXISTING, 0, NULL);
        if (pipe != INVALID_HANDLE_VALUE) break;
        WaitNamedPipeA(PIPE_NAME, 100);
        Sleep(100);
    }
    if (pipe == INVALID_HANDLE_VALUE) {
        write_trainer_result("ERROR pipe unavailable");
        log_message("[看门狗] Console 转发失败：无法连接 Trainer 管道");
        return;
    }

    DWORD mode = PIPE_READMODE_MESSAGE;
    SetNamedPipeHandleState(pipe, &mode, NULL, NULL);
    DWORD written = 0;
    int ok = WriteFile(pipe, command, (DWORD)strlen(command), &written, NULL) && written > 0;
    if (ok) {
        /* 给旧服务端至少数个轮询周期来消费命令，同时排空状态推送。 */
        for (int wait = 0; wait < 16; ++wait) {
            DWORD available = 0;
            if (PeekNamedPipe(pipe, NULL, 0, NULL, &available, NULL) && available > 0) {
                char discard[1024];
                DWORD got = 0;
                ReadFile(pipe, discard, sizeof(discard), &got, NULL);
            }
            Sleep(50);
        }
    }
    CloseHandle(pipe);
    write_trainer_result(ok ? "OK command forwarded" : "ERROR command write failed");
    log_message(ok ? "[看门狗] 已转发 Trainer 窗口命令" : "[看门狗] Trainer 窗口命令写入失败");
}

// ========== 终端命令（TUI 写 wd_cmd.txt：SHOW / HIDE，隐藏/显示终端窗口，内容保留，后台运行） ==========
void poll_terminal_cmd(void) {
    /* BUG-038: 用 rename 保证原子性（TUI 先写 .tmp 再 rename 为 .txt） */
    FILE* f = fopen("wd_cmd.txt", "rb");
    if (!f) return;
    char cmd[32] = "";
    if (fgets(cmd, sizeof(cmd), f)) {
        fclose(f);
        remove("wd_cmd.txt");
        if (g_hwnd) ShowWindow(g_hwnd, strstr(cmd, "SHOW") ? SW_SHOW : SW_HIDE);
        log_message(strstr(cmd, "SHOW") ? "[看门狗] 终端已显示" : "[看门狗] 终端已隐藏（后台运行）");
    } else {
        fclose(f);
    }
}

// ========== 崩坏检测（增加“良好性能”前置条件） ==========
int check_collapse(float lose_rate, int current_depth, int episode) {
    // Missing/invalid observations must never justify deleting checkpoints.
    if (!isfinite(lose_rate) || lose_rate < 0.0f || lose_rate > 1.0f ||
        current_depth < 0 || current_depth > 25 || episode < 0) return 0;
    /* 每个正式训练阶段的 episode 会重新从 1 开始。旧历史和“曾经良好”
       不能跨阶段继承，否则新阶段热身结束后可能把正常的难度跃迁误判成崩坏。 */
    if (last_eval_phase != collapse_phase) {
        char msg[256];
        snprintf(msg, sizeof(msg), "[看门狗] 训练阶段切换为 %d，重置崩坏基线", last_eval_phase);
        log_message(msg);
        history_count = 0;
        memset(lose_history, 0, sizeof(lose_history));
        has_good_performance = 0;
        last_known_depth = current_depth;
        collapse_phase = last_eval_phase;
        return 0;
    }
    // 热身期不检测
    if (episode < WARMUP_EPISODES) return 0;

    // Good performance is meaningful only against the same difficulty.
    if (current_depth != last_known_depth) {
        char msg[256];
        snprintf(msg, sizeof(msg), "[看门狗] 深度从 %d 变为 %d，重置崩坏历史", last_known_depth, current_depth);
        log_message(msg);
        history_count = 0;
        memset(lose_history, 0, sizeof(lose_history));
        last_known_depth = current_depth;
        has_good_performance = 0;
        return 0;
    }

    // ========== 1. 记录当前败率（循环数组，始终存储，不因单次低值而重置） ==========
    lose_history[history_count % PATIENCE] = lose_rate;
    history_count++;

    // 如果历史记录不足 PATIENCE 次，不进行任何判定
    if (history_count < PATIENCE) return 0;

    // ========== 2. 计算最近 PATIENCE 次败率的平均值 ==========
    float sum = 0.0f;
    for (int i = 0; i < PATIENCE; i++) {
        sum += lose_history[(history_count - PATIENCE + i) % PATIENCE];
    }
    float avg_lose = sum / PATIENCE;

    // ========== 3. 【核心修改】良好性能标准：过去5次均值 < 阈值 ==========
    // 只有连续5次的平均败率低于阈值，才认为模型真正达到了良好性能
    if (avg_lose < LOSE_RATE_THRESHOLD) {
        has_good_performance = 1;
        // 注意：不重置历史，继续记录后续数据
        return 0;
    }

    // 如果从未达到良好性能，不触发崩坏（避免在初期未学会时误判）
    if (!has_good_performance) return 0;

    // ========== 4. 【崩坏标准保持不变】检查最近 PATIENCE 次是否全部高于阈值 ==========
    int all_bad = 1;
    for (int i = 0; i < PATIENCE; i++) {
        if (lose_history[(history_count - PATIENCE + i) % PATIENCE] < LOSE_RATE_THRESHOLD) {
            all_bad = 0;
            break;
        }
    }

    if (all_bad) {
        char msg[256];
        snprintf(msg, sizeof(msg), "[看门狗] 最近 %d 次全部高于阈值 (均值 %.2f%%)，触发崩坏",
            PATIENCE, avg_lose * 100);
        log_message(msg);
        return 1;
    }

    return 0;
}

// ========== 主程序 ==========
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow) {
    (void)hInstance; (void)hPrevInstance; (void)lpCmdLine; (void)nCmdShow;
    /* 工作目录固定为 exe 所在目录（eval.dat 等在 logs\，从任意位置启动也正确） */
    {
        /* BUG-065: 使用 32768 缓冲区避免长路径截断 */
        wchar_t exePath[32768];
        DWORD len = GetModuleFileNameW(NULL, exePath, 32768);
        if (len > 0 && len < 32768) {
            wchar_t* slash = wcsrchr(exePath, L'\\');
            if (slash) { *slash = L'\0'; SetCurrentDirectoryW(exePath); }
        }
    }
    CreateDirectoryA(LOG_DIR, NULL);      /* 日志/存档分文件夹 */
    CreateDirectoryA(CKPT_DIR, NULL);
    CreateDirectoryA(CKPT_TEMP_DIR, NULL);
    win_init();   /* 自有终端窗口 */
    log_message("===== DQN 训练看门狗启动 =====");
    char msg[256];
    snprintf(msg, sizeof(msg), "败率阈值: %.0f%%, 连续 %d 次触发重启, 最大重启 %d 次, 热身期 %d 局",
        LOSE_RATE_THRESHOLD * 100, PATIENCE, MAX_RESTARTS, WARMUP_EPISODES);
    log_message(msg);
    log_message("[看门狗] 仅当模型曾达到良好性能（败率<阈值）后，再持续恶化才触发重启。");

    if (GetFileAttributesA(PARAM_FILE) == INVALID_FILE_ATTRIBUTES) adjust_parameters();

    log_message("[看门狗] 正在查找训练进程...");
    if (get_train_process_handle())
        log_message("[看门狗] 已附加到现有训练进程");
    else
        log_message("[看门狗] 未找到训练进程，等待...");

    while (watch_dog_running) {
        Sleep(CHECK_INTERVAL_MS);
        poll_watchdog_shutdown_request();
        if (!watch_dog_running) break;
        win_pump_messages();   /* 窗口消息泵（WM_CLOSE → watch_dog_running=0 退出） */
        poll_terminal_cmd();   /* 响应 TUI 的窗口显示/隐藏命令（wd_cmd.txt） */
        poll_snapshot_request(); /* 旧 Trainer 只读管道的受限快照代理 */
        poll_trainer_control_request(); /* 旧 Trainer 的 Console 显示/隐藏代理 */

        // An intentional stop covers the saving/failed-save interval too, not
        // only an already-dead process. Never kill a paused unsaved trainer on
        // the basis of an old evaluation while it waits for a storage retry.
        if (GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE) != INVALID_FILE_ATTRIBUTES)
            continue;
        if (GetFileAttributesA(TRAINING_FAULT_FILE) != INVALID_FILE_ATTRIBUTES) {
            static int fault_logged = 0;
            if (!fault_logged) {
                log_message("[看门狗] 训练因故障保护停机，保持停止，等待人工检查后重新启动");
                fault_logged = 1;
            }
            continue;
        }

        /* BUG-002 修复：分离句柄检查和进程查找，使重启逻辑可达。
           get_train_process_handle() 在句柄无效时关闭并返回 0，之前直接 continue 导致重启分支不可达。
           现在：先检查进程是否仍在运行，若已退出则进入重启逻辑，不 continue。 */
        int train_alive = 0;
        if (hTrainProcess) {
            DWORD exitCode;
            if (GetExitCodeProcess(hTrainProcess, &exitCode) && exitCode == STILL_ACTIVE) {
                train_alive = 1;
            } else {
                /* 训练进程已退出 */
                char msg2[256];
                snprintf(msg2, sizeof(msg2), "[看门狗] 训练进程已退出 (代码 %lu)，尝试重启", exitCode);
                log_message(msg2);
                CloseHandle(hTrainProcess); hTrainProcess = NULL; train_pid = 0;
            }
        }
        if (!train_alive) {
            /* 用户通过 TUI/Trainer 正常关闭时不是崩溃。这个判断放在退避前，
               也在退避后复查，封住“刚睡下时收到 shutdown”的竞态。 */
            if (GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE) != INVALID_FILE_ATTRIBUTES) {
                static int intentional_logged = 0;
                if (!intentional_logged) {
                    log_message("[看门狗] 训练会话已主动关闭，保持停止，不再自动拉起");
                    intentional_logged = 1;
                }
                continue;
            }
            if (GetFileAttributesA(TRAINING_FAULT_FILE) != INVALID_FILE_ATTRIBUTES)
                continue;
            /* BUG-002: 训练进程退出后真正重启（不只是等待） */
            static int _prev_running = 1;
            if (_prev_running) { log_message("[看门狗] 训练进程已退出，准备重启..."); _prev_running = 0; }
            /* 退避等待：避免快速重启循环（首次 3s，之后递增到 30s） */
            static int restart_backoff = 3;
            Sleep(restart_backoff * 1000);
            if (GetFileAttributesA(INTENTIONAL_SHUTDOWN_FILE) != INVALID_FILE_ATTRIBUTES)
                continue;
            if (GetFileAttributesA(TRAINING_FAULT_FILE) != INVALID_FILE_ATTRIBUTES)
                continue;
            if (restart_backoff < 30) restart_backoff += 3;
            if (start_train_process()) {
                log_message("[看门狗] 训练进程已重启");
                _prev_running = 1;
                restart_backoff = 3;   /* 成功后重置退避 */
                last_poll_time = time(NULL);   /* 重置轮询计时 */
            } else {
                log_message("[看门狗] 重启失败，稍后重试...");
            }
            continue;
        }

        // 定时轮询
        time_t now = time(NULL);
        if (now - last_poll_time < POLL_INTERVAL_SEC) continue;
        last_poll_time = now;

        float lose_rate; int episode; int depth;
        if (!read_latest_eval(&lose_rate, &episode, &depth)) continue;

        // 对比数据
        if (episode == last_episode && depth == last_depth &&
            fabs(lose_rate - last_lose_rate) < 0.0001f) {
            continue;
        }

        last_episode = episode;
        last_depth = depth;
        last_lose_rate = lose_rate;

        char msg3[256];
        snprintf(msg3, sizeof(msg3), "[看门狗] 当前败率: %3.2f%%, 深度: %d, Episode %d", lose_rate * 100, depth, episode);
        log_message(msg3);

        if (check_collapse(lose_rate, depth, episode)) {
            /* BUG-007: 先检查次数，再执行破坏性操作；BUG-006: adjust_parameters 在 restart_count++ 之后 */
            restart_count++;
            if (restart_count > MAX_RESTARTS) {
                log_message("[看门狗] 已超过最大重启次数，停止训练！");
                break;
            }
            log_message("[看门狗] ***** 训练崩坏检测！败率持续高于阈值，且曾达到良好性能 *****");
            stop_train_process();
            const int isolated_ok = delete_bad_checkpoints_extended(episode, restart_count);
            /* BUG-005 修复：回滚目标取实际保留的最新检查点（而非 episode-1），
               避免模型与日志不一致（模型回到 39000，日志截断到 39999）。
               枚举 checkpoints\ 取最新 (phase,ep) 的 ep 作为回滚目标。 */
            {
                WIN32_FIND_DATAA rfd;
                HANDLE rh = FindFirstFileA(CKPT_DIR "checkpoint_phase*_*.bin", &rfd);
                int best_ep = 0, best_phase = 1;
                if (rh != INVALID_HANDLE_VALUE) {
                    do {
                        int rp = 0, re = 0;
                        if (sscanf(rfd.cFileName, "checkpoint_phase%d_%d.bin", &rp, &re) == 2 &&
                            (rp > best_phase || (rp == best_phase && re > best_ep))) {
                            best_phase = rp; best_ep = re;
                        }
                    } while (FindNextFileA(rh, &rfd));
                    FindClose(rh);
                }
                const int phase_span = 100000;
                int global_target = (best_phase > 1 ? (best_phase - 1) * phase_span : 0) + best_ep;
                if (best_ep <= 0)
                    global_target = (last_eval_phase > 1 ? (last_eval_phase - 1) * phase_span : 0) +
                        ((episode > 0) ? episode - 1 : 0);
                write_rollback_marker(global_target);
            }
            adjust_parameters();

            // 重置历史，但保留良好性能标记（因为曾达到过）
            history_count = 0;
            memset(lose_history, 0, sizeof(lose_history));

            /* IND-03：隔离未完成（恢复点残留）时，不能按「恢复成功」继续启动
               训练，否则可能重新载入本应处置的异常状态。停止自动重启，等待人工。 */
            if (!isolated_ok) {
                log_message("[看门狗] 崩坏处置隔离未完成（恢复点残留），暂停自动重启，等待人工处理");
                break;
            }

            if (start_train_process()) log_message("[看门狗] 训练进程重启成功");
            else log_message("[看门狗] 训练进程重启失败");
        }
    }

    if (hTrainProcess) CloseHandle(hTrainProcess);
    log_message("[看门狗] 看门狗退出。");
    /* NoTerminal 后台版：移除"按任意键退出"阻塞，直接返回 */
    return 0;
}

/* 独立关闭 Watchdog，不触碰 Trainer。请求和确认均使用原子文件协议，
   这样无窗口运行时 TUI/脚本也能可靠等待退出。 */
static void poll_watchdog_shutdown_request(void) {
    FILE* request = fopen(WATCHDOG_SHUTDOWN_REQUEST, "rb");
    if (!request) return;
    fclose(request);
    remove(WATCHDOG_SHUTDOWN_REQUEST);
    FILE* ack = fopen("watchdog_shutdown.ack.tmp", "wb");
    if (ack) {
        fputs("OK watchdog graceful shutdown\n", ack);
        fflush(ack);
        _commit(_fileno(ack));
        fclose(ack);
        MoveFileExA("watchdog_shutdown.ack.tmp", WATCHDOG_SHUTDOWN_ACK,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    }
    log_message("[看门狗] 收到独立优雅关闭请求；Trainer 保持当前状态，不再监控或拉起。");
    watch_dog_running = 0;
}
