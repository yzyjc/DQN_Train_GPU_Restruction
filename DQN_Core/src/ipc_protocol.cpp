#include "core.h"
/* 发送命令（SET 或其它，不读取响应） */
int send_command(const char* cmd) {
    if (g_pipe == INVALID_HANDLE_VALUE) {
        g_pipe = CreateFileA(
            PIPE_NAME,
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, OPEN_EXISTING, 0, NULL
        );
        if (g_pipe == INVALID_HANDLE_VALUE) {
            time_t now = time(NULL);
            struct tm* tm = localtime(&now);
            char time_str[16];
            strftime(time_str, sizeof(time_str), "%H:%M:%S", tm);
            snprintf(log_lines[log_count % MAX_LOG_LINES], 256,
                "[%s] 命令发送失败：管道未连接", time_str);
            log_count++;
            need_log_refresh = 1;
            return 0;
        }
    }

    DWORD bytes_written;
    if (!WriteFile(g_pipe, cmd, (DWORD)strlen(cmd), &bytes_written, NULL) || bytes_written == 0) {
        CloseHandle(g_pipe);
        g_pipe = INVALID_HANDLE_VALUE;
        return 0;
    }
    return 1;
}

/* 解析推送的 JSON 并更新 status_lines */
void parse_status_push(const char* json) {
    int ep = 0, ph = 0, dp = 0;
    float wr = 0, lr2 = 0, epc = 0, lr = 0;

    const char* p;
    if ((p = strstr(json, "\"ep\":"))) { ep = atoi(p + 5); }
    if ((p = strstr(json, "\"ph\":"))) { ph = atoi(p + 5); }
    if ((p = strstr(json, "\"wr\":"))) { wr = atof(p + 5); }
    if ((p = strstr(json, "\"lr2\":"))) { lr2 = atof(p + 6); }
    if ((p = strstr(json, "\"dp\":"))) { dp = atoi(p + 5); }
    if ((p = strstr(json, "\"epc\":"))) { epc = atof(p + 6); }
    if ((p = strstr(json, "\"lr\":"))) { lr = atof(p + 5); }

    snprintf(status_lines[0], sizeof(status_lines[0]), "%d", ep);
    snprintf(status_lines[1], sizeof(status_lines[1]), "Phase %d", ph);
    snprintf(status_lines[2], sizeof(status_lines[2]), "%.2f%%", wr);
    snprintf(status_lines[3], sizeof(status_lines[3]), "%.2f%%", lr2);
    snprintf(status_lines[4], sizeof(status_lines[4]), "%d", dp);
    snprintf(status_lines[5], sizeof(status_lines[5]), "%.4f", epc);
    snprintf(status_lines[6], sizeof(status_lines[6]), "%.6f", lr);

    need_status_refresh = 1;
}