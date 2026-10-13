// dqn_replay.cpp - 对局回放（从 DQN_Train_Logreader.cpp 移植的纯数据逻辑）
// 自我对弈 game.dat + Minimax 对战 minimax_games.dat 的加载、棋盘重建、胜负判定
#define _CRT_SECURE_NO_WARNINGS
#include "dqn_replay.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 256
#define PHASE0_END 20000
#define LOG_DIR "logs\\"   /* 通信兼日志文件目录（game.dat/minimax_games.dat 等） */
#include "bin_reader.h"
#define PHASE1_END 120000

/* ---------- 阶段边界（与 core.h / logreader 一致） ---------- */
static int get_phase_start_for_ep(int global_ep) {
    if (global_ep < PHASE0_END) return 0;
    if (global_ep < PHASE1_END) return PHASE0_END;
    return PHASE1_END;
}

/* Count spaces to determine format (7-col has 6 spaces, 6-col has 5 spaces) */
static int count_spaces(const char* s) {
    int count = 0;
    while (*s) { if (*s == ' ') count++; s++; }
    return count;
}

/* ---------- 加载自我对弈棋谱（兼容 7/6 列 + global/local episode） ---------- */
static int load_game_episode(const char* filename, int episode, GameStep* buffer, int max_steps) {
    /* 从 metric_steps.bin 读取（ep, step, fields[4]={player, action, reward, done}） */
    BRRecord rec_buf[512];
    int n = br_read_episode(active_training_log_path(LOG_DIR, "metric_steps.bin").c_str(), episode, rec_buf, 512);
    int count = 0;
    for (int i = 0; i < n && count < max_steps; i++) {
        buffer[count].episode = rec_buf[i].episode;
        buffer[count].step    = rec_buf[i].step;
        buffer[count].player  = (int)rec_buf[i].fields[0];
        buffer[count].action  = (int)rec_buf[i].fields[1];
        buffer[count].reward  = rec_buf[i].fields[2];
        buffer[count].done    = (int)rec_buf[i].fields[3];
        count++;
        if (buffer[count - 1].done) break;
    }
    if (count > 0) return count;

    /* fallback: 按 local episode 匹配（旧 bin 文件无 episode 或 phase 偏移场景） */
    {
        int local_ep = episode - get_phase_start_for_ep(episode);
        if (local_ep >= 0 && local_ep != episode) {
            int n2 = br_read_episode(active_training_log_path(LOG_DIR, "metric_steps.bin").c_str(), local_ep, rec_buf, 512);
            for (int i = 0; i < n2 && count < max_steps; i++) {
                buffer[count].episode = episode;   /* 映射回全局 episode */
                buffer[count].step    = rec_buf[i].step;
                buffer[count].player  = (int)rec_buf[i].fields[0];
                buffer[count].action  = (int)rec_buf[i].fields[1];
                buffer[count].reward  = rec_buf[i].fields[2];
                buffer[count].done    = (int)rec_buf[i].fields[3];
                count++;
                if (buffer[count - 1].done) break;
            }
        }
    }
    return count;
}

/* ---------- 加载 Minimax 对战棋谱（从 metric_minimax.bin，fields: gid,turn,action,done） ---------- */
static int load_minimax_game(const char* filename, int episode, int game_id, GameStep* buffer, int max_steps) {
    (void)filename;
    BRRecord rec_buf[512];
    int n = br_read_episode(active_training_log_path(LOG_DIR, "metric_minimax.bin").c_str(), episode, rec_buf, 512);
    int count = 0;
    for (int i = 0; i < n && count < max_steps; i++) {
        int gid = (int)rec_buf[i].fields[0];
        if (gid != game_id) continue;
        buffer[count].episode = rec_buf[i].episode;
        buffer[count].step    = rec_buf[i].step;
        buffer[count].player  = (int)rec_buf[i].fields[1];
        buffer[count].action  = (int)rec_buf[i].fields[2];
        buffer[count].reward  = 0.0f;   /* metric_minimax.bin 无 reward 字段 */
        buffer[count].done    = (int)rec_buf[i].fields[3];
        count++;
        if (buffer[count - 1].done) break;
    }
    return count;
}

/* ---------- 胜负判定：0=无胜者 1=X胜 2=O胜 ---------- */
static int check_winner_from_board(char board[REPLAY_BOARD_SIZE][REPLAY_BOARD_SIZE]) {
    for (int r = 0; r < REPLAY_BOARD_SIZE; r++)
        for (int c = 0; c <= REPLAY_BOARD_SIZE - REPLAY_WIN_STREAK; c++) {
            char first = board[r][c];
            if (first == '.') continue;
            int count = 1;
            for (int k = 1; k < REPLAY_WIN_STREAK; k++) {
                if (board[r][c + k] == first) count++; else break;
            }
            if (count >= REPLAY_WIN_STREAK) return (first == 'X') ? 1 : 2;
        }
    for (int c = 0; c < REPLAY_BOARD_SIZE; c++)
        for (int r = 0; r <= REPLAY_BOARD_SIZE - REPLAY_WIN_STREAK; r++) {
            char first = board[r][c];
            if (first == '.') continue;
            int count = 1;
            for (int k = 1; k < REPLAY_WIN_STREAK; k++) {
                if (board[r + k][c] == first) count++; else break;
            }
            if (count >= REPLAY_WIN_STREAK) return (first == 'X') ? 1 : 2;
        }
    for (int r = 0; r <= REPLAY_BOARD_SIZE - REPLAY_WIN_STREAK; r++)
        for (int c = 0; c <= REPLAY_BOARD_SIZE - REPLAY_WIN_STREAK; c++) {
            char first = board[r][c];
            if (first == '.') continue;
            int count = 1;
            for (int k = 1; k < REPLAY_WIN_STREAK; k++) {
                if (board[r + k][c + k] == first) count++; else break;
            }
            if (count >= REPLAY_WIN_STREAK) return (first == 'X') ? 1 : 2;
        }
    for (int r = 0; r <= REPLAY_BOARD_SIZE - REPLAY_WIN_STREAK; r++)
        for (int c = REPLAY_WIN_STREAK - 1; c < REPLAY_BOARD_SIZE; c++) {
            char first = board[r][c];
            if (first == '.') continue;
            int count = 1;
            for (int k = 1; k < REPLAY_WIN_STREAK; k++) {
                if (board[r + k][c - k] == first) count++; else break;
            }
            if (count >= REPLAY_WIN_STREAK) return (first == 'X') ? 1 : 2;
        }
    return 0;
}

/* ---------- 内部缓冲 ---------- */
static GameStep g_steps[REPLAY_MAX_STEPS];
static int g_step_count = 0;
static int g_winner = -1;
static char g_meta[96] = "";

int tui_replay_load(const char* cmd) {
    g_step_count = 0;
    g_winner = -1;
    g_meta[0] = '\0';

    int episode, mode, idx;
    char color;
    if (sscanf(cmd, "g %d", &episode) == 1) {
        int n = load_game_episode(active_training_log_path(LOG_DIR, "game.dat").c_str(), episode, g_steps, REPLAY_MAX_STEPS);
        if (n <= 0) {
            snprintf(g_meta, sizeof(g_meta), "未找到对局 g %d", episode);
            return n;
        }
        g_step_count = n;
        snprintf(g_meta, sizeof(g_meta), "Ep %d 自我对弈", episode);
    } else if (sscanf(cmd, "m %d %d %c %d", &episode, &mode, &color, &idx) == 4) {
        if (mode < 0 || mode > 2) {
            snprintf(g_meta, sizeof(g_meta), "mode 需 0~2 (ε档位)");
            return -1;
        }
        if (color != 'b' && color != 'w') {
            snprintf(g_meta, sizeof(g_meta), "color 需 b(黑) 或 w(白)");
            return -1;
        }
        if (idx < 0 || idx > 199) {
            snprintf(g_meta, sizeof(g_meta), "index 需 0~199");
            return -1;
        }
        int game_id = mode * 400 + (color == 'b' ? 0 : 200) + idx;
        int n = load_minimax_game(active_training_log_path(LOG_DIR, "minimax_games.dat").c_str(), episode, game_id, g_steps, REPLAY_MAX_STEPS);
        if (n <= 0) {
            snprintf(g_meta, sizeof(g_meta), "未找到 Minimax 局 m %d %d %c %d", episode, mode, color, idx);
            return n;
        }
        g_step_count = n;
        snprintf(g_meta, sizeof(g_meta), "Ep %d Minimax mode%d %s 第%d局",
            episode, mode, color == 'b' ? "黑" : "白", idx);
    } else {
        snprintf(g_meta, sizeof(g_meta), "用法: g <ep> | m <ep> <mode> <color> <idx>");
        return -1;
    }

    /* 用全部落子重建棋盘，判定终局胜负 */
    char board[REPLAY_BOARD_SIZE][REPLAY_BOARD_SIZE];
    memset(board, '.', sizeof(board));
    int done = 0;
    for (int i = 0; i < g_step_count; i++) {
        int pos = g_steps[i].action;
        if (pos >= 0 && pos < REPLAY_BOARD_SIZE * REPLAY_BOARD_SIZE) {
            board[pos / REPLAY_BOARD_SIZE][pos % REPLAY_BOARD_SIZE] =
                (g_steps[i].player == 0) ? 'X' : 'O';
        }
        if (g_steps[i].done) done = 1;
    }
    g_winner = done ? check_winner_from_board(board) : -1;
    return g_step_count;
}

const GameStep* tui_replay_steps(void) { return g_steps; }
int tui_replay_count(void) { return g_step_count; }
int tui_replay_winner(void) { return g_winner; }
const char* tui_replay_meta(void) { return g_meta; }
