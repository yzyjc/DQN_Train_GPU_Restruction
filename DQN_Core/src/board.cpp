// board.cpp
#include "core.h"

void reset_game(GameState* g) {
    memset(g->board, 0, sizeof(g->board));
    g->turn = 0;
}

int is_legal_move(const GameState* g, int action) {
    return g && action >= 0 && action < CELL_COUNT && g->board[action] == 0 &&
        (g->turn == 0 || g->turn == 1);
}

void make_move(GameState* g, int action) {
    // Last line of defense. Callers still have to handle invalid actions rather
    // than treating this no-op as a completed move.
    if (!is_legal_move(g, action)) return;
    g->board[action] = g->turn + 1;
    g->turn = 1 - g->turn;
}

int check_winner(const GameState* g) {
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int r = 0; r < BOARD_SIZE; r++)
        for (int c = 0; c < BOARD_SIZE; c++) {
            int player = g->board[r * BOARD_SIZE + c];
            if (!player) continue;
            for (int d = 0; d < 4; d++) {
                int count = 1;
                int nr = r + dirs[d][0], nc = c + dirs[d][1];
                while (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE &&
                    g->board[nr * BOARD_SIZE + nc] == player) {
                    count++; nr += dirs[d][0]; nc += dirs[d][1];
                }
                nr = r - dirs[d][0]; nc = c - dirs[d][1];
                while (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE &&
                    g->board[nr * BOARD_SIZE + nc] == player) {
                    count++; nr -= dirs[d][0]; nc -= dirs[d][1];
                }
                if (count >= WIN_STREAK) return player;
            }
        }
    return 0;
}

int is_full(const GameState* g) {
    for (int i = 0; i < CELL_COUNT; i++)
        if (g->board[i] == 0) return 0;
    return 1;
}

int get_legal_moves(const GameState* g, int moves[]) {
    int cnt = 0;
    for (int i = 0; i < CELL_COUNT; i++)
        if (g->board[i] == 0) moves[cnt++] = i;
    return cnt;
}

void state_to_input(const GameState* g, float input[]) {
    for (int i = 0; i < CELL_COUNT; i++) {
        input[i] = (g->board[i] == 0) ? 0.0f : (g->board[i] == 1 ? 1.0f : -1.0f);
    }
    input[CELL_COUNT] = (float)g->turn;
}

void flip_state(const float* src, float* dst, int* flipped_action) {
    for (int i = 0; i < CELL_COUNT; i++) dst[i] = -src[24 - i];
    dst[CELL_COUNT] = 1.0f - src[CELL_COUNT];
    if (flipped_action) *flipped_action = 24 - (*flipped_action);
}

int perspective_action(int action, int perspective) {
    if (perspective == 0) return action;
    else return 24 - action;
}

// ===== 新增：视角归一化函数 =====
void normalize_state_to_self(const GameState* src, GameState* dst, int perspective) {
    *dst = *src;
    for (int i = 0; i < CELL_COUNT; i++) {
        if (src->board[i] == 0) {
            dst->board[i] = 0;
        }
        else {
            int is_self = (src->board[i] == (perspective + 1));
            dst->board[i] = is_self ? 1 : -1;
        }
    }
    dst->turn = 0; // 强制归一化视角
}
