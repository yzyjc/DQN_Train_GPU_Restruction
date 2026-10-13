#include "core.h"

int is_live_two(const GameState* g, int row, int col, int player) {
    if (g->board[row * BOARD_SIZE + col] != 0) return 0;
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int d = 0; d < 4; d++) {
        int count = 1, r = row + dirs[d][0], c = col + dirs[d][1];
        int front_open = 0, back_open = 0;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r += dirs[d][0]; c += dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) front_open = 1;
        r = row - dirs[d][0]; c = col - dirs[d][1];
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r -= dirs[d][0]; c -= dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) back_open = 1;
        if (count == 2 && front_open && back_open) return 1;
    }
    return 0;
}

int is_live_three(const GameState* g, int row, int col, int player) {
    if (g->board[row * BOARD_SIZE + col] != 0) return 0;
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int d = 0; d < 4; d++) {
        int count = 1, r = row + dirs[d][0], c = col + dirs[d][1];
        int front_open = 0, back_open = 0;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r += dirs[d][0]; c += dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) front_open = 1;
        r = row - dirs[d][0]; c = col - dirs[d][1];
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r -= dirs[d][0]; c -= dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) back_open = 1;
        if (count == 3 && front_open && back_open) return 1;
    }
    return 0;
}

int is_threat_three(const GameState* g, int row, int col, int player) {
    if (g->board[row * BOARD_SIZE + col] != 0) return 0;
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int d = 0; d < 4; d++) {
        int count = 1, r = row + dirs[d][0], c = col + dirs[d][1];
        int front_open = 0, back_open = 0;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r += dirs[d][0]; c += dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) front_open = 1;
        r = row - dirs[d][0]; c = col - dirs[d][1];
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r -= dirs[d][0]; c -= dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) back_open = 1;
        if (count == 3 && (front_open ^ back_open)) return 1;
    }
    return 0;
}

int is_threat_four(const GameState* g, int row, int col, int player) {
    if (g->board[row * BOARD_SIZE + col] != 0) return 0;
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int d = 0; d < 4; d++) {
        int count = 1, r = row + dirs[d][0], c = col + dirs[d][1];
        int front_open = 0, back_open = 0;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r += dirs[d][0]; c += dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) front_open = 1;
        r = row - dirs[d][0]; c = col - dirs[d][1];
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r -= dirs[d][0]; c -= dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) back_open = 1;
        if (count == 4 && (front_open || back_open)) return 1;
    }
    return 0;
}

int detect_double_threat(const GameState* g, int row, int col, int player) {
    if (g->board[row * BOARD_SIZE + col] != 0) return 0;
    int dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    int threat_count = 0;
    for (int d = 0; d < 4; d++) {
        int count = 1, r = row + dirs[d][0], c = col + dirs[d][1];
        int front_open = 0, back_open = 0;
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r += dirs[d][0]; c += dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) front_open = 1;
        r = row - dirs[d][0]; c = col - dirs[d][1];
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == player) {
            count++; r -= dirs[d][0]; c -= dirs[d][1];
        }
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE &&
            g->board[r * BOARD_SIZE + c] == 0) back_open = 1;
        if (count == 3 && front_open && back_open) threat_count++;
    }
    return threat_count >= 2;
}

int find_opponent_winning_move(const GameState* g, int opponent) {
    for (int i = 0; i < CELL_COUNT; i++) {
        if (g->board[i] == 0) {
            GameState test = *g;
            test.board[i] = opponent;
            if (check_winner(&test) == opponent) return i;
        }
    }
    return -1;
}

int find_opponent_open_three(const GameState* g, int opponent) {
    for (int i = 0; i < CELL_COUNT; i++) {
        if (g->board[i] == 0) {
            int r = i / BOARD_SIZE, c = i % BOARD_SIZE;
            if (is_live_three(g, r, c, opponent)) return i;
        }
    }
    return -1;
}

int find_opponent_double_threat(const GameState* g, int opponent) {
    for (int i = 0; i < CELL_COUNT; i++) {
        if (g->board[i] == 0) {
            int r = i / BOARD_SIZE, c = i % BOARD_SIZE;
            if (detect_double_threat(g, r, c, opponent)) return i;
        }
    }
    return -1;
}

int evaluate_board(const GameState* g, int player) {
    int score = 0, dirs[4][2] = { {1,0},{0,1},{1,1},{1,-1} };
    for (int r = 0; r < BOARD_SIZE; r++)
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (g->board[r * BOARD_SIZE + c] != player) continue;
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
                if (count >= 4) score += 100;
                else if (count == 3) score += 10;
                else if (count == 2) score += 1;
            }
        }
    return score;
}

int minimax(GameState* g, int depth, int alpha, int beta, int maximizing) {
    int winner = check_winner(g);
    if (winner) return (winner == 1) ? 10000 : -10000;
    if (depth == 0 || is_full(g))
        return evaluate_board(g, 1) - evaluate_board(g, 2);
    int moves[CELL_COUNT], cnt = get_legal_moves(g, moves);
    if (cnt == 0) return 0;
    if (maximizing) {
        int max_eval = -100000;
        for (int i = 0; i < cnt; i++) {
            GameState child = *g;
            make_move(&child, moves[i]);
            int eval = minimax(&child, depth - 1, alpha, beta, 0);
            if (eval > max_eval) max_eval = eval;
            if (eval > alpha) alpha = eval;
            if (beta <= alpha) break;
        }
        return max_eval;
    }
    else {
        int min_eval = 100000;
        for (int i = 0; i < cnt; i++) {
            GameState child = *g;
            make_move(&child, moves[i]);
            int eval = minimax(&child, depth - 1, alpha, beta, 1);
            if (eval < min_eval) min_eval = eval;
            if (eval < beta) beta = eval;
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

int minimax_best_move(GameState* g, int depth, float temperature, PRNGState* rng) {
    int moves[CELL_COUNT], cnt = get_legal_moves(g, moves);
    if (cnt == 0) return -1;

    // ===== 修复：depth <= 0 时直接使用启发式评估 =====
    if (depth <= 0) {
        float best_score = (g->turn == 0) ? -1e9f : 1e9f;
        int best_move = moves[0];
        for (int i = 0; i < cnt; i++) {
            GameState child = *g;
            make_move(&child, moves[i]);
            float score = (float)(evaluate_board(&child, 1) - evaluate_board(&child, 2));
            if (g->turn == 0) {
                if (score > best_score) { best_score = score; best_move = moves[i]; }
            }
            else {
                if (score < best_score) { best_score = score; best_move = moves[i]; }
            }
        }
        return best_move;
    }
    // ==================================================

    float values[CELL_COUNT];
    float best_val = (g->turn == 0) ? -1e9f : 1e9f;
    for (int i = 0; i < cnt; i++) {
        GameState child = *g;
        make_move(&child, moves[i]);
        if (g->turn == 0)
            values[i] = minimax(&child, depth - 1, -100000, 100000, 0);
        else
            values[i] = minimax(&child, depth - 1, -100000, 100000, 1);
        if ((g->turn == 0 && values[i] > best_val) ||
            (g->turn == 1 && values[i] < best_val))
            best_val = values[i];
    }
    if (temperature == 0.0f) {
        int best_moves[CELL_COUNT], best_count = 0;
        for (int i = 0; i < cnt; i++) {
            if ((g->turn == 0 && values[i] == best_val) ||
                (g->turn == 1 && values[i] == best_val))
                best_moves[best_count++] = moves[i];
        }
        if (best_count == 0) return moves[0];
        return best_moves[prng_int(rng, best_count)];
    }
    else {
        float sum = 0.0f, max_val = values[0];
        for (int i = 1; i < cnt; i++)
            if (values[i] > max_val) max_val = values[i];
        for (int i = 0; i < cnt; i++) {
            values[i] = expf((values[i] - max_val) / temperature);
            sum += values[i];
        }
        if (sum == 0.0f) return moves[0];
        float r = prng_float(rng) * sum, cum = 0.0f;
        for (int i = 0; i < cnt; i++) {
            cum += values[i];
            if (cum >= r) return moves[i];
        }
        return moves[cnt - 1];
    }
}
