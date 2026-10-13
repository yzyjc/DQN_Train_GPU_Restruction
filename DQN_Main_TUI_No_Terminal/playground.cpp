// playground.cpp - PLAYGROUND 人机对战快照模型
// 加载 Trainer 发布的完整模型包（网络权重 + 同代记忆库），
// 用 DQN_Core 的统一部署推理语义选择 AI 着法和分析候选。
#define _CRT_SECURE_NO_WARNINGS
#include "playground.h"
#include "core.h"
#include "independent_networks.h"
#include <memory>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static NeuralNet g_play_net;
static NeuralNet g_play_white;
static int g_play_independent = 0;
static SRWLOCK g_play_net_lock = SRWLOCK_INIT;
static int g_model_loaded = 0;
static char g_model_info[96] = "未加载";

int playground_load_snapshot(void) {
    /* Preferred format is one atomic generation containing both network and
       memory.  Raw current_snapshot.bin remains a compatibility fallback. */
    NeuralNet* incoming = (NeuralNet*)malloc(sizeof(NeuralNet));
    if (!incoming) { snprintf(g_model_info, sizeof(g_model_info), "快照加载内存不足"); return -1; }
    int package_episode = 0, independent = 0;
    auto white = std::make_unique<NeuralNet>();
    // Serialize the memory+network swap against every Playground inference.
    AcquireSRWLockExclusive(&g_play_net_lock);
    if (memory_load_model_package_pair(incoming, white.get(), "current_model_package.bin", &package_episode, &independent)) {
        memcpy(&g_play_net, incoming, sizeof(NeuralNet));
        g_play_white=*white;
        g_play_independent=independent;
        g_model_loaded = 1;
        ReleaseSRWLockExclusive(&g_play_net_lock);
        free(incoming);
        snprintf(g_model_info, sizeof(g_model_info), "%s模型包 (EP %d, 含记忆库)", independent ? "黑白双网" : "共享", package_episode);
        return 0;
    }
    ReleaseSRWLockExclusive(&g_play_net_lock);
    // A damaged authoritative package must not silently fall back to a stale single network.
    if (GetFileAttributesA("current_model_package.bin") != INVALID_FILE_ATTRIBUTES) {
        free(incoming); snprintf(g_model_info,sizeof(g_model_info),"模型包损坏，保留旧快照"); return -1;
    }
    FILE* f = fopen("current_snapshot.bin", "rb");
    if (!f) {
        free(incoming);
        snprintf(g_model_info, sizeof(g_model_info), "无快照 (current_snapshot.bin 不存在)");
        return -1;
    }
    _fseeki64(f, 0, SEEK_END);
    int64_t bytes = _ftelli64(f);
    rewind(f);
    if (bytes != (int64_t)sizeof(NeuralNet)) {
        fclose(f); free(incoming);
        snprintf(g_model_info, sizeof(g_model_info), "快照尺寸异常 (%lld/%llu)",
            (long long)bytes, (unsigned long long)sizeof(NeuralNet));
        return -1;
    }
    /* 先读到临时对象；失败时保留上一份可用模型，不污染正在对弈的网络。 */
    size_t n = fread(incoming, sizeof(NeuralNet), 1, f);
    fclose(f);
    bool finite=true;
    const float* values=(const float*)incoming;
    for(size_t i=0;i<sizeof(NeuralNet)/sizeof(float) && n==1;++i) if(!isfinite(values[i])) finite=false;
    if (n != 1 || !finite) {
        free(incoming);
        snprintf(g_model_info, sizeof(g_model_info), "快照读取失败");
        return -1;
    }
    AcquireSRWLockExclusive(&g_play_net_lock);
    memcpy(&g_play_net, incoming, sizeof(NeuralNet));
    g_play_white=*incoming; g_play_independent=0;
    g_model_loaded = 1;
    ReleaseSRWLockExclusive(&g_play_net_lock);
    free(incoming);
    snprintf(g_model_info, sizeof(g_model_info), "兼容裸快照 + 已加载记忆库");
    return 0;
}

int playground_analyze(const int board[25], int turn,
    float* qvals_out, PlayCand* cands, int cand_cap, int* cand_count, double* infer_ms) {
    if (!g_model_loaded || (turn!=0 && turn!=1)) return -1;
    GameState g;
    for (int i = 0; i < CELL_COUNT; i++) g.board[i] = board[i];
    g.turn = turn;
    float input[CELL_COUNT + 1];
    state_to_input(&g, input);
    float qvals[CELL_COUNT];
    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    AcquireSRWLockShared(&g_play_net_lock);
    forward_cpu(g_play_independent && turn==1 ? &g_play_white : &g_play_net, input, qvals);
    memory_apply_model_scores(&g, qvals);
    ReleaseSRWLockShared(&g_play_net_lock);
    QueryPerformanceCounter(&t1);
    if (infer_ms) *infer_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart;

    if (qvals_out) memcpy(qvals_out, qvals, CELL_COUNT * sizeof(float));

    int moves[CELL_COUNT], cnt = get_legal_moves(&g, moves);
    if (cnt <= 0) { if (cand_count) *cand_count = 0; return 0; }

    /* 按 Q 值降序排序合法着法 */
    int order[CELL_COUNT];
    for (int i = 0; i < cnt; i++) order[i] = moves[i];
    for (int i = 0; i < cnt - 1; i++)
        for (int j = i + 1; j < cnt; j++)
            if (qvals[order[j]] > qvals[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }

    /* policy = softmax(Q/0.5) over legal moves */
    float mx = -1e9f;
    for (int i = 0; i < cnt; i++) if (qvals[order[i]] > mx) mx = qvals[order[i]];
    float sum = 0;
    float soft[CELL_COUNT];
    for (int i = 0; i < cnt; i++) {
        soft[i] = expf((qvals[order[i]] - mx) / 0.5f);
        sum += soft[i];
    }
    int nc = cnt < cand_cap ? cnt : cand_cap;
    for (int i = 0; i < nc; i++) {
        cands[i].move = order[i];
        cands[i].q = qvals[order[i]];
        cands[i].policy = sum > 0 ? soft[i] / sum : 0;
        cands[i].win = 0.5f + 0.5f * tanhf(qvals[order[i]] * 2.0f);   /* Q 归一化胜率估算 */
    }
    if (cand_count) *cand_count = nc;
    return nc;
}

int playground_ai_move(const int board[25], int turn) {
    PlayCand cands[5];
    int cnt = 0;
    double ms = 0;
    if (playground_analyze(board, turn, NULL, cands, 5, &cnt, &ms) <= 0) {
        /* 快照首次加载失败时自动重试；仍失败也要保证对局状态机继续走。
           模型面板会保留真实加载错误，合法着兜底不会伪装成模型推理成功。 */
        if (!g_model_loaded && playground_load_snapshot() == 0 &&
            playground_analyze(board, turn, NULL, cands, 5, &cnt, &ms) > 0)
            return cands[0].move;
        GameState g;
        for (int i = 0; i < CELL_COUNT; i++) g.board[i] = board[i];
        g.turn = turn;
        int moves[CELL_COUNT], legal = get_legal_moves(&g, moves);
        if (legal <= 0) return -1;
        for (int i = 0; i < legal; i++) if (moves[i] == 12) return 12;
        return moves[0];
    }
    return cands[0].move;
}

const char* playground_model_info(void) { return g_model_info; }
int playground_loaded(void) { return g_model_loaded; }

/* 真实参数数（NeuralNet 可学习 float 数） */
int playground_param_count(void) {
    return (int)(sizeof(NeuralNet) / sizeof(float));
}

/* 快照创建时间（current_snapshot.bin 修改时间 HH:MM:SS），失败返回空 */
const char* playground_created_at(void) {
    static char buf[16] = "--:--:--";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(GetFileAttributesA("current_model_package.bin")!=INVALID_FILE_ATTRIBUTES ? "current_model_package.bin" : "current_snapshot.bin", &fd);
    if (h != INVALID_HANDLE_VALUE) {
        FILETIME lt;
        FileTimeToLocalFileTime(&fd.ftLastWriteTime, &lt);
        SYSTEMTIME st;
        FileTimeToSystemTime(&lt, &st);
        snprintf(buf, sizeof(buf), "%02d:%02d:%02d", st.wHour, st.wMinute, st.wSecond);
        FindClose(h);
    }
    return buf;
}

const NeuralNet* playground_get_net(void) {
    return g_model_loaded ? &g_play_net : NULL;
}

int playground_copy_net(NeuralNet* out) {
    if (!out) return 0;
    AcquireSRWLockShared(&g_play_net_lock);
    int ok = g_model_loaded;
    if (ok) memcpy(out, &g_play_net, sizeof(*out));
    ReleaseSRWLockShared(&g_play_net_lock);
    return ok;
}

// GPT6-Astra: skill evaluation receives both immutable players, not just black.
int playground_copy_agent(DQNAgent* out) {
    if(!out)return 0;
    memset(out, 0, sizeof(*out));
    AcquireSRWLockShared(&g_play_net_lock);
    int ok=g_model_loaded;
    if(ok){
        out->q_net=g_play_net; out->target_net=g_play_net;
        if(g_play_independent){
            auto n=independent_training::indep_ensure(out);
            if(!n)ok=0;
            else {n->seed_both_from(g_play_net,g_play_net);
                n->online(independent_training::Side::White)=g_play_white;
                n->target(independent_training::Side::White)=g_play_white;n->set_active(1);}
        } else if(out->indep) out->indep->set_active(0);
    }
    ReleaseSRWLockShared(&g_play_net_lock);return ok;
}
