// neural_net.c
// Neural network implementation (CPU initialization + forward/train)
#include "core.h"
#include "gpu_types.h"
#include "network_input.h"
#include "sentinel_guard.h"
#include "independent_networks.h"
#include <math.h>
#include <string.h>
#include <memory>

/* backward_update 分段计时 (extern "C" 统一符号名, 避免 clang/MSVC 名字修饰不匹配) */
extern "C" {
volatile double g_bwd_fc4_us = 0, g_bwd_fc3_us = 0, g_bwd_fc2_us = 0, g_bwd_fc1_us = 0;
volatile double g_bwd_pool_us = 0, g_bwd_attn_us = 0, g_bwd_embed_us = 0;
volatile int g_bwd_count = 0;
/* train_step_batch per-sample 开销分段 */
volatile double g_ts_setup_us = 0, g_ts_memset_us = 0, g_ts_fwd_us = 0, g_ts_hubler_us = 0, g_ts_bwd_us = 0;
volatile double g_ts_validate_us = 0;
volatile int g_ts_count = 0;
}

using independent_training::Side;
using independent_training::SideUpdateState;
using namespace independent_training;

static const float TRAIN_GRAD_CLIP = 1.0f;
static const float TRAIN_PARAM_ABS_LIMIT = 100.0f;
/* Targets are bounded to +/-5.  A Huber loss above 100 therefore indicates
   runaway Q activations, not merely a difficult sample.  Such a sample is
   observable to diagnostics but must not update the live model. */
static const float TRAIN_LOSS_REJECT_LIMIT = 100.0f;
static unsigned long long g_gpu_loss_reject_count = 0;

/* ===== 训练安全门：整批参数更新范数限制 =====
   逐参数裁剪（clip ±1）允许几十万参数同时同向漂移，总更新量无上限。
   这里在提交前检查「整批参数变化 L2 范数」——它等价于 lr × 全局梯度范数。
   维护一个 EMA，若本批范数相对 EMA 突增（如 LR 跳变/发散），拒绝提交。
   这是全局范数裁剪的提交侧实现，不重构反向传播即可限制总更新量。 */
static const double TRAIN_BATCH_DELTA_EMA_ALPHA = 0.1;
static const double TRAIN_BATCH_DELTA_SURGE_RATIO = 8.0;
static double g_batch_delta_ema = 0.0;
static int g_batch_delta_warmup = 2;   /* 跳过前 2 批：冷启动/首个 batch 的梯度范数异常大，不应污染 EMA */
unsigned long long g_batch_delta_reject_count = 0;   /* 实验统计：参数范数突增门拒绝次数 */

/* ===== 按方健康统计 =====
   上面那组静态量是进程全局的「共享网络」健康基线，保留给未接入按方路由的调用方。
   一旦开启按方路由，DELTA 门的 EMA / 预热 / 拒绝计数与学习率回退必须逐方归属：
   否则黑方一批的范数突增会把白方 EMA 拉高，进而拒绝白方合法更新；黑方的 LR 回退
   也会同时压低白方学习率。这里把按方决策集中到一处，GPU / CPU 两条路径共用。 */
struct DeltaGateDecision {
    int warmed_up = 0;      /* 本批只建立基线 */
    int allowed = 1;
    int rejected = 0;       /* allowed==0 且已计入按方拒绝 */
    double baseline = 0.0;
    double ema_alpha = TRAIN_BATCH_DELTA_EMA_ALPHA;
};

static DeltaGateDecision delta_gate_decide(DQNAgent* agent, Side side, double delta_norm) {
    DeltaGateDecision d;
    if (independent_training::indep_active(agent)) {
        d.ema_alpha = TRAIN_BATCH_DELTA_EMA_ALPHA;
        const SideUpdateState& u = agent->indep->updates(side);
        d.baseline = u.delta_ema;
        d.warmed_up = u.delta_warmup > 0;
        d.allowed = agent->indep->delta_gate_allows(
            side, delta_norm, TRAIN_BATCH_DELTA_SURGE_RATIO, 1e-6, nullptr);
        d.rejected = !d.allowed;
        return d;
    }
    d.warmed_up = g_batch_delta_warmup > 0;
    d.baseline = g_batch_delta_ema;
    if (d.warmed_up) return d;
    d.allowed = isfinite(delta_norm) &&
        delta_norm <= TRAIN_BATCH_DELTA_SURGE_RATIO * g_batch_delta_ema + 1e-6;
    d.rejected = !d.allowed;
    return d;
}

static void delta_gate_commit(DQNAgent* agent, Side side, int allowed, double delta_norm) {
    if (independent_training::indep_active(agent)) {
        if (!allowed) { agent->indep->note_rejected(side); return; }
        agent->indep->commit_delta(side, delta_norm, 1, TRAIN_BATCH_DELTA_EMA_ALPHA);
        return;
    }
    if (!allowed) { ++g_batch_delta_reject_count; return; }
    if (g_batch_delta_warmup > 0) { --g_batch_delta_warmup; g_batch_delta_ema = delta_norm; return; }
    g_batch_delta_ema += TRAIN_BATCH_DELTA_EMA_ALPHA * (delta_norm - g_batch_delta_ema);
}

static void note_loss_reject(DQNAgent* agent, Side side) {
    if (independent_training::indep_active(agent)) {
        agent->indep->updates(side).rejected_loss_batches++;
        agent->indep->note_rejected(side);
    }
}

static void note_committed(DQNAgent* agent, Side side) {
    if (independent_training::indep_active(agent)) agent->indep->note_committed(side);
}

static void pack_params(const NeuralNet* net, GpuNetParams* p);   /* 前向声明，定义见下 */

/* 整批参数更新范数：只对可学习参数（GpuNetParams 布局）做差。
   旧实现对整个 NeuralNet 做差，把前向缓存字段（tokens/attn_context/flat_cache 等）
   误当成参数，导致换一个合法棋盘也会被误判为"参数突增"。 */
static double batch_delta_norm(const NeuralNet* a, const NeuralNet* b) {
    std::unique_ptr<GpuNetParams> pa(new (std::nothrow) GpuNetParams);
    std::unique_ptr<GpuNetParams> pb(new (std::nothrow) GpuNetParams);
    if (!pa || !pb) return 0.0;
    pack_params(a, pa.get());
    pack_params(b, pb.get());
    double n2 = 0.0;
    const float* fa = (const float*)pa.get();
    const float* fb = (const float*)pb.get();
    const size_t n = sizeof(GpuNetParams) / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        double d = (double)fa[i] - (double)fb[i];
        n2 += d * d;
    }
    return sqrt(n2);
}

static float clipped_gradient(float value) {
    if (!isfinite(value)) return value; /* keep numeric faults observable */
    return fmaxf(-TRAIN_GRAD_CLIP, fminf(TRAIN_GRAD_CLIP, value));
}

// ============================================================================
// GPU 驻留神经网络（声明来自 gpu_neural_net.hip）
// ============================================================================
extern "C" {
    int gpu_nn_create(GpuNetHandle* out);
    void gpu_nn_destroy(GpuNetHandle* h);
    int gpu_nn_upload(GpuNetHandle h, const void* cpu_params, size_t size);
    int gpu_nn_download(GpuNetHandle h, void* cpu_params, size_t size);
    int gpu_nn_forward(GpuNetHandle h, const float* input, float* output);
    int gpu_nn_forward_batch(GpuNetHandle h, const float* states, int batch_size, float* out_qvals);
    int gpu_nn_train_step(GpuNetHandle h, const float* state, int action, float target, float lr, float* out_loss);
    int gpu_nn_train_step_batch(GpuNetHandle h, const float* states, const int* actions, const float* targets, int batch_size, float lr, float* out_loss);
    int gpu_nn_soft_update(GpuNetHandle target, GpuNetHandle source, float tau);
}

// GPU 激活状态 + 句柄。
// 按方路由生效时**四套网络同时驻留**：g_gpu_h[side][which]，side 0=黑 1=白，
// which 0=online 1=target。选招/训练/目标同步直接取对应句柄，不做「每批下载旧方、
// 上传新方」的切换。非独立模式只创建并维护 [0][*] 这一对，行为与历史逐位一致。
static int g_gpu_forward_active = 0;
static int g_gpu_untrusted = 0;      /* 进程级：任一设备调用失败即作废**全部**句柄 */
static int g_gpu_per_side = 0;       /* 激活时按方路由是否生效：决定 [1][*] 是否存在 */
static GpuNetHandle g_gpu_h[2][2] = { { {0}, {0} }, { {0}, {0} } };
static const GpuNetHandle g_gpu_invalid = {0};
static GpuNetParams g_cpu_params;  // 打包/解包中转缓冲

/* 句柄直取：side<0 表示历史共享的 [0][*]；side 越界或按方行未驻留时返回无效句柄。 */
static GpuNetHandle gpu_handle(int side, int use_target) {
    const int s = (side <= 0) ? 0 : 1;
    if (s == 1 && !g_gpu_per_side) return g_gpu_invalid;
    return g_gpu_h[s][use_target ? 1 : 0];
}

/* 释放两组（按方时是四组）句柄。 */
static void gpu_destroy_all_handles_locked(void) {
    for (int s = 0; s < 2; ++s)
        for (int w = 0; w < 2; ++w)
            if (g_gpu_h[s][w].is_valid) gpu_nn_destroy(&g_gpu_h[s][w]);
    g_gpu_per_side = 0;
}

/* 句柄是否可用于本次访问（进程未失效 + 按方行已创建）。 */
static int gpu_handles_ready(int side) {
    const int s = (side <= 0) ? 0 : 1;
    if (s == 1 && !g_gpu_per_side) return 0;
    return g_gpu_h[s][0].is_valid && g_gpu_h[s][1].is_valid;
}

// NeuralNet -> GpuNetParams 打包
static void pack_params(const NeuralNet* net, GpuNetParams* p) {
    memcpy(p->token_embed, net->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, net->pos_embed, sizeof(p->pos_embed));
    memcpy(p->w1, net->w1, sizeof(p->w1));
    memcpy(p->b1, net->b1, sizeof(p->b1));
    memcpy(p->w2, net->w2, sizeof(p->w2));
    memcpy(p->b2, net->b2, sizeof(p->b2));
    memcpy(p->w3, net->w3, sizeof(p->w3));
    memcpy(p->b3, net->b3, sizeof(p->b3));
    memcpy(p->w4, net->w4, sizeof(p->w4));
    memcpy(p->b4, net->b4, sizeof(p->b4));
    memcpy(p->attn_wq, net->attn_wq, sizeof(p->attn_wq));
    memcpy(p->attn_bq, net->attn_bq, sizeof(p->attn_bq));
    memcpy(p->attn_wk, net->attn_wk, sizeof(p->attn_wk));
    memcpy(p->attn_bk, net->attn_bk, sizeof(p->attn_bk));
    memcpy(p->attn_wv, net->attn_wv, sizeof(p->attn_wv));
    memcpy(p->attn_bv, net->attn_bv, sizeof(p->attn_bv));
    memcpy(p->attn_wo, net->attn_wo, sizeof(p->attn_wo));
    memcpy(p->attn_bo, net->attn_bo, sizeof(p->attn_bo));
    memcpy(p->pool_w, net->pool_w, sizeof(p->pool_w));
    memcpy(p->pool_b, net->pool_b, sizeof(p->pool_b));
}

// GpuNetParams -> NeuralNet 解包（只写参数，不动缓存字段）
static void unpack_params(const GpuNetParams* p, NeuralNet* net) {
    memcpy(net->token_embed, p->token_embed, sizeof(net->token_embed));
    memcpy(net->pos_embed, p->pos_embed, sizeof(net->pos_embed));
    memcpy(net->w1, p->w1, sizeof(net->w1));
    memcpy(net->b1, p->b1, sizeof(net->b1));
    memcpy(net->w2, p->w2, sizeof(net->w2));
    memcpy(net->b2, p->b2, sizeof(net->b2));
    memcpy(net->w3, p->w3, sizeof(net->w3));
    memcpy(net->b3, p->b3, sizeof(net->b3));
    memcpy(net->w4, p->w4, sizeof(net->w4));
    memcpy(net->b4, p->b4, sizeof(net->b4));
    memcpy(net->attn_wq, p->attn_wq, sizeof(net->attn_wq));
    memcpy(net->attn_bq, p->attn_bq, sizeof(net->attn_bq));
    memcpy(net->attn_wk, p->attn_wk, sizeof(net->attn_wk));
    memcpy(net->attn_bk, p->attn_bk, sizeof(net->attn_bk));
    memcpy(net->attn_wv, p->attn_wv, sizeof(net->attn_wv));
    memcpy(net->attn_bv, p->attn_bv, sizeof(net->attn_bv));
    memcpy(net->attn_wo, p->attn_wo, sizeof(net->attn_wo));
    memcpy(net->attn_bo, p->attn_bo, sizeof(net->attn_bo));
    memcpy(net->pool_w, p->pool_w, sizeof(net->pool_w));
    memcpy(net->pool_b, p->pool_b, sizeof(net->pool_b));
}

// ============================================================================
// GPU 网络管理接口（供训练程序调用）
// ============================================================================
static SRWLOCK g_gpu_api_lock = SRWLOCK_INIT;

int gpu_net_active(void) {
    AcquireSRWLockShared(&g_gpu_api_lock);
    int active = g_gpu_forward_active;
    ReleaseSRWLockShared(&g_gpu_api_lock);
    return active;
}

/* ============================================================================
   GPU 初始化：按方路由生效时**四套网络同时驻留**。
     g_gpu_h[0][0/1] = 黑方 online/target
     g_gpu_h[1][0/1] = 白方 online/target
   非独立模式只创建 [0][*]，载荷来自共享 q_net/target_net，行为与历史逐位一致。
   驻留之后不存在「设备上是谁」的歧义，也就不存在「把设备载荷写回某一方影子」
   这类跨方污染路径。
   ============================================================================ */
int gpu_net_activate(const DQNAgent* agent) {
    if (!agent) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    if (g_gpu_forward_active) {
        /* 重复激活：先释放旧句柄，避免泄漏与残留。 */
        gpu_destroy_all_handles_locked();
        g_gpu_forward_active = 0;
    }
    DQNAgent* mutable_agent = const_cast<DQNAgent*>(agent);
    const int per_side = indep_active(agent);
    const int rows = per_side ? 2 : 1;

    for (int s = 0; s < rows; ++s) {
        const Side side = (s == 0) ? Side::Black : Side::White;
        const NeuralNet* online = per_side ? agent_online_net(mutable_agent, side) : &agent->q_net;
        const NeuralNet* target = per_side ? agent_target_net(mutable_agent, side) : &agent->target_net;
        if (!online || !target) {
            gpu_destroy_all_handles_locked();
            ReleaseSRWLockExclusive(&g_gpu_api_lock); return -6;
        }
        for (int w = 0; w < 2; ++w) {
            if (gpu_nn_create(&g_gpu_h[s][w]) != 0) {
                gpu_destroy_all_handles_locked();
                ReleaseSRWLockExclusive(&g_gpu_api_lock); return -2;
            }
            pack_params(w == 0 ? online : target, &g_cpu_params);
            if (gpu_nn_upload(g_gpu_h[s][w], &g_cpu_params, sizeof(g_cpu_params)) != 0) {
                gpu_destroy_all_handles_locked();
                ReleaseSRWLockExclusive(&g_gpu_api_lock); return (w == 0) ? -4 : -5;
            }
        }
    }
    g_gpu_per_side = per_side;
    g_gpu_forward_active = 1;
    g_gpu_untrusted = 0;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    printf("[GPU] resident handles: %s (%s online/target each)\n",
        per_side ? "black + white" : "shared",
        per_side ? "4 handles" : "2 handles");
    return 0;
}

void gpu_net_deactivate(void) {
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    g_gpu_forward_active = 0;
    gpu_destroy_all_handles_locked();
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
}

// GPU 操作重试宏：失败时等待后重试，最多 max_retry 次
#define GPU_RETRY_LOOP(op, max_retry, label) do { \
    int _retry = 0; \
    for (; _retry < (max_retry); _retry++) { \
        if ((op) == 0) break; \
        printf("[GPU] %s 失败 (attempt %d/%d)，等待重试...\n", label, _retry + 1, max_retry); \
        Sleep(100 * (_retry + 1)); \
    } \
    if (_retry >= (max_retry)) { \
        printf("[GPU] %s 重试 %d 次均失败\n", label, max_retry); \
    } \
} while(0)

/* 按方参数镜像的前向声明；定义见下方「按方 GPU 参数镜像」段。 */
static int download_side(DQNAgent* agent, Side side);

// 从 GPU 下载参数到 agent（保存 checkpoint 前调用）
int gpu_net_download_agent(DQNAgent* agent) {
    if (!agent) return -1;
    /* 按方路由生效时把「当前绑定方」的影子同步到它自己的驻留句柄。
       四句柄驻留之后绑定只是一条兼容性的记录（见 gpu_net_bind_side），
       它**不再是设备访问的门禁**：句柄按 (side, which) 直取。 */
    if (indep_active(agent)) return download_side(agent, indep_bound(agent));
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    if (!g_gpu_forward_active || g_gpu_untrusted) { ReleaseSRWLockExclusive(&g_gpu_api_lock); return -1; }
    int result = 0;
    /* 重试3次：GPU 偶发超时/总线错误可通过重试恢复 */
    { int rc; GPU_RETRY_LOOP((rc = gpu_nn_download(gpu_handle(-1, 0), &g_cpu_params, sizeof(g_cpu_params))), 3, "download q_net"); if (rc != 0) result = -2; }
    if (result == 0) {
        unpack_params(&g_cpu_params, &agent->q_net);
        { int rc; GPU_RETRY_LOOP((rc = gpu_nn_download(gpu_handle(-1, 1), &g_cpu_params, sizeof(g_cpu_params))), 3, "download target_net"); if (rc != 0) result = -3; }
        if (result == 0) unpack_params(&g_cpu_params, &agent->target_net);
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail(1);
    return result;
}

/* 保存/导出用的「两方同一逻辑时点」回读：在同一把锁内把两方的四个驻留句柄
   分别写回各自影子。非独立模式退化为共享那一对（就是历史行为）。 */
int gpu_net_download_sides_pair(DQNAgent* agent) {
    if (!agent) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    if (!g_gpu_forward_active || g_gpu_untrusted) { ReleaseSRWLockExclusive(&g_gpu_api_lock); return -1; }
    int result = 0;
    const int rows = g_gpu_per_side ? 2 : 1;
    for (int s = 0; s < rows && result == 0; ++s) {
        const Side side = (s == 0) ? Side::Black : Side::White;
        NeuralNet* online = g_gpu_per_side ? agent_online_net(agent, side) : &agent->q_net;
        NeuralNet* target = g_gpu_per_side ? agent_target_net(agent, side) : &agent->target_net;
        if (!online || !target) { result = -1; break; }
        int rc = -1;
        GPU_RETRY_LOOP((rc = gpu_nn_download(g_gpu_h[s][0], &g_cpu_params, sizeof(g_cpu_params))), 3,
            "download pair online");
        if (rc != 0) { result = -2; break; }
        unpack_params(&g_cpu_params, online);
        GPU_RETRY_LOOP((rc = gpu_nn_download(g_gpu_h[s][1], &g_cpu_params, sizeof(g_cpu_params))), 3,
            "download pair target");
        if (rc != 0) { result = -3; break; }
        unpack_params(&g_cpu_params, target);
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail(1);
    return result;
}

/* 历史兼容：读「当前绑定方」的 online。按方路由未生效时读共享 online。 */
int gpu_net_snapshot_q(NeuralNet* destination) {
    if (!destination) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    const int side = g_gpu_per_side ? side_to_physical(indep_bound(g_current_agent)) : -1;
    GpuNetHandle h = gpu_handle(side, 0);
    int rc = (g_gpu_forward_active && !g_gpu_untrusted && h.is_valid) ?
        gpu_nn_download(h, &g_cpu_params, sizeof(g_cpu_params)) : -1;
    if (rc == 0) {
        memset(destination, 0, sizeof(*destination));
        unpack_params(&g_cpu_params, destination);
    }
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    return rc;
}

/* ============================================================================
   按方 GPU 参数镜像（调用方必须已持有 g_gpu_api_lock）

   四套网络同时驻留：g_gpu_h[side][0]=online / [side][1]=target。
   这些函数只按 side 直取该方自己的两个句柄，不读也不写另一方的句柄，
   因此不存在「切换时把设备载荷写回某一方影子」的路径。
   句柄不存在（按方行未驻留）时返回负值，调用方必须显式检查。
   ========================================================================= */
/* online / target 的 which 取值（0/1），可读性用。 */
static const int gpu_which_online = 0;
static const int gpu_which_target = 1;

static int download_one_locked(int side, int which, NeuralNet* dst, const char* label) {
    GpuNetHandle h = gpu_handle(side, which);
    if (!h.is_valid || !dst) return -2;
    int rc = -1;
    GPU_RETRY_LOOP((rc = gpu_nn_download(h, &g_cpu_params, sizeof(g_cpu_params))), 3, label);
    if (rc != 0) return -2;
    unpack_params(&g_cpu_params, dst);
    return 0;
}

static int upload_one_locked(int side, int which, const NeuralNet* src, const char* label) {
    GpuNetHandle h = gpu_handle(side, which);
    if (!h.is_valid || !src) return -2;
    pack_params(src, &g_cpu_params);
    int rc = -1;
    GPU_RETRY_LOOP((rc = gpu_nn_upload(h, &g_cpu_params, sizeof(g_cpu_params))), 3, label);
    return rc != 0 ? -2 : 0;
}

static int download_side_locked(DQNAgent* agent, Side side) {
    const int s = side_to_physical(side);
    int rc = download_one_locked(s, gpu_which_online, agent_online_net(agent, side), "download side online");
    if (rc != 0) return -2;
    rc = download_one_locked(s, gpu_which_target, agent_target_net(agent, side), "download side target");
    if (rc != 0) return -3;
    return 0;
}

static int upload_side_locked(DQNAgent* agent, Side side) {
    const int s = side_to_physical(side);
    int rc = upload_one_locked(s, gpu_which_online, agent_online_net(agent, side), "upload side online");
    if (rc != 0) return -2;
    rc = upload_one_locked(s, gpu_which_target, agent_target_net(agent, side), "upload side target");
    if (rc != 0) return -3;
    return 0;
}

/* 按方下载：把指定方的驻留句柄同步回该方的 CPU 影子。
   驻留模式下不再要求「会话绑定到同一方」——句柄就是按方直取的。 */
static int download_side(DQNAgent* agent, Side side) {
    if (!agent || !indep_active(agent)) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side_to_physical(side))) {
        result = download_side_locked(agent, side);
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail(1);
    return result;
}

static int upload_side(DQNAgent* agent, Side side) {
    if (!agent || !indep_active(agent)) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side_to_physical(side))) {
        result = upload_side_locked(agent, side);
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail(1);
    return result;
}

/* 公开入口：按方下载 / 上传 / 快照。参数非法、按方未接入或该方句柄未驻留时返回 -1，
   调用方必须显式检查返回值，绝不能拿 CPU 回退结果冒充 GPU 验证。 */
int gpu_net_download_side(DQNAgent* agent, int side) {
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    return download_side(agent, s);
}

int gpu_net_upload_side(DQNAgent* agent, int side) {
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    return upload_side(agent, s);
}

int gpu_net_snapshot_side(DQNAgent* agent, int side, NeuralNet* destination) {
    if (!destination || !agent || !indep_active(agent)) return -1;
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int rc = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side)) {
        rc = download_one_locked(side, gpu_which_online, destination, "snapshot side online");
    }
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    return rc;
}

// GPT6-Astra: one GPU lock covers both copies; no training can split this snapshot.
int gpu_net_snapshot_online_pair(DQNAgent* agent, NeuralNet* black, NeuralNet* white) {
    if (!agent || !black || !white || black == white || !indep_active(agent)) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int rc = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && g_gpu_per_side &&
        gpu_handles_ready(0) && gpu_handles_ready(1)) {
        rc = download_one_locked(0, 0, black, "snapshot black online");
        if (rc == 0) rc = download_one_locked(1, 0, white, "snapshot white online");
    }
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    return rc;
}

/* ---- gpu_net_bind_side：驻留模式下的兼容 shim ----
   四套网络同时驻留之后，「绑定」不再有意义：所有按方访问都按 (side, which) 直取
   句柄，没有任何「设备当前代表哪一方」的状态。这里只维护 `indep_bound` 这条兼容
   记录（存档里会 set_bound(Black)，旧调用点会读取它）并保证该方影子与句柄一致，
   不再做任何跨方搬运。返回 0 的语义从「切换设备」退化为「该方可用」。
   符号保留是为了不破坏调用点与存档；不要再把它当设备访问的门禁。 */
int gpu_net_bind_side(DQNAgent* agent, int side) {
    if (!agent || !indep_active(agent)) return -1;
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (!g_gpu_forward_active) {
        agent->indep->set_bound(s);
        result = 0;
    } else if (g_gpu_untrusted) {
        result = -1;
    } else if (gpu_handles_ready(side)) {
        // GPT6-Astra: logical selection only. Callers requiring CPU parameters
        // use explicit download/snapshot APIs; selecting a side moves no weights.
        agent->indep->set_bound(s);
        result = 0;
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail_detail(1, "gpu_bind_side", result);
    return result;
}

/* 纯只读：一次性回读指定方的 online + target 驻留参数。
   驻留模式下直接读该方自己的两个句柄，不再要求 bound == side。
   不写任何 CPU 影子、不改训练语义；守卫不满足时只返回非 0，
   刻意不置 g_gpu_untrusted —— 它只是取证读取，不是设备故障。 */
int gpu_net_snapshot_side_pair(DQNAgent* agent, int side, NeuralNet* online_out, NeuralNet* target_out) {
    if (!agent || !online_out || !target_out || !indep_active(agent)) return -1;
    Side s;
    try { s = side_from_physical(side); } catch (...) { return -1; }
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int rc = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side)) {
        rc = download_one_locked(side, gpu_which_online, online_out, "snapshot pair online");
        if (rc == 0) rc = download_one_locked(side, gpu_which_target, target_out, "snapshot pair target");
    }
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    return rc;
}

// 上传 agent 参数到 GPU（加载 checkpoint 后调用）
int gpu_net_upload_agent(const DQNAgent* agent) {
    if (!agent) return -1;
    if (indep_active(agent)) return upload_side(const_cast<DQNAgent*>(agent), indep_bound(agent));
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    if (!g_gpu_forward_active || g_gpu_untrusted) { ReleaseSRWLockExclusive(&g_gpu_api_lock); return -1; }
    int rc = upload_one_locked(-1, gpu_which_online, &agent->q_net, "upload q_net");
    if (rc != 0) { g_gpu_untrusted = 1; ReleaseSRWLockExclusive(&g_gpu_api_lock); training_fail(1); return -2; }
    rc = upload_one_locked(-1, gpu_which_target, &agent->target_net, "upload target_net");
    if (rc != 0) { g_gpu_untrusted = 1; ReleaseSRWLockExclusive(&g_gpu_api_lock); training_fail(1); return -3; }
    ReleaseSRWLockExclusive(&g_gpu_api_lock); return 0;
}

/* 上传两方（驻留模式下是四套）到各自的句柄。恢复/迁移后使用，保证设备载荷
   与两方影子一致，不再依赖「绑定方」这一状态。 */
int gpu_net_upload_sides_pair(DQNAgent* agent) {
    if (!agent) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    if (!g_gpu_forward_active || g_gpu_untrusted) { ReleaseSRWLockExclusive(&g_gpu_api_lock); return -1; }
    int result = 0;
    const int rows = g_gpu_per_side ? 2 : 1;
    for (int s = 0; s < rows && result == 0; ++s) {
        const Side side = (s == 0) ? Side::Black : Side::White;
        const NeuralNet* online = g_gpu_per_side ? agent_online_net(agent, side) : &agent->q_net;
        const NeuralNet* target = g_gpu_per_side ? agent_target_net(agent, side) : &agent->target_net;
        if (!online || !target) { result = -1; break; }
        result = upload_one_locked(s, gpu_which_online, online, "upload pair online");
        if (result == 0) result = upload_one_locked(s, gpu_which_target, target, "upload pair target");
    }
    if (result != 0) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (result != 0) training_fail(1);
    return result;
}

// GPU soft update: target = tau * online + (1-tau) * target
/* 按方同步：只动该方自己的两个句柄，另一方一个字节都不碰。 */
int gpu_net_soft_update_side(int side) {
    if (side < 0 || side > 1) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side)) {
        GPU_RETRY_LOOP((result = gpu_nn_soft_update(g_gpu_h[side][1], g_gpu_h[side][0], TAU)), 3,
            "soft_update_side");
    }
    if (result != 0 && g_gpu_forward_active) g_gpu_untrusted = 1;
    int fatal = g_gpu_untrusted;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (fatal) training_fail(1);
    return result;
}

int gpu_net_hard_update_side(int side) {
    if (side < 0 || side > 1) return -1;
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(side)) {
        if (gpu_nn_download(g_gpu_h[side][0], &g_cpu_params, sizeof(g_cpu_params)) == 0 &&
            gpu_nn_upload(g_gpu_h[side][1], &g_cpu_params, sizeof(g_cpu_params)) == 0) {
            result = 0;
        }
    }
    if (result != 0 && g_gpu_forward_active) g_gpu_untrusted = 1;
    int fatal = g_gpu_untrusted;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (fatal) training_fail(1);
    return result;
}

// 历史共享接口（非独立模式）：等价于 side=-1 的那一对句柄。
int gpu_net_soft_update(void) {
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(-1)) {
        GPU_RETRY_LOOP((result = gpu_nn_soft_update(g_gpu_h[0][1], g_gpu_h[0][0], TAU)), 3, "soft_update");
    }
    if (result != 0 && g_gpu_forward_active) g_gpu_untrusted = 1;
    int fatal = g_gpu_untrusted;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (fatal) training_fail(1);
    return result;
}

/* GPU hard update: target <- q。使用同一把 API 锁保证下载/上传之间
   不会夹入训练更新。 */
int gpu_net_hard_update(void) {
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted && gpu_handles_ready(-1) &&
        gpu_nn_download(g_gpu_h[0][0], &g_cpu_params, sizeof(g_cpu_params)) == 0 &&
        gpu_nn_upload(g_gpu_h[0][1], &g_cpu_params, sizeof(g_cpu_params)) == 0) {
        result = 0;
    }
    if (result != 0 && g_gpu_forward_active) g_gpu_untrusted = 1;
    int fatal = g_gpu_untrusted;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    if (fatal) training_fail(1);
    return result;
}

/* GPU 批量训练：side<0 表示历史共享句柄；否则用该方自己的 online 句柄。
   四句柄驻留下不存在「当前设备是谁」的问题，也就不需要绑定前置条件。 */
int gpu_net_train_batch_side(int side, const float* states, const int* actions, const float* targets,
    int batch_size, float lr, float* out_loss) {
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    GpuNetHandle h = gpu_handle(side, 0);
    int result = g_gpu_untrusted ? -100 : (!g_gpu_forward_active || !h.is_valid) ? -1 :
        gpu_nn_train_step_batch(h, states, actions, targets, batch_size, lr, out_loss);
    if (result == -100) g_gpu_untrusted = 1;
    ReleaseSRWLockExclusive(&g_gpu_api_lock);
    /* -100 means the final device commit is indeterminate.  Latch it at the
       GPU facade boundary so every caller gets the same exactly-once failure
       notification, not only callers that happen to use train_agent_batch(). */
    if (result == -100) training_fail_detail(1, "gpu_train_batch", result);
    return result;
}

/* 历史共享入口：非独立模式的等价调用。 */
int gpu_net_train_batch(const float* states, const int* actions, const float* targets,
    int batch_size, float lr, float* out_loss) {
    return gpu_net_train_batch_side(-1, states, actions, targets, batch_size, lr, out_loss);
}

/* GPU 批量前向：side<0 = 历史共享句柄；(side, use_target) 直取该方句柄。
   四句柄驻留下不再有「设备当前代表哪一方」的概念。 */
int gpu_net_forward_batch_side(int side, int use_target, const float* states, int batch_size,
    float* out_qvals) {
    AcquireSRWLockExclusive(&g_gpu_api_lock);
    int result = -1;
    if (g_gpu_forward_active && !g_gpu_untrusted) {
        GpuNetHandle h = gpu_handle(side, use_target);
        if (h.is_valid) {
            GPU_RETRY_LOOP((result = gpu_nn_forward_batch(h, states, batch_size, out_qvals)), 3, "forward_batch");
        }
    }
    ReleaseSRWLockExclusive(&g_gpu_api_lock); return result;
}

int gpu_net_forward_batch(int use_target, const float* states, int batch_size, float* out_qvals) {
    return gpu_net_forward_batch_side(-1, use_target, states, batch_size, out_qvals);
}

// Independent handle: parity checks use exactly the supplied frozen weights,
// never the trainer's mutable GPU handle or a global CPU/GPU mode switch.
int forward_gpu_snapshot(const NeuralNet* net, const float* states, int count, float* outputs) {
    if (!net || !states || !outputs || count < 1 || count > GPU_MAX_BATCH) return -1;
    GpuNetParams* params = (GpuNetParams*)malloc(sizeof(GpuNetParams));
    if (!params) return -1;
    pack_params(net, params);
    GpuNetHandle h = {0};
    int rc = gpu_nn_create(&h);
    if (rc == 0) rc = gpu_nn_upload(h, params, sizeof(*params));
    if (rc == 0) rc = gpu_nn_forward_batch(h, states, count, outputs);
    gpu_nn_destroy(&h);
    free(params);
    return rc;
}

void init_net(NeuralNet* net) {
    /* NeuralNet 里还包含仅用于调试/前向缓存的字段。Agent 由 malloc 创建时，
       这些字段过去会带着未初始化字节进入 checkpoint，甚至偶发 NaN 导致
       刚保存的 checkpoint 被完整性检查拒绝。初始化网络时先清零整个结构。 */
    memset(net, 0, sizeof(*net));
    float scale_embed = sqrtf(2.0f / (3 + TOKEN_EMBED_DIM));

    // Initialize token embedding matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
            net->token_embed[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale_embed;
        }
    }

    // Initialize position encoding
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
            net->pos_embed[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        }
    }

    // FC layer weights
    float scale1 = sqrtf(2.0f / (FEATURE_DIM + HIDDEN1_SIZE));
    float scale2 = sqrtf(2.0f / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    float scale3 = sqrtf(2.0f / (HIDDEN2_SIZE + HIDDEN3_SIZE));
    float scale4 = sqrtf(2.0f / (HIDDEN3_SIZE + CELL_COUNT));

    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        net->b1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < FEATURE_DIM; j++) {
            net->w1[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale1;
        }
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        net->b2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            net->w2[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale2;
        }
    }
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        net->b3[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            net->w3[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale3;
        }
    }
    for (int i = 0; i < CELL_COUNT; i++) {
        net->b4[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            net->w4[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * scale4;
        }
    }

    // Multi-head attention weights
    float attn_scale = sqrtf(2.0f / (TOKEN_EMBED_DIM + ATTENTION_EMBED));
    for (int i = 0; i < ATTENTION_EMBED; i++) {
        net->attn_bq[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        net->attn_bk[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        net->attn_bv[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
            net->attn_wq[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * attn_scale;
            net->attn_wk[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * attn_scale;
            net->attn_wv[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * attn_scale;
        }
    }
    float attn_out_scale = sqrtf(2.0f / (ATTENTION_EMBED + TOKEN_EMBED_DIM));
    for (int i = 0; i < TOKEN_EMBED_DIM; i++) {
        net->attn_bo[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < ATTENTION_EMBED; j++) {
            net->attn_wo[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * attn_out_scale;
        }
    }

    // Learnable pooling layer (output FEATURE_DIM = 128)
    float pool_scale = sqrtf(2.0f / (CELL_COUNT * TOKEN_EMBED_DIM + FEATURE_DIM));
    for (int i = 0; i < FEATURE_DIM; i++) {
        net->pool_b[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            net->pool_w[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 2 * pool_scale;
        }
    }
    memset(net->flat_cache, 0, sizeof(net->flat_cache));
}

void copy_net(NeuralNet* dst, const NeuralNet* src) {
    memcpy(dst, src, sizeof(NeuralNet));
}

void forward_cpu(const NeuralNet* net, const float input[], float output[]) {
    // GPU 优先：q_net / target_net（或按方 online/target）已驻留 GPU 时，推理走 GPU
    if (gpu_net_active() && g_current_agent) {
        /* 四句柄驻留：按 (side, which) 直取句柄。side 由「这张网属于谁」决定，
           不再依赖任何绑定状态。 */
        int side = -1, use_target = 0;
        DQNAgent* a = g_current_agent;
        if (net == &a->q_net) { side = -1; use_target = 0; }
        else if (net == &a->target_net) { side = -1; use_target = 1; }
        else if (indep_active(a)) {
            const int b = side_to_physical(Side::Black);
            const int w = side_to_physical(Side::White);
            if (net == agent_online_net(a, Side::Black)) { side = b; use_target = 0; }
            else if (net == agent_target_net(a, Side::Black)) { side = b; use_target = 1; }
            else if (net == agent_online_net(a, Side::White)) { side = w; use_target = 0; }
            else if (net == agent_target_net(a, Side::White)) { side = w; use_target = 1; }
        }
        if (side != -1 || net == &a->q_net || net == &a->target_net) {
            /* gpu_net_forward_batch already owns the three-attempt device
               retry boundary.  Wrapping it again caused up to nine attempts
               and long apparent freezes when the GPU was genuinely lost. */
            int rc = gpu_net_forward_batch_side(side, use_target, input, 1, output);
            if (rc == 0) {
                for (int i = 0; i < CELL_COUNT; ++i)
                    if (!isfinite(output[i])) { training_fail(2); break; }
                return;
            }
            // A CPU shadow is not a fallback model until both live networks
            // have been downloaded successfully. Never hide an unreadable GPU.
            if (gpu_net_download_agent(g_current_agent) != 0) {
                training_fail(1);
                for (int i = 0; i < CELL_COUNT; ++i) output[i] = NAN;
                return;
            }
            printf("[GPU] Single forward failed; using freshly synchronized CPU parameters.\n");
        }
        // opponent_net / pool 网络不在 GPU 上，落到 CPU
    }

    forward_cpu_only(net, input, output);
    for (int i = 0; i < CELL_COUNT; ++i)
        if (!isfinite(output[i])) { training_fail(2); break; }
}

void forward_cpu_only(const NeuralNet* net, const float input[], float output[]) {
    // 1. Build Token sequence in the side-to-move's perspective.
    float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++) {
        int state_idx = network_token_index(input, i);
        if (state_idx < 0) state_idx = 0;
        if (state_idx > 2) state_idx = 2;
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            tokens[i][d] = net->token_embed[state_idx][d] + net->pos_embed[i][d];
        }
    }

    // 2. Q, K, V projection
    float q[CELL_COUNT][ATTENTION_EMBED];
    float k[CELL_COUNT][ATTENTION_EMBED];
    float v[CELL_COUNT][ATTENTION_EMBED];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < ATTENTION_EMBED; d++) {
            q[i][d] = net->attn_bq[d];
            k[i][d] = net->attn_bk[d];
            v[i][d] = net->attn_bv[d];
            for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
                q[i][d] += net->attn_wq[d][j] * tokens[i][j];
                k[i][d] += net->attn_wk[d][j] * tokens[i][j];
                v[i][d] += net->attn_wv[d][j] * tokens[i][j];
            }
        }
    }

    // 3. Multi-head attention
    float head_out[ATTENTION_HEADS][CELL_COUNT][ATTENTION_HEAD_DIM];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        float qh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float kh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float vh[CELL_COUNT][ATTENTION_HEAD_DIM];
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                qh[i][d] = q[i][start + d];
                kh[i][d] = k[i][start + d];
                vh[i][d] = v[i][start + d];
            }
        }

        float scores[CELL_COUNT][CELL_COUNT];
        for (int i = 0; i < CELL_COUNT; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < CELL_COUNT; j++) {
                float dot = 0.0f;
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    dot += qh[i][d] * kh[j][d];
                }
                scores[i][j] = dot / sqrtf((float)ATTENTION_HEAD_DIM);
                if (scores[i][j] > 20.0f) scores[i][j] = 20.0f;
                if (scores[i][j] < -20.0f) scores[i][j] = -20.0f;
                if (scores[i][j] > max_val) max_val = scores[i][j];
            }
            float sum = 0.0f;
            for (int j = 0; j < CELL_COUNT; j++) {
                scores[i][j] = expf(scores[i][j] - max_val);
                sum += scores[i][j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < CELL_COUNT; j++) scores[i][j] /= sum;
            } else {
                for (int j = 0; j < CELL_COUNT; j++) scores[i][j] = 1.0f / CELL_COUNT;
            }
        }

        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                head_out[h][i][d] = 0.0f;
                for (int j = 0; j < CELL_COUNT; j++) {
                    head_out[h][i][d] += scores[i][j] * vh[j][d];
                }
            }
        }
    }

    // 4. Concatenate + output projection + residual
    float concat[CELL_COUNT][ATTENTION_EMBED];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                concat[i][start + d] = head_out[h][i][d];
            }
        }
    }

    float attn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            attn_out[i][d] = net->attn_bo[d];
            for (int j = 0; j < ATTENTION_EMBED; j++) {
                attn_out[i][d] += net->attn_wo[d][j] * concat[i][j];
            }
            attn_out[i][d] += tokens[i][d];
        }
    }

    // 5. Flatten Attention_O[25][16] → flat[400] (cell * TOKEN_EMBED_DIM + feature)
    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            flat[i * TOKEN_EMBED_DIM + d] = attn_out[i][d];

    // 6. Learnable pooling layer (output FEATURE_DIM), full 400-dim input
    float pooled_feat[FEATURE_DIM];
    for (int i = 0; i < FEATURE_DIM; i++) {
        pooled_feat[i] = net->pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            pooled_feat[i] += net->pool_w[i][j] * flat[j];
        }
        pooled_feat[i] = (pooled_feat[i] > 0) ? pooled_feat[i] : 0.0f;
    }

    // 7. FC layers (handwritten matrix multiplication)
    float hidden1[HIDDEN1_SIZE], hidden2[HIDDEN2_SIZE], hidden3[HIDDEN3_SIZE];

    // FC1: hidden1 = b1 + w1 * pooled_feat, then ReLU
    memcpy(hidden1, net->b1, sizeof(float) * HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < FEATURE_DIM; j++) {
            sum += net->w1[i][j] * pooled_feat[j];
        }
        hidden1[i] += sum;
    }
    for (int i = 0; i < HIDDEN1_SIZE; i++) { if (hidden1[i] < 0) hidden1[i] = 0; }

    // FC2: hidden2 = b2 + w2 * hidden1, then ReLU
    memcpy(hidden2, net->b2, sizeof(float) * HIDDEN2_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += net->w2[i][j] * hidden1[j];
        }
        hidden2[i] += sum;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) { if (hidden2[i] < 0) hidden2[i] = 0; }

    // FC3: hidden3 = b3 + w3 * hidden2, then ReLU
    memcpy(hidden3, net->b3, sizeof(float) * HIDDEN3_SIZE);
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            sum += net->w3[i][j] * hidden2[j];
        }
        hidden3[i] += sum;
    }
    for (int i = 0; i < HIDDEN3_SIZE; i++) { if (hidden3[i] < 0) hidden3[i] = 0; }

    // FC4: output = b4 + w4 * hidden3
    memcpy(output, net->b4, sizeof(float) * CELL_COUNT);
    for (int i = 0; i < CELL_COUNT; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            sum += net->w4[i][j] * hidden3[j];
        }
        output[i] += sum;
    }
}

// All callers share one update path. CPU fallback never redispatches to GPU.
static float train_cpu_inplace(NeuralNet*, const float[], int, float, float);
static bool safe_parameters(const NeuralNet* net) {
    if (!net) return false;
    /* NeuralNet also contains forward activations and caches.  Only the packed
       learnable image is persistent model state; rejecting a large but finite
       activation here used to cause false protective shutdowns. */
    std::unique_ptr<GpuNetParams> packed(new (std::nothrow) GpuNetParams());
    if (!packed) return false;
    pack_params(net, packed.get());
    const float* p = (const float*)packed.get();
    for (size_t i = 0; i < sizeof(*packed) / sizeof(float); ++i)
        if (!isfinite(p[i]) || fabsf(p[i]) > TRAIN_PARAM_ABS_LIMIT) return false;
    return true;
}

/* ============================================================================
   按方批训练：一个批次必须整体属于同一物理棋色。
   黑批只更新黑方 online，白批只更新白方 online；双方的 target、DELTA 门
   EMA/预热、提交/拒绝计数与学习率回退全部按方归属。
   未开启按方路由时 side 被忽略，走历史共享 q_net/target_net 路径。
   ========================================================================= */
int train_agent_batch_side(DQNAgent* agent, int side, const float* states,
    const int* actions, const float* targets, int count, float learning_rate, float* loss) {
    Side s;
    try { s = side_from_physical(side); } catch (...) { if (loss) *loss = 0.0f; return 0; }
    /* 按方路由未开启时忽略 side，保持历史调用方语义完全不变。 */
    if (!agent || !indep_active(agent)) {
        return train_agent_batch(agent, states, actions, targets, count, learning_rate, loss);
    }
    if (loss) *loss = 0.0f;
    if (g_training_fault) return 0;
    /* 安全暂停期间共享提交入口直接拒绝，本局剩余批次不再更新权重。 */
    if (sentinel_guard_paused()) return 2;
    if (!states || !actions || !targets || !loss ||
        count < 1 || count > GPU_MAX_BATCH || !isfinite(learning_rate) ||
        learning_rate <= 0.0f || learning_rate > 1.0f) { training_fail(2); return 0; }
    for (int i = 0; i < count; ++i) {
        if (actions[i] < 0 || actions[i] >= CELL_COUNT || !isfinite(targets[i])) { training_fail(2); return 0; }
        for (int j = 0; j <= CELL_COUNT; ++j)
            if (!isfinite(states[i * (CELL_COUNT + 1) + j])) { training_fail(2); return 0; }
    }
    /* 四套网络同时驻留：按 (side, which) 直取该方句柄，不存在「本批是否绑定了
       该方」这种状态，因此不再有 indep_side_not_bound 守卫（那条守卫是单方驻留
       + 切换时代的状态机产物，也是真实 Trainer 曾经崩溃的原因）。 */
    NeuralNet* const own = agent_online_net(agent, s);
    if (!own) { training_fail(2); return 0; }
    const int on_gpu = gpu_net_active();
    const float effective_lr = agent->indep->learning_rate_or(s, learning_rate);

    /* 哨兵基线：GPU 模式下先同步该方设备参数，得到权威的更新前状态。 */
    if (on_gpu && gpu_net_download_side(agent, side_to_physical(s)) != 0) {
        training_fail(1);
        return 0;
    }
    SentinelHealth before;
    sentinel_evaluate_net(own, &before);
    std::unique_ptr<NeuralNet> before_net(new (std::nothrow) NeuralNet(*own));
    if (!before_net) { training_fail(2); return 0; }

    if (on_gpu) {
        int rc = -1;
        float attempt_lr = effective_lr;
        int numeric_retries = 0;
        for (int attempt = 0; attempt < 4; ++attempt) {
            rc = gpu_net_train_batch_side(side_to_physical(s), states, actions, targets, count, attempt_lr, loss);
            if (rc == 0) break;
            if (rc == -102) {
                if (gpu_net_download_side(agent, side_to_physical(s)) != 0 ||
                    !safe_parameters(own)) {
                    training_fail_detail(2, "gpu_live_parameters", rc);
                    return 0;
                }
                ++g_gpu_loss_reject_count;
                note_loss_reject(agent, s);
                const unsigned long long n = agent->indep->updates(s).rejected_loss_batches;
                if (n <= 4 || (n & (n - 1)) == 0)
                    printf("[GPU] High-loss batch rejected before commit, side=%s total=%llu.\n",
                        side_name(s), n);
                *loss = 0.0f;
                return 2;
            }
            if (rc == -104) {
                numeric_retries++;
                attempt_lr *= 0.25f;
                printf("[GPU] Unsafe batch rejected (code %d); retrying with lr %.9g (%d/3)\n",
                    rc, attempt_lr, numeric_retries);
                continue;
            }
            if (rc == -100) break;
            Sleep(100 * (attempt + 1));
        }
        if (rc == 0) {
            if (!isfinite(*loss)) { training_fail(2); return 0; }
            if (gpu_net_download_side(agent, side_to_physical(s)) != 0) { training_fail(1); return 0; }
            const double delta_norm = batch_delta_norm(before_net.get(), own);
            const DeltaGateDecision gate = delta_gate_decide(agent, s, delta_norm);
            if (!gate.allowed) {
                *own = *before_net;
                if (gpu_net_upload_side(agent, side_to_physical(s)) != 0) { training_fail(1); return 0; }
                delta_gate_commit(agent, s, 0, delta_norm);
                printf("[GATE] Rejected GPU batch (side=%s): update norm %.6g surges vs EMA %.6g; "
                    "online restored, target untouched.\n", side_name(s), delta_norm, gate.baseline);
                *loss = 0.0f;
                return 2;
            }
            /* 哨兵拒绝必须连保护门基线一起回滚：delta_gate_commit 会把本次更新
               计入 EMA/预热并记一次接受，如果哨兵随后拒绝而只回滚权重，
               下一次接受/拒绝判断就建立在一次已被撤销的更新上。 */
            const independent_training::SideUpdateState gate_state_before =
                agent->indep->updates(s);
            delta_gate_commit(agent, s, 1, delta_norm);
            SentinelHealth after;
            sentinel_evaluate_net(own, &after);
            char reason[128];
            if (sentinel_reject_update(&before, &after, reason, sizeof(reason))) {
                *own = *before_net;
                if (gpu_net_upload_side(agent, side_to_physical(s)) != 0) { training_fail(1); return 0; }
                agent->indep->updates(s) = gate_state_before;
                ++agent->indep->updates(s).sentinel_consecutive_rejects;
                sentinel_rejection_escalate(reason);
                agent->indep->note_rejected(s);
                printf("[SENTINEL] Rejected GPU batch (side=%s, %s); online restored, target untouched.\n",
                    side_name(s), reason);
                *loss = 0.0f;
                return 2;
            }
            agent->indep->updates(s).sentinel_consecutive_rejects = 0;
            sentinel_rejection_reset();
            if (numeric_retries > 0 && attempt_lr < effective_lr) {
                agent->indep->set_learning_rate(s, attempt_lr);
                printf("[GPU] Learning rate safety backoff committed for side=%s: %.9g\n",
                    side_name(s), attempt_lr);
            }
            note_committed(agent, s);
            return 1;
        }
        if (rc == -100 || rc == -101 || rc == -103 || rc == -104 ||
            gpu_net_download_side(agent, side_to_physical(s)) != 0) {
            training_fail_detail((rc == -101 || rc == -103 || rc == -104) ? 2 : 1,
                "gpu_train_batch", rc);
            return 0;
        }
    }
    std::unique_ptr<NeuralNet> staged(new (std::nothrow) NeuralNet(*own));
    if (!staged) { training_fail(2); return 0; }
    double sum = 0;
    int accepted = 0;
    const float sample_lr = effective_lr / (float)count;
    for (int i = 0; i < count; ++i) {
        float value = train_cpu_inplace(staged.get(),
            states + i * (CELL_COUNT + 1), actions[i], targets[i], sample_lr);
        if (!isfinite(value) || value < 0.0f) { training_fail(2); return 0; }
        if (value > TRAIN_LOSS_REJECT_LIMIT) continue;
        sum += value;
        ++accepted;
    }
    if (!safe_parameters(staged.get())) { training_fail(2); return 0; }
    /* 同 GPU 路径：哨兵拒绝时保护门基线也必须回滚，所以快照要在提交之前取，
       并且必须活到下面的哨兵检查块（因此声明在这一层，不放内层作用域）。 */
    independent_training::SideUpdateState gate_state_before = agent->indep->updates(s);
    {
        const double delta_norm = batch_delta_norm(own, staged.get());
        const DeltaGateDecision gate = delta_gate_decide(agent, s, delta_norm);
        if (!gate.allowed) {
            delta_gate_commit(agent, s, 0, delta_norm);
            printf("[GATE] Rejected CPU batch (side=%s): update norm %.6g surges vs EMA %.6g; "
                "online unchanged, target untouched.\n", side_name(s), delta_norm, gate.baseline);
            *loss = 0.0f;
            return 2;
        }
        delta_gate_commit(agent, s, 1, delta_norm);
    }
    {
        SentinelHealth after;
        sentinel_evaluate_net(staged.get(), &after);
        char reason[128];
        if (sentinel_reject_update(&before, &after, reason, sizeof(reason))) {
            agent->indep->updates(s) = gate_state_before;
            ++agent->indep->updates(s).sentinel_consecutive_rejects;
            sentinel_rejection_escalate(reason);
            agent->indep->note_rejected(s);
            printf("[SENTINEL] Rejected CPU batch (side=%s, %s); online unchanged, target untouched.\n",
                side_name(s), reason);
            *loss = 0.0f;
            return 2;
        }
        agent->indep->updates(s).sentinel_consecutive_rejects = 0;
        sentinel_rejection_reset();
    }
    *own = *staged;
    if (on_gpu && gpu_net_upload_side(agent, side_to_physical(s)) != 0) {
        training_fail(1);
        return 0;
    }
    *loss = accepted ? (float)(sum / accepted) : 0.0f;
    note_committed(agent, s);
    return 1;
}

int train_agent_batch(DQNAgent* agent, const float* states, const int* actions,
    const float* targets, int count, float learning_rate, float* loss) {
    if (g_training_fault) return 0;
    /* Bug-7 修复：安全暂停期间，共享提交入口直接拒绝，本局剩余批次不再更新权重。 */
    if (sentinel_guard_paused()) { *loss = 0.0f; return 2; }
    if (!agent || !states || !actions || !targets || !loss ||
        count < 1 || count > GPU_MAX_BATCH || !isfinite(learning_rate) ||
        learning_rate <= 0.0f || learning_rate > 1.0f) { training_fail(2); return 0; }
    for (int i = 0; i < count; ++i) {
        if (actions[i] < 0 || actions[i] >= CELL_COUNT || !isfinite(targets[i])) { training_fail(2); return 0; }
        for (int j = 0; j <= CELL_COUNT; ++j)
            if (!isfinite(states[i * (CELL_COUNT + 1) + j])) { training_fail(2); return 0; }
    }
    const int on_gpu = gpu_net_active();

    /* ===== 哨兵基线：记录更新前的裸网络健康，并保存可回滚的 q_net 副本。
       GPU 模式下 CPU 影子可能落后于设备，先同步一次以获得权威基线。 ===== */
    if (on_gpu && gpu_net_download_agent(agent) != 0) { training_fail(1); return 0; }
    SentinelHealth before;
    sentinel_evaluate_net(&agent->q_net, &before);
    std::unique_ptr<NeuralNet> before_net(new (std::nothrow) NeuralNet(agent->q_net));
    if (!before_net) { training_fail(2); return 0; }

    if (on_gpu) {
        int rc = -1;
        float attempt_lr = learning_rate;
        int numeric_retries = 0;
        for (int attempt = 0; attempt < 4; ++attempt) {
            /* 历史共享路径：side<0 走共享句柄。 */
            rc = gpu_net_train_batch_side(-1, states, actions, targets, count, attempt_lr, loss);
            if (rc == 0) break;
            if (rc == -102) {
                /* The GPU trained a private parameter image and discarded it
                   before commit.  Loss is determined by the pre-update Q, so
                   retrying with a lower LR cannot make this sample valid. */
                if (gpu_net_download_agent(agent) != 0 || !safe_parameters(&agent->q_net)) {
                    training_fail_detail(2, "gpu_live_parameters", rc);
                    return 0;
                }
                ++g_gpu_loss_reject_count;
                const unsigned long long n = g_gpu_loss_reject_count;
                if (n <= 4 || (n & (n - 1)) == 0)
                    printf("[GPU] High-loss batch rejected before commit, total=%llu.\n", n);
                *loss = 0.0f;
                return 2;   /* 安全拒绝：无更新，非致命 */
            }
            if (rc == -104) {
                /* update_params is a private copy and was not committed, so a
                   lower-LR retry can safely recover a finite parameter-limit
                   violation.  Invalid loss/NaN codes are not LR-dependent. */
                numeric_retries++;
                attempt_lr *= 0.25f;
                printf("[GPU] Unsafe batch rejected (code %d); retrying with lr %.9g (%d/3)\n",
                    rc, attempt_lr, numeric_retries);
                continue;
            }
            if (rc == -100) break; /* final commit is indeterminate */
            Sleep(100 * (attempt + 1));
        }
        if (rc == 0) {
            if (!isfinite(*loss)) { training_fail(2); return 0; }
            /* 下载新参数，在提交给 live params 之后立即做哨兵校验。
               校验失败则回滚 q_net 到更新前，target_net 保持同代不变。 */
            if (gpu_net_download_agent(agent) != 0) { training_fail(1); return 0; }
            /* Bug-1 修复：GPU 提交后同样执行整批参数变化量安全门（与 CPU 一致）。 */
            {
                const double delta_norm = batch_delta_norm(before_net.get(), &agent->q_net);
                if (g_batch_delta_warmup > 0) {
                    --g_batch_delta_warmup;
                    g_batch_delta_ema = delta_norm;
                } else if (!isfinite(delta_norm) ||
                           delta_norm > TRAIN_BATCH_DELTA_SURGE_RATIO * g_batch_delta_ema + 1e-6) {
                    agent->q_net = *before_net;
                    if (gpu_net_upload_agent(agent) != 0) { training_fail(1); return 0; }
                    g_batch_delta_reject_count++;
                    printf("[GATE] Rejected GPU batch: update norm %.6g surges vs EMA %.6g; q_net restored.\n",
                        delta_norm, g_batch_delta_ema);
                    *loss = 0.0f;
                    return 2;
                } else {
                    g_batch_delta_ema += TRAIN_BATCH_DELTA_EMA_ALPHA * (delta_norm - g_batch_delta_ema);
                }
            }
            SentinelHealth after;
            sentinel_evaluate_net(&agent->q_net, &after);
            char reason[128];
            if (sentinel_reject_update(&before, &after, reason, sizeof(reason))) {
                agent->q_net = *before_net;
                if (gpu_net_upload_agent(agent) != 0) { training_fail(1); return 0; }
                sentinel_rejection_escalate(reason);
                printf("[SENTINEL] Rejected GPU batch (%s); q_net/target restored.\n", reason);
                *loss = 0.0f;
                return 2;
            }
            sentinel_rejection_reset();
            if (numeric_retries > 0 && attempt_lr < g_learning_rate) {
                g_learning_rate = attempt_lr;
                printf("[GPU] Learning rate safety backoff committed: %.9g\n", g_learning_rate);
            }
            return 1;   /* 已提交 */
        }
        // A failed device commit cannot be safely retried or saved.
        if (rc == -100 || rc == -101 || rc == -103 || rc == -104 || gpu_net_download_agent(agent) != 0) {
            training_fail_detail((rc == -101 || rc == -103 || rc == -104) ? 2 : 1,
                "gpu_train_batch", rc);
            return 0;
        }
    }
    std::unique_ptr<NeuralNet> staged(new (std::nothrow) NeuralNet(agent->q_net));
    if (!staged) { training_fail(2); return 0; }
    double sum = 0;
    int accepted = 0;
    const float sample_lr = learning_rate / (float)count;
    for (int i = 0; i < count; ++i) {
        float value = train_cpu_inplace(staged.get(),
            states + i * (CELL_COUNT + 1), actions[i], targets[i], sample_lr);
        if (!isfinite(value) || value < 0.0f) {
            training_fail(2);
            return 0;
        }
        /* train_cpu_inplace returns the measured loss without updating staged
           parameters when this threshold is exceeded. */
        if (value > TRAIN_LOSS_REJECT_LIMIT) continue;
        sum += value;
        ++accepted;
    }
    if (!safe_parameters(staged.get())) { training_fail(2); return 0; }
    /* 安全门：整批参数更新范数限制（等价于全局范数裁剪的提交侧）。
       先用「旧 EMA」判断突增，再更新 EMA；拒绝的批次不污染健康 EMA。 */
    {
        const double delta_norm = batch_delta_norm(&agent->q_net, staged.get());
        if (g_batch_delta_warmup > 0) {
            --g_batch_delta_warmup;
            g_batch_delta_ema = delta_norm;   /* 用最后一个 warmup 批初始化健康基线 */
        } else if (!isfinite(delta_norm) ||
                   delta_norm > TRAIN_BATCH_DELTA_SURGE_RATIO * g_batch_delta_ema + 1e-6) {
            g_batch_delta_reject_count++;
            printf("[GATE] Rejected CPU batch: update norm %.6g surges vs EMA %.6g; q_net unchanged.\n",
                delta_norm, g_batch_delta_ema);
            *loss = 0.0f;
            return 2;
        } else {
            g_batch_delta_ema += TRAIN_BATCH_DELTA_EMA_ALPHA * (delta_norm - g_batch_delta_ema);
        }
    }
    /* 哨兵校验：staged 是私有副本，未提交前可安全拒绝，q_net 保持更新前状态。 */
    {
        SentinelHealth after;
        sentinel_evaluate_net(staged.get(), &after);
        char reason[128];
        if (sentinel_reject_update(&before, &after, reason, sizeof(reason))) {
            sentinel_rejection_escalate(reason);
            printf("[SENTINEL] Rejected CPU batch (%s); q_net unchanged.\n", reason);
            *loss = 0.0f;
            return 2;
        }
        sentinel_rejection_reset();
    }
    agent->q_net = *staged;
    if (on_gpu && gpu_net_upload_agent(agent) != 0) {
        training_fail(1);
        return 0;
    }
    *loss = accepted ? (float)(sum / accepted) : 0.0f;
    return 1;
}

float train_step(NeuralNet* net, const float state[], int action, float target) {
    if (g_current_agent && net == &g_current_agent->q_net) {
        float loss = 0.0f;
        return train_agent_batch(g_current_agent, state, &action, &target, 1,
            g_learning_rate, &loss) ? loss : NAN;
    }
    return train_step_cpu_only(net, state, action, target, g_learning_rate);
}

float train_step_cpu_only(NeuralNet* net, const float state[], int action,
    float target, float learning_rate) {
    if (!net) return NAN;
    std::unique_ptr<NeuralNet> staged(new (std::nothrow) NeuralNet(*net));
    if (!staged) return NAN;
    float loss = train_cpu_inplace(staged.get(), state, action, target, learning_rate);
    if (!isfinite(loss) || loss < 0.0f ||
        !safe_parameters(staged.get())) return NAN;
    *net = *staged;
    return loss;
}

static float train_cpu_inplace(NeuralNet* net, const float state[], int action,
    float target, float learning_rate) {
    if (!net || !state || action < 0 || action >= CELL_COUNT ||
        !isfinite(target) || !isfinite(learning_rate) ||
        learning_rate <= 0.0f || learning_rate > 1.0f) return NAN;
    for (int i = 0; i <= CELL_COUNT; ++i) if (!isfinite(state[i])) return NAN;

    // 1. Build Token sequence + cache
    float tokens[CELL_COUNT][TOKEN_EMBED_DIM];
    int state_idx[CELL_COUNT];
    for (int i = 0; i < CELL_COUNT; i++) {
        state_idx[i] = network_token_index(state, i);
        if (state_idx[i] < 0) state_idx[i] = 0;
        if (state_idx[i] > 2) state_idx[i] = 2;
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            tokens[i][d] = net->token_embed[state_idx[i]][d] + net->pos_embed[i][d];
        }
    }
    memcpy(net->tokens, tokens, sizeof(tokens));

    // 2. Q, K, V projection
    float q[CELL_COUNT][ATTENTION_EMBED];
    float k[CELL_COUNT][ATTENTION_EMBED];
    float v[CELL_COUNT][ATTENTION_EMBED];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < ATTENTION_EMBED; d++) {
            q[i][d] = net->attn_bq[d];
            k[i][d] = net->attn_bk[d];
            v[i][d] = net->attn_bv[d];
            for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
                q[i][d] += net->attn_wq[d][j] * tokens[i][j];
                k[i][d] += net->attn_wk[d][j] * tokens[i][j];
                v[i][d] += net->attn_wv[d][j] * tokens[i][j];
            }
        }
    }

    // 3. Multi-head attention + cache scores
    float head_out[ATTENTION_HEADS][CELL_COUNT][ATTENTION_HEAD_DIM];
    float scores_cache[ATTENTION_HEADS][CELL_COUNT][CELL_COUNT];
    bool score_in_range[ATTENTION_HEADS][CELL_COUNT][CELL_COUNT];

    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        float qh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float kh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float vh[CELL_COUNT][ATTENTION_HEAD_DIM];
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                qh[i][d] = q[i][start + d];
                kh[i][d] = k[i][start + d];
                vh[i][d] = v[i][start + d];
            }
        }

        float raw_scores[CELL_COUNT][CELL_COUNT];
        for (int i = 0; i < CELL_COUNT; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < CELL_COUNT; j++) {
                raw_scores[i][j] = 0.0f;
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    raw_scores[i][j] += qh[i][d] * kh[j][d];
                }
                raw_scores[i][j] /= sqrtf((float)ATTENTION_HEAD_DIM);
                score_in_range[h][i][j] = raw_scores[i][j] > -20.0f && raw_scores[i][j] < 20.0f;
                raw_scores[i][j] = fmaxf(-20.0f, fminf(20.0f, raw_scores[i][j]));
                if (raw_scores[i][j] > max_val) max_val = raw_scores[i][j];
            }
            float sum = 0.0f;
            for (int j = 0; j < CELL_COUNT; j++) {
                raw_scores[i][j] = expf(raw_scores[i][j] - max_val);
                sum += raw_scores[i][j];
            }
            for (int j = 0; j < CELL_COUNT; j++) {
                scores_cache[h][i][j] = raw_scores[i][j] / sum;
            }
        }

        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                head_out[h][i][d] = 0.0f;
                for (int j = 0; j < CELL_COUNT; j++) {
                    head_out[h][i][d] += scores_cache[h][i][j] * vh[j][d];
                }
            }
        }
    }

    // 4. Concatenate + output projection + residual
    float concat[CELL_COUNT][ATTENTION_EMBED];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                concat[i][start + d] = head_out[h][i][d];
            }
        }
    }

    float attn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            attn_out[i][d] = net->attn_bo[d];
            for (int j = 0; j < ATTENTION_EMBED; j++) {
                attn_out[i][d] += net->attn_wo[d][j] * concat[i][j];
            }
            attn_out[i][d] += tokens[i][d];
        }
    }
    memcpy(net->attn_context, attn_out, sizeof(attn_out));

    // Flatten and cache
    float flat[CELL_COUNT * TOKEN_EMBED_DIM];
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            flat[i * TOKEN_EMBED_DIM + d] = attn_out[i][d];
        }
    }
    memcpy(net->flat_cache, flat, sizeof(flat));

    // 6. Learnable pooling layer forward (full 400-dim input)
    float pooled_feat[FEATURE_DIM];
    for (int i = 0; i < FEATURE_DIM; i++) {
        pooled_feat[i] = net->pool_b[i];
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            pooled_feat[i] += net->pool_w[i][j] * flat[j];
        }
        pooled_feat[i] = (pooled_feat[i] > 0) ? pooled_feat[i] : 0.0f;
    }

    // 7. FC forward (handwritten matrix multiplication)
    float hidden1[HIDDEN1_SIZE], hidden2[HIDDEN2_SIZE], hidden3[HIDDEN3_SIZE];

    // FC1: hidden1 = b1 + w1 * pooled_feat, then ReLU
    memcpy(hidden1, net->b1, sizeof(float) * HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < FEATURE_DIM; j++) {
            sum += net->w1[i][j] * pooled_feat[j];
        }
        hidden1[i] += sum;
    }
    for (int i = 0; i < HIDDEN1_SIZE; i++) { if (hidden1[i] < 0) hidden1[i] = 0; }

    // FC2: hidden2 = b2 + w2 * hidden1, then ReLU
    memcpy(hidden2, net->b2, sizeof(float) * HIDDEN2_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += net->w2[i][j] * hidden1[j];
        }
        hidden2[i] += sum;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) { if (hidden2[i] < 0) hidden2[i] = 0; }

    // FC3: hidden3 = b3 + w3 * hidden2, then ReLU
    memcpy(hidden3, net->b3, sizeof(float) * HIDDEN3_SIZE);
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            sum += net->w3[i][j] * hidden2[j];
        }
        hidden3[i] += sum;
    }
    for (int i = 0; i < HIDDEN3_SIZE; i++) { if (hidden3[i] < 0) hidden3[i] = 0; }

    // FC4: pred = b4 + w4 * hidden3
    float pred[CELL_COUNT];
    memcpy(pred, net->b4, sizeof(float) * CELL_COUNT);
    for (int i = 0; i < CELL_COUNT; i++) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            sum += net->w4[i][j] * hidden3[j];
        }
        pred[i] += sum;
    }

    // 8. Loss calculation (Huber)
    float error = pred[action] - target;
    float abs_error = fabsf(error);
    float delta = 0.5f;
    float loss;
    float delta_out[CELL_COUNT] = { 0 };
    if (!isfinite(error)) return NAN; // Reject invalid forward results before any update.
    if (abs_error <= delta) {
        loss = 0.5f * error * error;
        delta_out[action] = error;
    } else {
        loss = delta * (abs_error - 0.5f * delta);
        delta_out[action] = (error > 0) ? delta : -delta;
    }
    /* Return the diagnostic value, but leave this private model image exactly
       unchanged.  Callers can skip it without treating a finite outlier as a
       device failure or publishing any partial update. */
    if (loss > TRAIN_LOSS_REJECT_LIMIT) return loss;

    // 9. Backpropagation (handwritten matrix multiplication)
    // FC layer backward — delta computation (matrix transpose multiply)
    float delta_hidden3[HIDDEN3_SIZE];
    memset(delta_hidden3, 0, sizeof(float) * HIDDEN3_SIZE);
    for (int j = 0; j < HIDDEN3_SIZE; j++) {
        float sum = 0.0f;
        for (int i = 0; i < CELL_COUNT; i++) {
            sum += net->w4[i][j] * delta_out[i];
        }
        delta_hidden3[j] = sum;
    }
    for (int j = 0; j < HIDDEN3_SIZE; j++) { if (hidden3[j] <= 0) delta_hidden3[j] = 0; }

    float delta_hidden2[HIDDEN2_SIZE];
    memset(delta_hidden2, 0, sizeof(float) * HIDDEN2_SIZE);
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN3_SIZE; i++) {
            sum += net->w3[i][j] * delta_hidden3[i];
        }
        delta_hidden2[j] = sum;
    }
    for (int j = 0; j < HIDDEN2_SIZE; j++) { if (hidden2[j] <= 0) delta_hidden2[j] = 0; }

    float delta_hidden1[HIDDEN1_SIZE];
    memset(delta_hidden1, 0, sizeof(float) * HIDDEN1_SIZE);
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            sum += net->w2[i][j] * delta_hidden2[i];
        }
        delta_hidden1[j] = sum;
    }
    for (int k = 0; k < HIDDEN1_SIZE; k++) { if (hidden1[k] <= 0) delta_hidden1[k] = 0; }

    // 在更新 w1 以前计算传给池化层的梯度。反向传播必须使用产生本次
    // 前向结果的那组参数，不能混用刚刚更新后的参数。
    float dL_dpooled_feat[FEATURE_DIM];
    memset(dL_dpooled_feat, 0, sizeof(float) * FEATURE_DIM);
    for (int j = 0; j < FEATURE_DIM; j++) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            sum += net->w1[i][j] * delta_hidden1[i];
        }
        dL_dpooled_feat[j] = (pooled_feat[j] > 0.0f) ? sum : 0.0f;
    }

    // Update FC weights — outer product + bias
    float neg_lr = -learning_rate;

    // w4/b4
    for (int i = 0; i < CELL_COUNT; i++) {
        net->b4[i] += neg_lr * clipped_gradient(delta_out[i]);
        for (int j = 0; j < HIDDEN3_SIZE; j++) {
            net->w4[i][j] += neg_lr * clipped_gradient(delta_out[i] * hidden3[j]);
        }
    }

    // w3/b3
    for (int i = 0; i < HIDDEN3_SIZE; i++) {
        net->b3[i] += neg_lr * clipped_gradient(delta_hidden3[i]);
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            net->w3[i][j] += neg_lr * clipped_gradient(delta_hidden3[i] * hidden2[j]);
        }
    }

    // w2/b2
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        net->b2[j] += neg_lr * clipped_gradient(delta_hidden2[j]);
        for (int k = 0; k < HIDDEN1_SIZE; k++) {
            net->w2[j][k] += neg_lr * clipped_gradient(delta_hidden2[j] * hidden1[k]);
        }
    }

    // w1/b1
    for (int k = 0; k < HIDDEN1_SIZE; k++) {
        net->b1[k] += neg_lr * clipped_gradient(delta_hidden1[k]);
        for (int j = 0; j < FEATURE_DIM; j++) {
            net->w1[k][j] += neg_lr * clipped_gradient(delta_hidden1[k] * pooled_feat[j]);
        }
    }

    // 10/11. Pooling backward（展平层：梯度直接映射回 attn_out，无除法）
    float dL_dflat[CELL_COUNT * TOKEN_EMBED_DIM] = { 0 };
    for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
        for (int i = 0; i < FEATURE_DIM; i++) {
            dL_dflat[j] += dL_dpooled_feat[i] * net->pool_w[i][j];
        }
    }
    float dL_dattn_out[CELL_COUNT][TOKEN_EMBED_DIM];
    for (int c = 0; c < CELL_COUNT; c++)
        for (int d = 0; d < TOKEN_EMBED_DIM; d++)
            dL_dattn_out[c][d] = dL_dflat[c * TOKEN_EMBED_DIM + d];

    // Update pooling weights（全 400 列）
    for (int i = 0; i < FEATURE_DIM; i++) {
        float grad_b = dL_dpooled_feat[i];
        net->pool_b[i] -= learning_rate * clipped_gradient(grad_b);
        for (int j = 0; j < CELL_COUNT * TOKEN_EMBED_DIM; j++) {
            net->pool_w[i][j] -= learning_rate * clipped_gradient(grad_b * flat[j]);
        }
    }

    // 12. Attention output projection + residual backward
    float dL_dconcat[CELL_COUNT][ATTENTION_EMBED] = { 0 };
    float dL_dtokens[CELL_COUNT][TOKEN_EMBED_DIM] = { 0 };

    for (int i = 0; i < CELL_COUNT; i++) {
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            dL_dtokens[i][d] += dL_dattn_out[i][d];
            for (int j = 0; j < ATTENTION_EMBED; j++) {
                dL_dconcat[i][j] += dL_dattn_out[i][d] * net->attn_wo[d][j];
            }
        }
    }

    // Update attn_wo and attn_bo
    for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
        float grad_bias = 0.0f;
        for (int i = 0; i < CELL_COUNT; i++) {
            grad_bias += dL_dattn_out[i][d];
        }
        net->attn_bo[d] -= learning_rate * clipped_gradient(grad_bias);

        for (int j = 0; j < ATTENTION_EMBED; j++) {
            float grad = 0.0f;
            for (int i = 0; i < CELL_COUNT; i++) {
                grad += dL_dattn_out[i][d] * concat[i][j];
            }
            net->attn_wo[d][j] -= learning_rate * clipped_gradient(grad);
        }
    }

    // 13. Multi-head attention backward
    float dL_dhead[ATTENTION_HEADS][CELL_COUNT][ATTENTION_HEAD_DIM];
    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                dL_dhead[h][i][d] = dL_dconcat[i][start + d];
            }
        }
    }

    // Q/K/V 属于每个棋盘位置，不能先把 25 个位置压成一个向量。
    float dL_dq_global[CELL_COUNT][ATTENTION_EMBED] = {};
    float dL_dk_global[CELL_COUNT][ATTENTION_EMBED] = {};
    float dL_dv_global[CELL_COUNT][ATTENTION_EMBED] = {};

    for (int h = 0; h < ATTENTION_HEADS; h++) {
        int start = h * ATTENTION_HEAD_DIM;

        float qh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float kh[CELL_COUNT][ATTENTION_HEAD_DIM];
        float vh[CELL_COUNT][ATTENTION_HEAD_DIM];
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                qh[i][d] = q[i][start + d];
                kh[i][d] = k[i][start + d];
                vh[i][d] = v[i][start + d];
            }
        }

        float dL_dvh[CELL_COUNT][ATTENTION_HEAD_DIM] = { 0 };
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int j = 0; j < CELL_COUNT; j++) {
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    dL_dvh[j][d] += dL_dhead[h][i][d] * scores_cache[h][i][j];
                }
            }
        }

        float dL_dscores[CELL_COUNT][CELL_COUNT] = { 0 };
        for (int i = 0; i < CELL_COUNT; i++) {
            for (int j = 0; j < CELL_COUNT; j++) {
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    dL_dscores[i][j] += dL_dhead[h][i][d] * vh[j][d];
                }
            }
        }

        float grad_softmax[CELL_COUNT][CELL_COUNT] = { 0 };
        for (int i = 0; i < CELL_COUNT; i++) {
            float sum_k = 0.0f;
            for (int k2 = 0; k2 < CELL_COUNT; k2++) {
                sum_k += scores_cache[h][i][k2] * dL_dscores[i][k2];
            }
            for (int j = 0; j < CELL_COUNT; j++) {
                grad_softmax[i][j] = scores_cache[h][i][j] * (dL_dscores[i][j] - sum_k);
                if (!score_in_range[h][i][j]) grad_softmax[i][j] = 0.0f;
            }
        }

        float dL_dqh[CELL_COUNT][ATTENTION_HEAD_DIM] = { 0 };
        float dL_dkh[CELL_COUNT][ATTENTION_HEAD_DIM] = { 0 };
        float inv_sqrt = 1.0f / sqrtf((float)ATTENTION_HEAD_DIM);

        for (int i = 0; i < CELL_COUNT; i++) {
            for (int j = 0; j < CELL_COUNT; j++) {
                for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                    dL_dqh[i][d] += grad_softmax[i][j] * kh[j][d] * inv_sqrt;
                    dL_dkh[j][d] += grad_softmax[i][j] * qh[i][d] * inv_sqrt;
                }
            }
        }

        for (int i = 0; i < CELL_COUNT; i++) {
            for (int d = 0; d < ATTENTION_HEAD_DIM; d++) {
                int idx = start + d;
                dL_dq_global[i][idx] += dL_dqh[i][d];
                dL_dk_global[i][idx] += dL_dkh[i][d];
                dL_dv_global[i][idx] += dL_dvh[i][d];
            }
        }
    }

    // Q/K/V 投影同样是 token 路径的一部分，先把梯度传回 token。
    for (int i = 0; i < CELL_COUNT; i++) {
        for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
            float grad = 0.0f;
            for (int d = 0; d < ATTENTION_EMBED; d++) {
                grad += net->attn_wq[d][j] * dL_dq_global[i][d]
                      + net->attn_wk[d][j] * dL_dk_global[i][d]
                      + net->attn_wv[d][j] * dL_dv_global[i][d];
            }
            dL_dtokens[i][j] += grad;
        }
    }

    // 14. Update attention projection weights
    for (int d = 0; d < ATTENTION_EMBED; d++) {
        float grad_bq = 0.0f, grad_bk = 0.0f, grad_bv = 0.0f;
        for (int i = 0; i < CELL_COUNT; i++) {
            grad_bq += dL_dq_global[i][d];
            grad_bk += dL_dk_global[i][d];
            grad_bv += dL_dv_global[i][d];
        }
        net->attn_bq[d] -= learning_rate * clipped_gradient(grad_bq);
        net->attn_bk[d] -= learning_rate * clipped_gradient(grad_bk);
        net->attn_bv[d] -= learning_rate * clipped_gradient(grad_bv);
        for (int j = 0; j < TOKEN_EMBED_DIM; j++) {
            float grad_q = 0.0f, grad_k = 0.0f, grad_v = 0.0f;
            for (int i = 0; i < CELL_COUNT; i++) {
                grad_q += dL_dq_global[i][d] * tokens[i][j];
                grad_k += dL_dk_global[i][d] * tokens[i][j];
                grad_v += dL_dv_global[i][d] * tokens[i][j];
            }
            net->attn_wq[d][j] -= learning_rate * clipped_gradient(grad_q);
            net->attn_wk[d][j] -= learning_rate * clipped_gradient(grad_k);
            net->attn_wv[d][j] -= learning_rate * clipped_gradient(grad_v);
        }
    }

    // 15. Update token embedding and position encoding
    for (int i = 0; i < CELL_COUNT; i++) {
        int idx = state_idx[i];
        for (int d = 0; d < TOKEN_EMBED_DIM; d++) {
            const float grad = clipped_gradient(dL_dtokens[i][d]);
            net->token_embed[idx][d] -= learning_rate * grad;
            net->pos_embed[i][d] -= learning_rate * grad;
        }
    }

    return loss;
}
