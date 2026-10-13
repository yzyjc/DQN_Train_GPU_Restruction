/* neural_probe_protocol.h — Neural Probe 共享内存协议
   Trainer 和 Viewer 通过 Windows Shared Memory 交换神经网络快照数据。
   所有 offset 均相对于 shared memory base，不使用裸指针。 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 常量 ========== */
#define NEURAL_PROBE_MAGIC      0x4E505242   /* "NPRB" */
#define NEURAL_PROBE_VERSION    3
#define NEURAL_PROBE_NAME       "DQN_Neural_Probe"
#define MAX_PROBE_LAYERS        32
#define PROBE_PUBLISH_HZ        1            /* 有查看器时：每局结束最多发布一次 */
#define PROBE_PRETRAIN_PUBLISH_MS 1000        /* 预训练局数很快，按时间节流 */
#define PROBE_FORMAL_PUBLISH_STEPS 10         /* 正式训练每 10 个环境步 */
#define PROBE_CLIENT_POLL_MS    250          /* TUI 检查新快照的间隔 */
#define PROBE_VIEWER_TIMEOUT_MS 3000         /* 超时后停止 GPU 快照，避免无人查看时拖慢训练 */
#define PROBE_RESERVED_PUBLISHER_PID 0        /* header._reserved[] 索引 */
#define PROBE_RESERVED_HEARTBEAT_MS 1
#define PROBE_RESERVED_VIEWER_HEARTBEAT_MS 2
#define PROBE_ATTENTION_HEADS 4
#define PROBE_ATTENTION_TOKENS 25
#define PROBE_INPUT_SIZE 26
#define PROBE_DIAG_FORWARD_VALID  0x1u
#define PROBE_DIAG_GRADIENT_VALID 0x2u

/* ProbeLayerDesc._reserved[] stores the matrix-input activation vector used
   by the forward-contribution view. */
#define PROBE_LAYER_INPUT_ACTIVATION_OFFSET 0
#define PROBE_LAYER_INPUT_ACTIVATION_COUNT  1
/* 按方路由：这条快照描述的是**哪一方**的权重。0=黑，1=白，0xFFFFFFFF=共享网络
   （未启用按方路由）。放在 ProbeLayerDesc._reserved[2]（索引 1 已用于上面的
   activation 偏移），结构体大小不变，旧 Viewer 忽略它即可。
   没有这个字段时，TUI 展示的权重无法归因到某一方。 */
#define PROBE_LAYER_SIDE_OFFSET             2
#define PROBE_LAYER_SIDE_SHARED             0xFFFFFFFFu

/* ========== 层类型 ========== */
typedef enum ProbeLayerType {
    PROBE_LAYER_UNKNOWN = 0,
    PROBE_LAYER_INPUT,          /* 输入层（棋盘状态） */
    PROBE_LAYER_EMBED,          /* Token/Position Embedding */
    PROBE_LAYER_ATTENTION,      /* Multi-Head Attention */
    PROBE_LAYER_POOLING,        /* 可学习池化层 */
    PROBE_LAYER_LINEAR,         /* 全连接层 */
    PROBE_LAYER_RELU,           /* ReLU 激活（内嵌于 LINEAR） */
    PROBE_LAYER_OUTPUT          /* 输出层（Q 值） */
} ProbeLayerType;

/* ========== 全局状态（每个 snapshot buffer 开头） ========== */
typedef struct ProbeGlobalState {
    uint64_t episode;           /* 当前训练局号 */
    uint64_t train_step;        /* 全局训练步数 */
    float    epsilon;           /* 当前 epsilon */
    float    loss;              /* 最近一次 loss */
    float    learning_rate;     /* 当前学习率 */
    float    q_values[25];      /* 最近一次推理的 Q 值（25 个格子） */
    uint32_t current_player;    /* 最近一次推理的当前玩家 */
    uint32_t diagnostics_flags; /* PROBE_DIAG_*；首次快照没有有效梯度 */
    uint64_t gradient_span_steps; /* 有效梯度覆盖的 global_step 区间 */
    float    last_input[PROBE_INPUT_SIZE];
    /* Softmax 后的注意力概率。Viewer 默认对四个 head 求均值。 */
    float    attention_scores[PROBE_ATTENTION_HEADS]
                                  [PROBE_ATTENTION_TOKENS]
                                  [PROBE_ATTENTION_TOKENS];
} ProbeGlobalState;

/* ========== 层描述符 ========== */
typedef struct ProbeLayerDesc {
    uint32_t layer_id;          /* 层序号（0-based） */
    uint32_t type;              /* ProbeLayerType */
    char     name[32];          /* 可读名称，如 "FC1", "Attention" */

    uint32_t input_size;        /* 输入维度 */
    uint32_t output_size;       /* 输出维度 */
    uint32_t dim_count;         /* 张量维度数（1=向量, 2=矩阵） */
    uint32_t dims[4];           /* 各维度大小 */

    /* 以下 offset 均相对于当前 snapshot buffer 起始位置 */
    uint64_t weight_offset;     /* 权重数据偏移 */
    uint64_t weight_count;      /* 权重元素数（float） */

    uint64_t bias_offset;       /* 偏置数据偏移 */
    uint64_t bias_count;        /* 偏置元素数 */

    uint64_t activation_offset; /* 本层输出激活值偏移 */
    uint64_t activation_count;

    uint64_t gradient_offset;   /* 与 weight_count 对齐的有效参数梯度 */
    uint64_t gradient_count;

    uint64_t _reserved[4];      /* 预留扩展 */
} ProbeLayerDesc;

/* ========== 共享内存 Header ========== */
typedef struct NeuralProbeHeader {
    uint32_t magic;             /* NEURAL_PROBE_MAGIC */
    uint32_t version;           /* NEURAL_PROBE_VERSION */
    uint64_t total_size;        /* 共享内存总大小 */

    uint64_t snapshot_id;       /* 单调递增的快照 ID */
    uint64_t episode;           /* 最新 episode */
    uint64_t train_step;        /* 最新 train_step */

    uint32_t layer_count;       /* 层数（≤ MAX_PROBE_LAYERS） */
    volatile int32_t active_buffer; /* 0 或 1，原子切换（用 InterlockedExchange） */

    uint64_t topology_offset;   /* ProbeLayerDesc[layer_count] 的偏移 */
    uint64_t topology_size;     /* topology 数据大小 */

    uint64_t buffer0_offset;    /* Snapshot Buffer A 偏移 */
    uint64_t buffer1_offset;    /* Snapshot Buffer B 偏移 */
    uint64_t buffer_size;       /* 单个 snapshot buffer 大小 */

    uint32_t probe_enabled;     /* 探针是否启用 */
    uint32_t publish_hz;        /* 发布频率 */

    /* Seqlock generation: odd while a buffer/header update is in progress,
       even when readers may accept the snapshot. */
    volatile uint64_t publish_sequence;
    uint64_t _reserved[7];      /* [0]=publisher PID, [1]=publisher heartbeat, [2]=任一 viewer heartbeat */
} NeuralProbeHeader;

/* ========== 辅助宏 ========== */
/* 获取当前活跃 snapshot buffer 的指针 */
#define PROBE_ACTIVE_BUFFER(base, hdr) \
    ((uint8_t*)(base) + ((hdr)->active_buffer == 0 ? (hdr)->buffer0_offset : (hdr)->buffer1_offset))

/* 获取 topology 数组指针 */
#define PROBE_TOPOLOGY(base, hdr) \
    ((ProbeLayerDesc*)((uint8_t*)(base) + (hdr)->topology_offset))

/* 获取 layer 中的权重指针 */
#define PROBE_WEIGHT(base, buf_off, layer) \
    ((float*)((uint8_t*)(base) + (buf_off) + (layer)->weight_offset))

#define PROBE_BIAS(base, buf_off, layer) \
    ((float*)((uint8_t*)(base) + (buf_off) + (layer)->bias_offset))

/* 安全检查 offset */
#define PROBE_OFFSET_OK(hdr, offset, size) \
    ((offset) + (size) <= (hdr)->total_size)

#ifdef __cplusplus
}
#endif
