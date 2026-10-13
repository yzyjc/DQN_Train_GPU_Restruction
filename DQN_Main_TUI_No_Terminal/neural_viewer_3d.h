/* neural_viewer_3d.h — 3D Neural Viewer 公共 API
   Direct3D 11 渲染的神经网络概念图式可视化。
   子窗口模式：在 Probe 页面区域创建 HWND，D3D11 SwapChain 渲染。 */
#pragma once
#include <windows.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== 层布局配置 ========== */
#define NV_MAX_LAYERS     16
#define NV_MAX_NEURONS    2048   /* 每层最大节点数 */
#define NV_MAX_CONNECTIONS 512   /* 真实强连接候选（其余连接在渲染时抽样） */
#define NV_LAYER_SPACING  8.0f   /* 默认层间 Z 轴间距 */
#define NV_FC_LAYER_SPACING 12.0f  /* FC 层间更大间距（视觉区分） */
#define NV_NODE_BASE_SIZE 0.15f  /* 节点基础大小 */

/* 层类型（与 type_names[] 数组索引对齐） */
typedef enum {
    NV_LAYER_UNKNOWN = 0,
    NV_LAYER_INPUT,
    NV_LAYER_EMBED,
    NV_LAYER_ATTENTION,
    NV_LAYER_POOLING,
    NV_LAYER_LINEAR,
    NV_LAYER_RELU,
    NV_LAYER_OUTPUT
} NVLayerType;

/* 单个神经元 */
typedef struct {
    float x, y, z;      /* 3D 位置 */
    float activation;   /* 激活值（用于颜色编码） */
    float gradient;     /* 梯度值 */
    float bias;         /* 偏置 */
    float connection_strength; /* 连接强度 0~1（连线过滤时计算，渲染时映射透明度） */
    uint32_t color;     /* RGBA 颜色 */
    uint8_t selected;   /* 是否被选中 */
    uint8_t dead;       /* 是否为 dead neuron */
} NVNeuron;

/* 层描述 */
typedef struct {
    char name[32];
    NVLayerType type;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t neuron_count;
    uint32_t grid_w, grid_h;   /* 节点网格排列：grid_w × grid_h */
    float z_offset;            /* Z 轴位置 */
    NVNeuron neurons[NV_MAX_NEURONS];
} NVLayer;

/* 连接（用于 Top-N 渲染） */
typedef struct {
    uint16_t src_layer, src_neuron;
    uint16_t dst_layer, dst_neuron;
    float weight;
    uint32_t color;
} NVConnection;

/* 相机状态 */
typedef struct {
    float target_x, target_y, target_z;   /* 观察目标 */
    float distance;                        /* 距离目标的距离 */
    float yaw, pitch;                      /* 旋转角度（弧度） */
    float fov;                             /* 视场角（度） */
    float near_plane, far_plane;
} NVCamera;

/* 快照数据（从 probe 共享内存读取） */
typedef struct {
    uint32_t layer_count;
    NVLayer layers[NV_MAX_LAYERS];
    uint32_t connection_count;
    NVConnection connections[NV_MAX_CONNECTIONS];
    uint64_t snapshot_id;
    uint64_t episode;
    uint64_t train_step;
} NVSnapshot;

/* 显示模式 */
typedef enum {
    NV_MODE_ACTIVATION = 0,
    NV_MODE_WEIGHT,
    NV_MODE_GRADIENT,
    NV_MODE_CONTRIBUTION,
    NV_MODE_ATTENTION
} NVDisplayMode;

/* 选中信息 */
typedef struct {
    int layer_idx;
    int neuron_idx;
    int connection_idx;   /* -1 表示未选中连接 */
    float weight;
    float activation;
    float gradient;
    float bias;
    char layer_name[32];
} NVSelection;

/* ========== 公共 API ========== */

/* 初始化 3D Viewer（创建 D3D11 设备 + 子窗口） */
int nv_init(HWND parent_hwnd, int x, int y, int w, int h);

/* 关闭 3D Viewer */
void nv_shutdown(void);

/* 调整大小 */
void nv_resize(int w, int h);

/* 渲染一帧：0=成功，负数=设备/交换链已不可用，调用方应关闭并允许重试。 */
int nv_render(void);

/* Regression/diagnostic helper: count pixels that differ materially from the
   clear colour in the current back buffer; negative values indicate failure. */
int nv_debug_visible_pixel_count(void);
uint32_t nv_debug_last_connection_segment_count(void);
uint32_t nv_debug_last_strong_connection_segment_count(void);
uint32_t nv_debug_layer_label_count(void);
int nv_debug_connection_pixel_count(void);
int nv_debug_get_layer_grid(uint32_t layer, uint32_t* out_w, uint32_t* out_h, uint32_t* out_count);

/* 设置快照数据（从 probe 共享内存转换） */
void nv_set_snapshot(const NVSnapshot* snap);

/* 设置显示模式 */
void nv_set_display_mode(NVDisplayMode mode);

/* 获取当前选中信息 */
const NVSelection* nv_get_selection(void);

/* 相机控制 */
void nv_camera_reset(void);
void nv_camera_update(float dt); /* Advance interruptible camera/layout motion. */
void nv_camera_set_goal(float tx, float ty, float tz, float dist, float yaw, float pitch);  /* 设置相机目标（运镜用） */
void nv_camera_fit_network(uint32_t layer_count, float max_half_extent);
void nv_camera_focus_layer(int layer_idx);
const NVCamera* nv_camera_get(void);
void nv_camera_get_view_matrix(float* out_4x4);
void nv_camera_get_projection_matrix(float* out_4x4, float aspect);
/* 层间距（Shift+滚轮可调） */
float nv_get_layer_spacing(void);
float nv_get_fc_layer_spacing(void);
int nv_spacing_changed(void);   /* 间距变化标志，主循环检测后重新布局 */

/* 输入处理（由子窗口转发）：button 0=旋转，1/2=平移。 */
void nv_on_mouse_move(int x, int y, int buttons);
void nv_on_mouse_button(int button, int pressed, int x, int y);
void nv_on_mouse_wheel(int delta, int x, int y);
void nv_on_key(int key, int pressed);

/* 获取子窗口句柄 */
HWND nv_get_hwnd(void);

/* ========== 布局算法 ========== */
void nv_layout_compute(NVSnapshot* snap);

/* ========== 模型转换（probe → snapshot） */
struct NeuralProbeHeader;
struct ProbeLayerDesc;
struct ProbeGlobalState;
int nv_snapshot_from_probe(const struct NeuralProbeHeader* hdr,
    const struct ProbeLayerDesc* topology,
    const uint8_t* snapshot_buf,
    NVSnapshot* out);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
unsigned nv_camera_input_revision(void);
#ifdef __cplusplus
}
#endif
