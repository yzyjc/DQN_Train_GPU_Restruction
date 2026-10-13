/* neural_probe_client.cpp — Viewer 端 Neural Probe 客户端
   连接 Trainer 创建的共享内存，读取神经网络快照数据。
   所有 offset 均经过安全校验，不会因数据异常导致崩溃。 */
#include "neural_probe_protocol.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

int probe_client_update(void);
void probe_client_disconnect(void);

/* ========== 内部状态 ========== */
static HANDLE g_client_mapping = NULL;
static void*  g_client_base = NULL;
static int    g_client_connected = 0;
static uint64_t g_last_snapshot_id = 0;
static uint64_t g_client_view_size = 0;
static int      g_have_local_snapshot = 0;
static int      g_client_can_write = 0;

/* 本地 staging buffer（避免长时间锁住共享内存） */
static NeuralProbeHeader g_local_hdr;
static ProbeLayerDesc    g_local_topology[MAX_PROBE_LAYERS];
static uint8_t*          g_local_buf = NULL;
static uint64_t          g_local_buf_size = 0;
static ProbeGlobalState  g_legacy_global;

/* Protocol v2 is kept read-only so a newly built TUI can continue monitoring
   an already-running old Trainer.  Diagnostic arrays become available after
   that Trainer is safely restarted with protocol v3. */
typedef struct ProbeGlobalStateV2 {
    uint64_t episode;
    uint64_t train_step;
    float epsilon;
    float loss;
    float learning_rate;
    float q_values[25];
    uint32_t current_player;
    uint32_t pad[3];
} ProbeGlobalStateV2;

/* A shared timestamp represents "at least one viewer is alive".  Multiple
   readers may update it safely; expiration rather than a ref-count also
   recovers automatically when a viewer is killed or crashes. */
static void touch_viewer_heartbeat(void) {
    if (!g_client_can_write || !g_client_base) return;
    NeuralProbeHeader* shared = (NeuralProbeHeader*)g_client_base;
    InterlockedExchange64((volatile LONG64*)&shared->_reserved[PROBE_RESERVED_VIEWER_HEARTBEAT_MS],
        (LONG64)GetTickCount64());
}

/* ========== 内部函数 ========== */

static int span_ok(uint64_t total, uint64_t offset, uint64_t size) {
    return offset <= total && size <= total - offset;
}

static int validate_header(const NeuralProbeHeader* hdr, uint64_t mapped_size) {
    if (hdr->magic != NEURAL_PROBE_MAGIC) return -1;
    if (hdr->version != 2 && hdr->version != NEURAL_PROBE_VERSION) return -2;
    if (hdr->layer_count == 0 || hdr->layer_count > MAX_PROBE_LAYERS) return -3;
    if (hdr->total_size < sizeof(NeuralProbeHeader)) return -4;
    if (!span_ok(hdr->total_size, hdr->topology_offset, hdr->topology_size)) return -5;
    if (!span_ok(hdr->total_size, hdr->buffer0_offset, hdr->buffer_size)) return -6;
    if (!span_ok(hdr->total_size, hdr->buffer1_offset, hdr->buffer_size)) return -7;
    if (hdr->topology_size < hdr->layer_count * sizeof(ProbeLayerDesc)) return -8;
    const uint64_t state_size = hdr->version == 2 ?
        sizeof(ProbeGlobalStateV2) : sizeof(ProbeGlobalState);
    if (hdr->buffer_size < state_size || hdr->buffer_size > SIZE_MAX) return -9;
    if (hdr->active_buffer != 0 && hdr->active_buffer != 1) return -10;
    if (mapped_size && hdr->total_size > mapped_size) return -11;
    return 0;
}

static int validate_layer(const ProbeLayerDesc* layer, uint64_t buffer_size) {
    if (!layer || layer->type > PROBE_LAYER_OUTPUT || layer->dim_count > 4) return 0;
    if (!memchr(layer->name, '\0', sizeof(layer->name))) return 0;
    const uint64_t counts[] = { layer->weight_count, layer->bias_count,
        layer->activation_count, layer->gradient_count };
    const uint64_t offsets[] = { layer->weight_offset, layer->bias_offset,
        layer->activation_offset, layer->gradient_offset };
    for (int i = 0; i < 4; ++i) {
        if (counts[i] > UINT64_MAX / sizeof(float) ||
            !span_ok(buffer_size, offsets[i], counts[i] * sizeof(float))) return 0;
    }
    return 1;
}

/* The mapping object stays alive while any TUI still has it open. New
   publishers identify their owner so a stale mapping is not mistaken for a
   permanently frozen live Trainer. PID 0 means legacy/unknown and remains
   compatible. */
static int publisher_has_exited(const NeuralProbeHeader* hdr) {
    uint64_t raw_pid = hdr->_reserved[PROBE_RESERVED_PUBLISHER_PID];
    if (raw_pid == 0 || raw_pid > UINT32_MAX) return 0;
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)raw_pid);
    if (!process) return 0; /* Access denied is not evidence that it exited. */
    DWORD wait = WaitForSingleObject(process, 0);
    CloseHandle(process);
    return wait == WAIT_OBJECT_0;
}

/* 从 snapshot buffer 读取权重统计 */
static void calc_weight_stats(const float* data, uint64_t count,
    float* out_min, float* out_max, float* out_mean, float* out_std) {
    if (count == 0) { *out_min = *out_max = *out_mean = *out_std = 0; return; }
    float mn = data[0], mx = data[0];
    double sum = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    float mean = (float)(sum / count);
    double var = 0;
    for (uint64_t i = 0; i < count; i++) {
        double d = data[i] - mean;
        var += d * d;
    }
    *out_min = mn; *out_max = mx; *out_mean = mean;
    *out_std = (float)sqrt(var / count);
}

/* ========== 公共 API ========== */

int probe_client_connect(void) {
    if (g_client_connected) return 0;

    g_client_mapping = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, NEURAL_PROBE_NAME);
    g_client_can_write = g_client_mapping != NULL;
    if (!g_client_mapping)
        g_client_mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, NEURAL_PROBE_NAME);
    if (!g_client_mapping) return -1;  /* Trainer 未启动或未启用 probe */

    const DWORD map_access = g_client_can_write ? (FILE_MAP_READ | FILE_MAP_WRITE) : FILE_MAP_READ;
    g_client_base = MapViewOfFile(g_client_mapping, map_access, 0, 0, 0);
    if (!g_client_base) {
        CloseHandle(g_client_mapping); g_client_mapping = NULL;
        return -2;
    }

    MEMORY_BASIC_INFORMATION mbi = {};
    if (!VirtualQuery(g_client_base, &mbi, sizeof(mbi)) || mbi.RegionSize < sizeof(NeuralProbeHeader)) {
        UnmapViewOfFile(g_client_base); g_client_base = NULL;
        CloseHandle(g_client_mapping); g_client_mapping = NULL;
        return -3;
    }
    g_client_view_size = (uint64_t)mbi.RegionSize;

    /* 验证 header */
    const NeuralProbeHeader* hdr = (const NeuralProbeHeader*)g_client_base;
    int err = validate_header(hdr, g_client_view_size);
    if (err != 0) {
        printf("[PROBE CLIENT] Header 验证失败: %d\n", err);
        UnmapViewOfFile(g_client_base); g_client_base = NULL;
        CloseHandle(g_client_mapping); g_client_mapping = NULL;
        return -3;
    }

    /* 分配 local staging buffer */
    g_local_buf_size = hdr->buffer_size;
    g_local_buf = (uint8_t*)malloc((size_t)g_local_buf_size);
    if (!g_local_buf) {
        UnmapViewOfFile(g_client_base); g_client_base = NULL;
        CloseHandle(g_client_mapping); g_client_mapping = NULL;
        return -4;
    }

    g_client_connected = 1;
    g_have_local_snapshot = 0;
    g_last_snapshot_id = 0;
    touch_viewer_heartbeat();
    /* Snapshot id 0 is a valid initialized frame.  Do one immediate copy so
       callers never observe a zeroed local header after a successful connect. */
    err = probe_client_update();
    if (err < 0) {
        probe_client_disconnect();
        return -5;
    }
    printf("[PROBE CLIENT] 已连接共享内存 (%.1f MB, %u 层)\n",
        hdr->total_size / (1024.0 * 1024.0), hdr->layer_count);
    return 0;
}

void probe_client_disconnect(void) {
    if (g_local_buf) { free(g_local_buf); g_local_buf = NULL; }
    if (g_client_base) { UnmapViewOfFile(g_client_base); g_client_base = NULL; }
    if (g_client_mapping) { CloseHandle(g_client_mapping); g_client_mapping = NULL; }
    g_client_connected = 0;
    g_last_snapshot_id = 0;
    g_client_view_size = 0;
    g_local_buf_size = 0;
    g_have_local_snapshot = 0;
    g_client_can_write = 0;
    memset(&g_local_hdr, 0, sizeof(g_local_hdr));
    memset(g_local_topology, 0, sizeof(g_local_topology));
    memset(&g_legacy_global, 0, sizeof(g_legacy_global));
}

int probe_client_is_connected(void) {
    return g_client_connected;
}

/* 读取最新的 snapshot（原子切换后复制到本地 staging buffer） */
int probe_client_update(void) {
    if (!g_client_connected || !g_client_base || !g_local_buf) return -1;

    touch_viewer_heartbeat();

    const NeuralProbeHeader* shared = (const NeuralProbeHeader*)g_client_base;
    if (publisher_has_exited(shared)) return -4;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const uint64_t sequence_before = shared->publish_sequence;
        MemoryBarrier();
        if (sequence_before & 1u) continue;

        NeuralProbeHeader candidate;
        memcpy(&candidate, shared, sizeof(candidate));
        if (validate_header(&candidate, g_client_view_size) != 0 ||
            candidate.buffer_size != g_local_buf_size) return -2;
        if (g_have_local_snapshot && candidate.snapshot_id == g_last_snapshot_id) return 0;
        const uint64_t topo_size = candidate.layer_count * sizeof(ProbeLayerDesc);
        ProbeLayerDesc candidate_topology[MAX_PROBE_LAYERS];
        memset(candidate_topology, 0, sizeof(candidate_topology));
        memcpy(candidate_topology, (uint8_t*)g_client_base + candidate.topology_offset, (size_t)topo_size);
        for (uint32_t i = 0; i < candidate.layer_count; ++i)
            if (!validate_layer(&candidate_topology[i], candidate.buffer_size)) return -3;
        const uint64_t buf_off = candidate.active_buffer == 0 ?
            candidate.buffer0_offset : candidate.buffer1_offset;
        memcpy(g_local_buf, (uint8_t*)g_client_base + buf_off, (size_t)candidate.buffer_size);
        MemoryBarrier();
        const uint64_t sequence_after = shared->publish_sequence;
        if (sequence_before == sequence_after && !(sequence_after & 1u)) {
            g_local_hdr = candidate;
            memcpy(g_local_topology, candidate_topology, (size_t)topo_size);
            g_last_snapshot_id = candidate.snapshot_id;
            g_have_local_snapshot = 1;
            return 1;
        }
    }
    return 0; /* Publisher was busy; retry on the next UI tick. */
}

/* ========== 查询 API ========== */

const NeuralProbeHeader* probe_get_header(void) {
    return g_client_connected && g_have_local_snapshot ? &g_local_hdr : NULL;
}

const ProbeLayerDesc* probe_get_topology(void) {
    return g_client_connected && g_have_local_snapshot ? g_local_topology : NULL;
}

uint32_t probe_get_layer_count(void) {
    return g_client_connected && g_have_local_snapshot ? g_local_hdr.layer_count : 0;
}

const ProbeLayerDesc* probe_get_layer(uint32_t index) {
    if (!g_client_connected || !g_have_local_snapshot || index >= g_local_hdr.layer_count) return NULL;
    return &g_local_topology[index];
}

/* 获取某层的权重指针（来自本地 staging buffer） */
const float* probe_get_weights(uint32_t layer_index) {
    if (!g_client_connected || !g_have_local_snapshot || layer_index >= g_local_hdr.layer_count) return NULL;
    const ProbeLayerDesc* layer = &g_local_topology[layer_index];
    if (layer->weight_count == 0) return NULL;
    if (layer->weight_count > UINT64_MAX / sizeof(float) ||
        !span_ok(g_local_buf_size, layer->weight_offset, layer->weight_count * sizeof(float))) return NULL;
    return (const float*)(g_local_buf + layer->weight_offset);
}

const float* probe_get_biases(uint32_t layer_index) {
    if (!g_client_connected || !g_have_local_snapshot || layer_index >= g_local_hdr.layer_count) return NULL;
    const ProbeLayerDesc* layer = &g_local_topology[layer_index];
    if (layer->bias_count == 0) return NULL;
    if (layer->bias_count > UINT64_MAX / sizeof(float) ||
        !span_ok(g_local_buf_size, layer->bias_offset, layer->bias_count * sizeof(float))) return NULL;
    return (const float*)(g_local_buf + layer->bias_offset);
}

const float* probe_get_activations(uint32_t layer_index) {
    if (!g_client_connected || !g_have_local_snapshot || layer_index >= g_local_hdr.layer_count) return NULL;
    const ProbeLayerDesc* layer = &g_local_topology[layer_index];
    if (!layer->activation_count || layer->activation_count > UINT64_MAX / sizeof(float) ||
        !span_ok(g_local_buf_size, layer->activation_offset,
            layer->activation_count * sizeof(float))) return NULL;
    return (const float*)(g_local_buf + layer->activation_offset);
}

const float* probe_get_gradients(uint32_t layer_index) {
    if (!g_client_connected || !g_have_local_snapshot || layer_index >= g_local_hdr.layer_count) return NULL;
    const ProbeLayerDesc* layer = &g_local_topology[layer_index];
    if (!layer->gradient_count || layer->gradient_count > UINT64_MAX / sizeof(float) ||
        !span_ok(g_local_buf_size, layer->gradient_offset,
            layer->gradient_count * sizeof(float))) return NULL;
    return (const float*)(g_local_buf + layer->gradient_offset);
}

const float* probe_get_input_activations(uint32_t layer_index, uint64_t* out_count) {
    if (out_count) *out_count = 0;
    if (!g_client_connected || !g_have_local_snapshot || layer_index >= g_local_hdr.layer_count) return NULL;
    const ProbeLayerDesc* layer = &g_local_topology[layer_index];
    const uint64_t offset = layer->_reserved[PROBE_LAYER_INPUT_ACTIVATION_OFFSET];
    const uint64_t count = layer->_reserved[PROBE_LAYER_INPUT_ACTIVATION_COUNT];
    if (!count || count > UINT64_MAX / sizeof(float) ||
        !span_ok(g_local_buf_size, offset, count * sizeof(float))) return NULL;
    if (out_count) *out_count = count;
    return (const float*)(g_local_buf + offset);
}

const ProbeGlobalState* probe_get_global_state(void) {
    if (!g_client_connected || !g_have_local_snapshot) return NULL;
    if (g_local_hdr.version == 2) {
        if (g_local_buf_size < sizeof(ProbeGlobalStateV2)) return NULL;
        const ProbeGlobalStateV2* old_state = (const ProbeGlobalStateV2*)
            (g_local_buf + g_local_buf_size - sizeof(ProbeGlobalStateV2));
        memset(&g_legacy_global, 0, sizeof(g_legacy_global));
        g_legacy_global.episode = old_state->episode;
        g_legacy_global.train_step = old_state->train_step;
        g_legacy_global.epsilon = old_state->epsilon;
        g_legacy_global.loss = old_state->loss;
        g_legacy_global.learning_rate = old_state->learning_rate;
        memcpy(g_legacy_global.q_values, old_state->q_values, sizeof(old_state->q_values));
        g_legacy_global.current_player = old_state->current_player;
        return &g_legacy_global;
    }
    /* 全局状态在 buffer 末尾 */
    if (g_local_buf_size < sizeof(ProbeGlobalState)) return NULL;
    uint64_t off = g_local_buf_size - sizeof(ProbeGlobalState);
    return (const ProbeGlobalState*)(g_local_buf + off);
}

uint64_t probe_get_snapshot_id(void) {
    return g_last_snapshot_id;
}

/* 获取某层权重的统计信息 */
int probe_get_weight_stats(uint32_t layer_index,
    float* out_min, float* out_max, float* out_mean, float* out_std) {
    const float* w = probe_get_weights(layer_index);
    const ProbeLayerDesc* layer = probe_get_layer(layer_index);
    if (!w || !layer) return -1;
    calc_weight_stats(w, layer->weight_count, out_min, out_max, out_mean, out_std);
    return 0;
}

int probe_get_bias_stats(uint32_t layer_index,
    float* out_min, float* out_max, float* out_mean, float* out_std) {
    const float* b = probe_get_biases(layer_index);
    const ProbeLayerDesc* layer = probe_get_layer(layer_index);
    if (!b || !layer) return -1;
    calc_weight_stats(b, layer->bias_count, out_min, out_max, out_mean, out_std);
    return 0;
}

#ifdef __cplusplus
}
#endif
