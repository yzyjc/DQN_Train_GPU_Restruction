/* neural_viewer_layout.cpp — 层布局算法
   将网络拓扑转换为 3D 空间中的节点坐标。
   每层排列为二维网格（grid_w × grid_h），层间沿 Z 轴排列。 */
#include "neural_viewer_3d.h"
#include <math.h>
#include <string.h>

/* 自动计算网格尺寸：尽量接近正方形 */
static void compute_grid(uint32_t count, uint32_t* out_w, uint32_t* out_h) {
    if (count <= 0) { *out_w = *out_h = 0; return; }
    if (count <= 25) {
        /* Small visual layers should use the smallest square/near-square
           grid.  The old fixed 5x5 rule made a 16-node attention layer look
           like an incomplete 5x4 grid instead of the requested 4x4. */
        uint32_t w = (uint32_t)ceilf(sqrtf((float)count));
        *out_w = w;
        *out_h = (count + w - 1) / w;
        return;
    }
    /* 找最接近平方根的因子 */
    uint32_t sq = (uint32_t)sqrtf((float)count);
    for (uint32_t h = sq; h >= 1; h--) {
        if (count % h == 0) {
            *out_w = count / h; /* non-square layers lie horizontally */
            *out_h = h;
            return;
        }
    }
    *out_w = count; *out_h = 1;
}

/* 为一层的节点计算 3D 坐标 */
static void layout_layer_neurons(NVLayer* layer) {
    uint32_t gw = layer->grid_w, gh = layer->grid_h;
    if (gw == 0 || gh == 0) return;

    /* 间距：根据层类型和网格大小调整 */
    float spacing = 1.0f;
    if (layer->type == NV_LAYER_LINEAR) {
        /* FC 层：间距拉大，至少和池化层一样 */
        if (gw > 16) spacing = 0.55f;      /* 32×32 等大网格 */
        else if (gw > 8) spacing = 0.7f;   /* 16×16 等 */
        else spacing = 1.0f;
    } else {
        /* 其他层（Input/Output/Attention/Pooling） */
        if (gw > 16) spacing = 0.3f;
        else if (gw > 8) spacing = 0.5f;
    }

    float total_w = (gw - 1) * spacing;
    float total_h = (gh - 1) * spacing;
    float x0 = -total_w * 0.5f;
    float y0 = -total_h * 0.5f;

    for (uint32_t r = 0; r < gh && r * gw < layer->neuron_count; r++) {
        for (uint32_t c = 0; c < gw && r * gw + c < layer->neuron_count; c++) {
            uint32_t idx = r * gw + c;
            layer->neurons[idx].x = x0 + c * spacing;
            layer->neurons[idx].y = y0 + r * spacing;
            layer->neurons[idx].z = layer->z_offset;
        }
    }
}

/* ========== 公共 API ========== */

void nv_layout_compute(NVSnapshot* snap) {
    if (!snap) return;
    if (snap->layer_count > NV_MAX_LAYERS) snap->layer_count = NV_MAX_LAYERS;
    for (uint32_t i = 0; i < snap->layer_count; i++) {
        NVLayer* layer = &snap->layers[i];
        if (layer->neuron_count > NV_MAX_NEURONS) layer->neuron_count = NV_MAX_NEURONS;

        /* The legacy first layer contained exactly three vocabulary entries.
           Real TokenEmbedding visualization now exposes 16 feature channels,
           which should use the same square layout as every other tensor. */
        if (i == 0 && layer->neuron_count == 3) {
            layer->grid_h = 3;
            layer->grid_w = (layer->neuron_count + 2) / 3;
        } else {
            compute_grid(layer->neuron_count, &layer->grid_w, &layer->grid_h);
        }

        /* Z 轴位置：层序号 × 间距（FC 层用更大间距，Shift+滚轮可调） */
        float spacing = nv_get_layer_spacing();
        if (layer->type == NV_LAYER_LINEAR && i > 0 && snap->layers[i-1].type == NV_LAYER_LINEAR)
            spacing = nv_get_fc_layer_spacing();
        layer->z_offset = (i == 0) ? 0.0f : snap->layers[i-1].z_offset - spacing;
        const bool tensor_graph=snap->layer_count==13 && !strcmp(snap->layers[11].name,"EmbeddedTokens");
        if(tensor_graph) {
            const float s=nv_get_layer_spacing();
            if(i<=1)layer->z_offset=0;
            else if(i<=4)layer->z_offset=-2*s;
            else if(i==5)layer->z_offset=-4*s;
            else if(i==6)layer->z_offset=-5*s;
            else if(i==7)layer->z_offset=-6*s;
            else if(i==11)layer->z_offset=-s;
            else if(i==12)layer->z_offset=-3*s;
        }

        /* 计算每个节点的 3D 坐标 */
        layout_layer_neurons(layer);
        if(tensor_graph) {
            const float shift=i==0?-4.5f:i==1?4.5f:i==2?-6.f:i==4?6.f:0.f;
            for(uint32_t n=0;n<layer->neuron_count;++n)layer->neurons[n].x+=shift;
        }

        /* 默认颜色：层类型决定基础色 */
        uint32_t base_color = 0xFF00CCFF;  /* 默认青色 */
        switch (layer->type) {
            case NV_LAYER_INPUT:    base_color = 0xFF00FF88; break;  /* 绿色 */
            case NV_LAYER_OUTPUT:   base_color = 0xFFFF6600; break;  /* 橙色 */
            case NV_LAYER_ATTENTION:base_color = 0xFFFF00FF; break;  /* 紫色 */
            default: break;
        }
        for (uint32_t j = 0; j < layer->neuron_count; j++) {
            if (layer->neurons[j].color == 0)
                layer->neurons[j].color = base_color;
        }
    }
}

/* ========== 模型转换（probe → snapshot） ========== */

int nv_snapshot_from_probe(const void* hdr_ptr,
    const void* topology_ptr,
    const uint8_t* snapshot_buf,
    NVSnapshot* out)
{
    /* 前向声明避免头文件循环依赖，使用 void* 在实现中转换 */
    /* 实际使用时由调用者传入正确类型 */
    (void)hdr_ptr; (void)topology_ptr; (void)snapshot_buf;
    (void)out;
    /* TODO Phase 4: 从 probe 共享内存填充 NVSnapshot */
    return 0;
}
