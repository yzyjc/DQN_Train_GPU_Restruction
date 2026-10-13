/* neural_viewer_3d.cpp — 3D Neural Viewer 主入口
   D3D11 渲染：设备/交换链/着色器/顶点缓冲/渲染循环。
   子窗口模式：在父窗口中创建 HWND 子窗口。 */
#include "neural_viewer_3d.h"
#include "neural_probe_protocol.h"
#include <d3d11.h>
#include <d3d11sdklayers.h>
#include <d3dcompiler.h>
#include <windowsx.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

/* ========== 内部状态 ========== */
static HWND g_nv_hwnd = NULL;
static HWND g_nv_parent = NULL;

/* D3D11 对象 */
static ID3D11Device*           g_dev = NULL;
static ID3D11DeviceContext*    g_ctx = NULL;
static IDXGISwapChain*         g_sc = NULL;
static ID3D11RenderTargetView* g_rtv = NULL;
static ID3D11DepthStencilView* g_dsv = NULL;
static ID3D11DepthStencilState* g_dss = NULL;
static ID3D11RasterizerState*   g_rs = NULL;
static ID3D11BlendState*        g_blend = NULL;

/* 着色器 */
static ID3D11VertexShader* g_vs = NULL;
static ID3D11PixelShader*  g_ps = NULL;
static ID3D11VertexShader* g_vs_line = NULL;
static ID3D11PixelShader*  g_ps_line = NULL;
static ID3D11InputLayout*  g_il = NULL;
static ID3D11InputLayout*  g_il_line = NULL;

/* 常量缓冲 */
static ID3D11Buffer* g_cb = NULL;

/* 状态 */
static int g_nv_w = 800, g_nv_h = 600;
static NVSnapshot g_snap;
static NVDisplayMode g_mode = NV_MODE_WEIGHT;

/* 扫描光点动画状态 */
static float g_scan_z = 0.0f;           /* 当前扫描平面 Z 位置 */
static int g_scan_active = 1;           /* 是否激活扫描 */
static int g_scan_steps_total = 20;     /* 从首层到末层的总步数 */
static int g_scan_step_current = 0;     /* 当前步 */
static NVSelection g_sel = { -1, -1, -1, 0, 0, 0, 0, "" };
static int g_nv_initialized = 0;
static HRESULT g_nv_last_error = S_OK;
static int g_nv_debug_capture_requested = 0;
static int g_nv_debug_visible_pixels = -1;
static int g_nv_debug_connection_pixels = -1;
static uint32_t g_nv_last_connection_segments = 0;
static uint32_t g_nv_last_strong_segments = 0;
static HWND g_nv_layer_labels[NV_MAX_LAYERS] = {};

/* ========== 3D 入场动画 ========== */
#define NV_INTRO_MS 1400  /* 入场总时长 1.2秒 */
#define NV_INTRO_NEURON_MS 800  /* 神经元聚拢时长 */
#define NV_INTRO_CONN_MS 600   /* 连线生长时长 */

static int g_nv_intro_active = 0;
static int g_nv_intro_force_next = 0;  /* 首次收到真实快照时强制触发入场 */
static DWORD g_nv_intro_start = 0;
static DWORD g_nv_intro_elapsed = 0;
static unsigned g_nv_intro_input_revision = 0;

/* 入场后运镜动画 */
static int g_nv_cam_flyover = 0;
static DWORD g_nv_cam_flyover_start = 0;
static float g_nv_cam_flyover_from_x, g_nv_cam_flyover_from_y, g_nv_cam_flyover_from_z;
static float g_nv_cam_flyover_from_dist;
static float g_nv_cam_flyover_to_x, g_nv_cam_flyover_to_y, g_nv_cam_flyover_to_z;
static float g_nv_cam_flyover_to_dist;
static float g_nv_cam_flyover_to_yaw, g_nv_cam_flyover_to_pitch;
#define NV_FLYOVER_MS 800  /* 运镜时长 */
static float g_nv_intro_neuron_x[NV_MAX_LAYERS][NV_MAX_NEURONS];  /* 初始外围位置 */
static float g_nv_intro_neuron_y[NV_MAX_LAYERS][NV_MAX_NEURONS];
static float g_nv_intro_neuron_z[NV_MAX_LAYERS][NV_MAX_NEURONS];
static int g_nv_intro_neuron_count[NV_MAX_LAYERS];
static int g_nv_intro_layer_count = 0;
static HFONT g_nv_label_font = NULL;
static uint32_t g_nv_label_count = 0;

/* 顶点结构 */
struct NVVertex {
    float pos[3];
    float size;
    uint32_t color;
};

static ID3D11Buffer* g_node_stream = NULL;
static std::vector<NVVertex> g_node_vertices;

struct NVLineVertex {
    float pos[3];
    uint32_t color;
};

static uint32_t nv_pack_rgba8(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    /* DXGI_FORMAT_R8G8B8A8_UNORM consumes bytes in R,G,B,A order. */
    return (r & 255u) | ((g & 255u) << 8) | ((b & 255u) << 16) | ((a & 255u) << 24);
}

/* 常量缓冲布局 */
struct NVConstantBuffer {
    float view[16];
    float proj[16];
    float camera_pos[4];
    float params[4];   /* x=point_size, y=time, z,w=reserved */
};

/* ========== HLSL 着色器 ========== */
/* 节点着色器：Point Sprite */
static const char* g_vs_code = R"(
cbuffer CB : register(b0) {
    /* The CPU camera helpers produce DirectX row-vector matrices.  Make the
       storage convention explicit; the HLSL default is column-major and used
       to transpose both matrices silently, leaving every point off-screen. */
    row_major float4x4 g_view;
    row_major float4x4 g_proj;
    float4 g_camera;
    float4 g_params;
};
struct VS_IN {
    float3 pos : POSITION;
    float size : PSIZE;
    float4 color : COLOR;
};
struct VS_OUT {
    float4 pos : SV_Position;
    float4 color : COLOR;
};
VS_OUT main(VS_IN v) {
    VS_OUT o;
    float4 world = float4(v.pos, 1.0);
    float4 view = mul(world, g_view);
    o.pos = mul(view, g_proj);
    o.color = v.color;
    return o;
}
)";

static const char* g_ps_code = R"(
struct PS_IN {
    float4 pos : SV_Position;
    float4 color : COLOR;
};
float4 main(PS_IN i) : SV_Target {
    return i.color;
}
)";

/* 连线着色器 */
static const char* g_vs_line_code = R"(
cbuffer CB : register(b0) {
    row_major float4x4 g_view;
    row_major float4x4 g_proj;
    float4 g_camera;
    float4 g_params;
};
struct VS_IN {
    float3 pos : POSITION;
    float4 color : COLOR;
};
struct VS_OUT {
    float4 pos : SV_Position;
    float4 color : COLOR;
};
VS_OUT main(VS_IN v) {
    VS_OUT o;
    float4 world = float4(v.pos, 1.0);
    o.pos = mul(mul(world, g_view), g_proj);
    o.color = v.color;
    return o;
}
)";

static const char* g_ps_line_code = R"(
struct PS_IN {
    float4 pos : SV_Position;
    float4 color : COLOR;
};
float4 main(PS_IN i) : SV_Target {
    return i.color;
}
)";

/* ========== 内部函数 ========== */

static void nv_cleanup_d3d(void) {
    if (g_node_stream) { g_node_stream->Release(); g_node_stream = NULL; }
    std::vector<NVVertex>().swap(g_node_vertices);
    if (g_ctx) {
        g_ctx->ClearState();
        g_ctx->Flush();
    }
    if (g_cb) { g_cb->Release(); g_cb = NULL; }
    if (g_blend) { g_blend->Release(); g_blend = NULL; }
    if (g_rs) { g_rs->Release(); g_rs = NULL; }
    if (g_dss) { g_dss->Release(); g_dss = NULL; }
    if (g_dsv) { g_dsv->Release(); g_dsv = NULL; }
    if (g_rtv) { g_rtv->Release(); g_rtv = NULL; }
    if (g_sc) { g_sc->Release(); g_sc = NULL; }
    if (g_vs) { g_vs->Release(); g_vs = NULL; }
    if (g_ps) { g_ps->Release(); g_ps = NULL; }
    if (g_vs_line) { g_vs_line->Release(); g_vs_line = NULL; }
    if (g_ps_line) { g_ps_line->Release(); g_ps_line = NULL; }
    if (g_il) { g_il->Release(); g_il = NULL; }
    if (g_il_line) { g_il_line->Release(); g_il_line = NULL; }
    if (g_ctx) { g_ctx->Release(); g_ctx = NULL; }
    if (g_dev) { g_dev->Release(); g_dev = NULL; }
}

static ID3DBlob* compile_shader(const char* code, const char* entry, const char* target) {
    ID3DBlob* blob = NULL;
    ID3DBlob* err = NULL;
    HRESULT hr = D3DCompile(code, strlen(code), NULL, NULL, NULL,
        entry, target, 0, 0, &blob, &err);
    if (FAILED(hr) && err) {
        printf("[NV3D] Shader compile error: %s\n", (char*)err->GetBufferPointer());
        err->Release();
        return NULL;
    }
    if (err) err->Release();
    return blob;
}

static int create_render_target(void) {
    if (!g_sc || !g_dev || g_nv_w < 1 || g_nv_h < 1) return -1;
    ID3D11Texture2D* bb = NULL;
    if (FAILED(g_sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb))) return -1;
    if (FAILED(g_dev->CreateRenderTargetView(bb, NULL, &g_rtv))) { bb->Release(); return -2; }
    bb->Release();

    /* 深度缓冲 */
    D3D11_TEXTURE2D_DESC dd = {};
    dd.Width = g_nv_w; dd.Height = g_nv_h;
    dd.MipLevels = 1; dd.ArraySize = 1;
    dd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dd.SampleDesc.Count = 1;
    dd.Usage = D3D11_USAGE_DEFAULT;
    dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    ID3D11Texture2D* dt = NULL;
    if (FAILED(g_dev->CreateTexture2D(&dd, NULL, &dt))) {
        g_rtv->Release(); g_rtv = NULL;
        return -3;
    }
    if (FAILED(g_dev->CreateDepthStencilView(dt, NULL, &g_dsv))) {
        dt->Release();
        g_rtv->Release(); g_rtv = NULL;
        return -4;
    }
    dt->Release();
    return 0;
}

/* ========== 窗口过程 ========== */
static LRESULT CALLBACK nv_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        g_nv_w = LOWORD(lp); g_nv_h = HIWORD(lp);
        if (g_nv_w < 1) g_nv_w = 1;
        if (g_nv_h < 1) g_nv_h = 1;
        nv_resize(g_nv_w, g_nv_h);
        return 0;
    case WM_LBUTTONDOWN:
        SetCapture(hwnd);
        nv_on_mouse_button(0, 1, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_LBUTTONUP:
        ReleaseCapture();
        nv_on_mouse_button(0, 0, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_RBUTTONDOWN:
        SetCapture(hwnd);
        nv_on_mouse_button(2, 1, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_RBUTTONUP:
        ReleaseCapture();
        nv_on_mouse_button(2, 0, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_MBUTTONDOWN:
        SetCapture(hwnd);
        nv_on_mouse_button(1, 1, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_MBUTTONUP:
        ReleaseCapture();
        nv_on_mouse_button(1, 0, GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_MOUSEMOVE:
        nv_on_mouse_move(GET_X_LPARAM(lp), GET_Y_LPARAM(lp), (int)wp);
        return 0;
    case WM_MOUSEWHEEL:
        nv_on_mouse_wheel(GET_WHEEL_DELTA_WPARAM(wp), GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        return 0;
    case WM_KEYDOWN:
        nv_on_key((int)wp, 1);
        return 0;
    case WM_KEYUP:
        nv_on_key((int)wp, 0);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        if (g_nv_hwnd == hwnd) g_nv_hwnd = NULL;
        g_nv_initialized = 0;
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

/* Layer captions are separate child windows above the D3D surface.  This
   gives us proper Chinese glyph rendering without baking a font atlas into
   the shader and keeps labels crisp at every DPI. */
static LRESULT CALLBACK nv_label_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_NCHITTEST:
        return HTTRANSPARENT; /* mouse gestures pass through to the 3D view */
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        RECT rc = {};
        GetClientRect(hwnd, &rc);
        HBRUSH bg = CreateSolidBrush(RGB(8, 22, 28));
        HBRUSH border = CreateSolidBrush(RGB(36, 110, 122));
        FillRect(dc, &rc, bg);
        FrameRect(dc, &rc, border);
        DeleteObject(border);
        DeleteObject(bg);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(130, 235, 235));
        HFONT old_font = g_nv_label_font ? (HFONT)SelectObject(dc, g_nv_label_font) : NULL;
        wchar_t text[128] = {};
        GetWindowTextW(hwnd, text, (int)(sizeof(text) / sizeof(text[0])));
        DrawTextW(dc, text, -1, &rc, DT_CENTER | DT_VCENTER | DT_WORDBREAK | DT_NOPREFIX);
        if (old_font) SelectObject(dc, old_font);
        EndPaint(hwnd, &ps);
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void nv_cleanup_layer_labels(void) {
    for (uint32_t i = 0; i < NV_MAX_LAYERS; ++i) {
        if (g_nv_layer_labels[i] && IsWindow(g_nv_layer_labels[i])) DestroyWindow(g_nv_layer_labels[i]);
        g_nv_layer_labels[i] = NULL;
    }
    if (g_nv_label_font) { DeleteObject(g_nv_label_font); g_nv_label_font = NULL; }
    g_nv_label_count = 0;
}

static int nv_ensure_label_windows(void) {
    if (!g_nv_hwnd || !IsWindow(g_nv_hwnd)) return 0;
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = nv_label_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    wc.lpszClassName = L"NeuralViewer3DLayerLabel";
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return 0;
    if (!g_nv_label_font)
        g_nv_label_font = CreateFontW(-15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Microsoft YaHei UI");
    for (uint32_t i = 0; i < NV_MAX_LAYERS; ++i) {
        if (g_nv_layer_labels[i] && IsWindow(g_nv_layer_labels[i])) continue;
        g_nv_layer_labels[i] = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TRANSPARENT,
            wc.lpszClassName, L"", WS_CHILD | WS_CLIPSIBLINGS,
            0, 0, 180, 40, g_nv_hwnd, NULL, wc.hInstance, NULL);
        if (!g_nv_layer_labels[i]) return 0;
    }
    return 1;
}

static const wchar_t* nv_layer_role_zh(const NVLayer* layer) {
    if (!layer) return L"网络层：特征处理";
    if (strstr(layer->name, "TokenEmbedding_16D"))
        return L"嵌入查表：25格输出汇总为16个特征维";
    if (strstr(layer->name, "EmbeddedTokens"))return L"词嵌入+位置嵌入；灰线为计算依赖";
    if (strstr(layer->name, "AttentionMix"))return L"softmax(QK)乘V；灰线不是梯度";
    if (strstr(layer->name, "PositionEmbedding"))
        return L"位置嵌入：25个棋盘位置的编码强度";
    if (layer->type == NV_LAYER_INPUT) return L"输入层：棋盘状态";
    if (layer->type == NV_LAYER_OUTPUT) return L"输出层：动作 Q 值";
    if (layer->type == NV_LAYER_ATTENTION) return L"注意力层：全局关系";
    if (layer->type == NV_LAYER_POOLING) return L"池化层：特征汇总";
    if (strstr(layer->name, "Embed") || strstr(layer->name, "embed")) return L"嵌入层：特征编码";
    if (strstr(layer->name, "ReLU") || strstr(layer->name, "relu")) return L"激活层：非线性变换";
    return L"全连接层：特征变换";
}

static void nv_row_vec_mul(const float in[4], const float matrix[16], float out[4]) {
    for (int col = 0; col < 4; ++col) {
        out[col] = 0.0f;
        for (int row = 0; row < 4; ++row) out[col] += in[row] * matrix[row * 4 + col];
    }
}

static void nv_update_layer_labels(uint32_t layer_count) {
    if (!nv_ensure_label_windows()) return;
    float view[16], proj[16];
    nv_camera_get_view_matrix(view);
    nv_camera_get_projection_matrix(proj, g_nv_h > 0 ? (float)g_nv_w / g_nv_h : 1.0f);
    g_nv_label_count = 0;
    for (uint32_t i = 0; i < NV_MAX_LAYERS; ++i) {
        HWND label = g_nv_layer_labels[i];
        if (i >= layer_count || !g_snap.layers[i].neuron_count) {
            ShowWindow(label, SW_HIDE);
            continue;
        }
        NVLayer* layer = &g_snap.layers[i];
        float max_y = -1e20f;
        const uint32_t n = layer->neuron_count > NV_MAX_NEURONS ? NV_MAX_NEURONS : layer->neuron_count;
        for (uint32_t j = 0; j < n; ++j) if (layer->neurons[j].y > max_y) max_y = layer->neurons[j].y;
        float world[4] = { 0.0f, max_y + 1.2f, layer->z_offset, 1.0f };
        float eye[4], clip[4];
        nv_row_vec_mul(world, view, eye);
        nv_row_vec_mul(eye, proj, clip);
        if (clip[3] <= 0.0f || clip[2] < 0.0f || clip[2] > clip[3]) {
            ShowWindow(label, SW_HIDE);
            continue;
        }
        const float nx = clip[0] / clip[3];
        const float ny = clip[1] / clip[3];
        const int px = (int)((nx * 0.5f + 0.5f) * g_nv_w);
        const int py = (int)((0.5f - ny * 0.5f) * g_nv_h);
        const int lw = 190, lh = 42;
        if (px < -lw || px > g_nv_w + lw || py < -lh || py > g_nv_h + lh) {
            ShowWindow(label, SW_HIDE);
            continue;
        }
        wchar_t layer_name[48] = {};
        if (!MultiByteToWideChar(CP_UTF8, 0, layer->name, -1, layer_name, 48))
            swprintf_s(layer_name, L"Layer %u", i);
        wchar_t caption[128] = {};
        swprintf_s(caption, L"%s\n%s", layer_name, nv_layer_role_zh(layer));
        wchar_t old_caption[128] = {};
        GetWindowTextW(label, old_caption, 128);
        if (wcscmp(old_caption, caption) != 0) SetWindowTextW(label, caption);
        SetWindowPos(label, HWND_TOP, px - lw / 2, py - lh - 4, lw, lh,
            SWP_NOACTIVATE | SWP_SHOWWINDOW);
        InvalidateRect(label, NULL, FALSE);
        ++g_nv_label_count;
    }
}

/* ========== 公共 API ========== */

int nv_init(HWND parent_hwnd, int x, int y, int w, int h) {
    /* A failed/partial previous initialization must never leak into a retry. */
    nv_shutdown();
    if (!parent_hwnd || !IsWindow(parent_hwnd) || w < 1 || h < 1) return -1;
    g_nv_parent = parent_hwnd;
    g_nv_w = w; g_nv_h = h;
    g_nv_last_error = S_OK;

    /* 注册窗口类 */
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = nv_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"NeuralViewer3D";
    wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        return -1;

    /* 创建子窗口 */
    g_nv_hwnd = CreateWindowExW(0, L"NeuralViewer3D", L"",
        WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, x, y, w, h,
        parent_hwnd, NULL, GetModuleHandleW(NULL), NULL);
    if (!g_nv_hwnd) return -1;

    /* D3D11 设备 + 交换链 */
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 1;
    sd.BufferDesc.Width = w;
    sd.BufferDesc.Height = h;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_nv_hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    char debug_env[2] = {};
    if (GetEnvironmentVariableA("DQN_NV3D_DEBUG", debug_env, sizeof(debug_env)) > 0)
        flags |= D3D11_CREATE_DEVICE_DEBUG;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
        flags, &fl, 1, D3D11_SDK_VERSION, &sd, &g_sc, &g_dev, NULL, &g_ctx);
    if (FAILED(hr)) {
        printf("[NV3D] Hardware D3D failed (0x%lX), trying WARP...\n", hr);
        hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_WARP, NULL,
            0, &fl, 1, D3D11_SDK_VERSION, &sd, &g_sc, &g_dev, NULL, &g_ctx);
    }
    if (FAILED(hr)) {
        printf("[NV3D] WARP failed (0x%lX), trying Reference...\n", hr);
        hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_REFERENCE, NULL,
            0, &fl, 1, D3D11_SDK_VERSION, &sd, &g_sc, &g_dev, NULL, &g_ctx);
    }
    if (FAILED(hr)) {
        g_nv_last_error = hr;
        printf("[NV3D] All D3D drivers failed: 0x%lX\n", hr);
        nv_shutdown();
        return -2;
    }

    /* 渲染目标 + 深度缓冲 */
    if (create_render_target() != 0) { nv_shutdown(); return -3; }

    /* 深度模板状态 */
    D3D11_DEPTH_STENCIL_DESC dsd = {};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsd.DepthFunc = D3D11_COMPARISON_LESS;
    if (FAILED(g_dev->CreateDepthStencilState(&dsd, &g_dss))) {
        nv_shutdown(); return -4;
    }

    /* Neural layers are diagnostic billboards, not closed meshes.  Render
       both windings so a camera orbit can never cull an entire layer. */
    D3D11_RASTERIZER_DESC rsd = {};
    rsd.FillMode = D3D11_FILL_SOLID;
    rsd.CullMode = D3D11_CULL_NONE;
    rsd.DepthClipEnable = TRUE;
    if (FAILED(g_dev->CreateRasterizerState(&rsd, &g_rs))) {
        nv_shutdown(); return -4;
    }

    D3D11_BLEND_DESC blend_desc = {};
    blend_desc.RenderTarget[0].BlendEnable = TRUE;
    blend_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blend_desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blend_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blend_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blend_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blend_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blend_desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(g_dev->CreateBlendState(&blend_desc, &g_blend))) {
        nv_shutdown(); return -4;
    }

    /* 编译着色器 */
    ID3DBlob* vs_blob = compile_shader(g_vs_code, "main", "vs_5_0");
    ID3DBlob* ps_blob = compile_shader(g_ps_code, "main", "ps_5_0");
    ID3DBlob* vs_line_blob = compile_shader(g_vs_line_code, "main", "vs_5_0");
    ID3DBlob* ps_line_blob = compile_shader(g_ps_line_code, "main", "ps_5_0");
    if (!vs_blob || !ps_blob || !vs_line_blob || !ps_line_blob) {
        if (vs_blob) vs_blob->Release();
        if (ps_blob) ps_blob->Release();
        if (vs_line_blob) vs_line_blob->Release();
        if (ps_line_blob) ps_line_blob->Release();
        nv_shutdown();
        return -5;
    }

    bool shader_ok =
        SUCCEEDED(g_dev->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), NULL, &g_vs)) &&
        SUCCEEDED(g_dev->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), NULL, &g_ps)) &&
        SUCCEEDED(g_dev->CreateVertexShader(vs_line_blob->GetBufferPointer(), vs_line_blob->GetBufferSize(), NULL, &g_vs_line)) &&
        SUCCEEDED(g_dev->CreatePixelShader(ps_line_blob->GetBufferPointer(), ps_line_blob->GetBufferSize(), NULL, &g_ps_line));

    /* 输入布局：节点 */
    D3D11_INPUT_ELEMENT_DESC ie[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA },
        { "PSIZE",    0, DXGI_FORMAT_R32_FLOAT,        0, 12, D3D11_INPUT_PER_VERTEX_DATA },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,   0, 16, D3D11_INPUT_PER_VERTEX_DATA },
    };
    bool layout_ok = SUCCEEDED(g_dev->CreateInputLayout(ie, 3,
        vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), &g_il));

    /* 输入布局：连线 */
    D3D11_INPUT_ELEMENT_DESC ie_line[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,   0, 12, D3D11_INPUT_PER_VERTEX_DATA },
    };
    layout_ok = layout_ok && SUCCEEDED(g_dev->CreateInputLayout(ie_line, 2,
        vs_line_blob->GetBufferPointer(), vs_line_blob->GetBufferSize(), &g_il_line));

    vs_blob->Release(); ps_blob->Release();
    vs_line_blob->Release(); ps_line_blob->Release();
    if (!shader_ok || !layout_ok) { nv_shutdown(); return -6; }

    /* 常量缓冲 */
    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(NVConstantBuffer);
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(g_dev->CreateBuffer(&cbd, NULL, &g_cb))) {
        nv_shutdown(); return -7;
    }

    /* 初始化相机 */
    nv_camera_reset();

    /* 初始化快照 */
    memset(&g_snap, 0, sizeof(g_snap));

    /* 设置默认测试数据。只在全局 g_snap 中逐层初始化，避免把五个
       各含 2048 神经元的 NVLayer 再压到主线程栈上。 */
    static const struct {
        const char* name; NVLayerType type;
        uint32_t input_size, output_size, neuron_count, grid_w, grid_h;
    } layers[] = {
        { "Input",  NV_LAYER_INPUT,  25,   25,   25,   5,  5 },
        { "FC1",    NV_LAYER_LINEAR, 128,  1024, 1024, 32, 32 },
        { "FC2",    NV_LAYER_LINEAR, 1024, 512,  512,  32, 16 },
        { "FC3",    NV_LAYER_LINEAR, 512,  256,  256,  16, 16 },
        { "Output", NV_LAYER_OUTPUT, 256,  25,   25,   5,  5 },
    };
    g_snap.layer_count = 5;
    for (int i = 0; i < 5; i++) {
        NVLayer* layer = &g_snap.layers[i];
        snprintf(layer->name, sizeof(layer->name), "%s", layers[i].name);
        layer->type = layers[i].type;
        layer->input_size = layers[i].input_size;
        layer->output_size = layers[i].output_size;
        layer->neuron_count = layers[i].neuron_count;
        layer->grid_w = layers[i].grid_w;
        layer->grid_h = layers[i].grid_h;
        /* 为每个节点设置默认激活值（随机模拟） */
        for (uint32_t j = 0; j < layer->neuron_count; j++) {
            layer->neurons[j].activation = 0.5f;
            layer->neurons[j].gradient = 0.0f;
        }
    }
    nv_layout_compute(&g_snap);
    nv_camera_fit_network(g_snap.layer_count, 10.0f);

    g_nv_initialized = 1;
    /* 立即启动入场动画，首帧就从外围飞入 */
    g_nv_intro_active = 1;
    g_nv_intro_start = GetTickCount();
    g_nv_intro_input_revision = nv_camera_input_revision();
    g_nv_intro_layer_count = g_snap.layer_count;
    for (uint32_t i = 0; i < g_snap.layer_count; ++i) {
        const NVLayer* layer = &g_snap.layers[i];
        g_nv_intro_neuron_count[i] = layer->neuron_count;
        for (uint32_t j = 0; j < layer->neuron_count; ++j) {
            float angle = (float)(i * 7 + j * 13) * 0.618f * 3.14159f;
            float radius = 20.0f + (float)(i * 3 + j) * 0.5f;
            g_nv_intro_neuron_x[i][j] = cosf(angle) * radius;
            g_nv_intro_neuron_y[i][j] = sinf(angle) * radius;
            g_nv_intro_neuron_z[i][j] = layer->neurons[j].z;
        }
    }
    printf("[NV3D] 初始化成功 (%dx%d)\n", w, h);
    return 0;
}

void nv_shutdown(void) {
    HWND old_hwnd = g_nv_hwnd;
    nv_cleanup_layer_labels();
    g_nv_hwnd = NULL;
    g_nv_initialized = 0;
    nv_cleanup_d3d();
    if (old_hwnd && IsWindow(old_hwnd)) DestroyWindow(old_hwnd);
    g_nv_parent = NULL;
}

void nv_resize(int w, int h) {
    if (!g_sc || !g_ctx || w < 1 || h < 1) return;
    g_nv_w = w; g_nv_h = h;
    g_ctx->OMSetRenderTargets(0, NULL, NULL);
    if (g_rtv) { g_rtv->Release(); g_rtv = NULL; }
    if (g_dsv) { g_dsv->Release(); g_dsv = NULL; }
    HRESULT hr = g_sc->ResizeBuffers(1, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    if (FAILED(hr) || create_render_target() != 0) {
        g_nv_last_error = FAILED(hr) ? hr : E_FAIL;
    }
}

void nv_set_snapshot(const NVSnapshot* snap) {
    if (snap) {
        bool geometry_changed = g_snap.layer_count != snap->layer_count;
        if (!geometry_changed) {
            const uint32_t compare_count = snap->layer_count > NV_MAX_LAYERS ? NV_MAX_LAYERS : snap->layer_count;
            for (uint32_t i = 0; i < compare_count; ++i) {
                if (g_snap.layers[i].neuron_count != snap->layers[i].neuron_count ||
                    g_snap.layers[i].type != snap->layers[i].type) {
                    geometry_changed = true;
                    break;
                }
            }
        }
        g_snap = *snap;
        if (g_snap.layer_count > NV_MAX_LAYERS) g_snap.layer_count = NV_MAX_LAYERS;
        if (g_snap.connection_count > NV_MAX_CONNECTIONS) g_snap.connection_count = NV_MAX_CONNECTIONS;
        for (uint32_t i = 0; i < g_snap.layer_count; ++i)
            if (g_snap.layers[i].neuron_count > NV_MAX_NEURONS)
                g_snap.layers[i].neuron_count = NV_MAX_NEURONS;
        nv_layout_compute(&g_snap);
        if (geometry_changed || g_nv_intro_force_next) {
            g_nv_intro_force_next = 0;
            float max_extent = 1.0f;
            for (uint32_t i = 0; i < g_snap.layer_count; ++i) {
                const NVLayer* layer = &g_snap.layers[i];
                for (uint32_t j = 0; j < layer->neuron_count; ++j) {
                    const float ax = fabsf(layer->neurons[j].x);
                    const float ay = fabsf(layer->neurons[j].y);
                    if (ax > max_extent) max_extent = ax;
                    if (ay > max_extent) max_extent = ay;
                }
            }
            nv_camera_fit_network(g_snap.layer_count, max_extent);
            /* 启动入场动画：记录每个神经元的目标位置，设置初始外围位置 */
            g_nv_intro_active = 1;
            g_nv_intro_start = GetTickCount();
    g_nv_intro_input_revision = nv_camera_input_revision();
            g_nv_intro_layer_count = g_snap.layer_count;
            for (uint32_t i = 0; i < g_snap.layer_count; ++i) {
                const NVLayer* layer = &g_snap.layers[i];
                g_nv_intro_neuron_count[i] = layer->neuron_count;
                for (uint32_t j = 0; j < layer->neuron_count; ++j) {
                    /* 初始位置：在外围随机分布 */
                    float angle = (float)(i * 7 + j * 13) * 0.618f * 3.14159f;  /* 确定性分布 */
                    float radius = 20.0f + (float)(i * 3 + j) * 0.5f;
                    g_nv_intro_neuron_x[i][j] = cosf(angle) * radius;
                    g_nv_intro_neuron_y[i][j] = sinf(angle) * radius;
                    g_nv_intro_neuron_z[i][j] = layer->neurons[j].z;  /* Z 保持层位置 */
                }
            }
        }
        /* 新快照到达：扫描光点前进一步 */
        if (g_snap.layer_count >= 2) {
            g_scan_step_current++;
            if (g_scan_step_current >= g_scan_steps_total) g_scan_step_current = 0;
        }
    }
}

void nv_set_display_mode(NVDisplayMode mode) {
    g_mode = mode;
}

/* 强制触发入场动画（页面切换完成后调用） */
void nv_force_intro_animation(void) {
    g_nv_intro_active = 1;
    g_nv_intro_start = GetTickCount();
    g_nv_intro_input_revision = nv_camera_input_revision();
    g_nv_intro_layer_count = g_snap.layer_count;
    for (uint32_t i = 0; i < g_snap.layer_count; ++i) {
        const NVLayer* layer = &g_snap.layers[i];
        g_nv_intro_neuron_count[i] = layer->neuron_count;
        for (uint32_t j = 0; j < layer->neuron_count; ++j) {
            float angle = (float)(i * 7 + j * 13) * 0.618f * 3.14159f;
            float radius = 20.0f + (float)(i * 3 + j) * 0.5f;
            g_nv_intro_neuron_x[i][j] = cosf(angle) * radius;
            g_nv_intro_neuron_y[i][j] = sinf(angle) * radius;
            g_nv_intro_neuron_z[i][j] = layer->neurons[j].z;
        }
    }
}

const NVSelection* nv_get_selection(void) {
    return &g_sel;
}

HWND nv_get_hwnd(void) {
    return g_nv_hwnd;
}

/* ========== 渲染 ========== */

static void update_constant_buffer(void) {
    if (!g_cb) return;
    D3D11_MAPPED_SUBRESOURCE ms;
    if (SUCCEEDED(g_ctx->Map(g_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        NVConstantBuffer* cb = (NVConstantBuffer*)ms.pData;
        nv_camera_get_view_matrix(cb->view);
        float aspect = (g_nv_h > 0) ? (float)g_nv_w / g_nv_h : 1.0f;
        nv_camera_get_projection_matrix(cb->proj, aspect);
        /* 获取相机位置（从 camera 模块的静态状态） */
        const NVCamera* cam = nv_camera_get();
        float cp = cosf(cam->pitch), sp = sinf(cam->pitch);
        float cy2 = cosf(cam->yaw), sy = sinf(cam->yaw);
        cb->camera_pos[0] = cam->target_x + cam->distance * cp * sy;
        cb->camera_pos[1] = cam->target_y + cam->distance * sp;
        cb->camera_pos[2] = cam->target_z + cam->distance * cp * cy2;
        cb->camera_pos[3] = 1.0f;
        cb->params[0] = NV_NODE_BASE_SIZE;
        cb->params[1] = (float)GetTickCount() * 0.001f;
        g_ctx->Unmap(g_cb, 0);
    }
}

static int nv_count_visible_pixels_in_back_buffer(void);

static float nv_connection_intro_progress(uint32_t layer, uint32_t layer_count) {
    if (!g_nv_intro_active) return 1.0f;
    float delay = (float)layer / (float)(layer_count > 1 ? layer_count - 1 : 1) * 260.0f;
    float t = ((float)g_nv_intro_elapsed - NV_INTRO_NEURON_MS - delay) / 340.0f;
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    return t*t*(3-2*t);
}
int nv_render(void) {
    if (!g_nv_initialized) return -1;
    if (!g_dev || !g_ctx || !g_sc || !g_rtv || !g_dsv || !g_dss || !g_rs || !g_blend ||
        !g_cb || !g_vs || !g_ps || !g_il) return -2;

    static ULONGLONG last_camera_tick = 0;
    ULONGLONG camera_tick = GetTickCount64();
    float camera_dt = last_camera_tick ? (float)(camera_tick - last_camera_tick) * 0.001f : 1.0f / 60.0f;
    last_camera_tick = camera_tick;
    nv_camera_update(camera_dt);
    g_nv_intro_elapsed = GetTickCount() - g_nv_intro_start;
    /* 层间距变化时重新布局 */
    if (nv_spacing_changed()) {
        nv_layout_compute(&g_snap);
    }

    update_constant_buffer();

    /* 清屏：深色背景 */
    float bg[4] = { 0.05f, 0.05f, 0.08f, 1.0f };
    g_ctx->ClearRenderTargetView(g_rtv, bg);
    g_ctx->ClearDepthStencilView(g_dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    D3D11_VIEWPORT vp = { 0, 0, (float)g_nv_w, (float)g_nv_h, 0, 1 };
    g_ctx->RSSetViewports(1, &vp);
    g_ctx->RSSetState(g_rs);
    g_ctx->OMSetRenderTargets(1, &g_rtv, g_dsv);
    g_ctx->OMSetDepthStencilState(g_dss, 0);
    float blend_factor[4] = {};
    g_ctx->OMSetBlendState(g_blend, blend_factor, 0xFFFFFFFFu);

    uint32_t layer_count = g_snap.layer_count > NV_MAX_LAYERS ? NV_MAX_LAYERS : g_snap.layer_count;

    /* Draw a representative structural subset of adjacent-layer edges.
       Fully-connected layers may contain millions of weights; rendering all
       of them would hide the nodes and stall the TUI.  Half of the sampled
       structure changes with snapshot_id, while the other half stays stable
       as a visual reference. */
    g_nv_last_connection_segments = 0;
    g_nv_last_strong_segments = 0;
    if (layer_count > 1 && g_vs_line && g_ps_line && g_il_line) {
        /* Only the weight overview gets decorative structural sampling.
           Diagnostic modes must show exclusively measured signal, otherwise
           unchanged sample lines can be mistaken for gradients/contribution. */
        const uint32_t sampled_limit = g_mode == NV_MODE_WEIGHT && layer_count < 13 ?
            (layer_count - 1u) * 96u : 0u;
        const uint32_t connection_count = g_snap.connection_count > NV_MAX_CONNECTIONS ?
            NV_MAX_CONNECTIONS : g_snap.connection_count;
        const uint32_t max_segments = sampled_limit + connection_count * 3u;
        NVLineVertex* line_verts = (NVLineVertex*)malloc((size_t)max_segments * 2u * sizeof(NVLineVertex));
        if (line_verts) {
            uint32_t line_vertex_count = 0;
            for (uint32_t l = 0; l + 1 < layer_count; ++l) {
                NVLayer* src = &g_snap.layers[l];
                NVLayer* dst = &g_snap.layers[l + 1];
                const uint32_t src_n = src->neuron_count > NV_MAX_NEURONS ? NV_MAX_NEURONS : src->neuron_count;
                const uint32_t dst_n = dst->neuron_count > NV_MAX_NEURONS ? NV_MAX_NEURONS : dst->neuron_count;
                if (!src_n || !dst_n) continue;
                float growth = nv_connection_intro_progress(l, layer_count);
                if (growth <= 0) continue;
                uint32_t samples = src_n > dst_n ? src_n : dst_n;
                if (samples > 64u) samples = 64u;
                for (uint32_t s = 0; s < samples; ++s) {
                    const uint32_t si = (uint32_t)(((uint64_t)s * src_n) / samples);
                    const uint32_t di = (uint32_t)(((uint64_t)s * dst_n) / samples);
                    const uint64_t snapshot_seed = g_snap.snapshot_id * 0x9E3779B97F4A7C15ull;
                    uint32_t destinations[2] = { di, (uint32_t)(((uint64_t)s * 37u +
                        l * 17u + snapshot_seed) % dst_n) };
                    const uint32_t edge_count = (s & 1u) ? 1u : 2u;
                    for (uint32_t e = 0; e < edge_count && g_nv_last_connection_segments < sampled_limit; ++e) {
                        NVNeuron* a = &src->neurons[si];
                        NVNeuron* b = &dst->neurons[destinations[e]];
                        float activity = (a->activation + b->activation) * 0.5f;
                        if (!isfinite(activity)) activity = 0.0f;
                        if (activity < 0.0f) activity = 0.0f;
                        if (activity > 1.0f) activity = 1.0f;
                        const uint32_t color = nv_pack_rgba8(28u,
                            64u + (uint32_t)(activity * 48.0f),
                            88u + (uint32_t)(activity * 72.0f), 96u);
                        NVLineVertex* v = &line_verts[line_vertex_count];
                        v[0].pos[0] = a->x; v[0].pos[1] = a->y; v[0].pos[2] = a->z; v[0].color = color;
                        v[1].pos[0] = a->x+(b->x-a->x)*growth;
                        v[1].pos[1] = a->y+(b->y-a->y)*growth;
                        v[1].pos[2] = a->z+(b->z-a->z)*growth; v[1].color = color;
                        line_vertex_count += 2;
                        ++g_nv_last_connection_segments;
                    }
                }
            }
            float max_abs_weight = 1e-12f;
            for (uint32_t ci = 0; ci < connection_count; ++ci) {
                const float aw = fabsf(g_snap.connections[ci].weight);
                if (isfinite(aw) && aw > max_abs_weight) max_abs_weight = aw;
            }
            /* 用 snapshot 中已有的连线权重计算每个节点的连接强度 */
            for (uint32_t li = 0; li < layer_count; ++li) {
                NVLayer* ly = &g_snap.layers[li];
                for (uint32_t ni = 0; ni < ly->neuron_count; ++ni)
                    ly->neurons[ni].connection_strength = 0.0f;
            }
            /* 从连线数据累加（src 和 dst 都累加） */
            for (uint32_t ci = 0; ci < connection_count; ++ci) {
                const NVConnection* c = &g_snap.connections[ci];
                if (c->src_layer < layer_count && c->dst_layer < layer_count) {
                    NVLayer* src = &g_snap.layers[c->src_layer];
                    NVLayer* dst = &g_snap.layers[c->dst_layer];
                    if (c->src_neuron < src->neuron_count && c->dst_neuron < dst->neuron_count) {
                        float aw = fabsf(c->weight);
                        if (isfinite(aw) && aw > 0) {
                            src->neurons[c->src_neuron].connection_strength += aw;
                            dst->neurons[c->dst_neuron].connection_strength += aw;
                        }
                    }
                }
            }
            /* 如果连线数据不足，用 activation 作为 fallback */
            int has_any_strength = 0;
            for (uint32_t li = 0; li < layer_count && !has_any_strength; ++li)
                for (uint32_t ni = 0; ni < g_snap.layers[li].neuron_count; ++ni)
                    if (g_snap.layers[li].neurons[ni].connection_strength > 0) { has_any_strength = 1; break; }
            if (!has_any_strength) {
                /* Fallback: 用 activation 值作为连接强度代理 */
                for (uint32_t li = 0; li < layer_count; ++li) {
                    NVLayer* ly = &g_snap.layers[li];
                    for (uint32_t ni = 0; ni < ly->neuron_count; ++ni) {
                        float act = ly->neurons[ni].activation;
                        ly->neurons[ni].connection_strength = isfinite(act) ? fabsf(act) : 0.0f;
                    }
                }
            }
            /* 每层独立归一化到 0~1 */
            for (uint32_t li = 0; li < layer_count; ++li) {
                NVLayer* ly = &g_snap.layers[li];
                float max_cs = 1e-12f;
                for (uint32_t ni = 0; ni < ly->neuron_count; ++ni) {
                    float cs = ly->neurons[ni].connection_strength;
                    if (cs > max_cs) max_cs = cs;
                }
                for (uint32_t ni = 0; ni < ly->neuron_count; ++ni)
                    ly->neurons[ni].connection_strength /= max_cs;
            }

            /* Top-|weight| edges are bright/thick. */
            for (uint32_t ci = 0; ci < connection_count; ++ci) {
                const NVConnection* c = &g_snap.connections[ci];
                if (c->src_layer >= layer_count || c->dst_layer >= layer_count) continue;
                NVLayer* src = &g_snap.layers[c->src_layer];
                NVLayer* dst = &g_snap.layers[c->dst_layer];
                if (c->src_neuron >= src->neuron_count || c->dst_neuron >= dst->neuron_count) continue;
                const NVNeuron* a = &src->neurons[c->src_neuron];
                const NVNeuron* b = &dst->neurons[c->dst_neuron];
                float strength = fabsf(c->weight) / max_abs_weight;
                if (!isfinite(strength)) strength = 0.0f;
                if (strength < 0.0f) strength = 0.0f;
                if (strength > 1.0f) strength = 1.0f;
                const bool strong = c->color == 1;
                const bool structural = c->color == 2;
                const uint32_t alpha = strong ? 180u + (uint32_t)(strength * 75.0f) :
                    48u + (uint32_t)(strength * 112.0f);
                const uint32_t color = structural ? nv_pack_rgba8(120u,120u,120u,80u) : strong ? nv_pack_rgba8(70u, 220u, 255u, alpha) :
                    nv_pack_rgba8(32u, 132u, 178u, alpha);
                const int thick_begin = strong ? -1 : 0;
                const int thick_end = strong ? 1 : 0;
                /* 入场动画：连线从源端向目标端生长 */
                float bx = b->x, by = b->y, bz = b->z;
                if (g_nv_intro_active) {
                    float local_t = nv_connection_intro_progress(c->src_layer, layer_count);
                    if (local_t <= 0) continue;
                    bx = a->x + (b->x - a->x) * local_t;
                    by = a->y + (b->y - a->y) * local_t;
                    bz = a->z + (b->z - a->z) * local_t;
                }
                for (int thick = thick_begin; thick <= thick_end &&
                     g_nv_last_connection_segments < max_segments; ++thick) {
                    const float offset = (float)thick * 0.055f;
                    NVLineVertex* v = &line_verts[line_vertex_count];
                    v[0].pos[0] = a->x + offset; v[0].pos[1] = a->y; v[0].pos[2] = a->z; v[0].color = color;
                    v[1].pos[0] = bx + offset;   v[1].pos[1] = by;   v[1].pos[2] = bz;   v[1].color = color;
                    line_vertex_count += 2;
                    ++g_nv_last_connection_segments;
                    if (strong) ++g_nv_last_strong_segments;
                }
            }
            /* 扫描光点：按采样间隔移动（每次新快照到达时移动一步） */
            if (layer_count >= 2 && g_scan_active && !g_nv_intro_active) {
                float z_first = g_snap.layers[0].z_offset;
                float z_last = g_snap.layers[layer_count - 1].z_offset;
                float z_range = z_last - z_first;
                if (z_range < 0) z_range = -z_range;
                if (z_range > 0.1f) {
                    /* 线性插值：从 z_first 到 z_last，分 g_scan_steps_total 步 */
                    float t = (float)g_scan_step_current / (float)(g_scan_steps_total - 1);
                    g_scan_z = z_first + t * (z_last - z_first);
                    g_scan_step_current++;
                    if (g_scan_step_current >= g_scan_steps_total) {
                        g_scan_step_current = 0;   /* 循环 */
                    }
                }
            }

            /* 扫描光点：遍历高亮连接找交点，渲染为小正方体 */
            uint32_t scan_vertex_count = 0;
            const uint32_t kMaxScanPoints = 512;
            const uint32_t verts_per_cube = 36u;
            NVVertex* scan_verts = (NVVertex*)malloc(kMaxScanPoints * verts_per_cube * sizeof(NVVertex));
            if (scan_verts) {
                /* 正方体 6 面 × 2 三角 = 36 顶点 */
                const float sd = 0.12f;  /* 半边长，与 dot_half=0.15 接近 */
                static const float cube_verts[36][3] = {
                    /* 前 +Z */ {-sd,-sd, sd},{ sd,-sd, sd},{ sd, sd, sd},{-sd,-sd, sd},{ sd, sd, sd},{-sd, sd, sd},
                    /* 后 -Z */ { sd,-sd,-sd},{-sd,-sd,-sd},{-sd, sd,-sd},{ sd,-sd,-sd},{-sd, sd,-sd},{ sd, sd,-sd},
                    /* 左 -X */ {-sd,-sd,-sd},{-sd,-sd, sd},{-sd, sd, sd},{-sd,-sd,-sd},{-sd, sd, sd},{-sd, sd,-sd},
                    /* 右 +X */ { sd,-sd, sd},{ sd,-sd,-sd},{ sd, sd,-sd},{ sd,-sd, sd},{ sd, sd,-sd},{ sd, sd, sd},
                    /* 上 +Y */ {-sd, sd, sd},{ sd, sd, sd},{ sd, sd,-sd},{-sd, sd, sd},{ sd, sd,-sd},{-sd, sd,-sd},
                    /* 下 -Y */ {-sd,-sd,-sd},{ sd,-sd,-sd},{ sd,-sd, sd},{-sd,-sd,-sd},{ sd,-sd, sd},{-sd,-sd, sd},
                };
                for (uint32_t ci = 0; ci < connection_count && scan_vertex_count < kMaxScanPoints; ++ci) {
                    const NVConnection* c = &g_snap.connections[ci];
                    if (c->src_layer >= layer_count || c->dst_layer >= layer_count) continue;
                    const NVLayer* src = &g_snap.layers[c->src_layer];
                    const NVLayer* dst = &g_snap.layers[c->dst_layer];
                    if (c->src_neuron >= src->neuron_count || c->dst_neuron >= dst->neuron_count) continue;
                    const NVNeuron* a = &src->neurons[c->src_neuron];
                    const NVNeuron* b = &dst->neurons[c->dst_neuron];
                    float z1 = a->z, z2 = b->z;
                    float z_min = fminf(z1, z2), z_max = fmaxf(z1, z2);
                    if (g_scan_z < z_min || g_scan_z > z_max) continue;
                    if (fabsf(z2 - z1) < 0.001f) continue;
                    float t = (g_scan_z - z1) / (z2 - z1);
                    float ix = a->x + t * (b->x - a->x);
                    float iy = a->y + t * (b->y - a->y);
                    float iz = g_scan_z;
                    uint32_t dot_color = nv_pack_rgba8(180u, 255u, 255u, 240u);
                    for (uint32_t k = 0; k < verts_per_cube; ++k) {
                        NVVertex* v = &scan_verts[scan_vertex_count * verts_per_cube + k];
                        v->pos[0] = ix + cube_verts[k][0];
                        v->pos[1] = iy + cube_verts[k][1];
                        v->pos[2] = iz + cube_verts[k][2];
                        v->size = 0.1f;
                        v->color = dot_color;
                    }
                    scan_vertex_count++;
                }
            }

            if (line_vertex_count) {
                D3D11_BUFFER_DESC bd = {};
                bd.ByteWidth = line_vertex_count * sizeof(NVLineVertex);
                bd.Usage = D3D11_USAGE_IMMUTABLE;
                bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                D3D11_SUBRESOURCE_DATA sd = { line_verts, 0, 0 };
                ID3D11Buffer* vb = NULL;
                if (SUCCEEDED(g_dev->CreateBuffer(&bd, &sd, &vb))) {
                    UINT stride = sizeof(NVLineVertex), offset = 0;
                    g_ctx->IASetVertexBuffers(0, 1, &vb, &stride, &offset);
                    g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
                    g_ctx->IASetInputLayout(g_il_line);
                    g_ctx->VSSetShader(g_vs_line, NULL, 0);
                    g_ctx->VSSetConstantBuffers(0, 1, &g_cb);
                    g_ctx->PSSetShader(g_ps_line, NULL, 0);
                    g_ctx->Draw(line_vertex_count, 0);
                    vb->Release();
                }
            }
            free(line_verts);

            /* 渲染扫描光点 */
            if (scan_vertex_count > 0 && scan_verts) {
                D3D11_BUFFER_DESC sbd = {};
                sbd.ByteWidth = scan_vertex_count * verts_per_cube * sizeof(NVVertex);
                sbd.Usage = D3D11_USAGE_IMMUTABLE;
                sbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                D3D11_SUBRESOURCE_DATA ssd = { scan_verts, 0, 0 };
                ID3D11Buffer* svb = NULL;
                if (SUCCEEDED(g_dev->CreateBuffer(&sbd, &ssd, &svb))) {
                    UINT stride = sizeof(NVVertex), offset = 0;
                    g_ctx->IASetVertexBuffers(0, 1, &svb, &stride, &offset);
                    g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    g_ctx->IASetInputLayout(g_il);
                    g_ctx->VSSetShader(g_vs, NULL, 0);
                    g_ctx->VSSetConstantBuffers(0, 1, &g_cb);
                    g_ctx->PSSetShader(g_ps, NULL, 0);
                    g_ctx->Draw(scan_vertex_count * verts_per_cube, 0);
                    svb->Release();
                }
            }
            free(scan_verts);
        }
    }

    /* Draw nodes after edges so layer structure remains legible. */
    for (uint32_t l = 0; l < layer_count; l++) {
        NVLayer* layer = &g_snap.layers[l];
        uint32_t neuron_count = layer->neuron_count > NV_MAX_NEURONS ? NV_MAX_NEURONS : layer->neuron_count;
        if (neuron_count == 0) continue;

        /* 每个神经元画一个长方体（6面×2三角×3顶点=36顶点） */
        const uint32_t verts_per_neuron = 36u;
        const uint32_t vertex_count = neuron_count * verts_per_neuron;
        try { g_node_vertices.resize(vertex_count); } catch (...) { continue; }
        NVVertex* verts = g_node_vertices.data();
        /* Snapshot conversion already normalizes measured values to [0,1].
           A second per-layer min/max stretch made small differences appear as
           fully-off versus fully-on nodes (most visible on three token rows). */
        /* 长方体 6 面，每面 2 三角 = 6 顶点 */
        const float hx = 0.22f, hy = 0.22f, hz = 0.12f;
        static const float cube_faces[36][3] = {
            /* 前 +Z */  {-hx,-hy, hz}, { hx,-hy, hz}, { hx, hy, hz}, {-hx,-hy, hz}, { hx, hy, hz}, {-hx, hy, hz},
            /* 后 -Z */  { hx,-hy,-hz}, {-hx,-hy,-hz}, {-hx, hy,-hz}, { hx,-hy,-hz}, {-hx, hy,-hz}, { hx, hy,-hz},
            /* 左 -X */  {-hx,-hy,-hz}, {-hx,-hy, hz}, {-hx, hy, hz}, {-hx,-hy,-hz}, {-hx, hy, hz}, {-hx, hy,-hz},
            /* 右 +X */  { hx,-hy, hz}, { hx,-hy,-hz}, { hx, hy,-hz}, { hx,-hy, hz}, { hx, hy,-hz}, { hx, hy, hz},
            /* 上 +Y */  {-hx, hy, hz}, { hx, hy, hz}, { hx, hy,-hz}, {-hx, hy, hz}, { hx, hy,-hz}, {-hx, hy,-hz},
            /* 下 -Y */  {-hx,-hy,-hz}, { hx,-hy,-hz}, { hx,-hy, hz}, {-hx,-hy,-hz}, { hx,-hy, hz}, {-hx,-hy, hz},
        };
        for (uint32_t i = 0; i < neuron_count; i++) {
            NVNeuron* n = &layer->neurons[i];
            float act = n->activation;
            if (!isfinite(act)) act = 0.0f;
            if (act < 0.0f) act = 0.0f;
            if (act > 1.0f) act = 1.0f;
            const uint32_t alpha = 255u;
            uint32_t r, g, b;
            if (act > 0.7f) {
                r = (uint32_t)(150 + act * 105);
                g = (uint32_t)(230 + act * 25);
                b = 255u;
            } else if (act > 0.3f) {
                r = (uint32_t)(50 + act * 150);
                g = (uint32_t)(180 + act * 75);
                b = (uint32_t)(220 + act * 35);
            } else {
                r = (uint32_t)(20 + act * 100);
                g = (uint32_t)(100 + act * 250);
                b = (uint32_t)(140 + act * 300);
            }
            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;
            const uint32_t color = nv_pack_rgba8(r, g, b, alpha);
            /* 入场动画：计算当前位置（外围→最终位置） */
            float nx = n->x, ny = n->y, nz = n->z;
            if (g_nv_intro_active && l < (uint32_t)g_nv_intro_layer_count &&
                i < (uint32_t)g_nv_intro_neuron_count[l]) {
                float layer_delay = (float)l / (float)(g_nv_intro_layer_count > 1 ? g_nv_intro_layer_count - 1 : 1) * 160.0f;
                float node_delay = (float)((i * 37u) % 101u) / 100.0f * 60.0f;
                float local_t = ((float)g_nv_intro_elapsed - layer_delay - node_delay) / 580.0f;
                if (local_t < 0) local_t = 0;
                if (local_t > 1) local_t = 1;
                local_t = local_t * local_t * local_t * (local_t * (local_t * 6 - 15) + 10);
                float inv_t = 1.0f - local_t;
                nx = g_nv_intro_neuron_x[l][i] * inv_t + n->x * local_t;
                ny = g_nv_intro_neuron_y[l][i] * inv_t + n->y * local_t;
                nz = g_nv_intro_neuron_z[l][i] * inv_t + n->z * local_t;
            }
            for (uint32_t k = 0; k < 36; ++k) {
                NVVertex* v = &verts[(size_t)i * verts_per_neuron + k];
                v->pos[0] = nx + cube_faces[k][0];
                v->pos[1] = ny + cube_faces[k][1];
                v->pos[2] = nz + cube_faces[k][2];
                v->size = NV_NODE_BASE_SIZE;
                v->color = color;
            }
        }

        // Reuse a dynamic stream rather than create a GPU resource per layer/frame.
        if (!g_node_stream) {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = NV_MAX_NEURONS * 36u * sizeof(NVVertex);
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            g_dev->CreateBuffer(&bd, NULL, &g_node_stream);
        }
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (g_node_stream && SUCCEEDED(g_ctx->Map(g_node_stream, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            memcpy(mapped.pData, verts, vertex_count * sizeof(NVVertex));
            g_ctx->Unmap(g_node_stream, 0);
            UINT stride = sizeof(NVVertex), offset = 0;
            g_ctx->IASetVertexBuffers(0, 1, &g_node_stream, &stride, &offset);
            g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            g_ctx->IASetInputLayout(g_il);
            g_ctx->VSSetShader(g_vs, NULL, 0);
            g_ctx->VSSetConstantBuffers(0, 1, &g_cb);
            g_ctx->PSSetShader(g_ps, NULL, 0);
            g_ctx->Draw(vertex_count, 0);
        }
    }

    /* TODO Phase 3: Picking */

    nv_update_layer_labels(g_nv_intro_active && g_nv_intro_elapsed < NV_INTRO_NEURON_MS ? 0 : layer_count);

    /* 检查入场动画是否完成 → 启动运镜 */
    if (g_nv_intro_active) {
        DWORD elapsed = GetTickCount() - g_nv_intro_start;
        if (elapsed >= (DWORD)NV_INTRO_MS) {
            g_nv_intro_active = 0;
            /* 启动运镜：从当前视角飞到全景视角（输入层在右，输出在左，网络占窗口70%） */
            const NVCamera* cam = nv_camera_get();
            g_nv_cam_flyover_from_x = cam->target_x;
            g_nv_cam_flyover_from_y = cam->target_y;
            g_nv_cam_flyover_from_z = cam->target_z;
            g_nv_cam_flyover_from_dist = cam->distance;
            /* 目标：网络横在面前，输入层在左，输出层(动作层)在右，大幅放大 */
            float z_first = 0.0f, z_last = 0.0f;
            if (g_snap.layer_count >= 2) {
                z_first = g_snap.layers[0].z_offset;
                z_last = g_snap.layers[g_snap.layer_count - 1].z_offset;
            }
            float z_center = (z_first + z_last) * 0.5f;
            g_nv_cam_flyover_to_x = 0.0f;
            g_nv_cam_flyover_to_y = 0.0f;
            g_nv_cam_flyover_to_z = z_center;
            /* 距离：先算70%窗口的距离，再缩小约15次滚轮 */
            float z_span = fabsf(z_last - z_first);
            float fov_rad = cam->fov * 3.14159f / 180.0f;
            float network_visual_width = z_span + 8.0f;
            float dist_70 = network_visual_width / (2.0f * tanf(fov_rad * 0.5f) * 0.7f);
            g_nv_cam_flyover_to_dist = dist_70 * 0.37f;  /* 约9-10次滚轮缩放 */
            if (g_nv_cam_flyover_to_dist < 8.0f) g_nv_cam_flyover_to_dist = 8.0f;
            if (g_nv_cam_flyover_to_dist > 80.0f) g_nv_cam_flyover_to_dist = 80.0f;
            /* yaw=-90° 让输入层在左、输出层(动作层)在右，pitch=0 水平看 */
            g_nv_cam_flyover_to_yaw = -1.5708f;
            g_nv_cam_flyover_to_pitch = 0.0f;
            g_nv_cam_flyover_start = GetTickCount();
            g_nv_cam_flyover = nv_camera_input_revision() == g_nv_intro_input_revision;
        }
    }

    /* 运镜更新：直接设置相机 goal，让相机系统自己插值 */
    if (g_nv_cam_flyover) {
        // Submit once: subsequent manual input must own the camera immediately.
        g_nv_cam_flyover = 0;
        nv_camera_set_goal(g_nv_cam_flyover_to_x, g_nv_cam_flyover_to_y,
            g_nv_cam_flyover_to_z, g_nv_cam_flyover_to_dist,
            g_nv_cam_flyover_to_yaw, g_nv_cam_flyover_to_pitch);
    }

    if (g_nv_debug_capture_requested) {
        g_nv_debug_capture_requested = 0;
        g_nv_debug_visible_pixels = nv_count_visible_pixels_in_back_buffer();
    }

    HRESULT hr = g_sc->Present(1, 0);
    if (FAILED(hr)) {
        g_nv_last_error = hr;
        return -3;
    }
    return 0;
}

static int nv_count_visible_pixels_in_back_buffer(void) {
    if (!g_nv_initialized || !g_dev || !g_ctx || !g_sc) return -1;
    ID3D11Texture2D* back_buffer = NULL;
    HRESULT hr = g_sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
    if (FAILED(hr) || !back_buffer) return -2;

    D3D11_TEXTURE2D_DESC desc = {};
    back_buffer->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    ID3D11Texture2D* staging = NULL;
    hr = g_dev->CreateTexture2D(&desc, NULL, &staging);
    if (FAILED(hr) || !staging) {
        back_buffer->Release();
        return -3;
    }

    /* A resource cannot be copied reliably while still bound for output.
       Unbind and flush so the staging texture observes this exact frame. */
    g_ctx->OMSetRenderTargets(0, NULL, NULL);
    g_ctx->Flush();
    g_ctx->CopyResource(staging, back_buffer);
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    hr = g_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
    int visible = 0;
    int connection_pixels = 0;
    if (SUCCEEDED(hr)) {
        for (UINT y = 0; y < desc.Height; ++y) {
            const uint8_t* row = (const uint8_t*)mapped.pData + (size_t)y * mapped.RowPitch;
            for (UINT x = 0; x < desc.Width; ++x) {
                const uint8_t* pixel = row + (size_t)x * 4;
                /* Clear colour is approximately RGB(13,13,20).  A generous
                   threshold avoids treating conversion rounding as geometry. */
                if (pixel[0] > 32 || pixel[1] > 32 || pixel[2] > 40) ++visible;
                const bool sampled_line = pixel[0] >= 17 && pixel[0] <= 22 &&
                    pixel[1] >= 28 && pixel[1] <= 55 &&
                    pixel[2] >= 40 && pixel[2] <= 80;
                const bool strong_line = pixel[0] >= 50 && pixel[0] <= 80 &&
                    pixel[1] >= 160 && pixel[2] >= 190;
                if (sampled_line || strong_line) ++connection_pixels;
            }
        }
        g_ctx->Unmap(staging, 0);
    } else {
        visible = -4;
    }
    staging->Release();
    back_buffer->Release();
    g_nv_debug_connection_pixels = connection_pixels;
    return visible;
}

static void nv_debug_dump_messages(void) {
    if (!g_dev) return;
    ID3D11InfoQueue* queue = NULL;
    if (FAILED(g_dev->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&queue)) || !queue) return;
    const UINT64 count = queue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T length = 0;
        if (FAILED(queue->GetMessage(i, NULL, &length)) || !length) continue;
        D3D11_MESSAGE* msg = (D3D11_MESSAGE*)malloc(length);
        if (!msg) break;
        if (SUCCEEDED(queue->GetMessage(i, msg, &length)))
            printf("[NV3D DEBUG] %s\n", msg->pDescription ? msg->pDescription : "(no description)");
        free(msg);
    }
    queue->ClearStoredMessages();
    queue->Release();
}

int nv_debug_visible_pixel_count(void) {
    if (!g_nv_initialized) return -1;
    g_nv_debug_visible_pixels = -1;
    g_nv_debug_capture_requested = 1;
    const int render_rc = nv_render();
    if (render_rc != 0) return -10 + render_rc;
    if (g_nv_debug_visible_pixels <= 0) nv_debug_dump_messages();
    return g_nv_debug_visible_pixels;
}

uint32_t nv_debug_last_connection_segment_count(void) {
    return g_nv_last_connection_segments;
}

uint32_t nv_debug_last_strong_connection_segment_count(void) {
    return g_nv_last_strong_segments;
}

uint32_t nv_debug_layer_label_count(void) {
    return g_nv_label_count;
}

int nv_debug_connection_pixel_count(void) {
    return g_nv_debug_connection_pixels;
}

int nv_debug_get_layer_grid(uint32_t layer, uint32_t* out_w, uint32_t* out_h, uint32_t* out_count) {
    if (layer >= g_snap.layer_count || !out_w || !out_h || !out_count) return -1;
    *out_w = g_snap.layers[layer].grid_w;
    *out_h = g_snap.layers[layer].grid_h;
    *out_count = g_snap.layers[layer].neuron_count;
    return 0;
}
