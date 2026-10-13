/* neural_viewer_camera.cpp — 轨道相机
   左键拖拽旋转，中键/右键拖拽平移，滚轮缩放，R 键重置 */
#include "neural_viewer_3d.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 相机状态 */
static NVCamera g_cam = {
    0.0f, 0.0f, -16.0f,   /* target: 网络中心 */
    50.0f,                  /* distance */
    0.3f, -0.3f,           /* yaw, pitch */
    60.0f,                  /* fov */
    0.1f, 500.0f           /* near, far */
};
static NVCamera g_cam_goal = g_cam;
static float g_camera_response = 16.0f;
static unsigned g_camera_input_revision = 0;
unsigned nv_camera_input_revision(void) { return g_camera_input_revision; }
static float g_key_dt = 1.0f / 60.0f;
static float g_spacing_goal = NV_LAYER_SPACING;
static float g_fc_spacing_goal = NV_FC_LAYER_SPACING;
static float g_reset_target_z = -16.0f;
static float g_reset_distance = 50.0f;

/* 鼠标状态 */
static int g_cam_dragging = 0;   /* 0=无, 1=旋转, 2=平移 */
static int g_cam_last_x = 0, g_cam_last_y = 0;

/* ========== 内部函数 ========== */

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* 从球坐标计算相机位置 */
static void cam_get_position(float* out_x, float* out_y, float* out_z) {
    float cp = cosf(g_cam.pitch), sp = sinf(g_cam.pitch);
    float cy = cosf(g_cam.yaw), sy = sinf(g_cam.yaw);
    *out_x = g_cam.target_x + g_cam.distance * cp * sy;
    *out_y = g_cam.target_y + g_cam.distance * sp;
    *out_z = g_cam.target_z + g_cam.distance * cp * cy;
}

/* ========== 公共 API ========== */

void nv_camera_reset(void) {
    g_camera_response = 12.0f;
    g_cam_goal.target_x = 0.0f;
    g_cam_goal.target_y = 0.0f;
    g_cam_goal.target_z = g_reset_target_z;
    g_cam_goal.distance = g_reset_distance;
    g_cam_goal.yaw = 0.3f;
    g_cam_goal.pitch = -0.3f;
    g_cam_goal.fov = 60.0f;
}

void nv_camera_set_goal(float tx, float ty, float tz, float dist, float yaw, float pitch) {
    g_camera_response = 8.0f;
    g_cam_goal.target_x = tx;
    g_cam_goal.target_y = ty;
    g_cam_goal.target_z = tz;
    g_cam_goal.distance = dist;
    g_cam_goal.yaw = yaw;
    g_cam_goal.pitch = pitch;
}

void nv_camera_fit_network(uint32_t layer_count, float max_half_extent) {
    if (layer_count < 1) layer_count = 1;
    const float depth_span = (float)(layer_count - 1u) * NV_LAYER_SPACING;
    g_reset_target_z = -depth_span * 0.5f;
    g_reset_distance = depth_span * 1.15f + max_half_extent * 2.0f + 10.0f;
    if (g_reset_distance < 50.0f) g_reset_distance = 50.0f;
    if (g_reset_distance > 200.0f) g_reset_distance = 200.0f;
    nv_camera_reset();
}

void nv_camera_focus_layer(int layer_idx) {
    g_camera_response = 12.0f;
    g_cam_goal.target_z = -(float)layer_idx * NV_LAYER_SPACING;
    g_cam_goal.distance = 30.0f;
}

const NVCamera* nv_camera_get(void) {
    return &g_cam;
}

void nv_camera_get_view_matrix(float* out_4x4) {
    float ex, ey, ez;
    cam_get_position(&ex, &ey, &ez);

    float fx = g_cam.target_x - ex;
    float fy = g_cam.target_y - ey;
    float fz = g_cam.target_z - ez;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz);
    if (fl < 1e-6f) fl = 1.0f;
    fx /= fl; fy /= fl; fz /= fl;

    /* right = normalize(cross(forward, world_up)).  The old expansion used
       fy for X and zero for both other components, producing a degenerate
       camera basis regardless of yaw. */
    float rx = -fz;
    float ry = 0.0f;
    float rz = fx;
    float rl = sqrtf(rx*rx + ry*ry + rz*rz);
    if (rl < 1e-6f) rl = 1.0f;
    rx /= rl; ry /= rl; rz /= rl;

    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    out_4x4[0]  = rx;  out_4x4[1]  = ux;  out_4x4[2]  = -fx; out_4x4[3]  = 0;
    out_4x4[4]  = ry;  out_4x4[5]  = uy;  out_4x4[6]  = -fy; out_4x4[7]  = 0;
    out_4x4[8]  = rz;  out_4x4[9]  = uz;  out_4x4[10] = -fz; out_4x4[11] = 0;
    out_4x4[12] = -(rx*ex + ry*ey + rz*ez);
    out_4x4[13] = -(ux*ex + uy*ey + uz*ez);
    out_4x4[14] = -(-fx*ex + -fy*ey + -fz*ez);
    out_4x4[15] = 1.0f;
}

void nv_camera_get_projection_matrix(float* out_4x4, float aspect) {
    float fov_rad = g_cam.fov * (float)M_PI / 180.0f;
    float f = 1.0f / tanf(fov_rad * 0.5f);
    float n = g_cam.near_plane, fp = g_cam.far_plane;
    float range_inv = 1.0f / (n - fp);

    for (int i = 0; i < 16; i++) out_4x4[i] = 0;
    out_4x4[0]  = f / aspect;
    out_4x4[5]  = f;
    /* Right-handed Direct3D depth range [0,1], row-vector form. */
    out_4x4[10] = fp * range_inv;
    out_4x4[11] = -1.0f;
    out_4x4[14] = fp * n * range_inv;
}

/* ========== 输入处理 ========== */

void nv_on_mouse_button(int button, int pressed, int x, int y) {
    (void)x; (void)y;
    if (pressed) {
        ++g_camera_input_revision;
        g_cam_goal = g_cam; // Grabbing the view interrupts any focus/reset flight.
        g_camera_response = 32.0f;
        g_cam_dragging = (button == 0) ? 1 : (button == 1 || button == 2) ? 2 : 0;
        g_cam_last_x = x;
        g_cam_last_y = y;
    } else {
        g_cam_dragging = 0;
    }
}

/* 计算相机局部坐标系的 right 和 up 向量 */
static void cam_get_local_axes(float* rx, float* rz, float* ux, float* uy, float* uz) {
    float cy = cosf(g_cam.yaw), sy = sinf(g_cam.yaw);
    float cp = cosf(g_cam.pitch), sp = sinf(g_cam.pitch);
    /* forward = (cp*sy, sp, cp*cy) */
    float fx = cp * sy, fy = sp, fz = cp * cy;
    /* right = normalize(cross(world_up, forward)) = (fz, 0, -fx) */
    *rx = fz; *rz = -fx;
    float rl = sqrtf((*rx) * (*rx) + (*rz) * (*rz));
    if (rl < 1e-6f) { *rx = 1.0f; *rz = 0.0f; } else { *rx /= rl; *rz /= rl; }
    /* up = cross(right, forward), right=(rx,0,rz) */
    *ux = 0.0f * fz - (*rz) * fy;
    *uy = (*rz) * fx - (*rx) * fz;
    *uz = (*rx) * fy - 0.0f * fx;
}

void nv_on_mouse_move(int x, int y, int buttons) {
    (void)buttons;
    if (!g_cam_dragging) return;

    int dx = x - g_cam_last_x;
    int dy = y - g_cam_last_y;
    g_cam_last_x = x;
    g_cam_last_y = y;

    if (g_cam_dragging == 1) {
        g_cam_goal.yaw   += dx * 0.005f;
        g_cam_goal.pitch  -= dy * 0.005f;
        g_cam_goal.pitch = clampf(g_cam_goal.pitch, -1.5f, 1.5f);
    } else if (g_cam_dragging == 2) {
        /* 相机局部坐标系平移：right 控制水平，up 控制垂直 */
        float rx, rz, ux, uy, uz;
        cam_get_local_axes(&rx, &rz, &ux, &uy, &uz);
        float scale = g_cam_goal.distance * 0.002f;
        g_cam_goal.target_x -= (dx * rx + dy * ux) * scale;
        g_cam_goal.target_y -= (dy * uy) * scale;
        g_cam_goal.target_z -= (dx * rz + dy * uz) * scale;
    }
}

/* 层间距可调（Shift+滚轮） */
static float g_nv_layer_spacing = NV_LAYER_SPACING;
static float g_nv_fc_layer_spacing = NV_FC_LAYER_SPACING;

float nv_get_layer_spacing(void) { return g_nv_layer_spacing; }
float nv_get_fc_layer_spacing(void) { return g_nv_fc_layer_spacing; }

static int g_spacing_changed = 0;   /* 间距变化标志，主循环检测后重新布局 */
int nv_spacing_changed(void) { int r = g_spacing_changed; g_spacing_changed = 0; return r; }

void nv_on_mouse_wheel(int delta, int x, int y) {
    if (!delta) return;
    ++g_camera_input_revision;
    (void)x; (void)y;
    /* Shift+滚轮：调整层间距 */
    if (GetKeyState(VK_SHIFT) & 0x8000) {
        float factor = (delta > 0) ? 0.9f : 1.1f;
        g_spacing_goal *= factor;
        g_fc_spacing_goal *= factor;
        g_spacing_goal = clampf(g_spacing_goal, 2.0f, 30.0f);
        g_fc_spacing_goal = clampf(g_fc_spacing_goal, 4.0f, 50.0f);
        g_spacing_changed = 1;
        return;
    }
    /* 普通滚轮：缩放 */
    g_camera_response = 20.0f;
    g_cam_goal.distance *= (delta > 0) ? 0.9f : 1.1f;
    g_cam_goal.distance = clampf(g_cam_goal.distance, 5.0f, 200.0f);
}

void nv_on_key(int key, int pressed) {
    if (!pressed) return;
    ++g_camera_input_revision;
    if (key == 'R' || key == 'r') { nv_camera_reset(); return; }
    /* WASD + Space/Shift：相机局部坐标系移动 */
    float rx, rz, ux, uy, uz;
    cam_get_local_axes(&rx, &rz, &ux, &uy, &uz);
    float cy = cosf(g_cam_goal.yaw), sy = sinf(g_cam_goal.yaw);
    float cp = cosf(g_cam_goal.pitch), sp = sinf(g_cam_goal.pitch);
    float fx = cp * sy, fz = cp * cy;   /* forward 水平分量 */
    g_camera_response = 24.0f;
    float speed = g_cam_goal.distance * 1.5f * g_key_dt;
    /* W/S：target 沿相机视线反方向移动（W=相机往场景深处走） */
    if (key == 'W' || key == 'w') { g_cam_goal.target_x -= fx * speed; g_cam_goal.target_z -= fz * speed; }
    if (key == 'S' || key == 's') { g_cam_goal.target_x += fx * speed; g_cam_goal.target_z += fz * speed; }
    if (key == 'A' || key == 'a') { g_cam_goal.target_x -= rx * speed; g_cam_goal.target_z -= rz * speed; }
    if (key == 'D' || key == 'd') { g_cam_goal.target_x += rx * speed; g_cam_goal.target_z += rz * speed; }
    /* Space 上升，左 Shift 下降 */
    if (key == VK_SPACE) g_cam_goal.target_y += speed;
    if (key == VK_LSHIFT) g_cam_goal.target_y -= speed;
}

// Time-based damping preserves the current position when the target changes.
// No queued animation and no overshoot; long pauses converge directly.
static bool camera_approach(float& value, float goal, float alpha) {
    float before = value;
    value += (goal - value) * alpha;
    if (fabsf(goal - value) < 0.0001f) value = goal;
    return before != value;
}
void nv_camera_update(float dt) {
    if (!isfinite(dt) || dt <= 0) return;
    g_key_dt = clampf(dt, 0.001f, 0.05f);
    float alpha = 1.0f - expf(-g_camera_response * dt);
    camera_approach(g_cam.target_x, g_cam_goal.target_x, alpha);
    camera_approach(g_cam.target_y, g_cam_goal.target_y, alpha);
    camera_approach(g_cam.target_z, g_cam_goal.target_z, alpha);
    camera_approach(g_cam.distance, g_cam_goal.distance, alpha);
    camera_approach(g_cam.yaw, g_cam_goal.yaw, alpha);
    camera_approach(g_cam.pitch, g_cam_goal.pitch, alpha);
    camera_approach(g_cam.fov, g_cam_goal.fov, alpha);
    float spacing_alpha = 1.0f - expf(-14.0f * dt);
    bool changed = camera_approach(g_nv_layer_spacing, g_spacing_goal, spacing_alpha);
    changed |= camera_approach(g_nv_fc_layer_spacing, g_fc_spacing_goal, spacing_alpha);
    if (changed) g_spacing_changed = 1;
}
