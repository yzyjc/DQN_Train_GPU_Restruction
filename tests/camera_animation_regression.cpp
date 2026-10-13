#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "../DQN_Main_TUI_No_Terminal/neural_viewer_camera.cpp"
int main() {
    nv_camera_reset(); nv_camera_update(2.0f);
    float initial = nv_camera_get()->distance;
    nv_on_mouse_wheel(120, 0, 0);
    assert(nv_camera_get()->distance == initial);
    nv_camera_update(0.016f);
    float middle = nv_camera_get()->distance;
    assert(middle < initial && middle > initial * 0.9f);
    nv_on_mouse_wheel(-120, 0, 0);
    assert(nv_camera_get()->distance == middle);
    nv_camera_update(0.016f);
    assert(nv_camera_get()->distance > middle);
    nv_camera_focus_layer(5);
    nv_camera_update(0.040f);
    float moving_z = nv_camera_get()->target_z;
    nv_camera_reset();
    assert(nv_camera_get()->target_z == moving_z);
    nv_camera_update(2.0f);
    assert(nv_camera_get()->target_z == -16.0f);
    nv_on_mouse_button(0, 1, 0, 0);
    nv_on_mouse_move(100, 100, 1);
    float before = nv_camera_get()->yaw;
    nv_camera_update(0.016f);
    assert(nv_camera_get()->yaw > before);
    nv_on_mouse_button(0, 0, 100, 100);
    nv_camera_update(2.0f);
    assert(fabsf(nv_camera_get()->yaw - 0.8f) < 0.001f);
    // The same elapsed time converges identically at different frame rates.
    nv_camera_reset(); nv_camera_update(2.0f); nv_camera_focus_layer(5);
    for (int i = 0; i < 30; ++i) nv_camera_update(1.0f / 60);
    float sixty = nv_camera_get()->target_z;
    nv_camera_reset(); nv_camera_update(2.0f); nv_camera_focus_layer(5);
    for (int i = 0; i < 15; ++i) nv_camera_update(1.0f / 30);
    assert(fabsf(sixty - nv_camera_get()->target_z) < 0.0001f);
    g_spacing_goal = 16.0f;
    nv_camera_update(0.016f);
    assert(nv_get_layer_spacing() > 8 && nv_get_layer_spacing() < 16);
    assert(nv_spacing_changed());
    nv_camera_update(2.0f);
    assert(nv_get_layer_spacing() == 16);
    puts("PASS: zoom reversal, focus/reset interruption, drag, frame-rate independence, spacing");
}
