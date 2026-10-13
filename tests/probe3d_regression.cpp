#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <memory>
#include "../DQN_Main_TUI_No_Terminal/neural_viewer_3d.h"
#include "../DQN_Core/include/neural_probe_protocol.h"

extern "C" {
int probe_client_connect(void);
void probe_client_disconnect(void);
const NeuralProbeHeader* probe_get_header(void);
const ProbeGlobalState* probe_get_global_state(void);
const float* probe_get_activations(uint32_t layer_index);
const float* probe_get_gradients(uint32_t layer_index);
const float* probe_get_input_activations(uint32_t layer_index, uint64_t* out_count);
}

static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static LRESULT CALLBACK parent_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void row_vec_mul(const float in[4], const float m[16], float out[4]) {
    for (int j = 0; j < 4; ++j) {
        out[j] = 0.0f;
        for (int i = 0; i < 4; ++i) out[j] += in[i] * m[i * 4 + j];
    }
}

static void camera_projection_test(void) {
    nv_camera_reset();
    nv_camera_update(2.0f);
    float view[16], proj[16], world[4] = { 0, 0, -16, 1 }, eye_space[4], clip[4];
    nv_camera_get_view_matrix(view);
    nv_camera_get_projection_matrix(proj, 4.0f / 3.0f);
    row_vec_mul(world, view, eye_space);
    row_vec_mul(eye_space, proj, clip);
    printf("Camera target -> eye=(%.3f,%.3f,%.3f,%.3f), clip=(%.3f,%.3f,%.3f,%.3f)\n",
        eye_space[0], eye_space[1], eye_space[2], eye_space[3],
        clip[0], clip[1], clip[2], clip[3]);
    CHECK(clip[3] > 0.0f);
    CHECK(clip[2] >= 0.0f && clip[2] <= clip[3]);
    CHECK(fabsf(clip[0] / clip[3]) < 0.01f && fabsf(clip[1] / clip[3]) < 0.01f);

    const NVCamera before = *nv_camera_get();
    nv_on_mouse_button(1, 1, 10, 10); /* middle drag pans */
    nv_on_mouse_move(35, 24, 0);
    nv_on_mouse_button(1, 0, 35, 24);
    nv_camera_update(1.0f / 60.0f);
    const NVCamera after = *nv_camera_get();
    CHECK(before.target_x != after.target_x || before.target_y != after.target_y || before.target_z != after.target_z);
    nv_camera_reset();
    nv_camera_update(2.0f);
}

static void probe_snapshot_zero_id_test(void) {
    const uint64_t topology_size = sizeof(ProbeLayerDesc);
    const uint64_t diagnostic_bytes = 4 * sizeof(float);
    const uint64_t buffer_size = diagnostic_bytes + sizeof(ProbeGlobalState);
    const uint64_t total_size = sizeof(NeuralProbeHeader) + topology_size + 2 * buffer_size;
    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        0, (DWORD)total_size, NEURAL_PROBE_NAME);
    const DWORD create_error = GetLastError();
    CHECK(mapping != NULL);
    if (!mapping) return;
    if (create_error == ERROR_ALREADY_EXISTS) {
        printf("SKIP: a live Trainer owns %s; shared-memory fixture was not touched.\n", NEURAL_PROBE_NAME);
        CloseHandle(mapping);
        return;
    }
    unsigned char* base = (unsigned char*)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    CHECK(base != NULL);
    if (!base) { CloseHandle(mapping); return; }
    memset(base, 0, (size_t)total_size);

    NeuralProbeHeader* hdr = (NeuralProbeHeader*)base;
    hdr->magic = NEURAL_PROBE_MAGIC;
    hdr->version = NEURAL_PROBE_VERSION;
    hdr->total_size = total_size;
    hdr->snapshot_id = 0; /* first initialized frame is legitimately id zero */
    hdr->episode = 42;
    hdr->layer_count = 1;
    hdr->active_buffer = 0;
    hdr->topology_offset = sizeof(*hdr);
    hdr->topology_size = topology_size;
    hdr->buffer0_offset = sizeof(*hdr) + topology_size;
    hdr->buffer1_offset = hdr->buffer0_offset + buffer_size;
    hdr->buffer_size = buffer_size;
    hdr->probe_enabled = 1;
    hdr->publish_sequence = 0;

    ProbeLayerDesc* layer = (ProbeLayerDesc*)(base + hdr->topology_offset);
    layer->layer_id = 0;
    layer->type = PROBE_LAYER_INPUT;
    strcpy_s(layer->name, "Input");
    layer->input_size = layer->output_size = 25;
    layer->activation_offset = 0;
    layer->activation_count = 1;
    layer->gradient_offset = sizeof(float);
    layer->gradient_count = 1;
    layer->_reserved[PROBE_LAYER_INPUT_ACTIVATION_OFFSET] = 2 * sizeof(float);
    layer->_reserved[PROBE_LAYER_INPUT_ACTIVATION_COUNT] = 2;
    float* diagnostic = (float*)(base + hdr->buffer0_offset);
    diagnostic[0] = 0.25f;
    diagnostic[1] = -0.5f;
    diagnostic[2] = 0.75f;
    diagnostic[3] = -1.0f;
    ProbeGlobalState* state = (ProbeGlobalState*)(base + hdr->buffer0_offset + diagnostic_bytes);
    state->episode = 42;
    state->epsilon = 0.75f;
    state->diagnostics_flags = PROBE_DIAG_FORWARD_VALID | PROBE_DIAG_GRADIENT_VALID;

    CHECK(probe_client_connect() == 0);
    const NeuralProbeHeader* local = probe_get_header();
    const ProbeGlobalState* local_state = probe_get_global_state();
    CHECK(local != NULL && local->snapshot_id == 0 && local->episode == 42 && local->layer_count == 1);
    CHECK(local_state != NULL && local_state->episode == 42 && local_state->epsilon == 0.75f);
    const float* activation = probe_get_activations(0);
    const float* gradient = probe_get_gradients(0);
    uint64_t input_count = 0;
    const float* input_activation = probe_get_input_activations(0, &input_count);
    CHECK(activation && activation[0] == 0.25f);
    CHECK(gradient && gradient[0] == -0.5f);
    CHECK(input_activation && input_count == 2 && input_activation[0] == 0.75f && input_activation[1] == -1.0f);
    CHECK(hdr->_reserved[PROBE_RESERVED_VIEWER_HEARTBEAT_MS] != 0);
    probe_client_disconnect();
    UnmapViewOfFile(base);
    CloseHandle(mapping);
}

static void live_probe_compatibility_test(void) {
    const int rc = probe_client_connect();
    if (rc != 0) {
        printf("SKIP: no live Probe publisher for compatibility check (%d).\n", rc);
        return;
    }
    const NeuralProbeHeader* hdr = probe_get_header();
    const ProbeGlobalState* state = probe_get_global_state();
    CHECK(hdr != NULL && (hdr->version == 2 || hdr->version == NEURAL_PROBE_VERSION));
    CHECK(state != NULL);
    if (hdr)
        printf("Live Probe compatibility: v%u, snapshot %llu, episode %llu\n",
            hdr->version, (unsigned long long)hdr->snapshot_id,
            (unsigned long long)hdr->episode);
    probe_client_disconnect();
}

static void viewer_lifecycle_test(void) {
    static_assert(sizeof(NVSnapshot) > 1024 * 1024,
        "The regression must exercise a snapshot larger than the default 1 MiB stack");
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = parent_wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"Probe3DRegressionParent";
    CHECK(RegisterClassExW(&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS);
    HWND parent = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW,
        0, 0, 640, 480, NULL, NULL, wc.hInstance, NULL);
    CHECK(parent != NULL);
    if (!parent) return;

    CHECK(nv_init(parent, 0, 0, 320, 240) == 0);
    std::unique_ptr<NVSnapshot> snap(new (std::nothrow) NVSnapshot{});
    CHECK(snap != NULL);
    if (snap) {
        snap->layer_count = 11;
        snap->layers[0].type = NV_LAYER_INPUT;
        snap->layers[0].neuron_count = 25;
        snap->layers[0].output_size = 25;
        snap->layers[1].type = NV_LAYER_ATTENTION;
        snap->layers[1].neuron_count = 16;
        snap->layers[1].output_size = 32;
        snap->layers[2].type = NV_LAYER_LINEAR;
        snap->layers[2].neuron_count = 1024;
        snap->layers[2].output_size = 1024;
        snap->layers[3].type = NV_LAYER_LINEAR;
        snap->layers[3].neuron_count = 512;
        snap->layers[3].output_size = 512;
        snap->layers[4].type = NV_LAYER_OUTPUT;
        snap->layers[4].neuron_count = 25;
        snap->layers[4].output_size = 25;
        snap->layers[5].type = NV_LAYER_ATTENTION;
        snap->layers[5].neuron_count = 16;
        snap->layers[6].type = NV_LAYER_POOLING;
        snap->layers[6].neuron_count = 128;
        snap->layers[7].type = NV_LAYER_LINEAR;
        snap->layers[7].neuron_count = 1024;
        snap->layers[8].type = NV_LAYER_LINEAR;
        snap->layers[8].neuron_count = 512;
        snap->layers[9].type = NV_LAYER_LINEAR;
        snap->layers[9].neuron_count = 256;
        snap->layers[10].type = NV_LAYER_OUTPUT;
        snap->layers[10].neuron_count = 25;
        snap->connection_count = 2;
        snap->connections[0] = { 0, 0, 1, 0, 0.9f, 1 };
        snap->connections[1] = { 1, 1, 2, 4, -0.8f, 1 };
        nv_set_snapshot(snap.get());
        CHECK(nv_render() == 0);
        CHECK(nv_debug_last_connection_segment_count() == 0);
        // Manual input during entry must cancel the later automatic flyover.
        nv_on_mouse_wheel(120, 0, 0);
        Sleep(1500); // Exercise the settled real renderer after the timed entry.
        uint32_t gw = 0, gh = 0, gn = 0;
        CHECK(nv_debug_get_layer_grid(0, &gw, &gh, &gn) == 0 && gw == 5 && gh == 5 && gn == 25);
        CHECK(nv_debug_get_layer_grid(1, &gw, &gh, &gn) == 0 && gw == 4 && gh == 4 && gn == 16);
        CHECK(nv_debug_get_layer_grid(2, &gw, &gh, &gn) == 0 && gw == 32 && gh == 32 && gn == 1024);
        CHECK(nv_debug_get_layer_grid(3, &gw, &gh, &gn) == 0 && gw == 32 && gh == 16 && gn == 512);
        nv_camera_update(2.0f);
        CHECK(fabsf(nv_camera_get()->target_z + 40.0f) < 0.01f);
        CHECK(nv_render() == 0);
        CHECK(nv_debug_last_connection_segment_count() > 0);
        CHECK(nv_debug_last_strong_connection_segment_count() == 6);
        CHECK(nv_debug_layer_label_count() >= 3);
        int visible = nv_debug_visible_pixel_count();
        printf("Visible pixels (320x240): %d\n", visible);
        CHECK(visible > 0);
        printf("Connection pixels: %d, segments: %u (strong %u), labels: %u\n",
            nv_debug_connection_pixel_count(), nv_debug_last_connection_segment_count(),
            nv_debug_last_strong_connection_segment_count(), nv_debug_layer_label_count());
        CHECK(nv_debug_connection_pixel_count() > 0);
        const int first_connection_pixels = nv_debug_connection_pixel_count();
        nv_set_display_mode(NV_MODE_ATTENTION);
        snap->snapshot_id++;
        nv_set_snapshot(snap.get());
        CHECK(nv_render() == 0);
        CHECK(nv_debug_last_connection_segment_count() == 6);
        CHECK(nv_debug_last_strong_connection_segment_count() == 6);
        CHECK(nv_debug_connection_pixel_count() > 0);
        nv_set_display_mode(NV_MODE_WEIGHT);
        snap->snapshot_id++;
        nv_set_snapshot(snap.get());
        CHECK(nv_debug_visible_pixel_count() > 0);
        const int next_connection_pixels = nv_debug_connection_pixel_count();
        printf("Connection refresh: snapshot pixels %d -> %d\n",
            first_connection_pixels, next_connection_pixels);
        CHECK(next_connection_pixels > 0 && next_connection_pixels != first_connection_pixels);
        nv_resize(400, 260);
        CHECK(nv_render() == 0);
        visible = nv_debug_visible_pixel_count();
        printf("Visible pixels (400x260): %d\n", visible);
        CHECK(visible > 0);

        /* TokenEmbedding now exposes its 16 feature dimensions as a 4x4
           first layer, instead of treating three vocabulary rows as neurons. */
        snap->layers[0].neuron_count = 16;
        snap->layers[0].output_size = 16;
        snap->snapshot_id++;
        nv_set_snapshot(snap.get());
        CHECK(nv_debug_get_layer_grid(0, &gw, &gh, &gn) == 0 && gw == 4 && gh == 4 && gn == 16);
        CHECK(nv_render() == 0);
    }
    nv_shutdown();
    nv_shutdown(); /* shutdown and partial-init cleanup are deliberately idempotent */
    DestroyWindow(parent);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
}

int main(void) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetEnvironmentVariableA("DQN_NV3D_DEBUG", "1");
    camera_projection_test();
    live_probe_compatibility_test();
    probe_snapshot_zero_id_test();
    viewer_lifecycle_test();
    printf("PROBE 3D REGRESSION: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
