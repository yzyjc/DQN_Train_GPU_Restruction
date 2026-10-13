#include <windows.h>
#include <assert.h>
#include <stdio.h>
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#include "../DQN_Main_TUI_No_Terminal/terminal_window.cpp"
int g_running = 1;
int g_skip_scene_crossfade = 0;
int main() {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"TUIAnimationRegression";
    assert(RegisterClassW(&wc));
    g_hwnd = CreateWindowW(wc.lpszClassName, L"Animation regression", WS_OVERLAPPEDWINDOW,
        0, 0, 640, 480, NULL, NULL, wc.hInstance, NULL);
    assert(g_hwnd);
    vt_init(50, 20);
    win_register_renderer();
    assert(g_rt);
    win_set_scene(0);
    vt_printf("First page");
    vt_present();
    assert(g_scene_presented && !g_scene_from);
    win_set_scene(1);
    assert(g_scene_from); // Hwnd target supports the actual GPU snapshot copy.
    vt_clear_screen(); vt_printf("Second page"); vt_present();
    assert(g_scene_from && vt_has_dirty());
    g_scene_started -= 80;
    vt_present();
    win_set_scene(2); // Capture the unfinished blend rather than the old page.
    assert(g_scene_from);
    ULONGLONG started = g_scene_started;
    win_set_scene(2);
    assert(g_scene_started == started);
    vt_clear_screen(); vt_printf("Third page"); vt_present();
    g_scene_started -= 250;
    vt_present();
    assert(!g_scene_from);
    win_set_scene(3); assert(g_scene_from);
    SendMessageW(g_hwnd, WM_SIZE, 0, MAKELPARAM(600, 400));
    assert(!g_scene_from && !g_scene_presented);
    win_shutdown();
    puts("PASS: D2D capture, interrupted scene fade, identical scene, completion, resize cleanup");
}
