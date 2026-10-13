// Simulate monitor DPI transitions against the production terminal window proc.
#define _CRT_SECURE_NO_WARNINGS
#include "../DQN_Main_TUI_No_Terminal/terminal_window.cpp"
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#include <cassert>
#include <cstdio>

int g_running = 1;
int g_skip_scene_crossfade = 1;

int main() {
    const wchar_t* class_name = L"DQN_TUI_DPI_REGRESSION";
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = class_name;
    assert(RegisterClassExW(&wc));

    HWND hwnd = CreateWindowExW(0, class_name, L"hidden DPI regression",
        WS_POPUP, 0, 0, 1200, 800, nullptr, nullptr, wc.hInstance, nullptr);
    assert(hwnd);
    g_hwnd = hwnd;
    g_cell_w = 11;
    g_cell_h = 21;
    vt_init(80, 24);

    const struct { UINT dpi; int width; int height; } transitions[] = {
        {144, 1200, 800}, {96, 1600, 1000}, {192, 1000, 700}, {144, 1200, 800}
    };
    for (const auto& t : transitions) {
        RECT suggested = {100, 100, 100 + t.width, 100 + t.height};
        WndProc(hwnd, WM_DPICHANGED, MAKELONG(t.dpi, t.dpi),
            reinterpret_cast<LPARAM>(&suggested));

        RECT client = {};
        assert(GetClientRect(hwnd, &client));
        assert(g_cell_w == 11 && g_cell_h == 21);
        assert(vt_get_width() == (client.right / 11 < 40 ? 40 : client.right / 11));
        assert(vt_get_height() == (client.bottom / 21 < 15 ? 15 : client.bottom / 21));
    }

    DestroyWindow(hwnd);
    UnregisterClassW(class_name, wc.hInstance);
    std::puts("PASS: DPI transitions keep measured cell metrics and resize the VT grid");
    return 0;
}
