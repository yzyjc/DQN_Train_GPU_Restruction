#define UNICODE
#define _UNICODE
#define wWinMain unused_tui_entry
#include "../DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp"
#undef wWinMain
#undef printf
#include <cassert>
int main() {
    vt_init(200, 60); g_console_width = 200; g_console_height = 60;
    auto clipped = control_clip(0, 0, 5, "中文AB", true, true);
    assert(clipped.width == 5 && clipped.contains(4, 0) && !clipped.contains(5, 0));
    assert(vt_get_back_cell(0, 0)->fg == 0x39C5BB);
    print_roll_clip(0, 1, 5, "%s", "123456789");
    assert(vt_get_back_cell(0, 1)->attr & VT_ATTR_ROLL);
    assert(vt_get_back_cell(4, 1)->glyph == L'…');
    vt_clear_screen();
    print_clip(0, 1, 5, "%s", "12345");
    assert(!(vt_get_back_cell(0, 1)->attr & VT_ATTR_ROLL));
    StatusFrames frames;
    std::string active = "{\"ep\":166100,\"pretrain\":0,\"eval_active\":1,\"eval_completed\":345,\"eval_total\":1200}";
    int received = 0;
    auto consume = [&](const char* json) { ++received; parse_status_push(json); };
    for (size_t i = 0; i < active.size(); ++i) {
        frames.feed(active.data() + i, 1, consume);
        assert(received == (i + 1 == active.size() ? 1 : 0));
    }
    assert(g_ui_eval_active == 1 && g_ui_eval_completed == 345);
    frames.feed("{\"status\":\"ok\"}", 15, consume);
    assert(received == 1 && g_ui_eval_active == 1);
    for (int width : {85, 120, 200}) {
        vt_resize(width, 60); g_console_width = width;
        compute_merge_layout(); g_active_tab = 0; draw_tab_bar();
        assert(vt_get_back_cell(g_tab_x[0], TAB_BAR_Y)->fg == 0x39C5BB);
        assert(vt_get_back_cell(g_tab_x[1], TAB_BAR_Y)->fg == 0x767676);
        g_active_tab = 3; g_force_redraw = 1;
        render_gpu(0, 0, width - 4, 30);
        assert(vt_get_back_cell(6, CONTENT_Y + 10)->attr & VT_ATTR_ROLL);
    }
    for (int width : {85, 120, 200}) {
        vt_resize(width, 60); g_console_width = width;
        parse_status_push("{\"ep\":166100,\"pretrain\":0,\"eval_active\":1,\"eval_completed\":345,\"eval_total\":1200}");
        assert(g_ui_eval_active == 1 && g_ui_eval_completed == 345 && g_ui_eval_total == 1200);
        draw_progress_bar(2, width - 4);
        int y = g_console_height - 6;
        assert(vt_get_back_cell(4, y + 2)->glyph == L'M');
        assert(vt_get_back_cell(4, y + 2)->attr & VT_ATTR_ROLL);
    }
    vt_resize(200, 60); g_console_width = 200;
    g_snap_test_running = 0; g_snap_test_done = 0;
    strcpy(g_snap_test_error, "Test message moves the RUN button down");
    PlayPanel panel = {2, 4, 70, 20};
    draw_play_test(panel);
    assert(g_skill_run_hits[0].y == panel.y + 2);
    assert(!g_skill_run_hits[0].contains(panel.x + 2, panel.y + 1));
    assert(g_skill_run_hits[1].contains(panel.x + 3, panel.y + 3));
    // Never click: renderer tests must not start tests or send training commands.
    puts("PASS clipped controls, static/rolling separation, 3 window sizes, navigation, skill button geometry");
}
