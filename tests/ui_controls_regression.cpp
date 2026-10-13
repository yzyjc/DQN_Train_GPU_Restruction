#include <windows.h>
#include <cassert>
#include "../DQN_Main_TUI_No_Terminal/virtual_terminal.cpp"
#include "../DQN_Main_TUI_No_Terminal/ui_controls.h"
void win_pump_messages(void) {}
int main() {
    vt_init(80, 5);
    auto lamp = ui_lamp_button();
    auto rect = ui_button(2, 1, "Trainer", {false, false}, lamp);
    assert(vt_get_back_cell(2, 1)->glyph == L'○');
    assert(vt_get_back_cell(4, 1)->fg == lamp.idle);
    assert(rect.width == 9 && rect.contains(10, 1) && !rect.contains(11, 1));
    ui_button(2, 1, "Trainer", {true, false}, lamp);
    assert(vt_get_back_cell(2, 1)->glyph == L'●');
    assert(vt_get_back_cell(2, 1)->fg == 0x39C5BB);
    assert(vt_get_back_cell(4, 1)->fg == 0xFFFFFF);
    auto toggle = ui_toggle_button();
    ui_button(0, 2, "Memory", {true, false}, toggle);
    assert(vt_get_back_cell(0, 2)->fg == 0x39C5BB);
    ui_button(0, 2, "Memory", {false, true}, toggle);
    assert(vt_get_back_cell(0, 2)->fg == toggle.idle);
    auto push = ui_push_button(); push.padding = 1;
    assert(ui_button_width("应用全部", push) == 10);
    ui_button(0, 3, "Apply", {false, true}, push);
    assert(vt_get_back_cell(1, 3)->fg == 0x39C5BB);
    ui_button(0, 3, "Apply", {true, false}, push);
    assert(vt_get_back_cell(1, 3)->fg == push.idle);
    ui_number(0, 4, "12.3");
    assert(vt_get_back_cell(0, 4)->attr & VT_ATTR_ROLL);
    ui_number(0, 4, "12.3", false);
    assert(!(vt_get_back_cell(0, 4)->attr & VT_ATTR_ROLL));
    puts("PASS button presets, press/release, UTF-8 width, hit bounds, rolling opt-in");
}
