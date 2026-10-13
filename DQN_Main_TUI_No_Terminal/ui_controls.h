#pragma once
#include <windows.h>
#include <cstdarg>
#include <cstdio>
#include "virtual_terminal.h"

// One renderer, three presets: lamp toggle, text toggle, momentary button.
struct UiButtonStyle {
    bool indicator = false;
    bool toggle = false;
    int padding = 0;
    uint32_t idle = 0x767676, active = 0x39C5BB;
    uint32_t lamp = 0x39C5BB, background = 0x0C0C0C;
};
inline UiButtonStyle ui_lamp_button() {
    UiButtonStyle s; s.indicator = true; s.toggle = true; s.active = 0xFFFFFF; return s;
}
inline UiButtonStyle ui_toggle_button() { UiButtonStyle s; s.toggle = true; return s; }
inline UiButtonStyle ui_push_button() { return {}; }
struct UiButtonState { bool active = false, pressed = false; };
struct UiControlRect {
    int x, y, width;
    bool contains(int px, int py) const { return py == y && px >= x && px < x + width; }
};
inline int ui_text_width(const char* text) {
    int width = 0;
    while (*text) {
        unsigned char c = (unsigned char)*text;
        int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        wchar_t glyph[2] = {};
        int available = 0; while (available < n && text[available]) ++available;
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, available, glyph, 2) > 0)
            width += vt_char_width(glyph[0]);
        else ++width;
        text += available;
    }
    return width;
}
inline int ui_button_width(const char* label, const UiButtonStyle& style) {
    return ui_text_width(label) + (style.indicator ? 2 : 0) + 2 * style.padding;
}
inline void ui_color(uint32_t color) { vt_set_truecolor_fg(color >> 16, (color >> 8) & 255, color & 255); }
inline UiControlRect ui_button(int x, int y, const char* label, UiButtonState state, UiButtonStyle style) {
    vt_gotoxy(x, y);
    vt_set_truecolor_bg(style.background >> 16, (style.background >> 8) & 255, style.background & 255);
    ui_color(style.idle);
    for (int i = 0; i < style.padding; ++i) vt_printf(" ");
    bool on = style.toggle ? state.active : state.pressed;
    if (style.indicator) {
        ui_color(on ? style.lamp : style.idle); vt_printf("%s ", on ? "●" : "○");
    }
    ui_color(on ? style.active : style.idle); vt_printf("%s", label);
    for (int i = 0; i < style.padding; ++i) vt_printf(" ");
    vt_reset_color();
    return { x, y, ui_button_width(label, style) };
}

// Call while drawing a page; the terminal advances/intercepts the roll each frame.
// Non-digits stay fixed. Scene changes and resize reset the terminal's motion.
inline void ui_number(int x, int y, const char* formatted, bool rolling = true) {
    vt_gotoxy(x, y);
    if (rolling) vt_roll_printf("%s", formatted);
    else vt_printf("%s", formatted);
}
inline int ui_number_printf(const char* format, ...) {
    char text[4096]; va_list args; va_start(args, format);
    int result = vsnprintf(text, sizeof(text), format, args); va_end(args);
    if (result >= 0) vt_roll_printf("%s", text);
    return result;
}
