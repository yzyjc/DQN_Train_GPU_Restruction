// GPT6-Astra: touch gesture recognition, independent of Win32 and rendering.
#pragma once
#include <cstdlib>
struct TouchScrollGesture {
    bool active = false, moved = false;
    int start_x = 0, start_y = 0, last_y = 0, remainder = 0;
    int threshold = 8, step = 20;
    void begin(int x, int y, int cell_height) {
        active = true; moved = false;
        start_x = x; start_y = last_y = y; remainder = 0;
        step = cell_height > 8 ? cell_height : 8;
        threshold = step / 2;
        if (threshold < 8) threshold = 8;
    }
    int update(int x, int y) {
        if (!active) return 0;
        if (!moved) {
            if (std::abs(x-start_x) < threshold && std::abs(y-start_y) < threshold) return 0;
            moved = true; // Horizontal drags also suppress accidental clicks.
        }
        remainder += y-last_y;
        last_y = y;
        int ticks = remainder / step;
        remainder -= ticks * step;
        return ticks;
    }
    bool end() {
        bool tap = active && !moved;
        active = false;
        return tap;
    }
    void cancel() { active = false; moved = false; remainder = 0; }
};
