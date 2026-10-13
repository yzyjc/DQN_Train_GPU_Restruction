#pragma once
#include <stdint.h>
#include <math.h>

// A wheel always has two adjacent faces, exactly one cell apart.
// Fast updates replace the queued destination, never the visible moving faces.
struct RollingDigit {
    struct Layer { wchar_t glyph = 0; float from = 0, to = 0; } layers[4];
    wchar_t target = 0;
    uint64_t started = 0, changed_at = 0;
    bool active = false, linear = false;
    int direction = 1;
    float progress(uint64_t now) const {
        float t = now <= started ? 0.0f : now >= started + 180 ? 1.0f : float(now - started) / 180.0f;
        return linear ? t : 1.0f - (1.0f-t)*(1.0f-t)*(1.0f-t);
    }
    float position(int i, uint64_t now) const {
        float t = progress(now);
        return layers[i].from + (layers[i].to-layers[i].from)*t;
    }
    void reset(wchar_t glyph) { *this = {}; target = glyph; }
    void begin(wchar_t from, wchar_t to, uint64_t now, bool continuous) {
        if (!continuous) direction = ((to-from+10)%10 <= 5) ? 1 : -1;
        layers[0] = {from, 0, -float(direction)};
        layers[1] = {to, float(direction), 0};
        layers[2] = {}; layers[3] = {};
        started = now; active = true; linear = continuous;
    }
    void update(wchar_t glyph, uint64_t now, bool enabled) {
        if (!enabled || target < L'0' || target > L'9' || glyph < L'0' || glyph > L'9') {
            reset(glyph); return;
        }
        wchar_t previous = target;
        if (glyph != target) { target = glyph; changed_at = now; }
        if (!active) {
            if (glyph != previous) begin(previous, glyph, now, false);
            return;
        }
        // Finish the current face before feeding in the latest queued value.
        // Preserve direction and use uniform speed while changes arrive <180ms apart.
        if (now >= started + 180) {
            wchar_t arrived = layers[1].glyph;
            uint64_t boundary = started + 180;
            if (arrived == target) { reset(target); return; }
            bool continuous = now >= changed_at && now - changed_at < 180;
            begin(arrived, target, boundary, continuous);
            if (now >= started + 180) reset(target);
        }
    }
};
