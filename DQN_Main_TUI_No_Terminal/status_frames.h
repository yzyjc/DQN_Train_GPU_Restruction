#pragma once
#include <string>

// Pipe reads are not frame boundaries (including byte-mode compatibility).
// Keep incomplete JSON until the next read instead of parsing/resetting fields.
struct StatusFrames {
    std::string pending;
    void reset() { pending.clear(); }
    template<class Consumer> void feed(const char* bytes, size_t size, Consumer consume) {
        pending.append(bytes, size);
        for (;;) {
            size_t start = pending.find('{');
            if (start == std::string::npos) { pending.clear(); return; }
            if (start) pending.erase(0, start);
            bool quoted = false, escaped = false;
            int depth = 0;
            size_t end = 0;
            for (; end < pending.size(); ++end) {
                char c = pending[end];
                if (quoted) {
                    if (escaped) escaped = false;
                    else if (c == '\\') escaped = true;
                    else if (c == '"') quoted = false;
                } else if (c == '"') quoted = true;
                else if (c == '{') ++depth;
                else if (c == '}' && --depth == 0) break;
            }
            if (end == pending.size()) {
                if (pending.size() > 16384) pending.clear();
                return;
            }
            std::string frame = pending.substr(0, end + 1);
            pending.erase(0, end + 1);
            if (frame.compare(0, 5, "{\"ep\"") == 0) consume(frame.c_str());
        }
    }
};
