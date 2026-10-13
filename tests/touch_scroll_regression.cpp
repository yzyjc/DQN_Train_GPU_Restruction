// GPT6-Astra: deterministic gesture tests; no touchscreen required.
#include "../DQN_Main_TUI_No_Terminal/touch_scroll.h"
#include <cassert>
#include <cstdio>
int main() {
    TouchScrollGesture g;
    g.begin(100,100,20);
    assert(g.update(103,104)==0); assert(g.end());
    g.begin(100,100,20);
    assert(g.update(100,79)==-1); assert(!g.end());
    g.begin(100,100,20);
    assert(g.update(100,161)==3); assert(!g.end());
    g.begin(100,100,20);
    assert(g.update(100,90)==0); assert(g.update(100,80)==-1);
    assert(g.update(100,100)==1); assert(!g.end());
    g.begin(100,100,20);
    assert(g.update(125,100)==0); assert(!g.end());
    g.begin(100,100,20); g.cancel(); assert(!g.end());
    assert(g.update(100,500)==0);
    g.begin(100,100,40); assert(g.update(100,140)==1); assert(!g.end());
    g.begin(100,100,0); assert(g.update(100,108)==1); assert(!g.end());
    puts("PASS: tap, jitter, scroll directions, accumulated motion, horizontal suppression, cancellation, DPI, minimum step");
}
