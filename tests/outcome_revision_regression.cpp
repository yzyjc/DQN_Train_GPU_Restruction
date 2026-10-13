// Collaboration signature: GPT6-Astra.
#include "step_outcome.h"
#include <cassert>
#include <cstdio>
int main() {
    StepOutcomeConfig c; c.enabled=1;
    assert(step_outcome_set(c,2,.15f) && c.black_early_loss_scale==0);
    // Historical layouts can still contain a nonzero deprecated field.
    c.black_early_loss_scale=.05f;
    for(int steps=-2;steps<=27;++steps) {
        float bw=.85f+step_outcome_bonus(c,0,1,steps,25);
        float bl=-.5f+step_outcome_bonus(c,0,2,steps,25);
        float wl=-.5f+step_outcome_bonus(c,1,1,steps,25);
        float wd=step_outcome_bonus(c,1,0,steps,25);
        assert(bl==-.5f && wl<0 && bw>wd && wd>wl && 0>bl);
        if(steps>0 && steps<=25) {
            assert(step_outcome_bonus(c,0,1,steps,25)<step_outcome_bonus(c,0,1,steps-1,25));
            assert(step_outcome_bonus(c,1,1,steps,25)>step_outcome_bonus(c,1,1,steps-1,25));
        }
    }
    for(int win=0;win<3;++win) for(int side=0;side<2;++side) {
        assert(std::isnan(step_outcome_bonus(c,side,win,5,0)));
    }
    assert(!step_outcome_set(c,1,NAN) && !step_outcome_set(c,3,.5f));
    puts("PASS: no early-loss penalty, monotonic shaping, hierarchy, legacy field, boundaries");
}
