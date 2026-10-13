#include "probe_diagnostics.h"
#include <cassert>
#include <cstdio>
int main(){
    ProbeParameterHistory h;
    h.at(0).bytes={1,2};h.at(0).updates=4;
    h.at(1).bytes={8,9};h.at(1).updates=20;
    assert(h.at(0).bytes[0]==1 && h.at(0).updates==4);
    assert(h.at(1).bytes[0]==8 && h.at(1).updates==20);
    float g=ProbeParameterHistory::mean_effective_gradient(1,.9f,.1f,2);
    assert(std::fabs(g-.5f)<1e-6f);
    assert(std::fabs(g*.1f*2-.1f)<1e-6f);
    assert(ProbeParameterHistory::mean_effective_gradient(1,.9f,0,2)==0);
    assert(probe_parameter_source(2)==11 && probe_parameter_source(3)==11 && probe_parameter_source(4)==11);
    assert(probe_parameter_source(5)==12 && probe_parameter_source(7)==6);
    h.reset();assert(h.at(0).bytes.empty() && h.at(1).bytes.empty());
    puts("PASS: per-side history, mean effective gradient/update identity, real QKV branches");
}
