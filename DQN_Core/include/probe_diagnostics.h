#pragma once
// Collaboration signature: GPT6-Astra. Measurement helpers, no training writes.
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
struct ProbeParameterHistory {
    struct Entry { std::vector<uint8_t> bytes; uint64_t updates=0; };
    std::array<Entry,3> sides;
    void reset(){for(auto& e:sides){e.bytes.clear();e.updates=0;}}
    static unsigned index(uint32_t side){return side<2?side:2;}
    Entry& at(uint32_t side){return sides[index(side)];}
    static float mean_effective_gradient(float old,float current,float lr,uint64_t span) {
        if(!span || !std::isfinite(lr) || lr<=0)return 0;
        double g=((double)old-current)/((double)lr*span);
        return std::isfinite(g) && std::fabs(g)<=3.402823466e38 ? (float)g : 0;
    }
};
// The real parameter matrices are not an adjacent-layer chain.
inline unsigned probe_parameter_source(unsigned layer) {
    if(layer>=2 && layer<=4)return 11; // merged token+position features
    if(layer==5)return 12;             // concatenated attention heads
    return layer-1;
}
