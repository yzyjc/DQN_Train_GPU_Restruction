// GPT6-Astra: backend failure contract and real-device sampling.
#include "../GPU_Analyzer/include/gpu_sensors.h"
#include <cassert>
#include <cstdio>
#include <cmath>
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0); puts("start");
    GpuTelemetrySample s={};s.gpu_temp_c=90;s.gpu_power_w=80;s.gpu_clock_mhz=1000;
    s.gpu_memory_clock_mhz=1200;s.gpu_hotspot_temp_c=100;
    gpu_sensors_sample(nullptr,&s);
    assert(s.gpu_temp_c==-1 && s.gpu_power_w==-1 && s.gpu_clock_mhz==-1 &&
           s.gpu_memory_clock_mhz==-1 && s.gpu_hotspot_temp_c==-1);
    assert(s._reserved==GPU_TELEM_SENSOR_EXT_MAGIC);
    void* mismatch=gpu_sensors_create("definitely not this adapter");
    assert(!mismatch);gpu_sensors_destroy(mismatch);
    if(argc<2){puts("PASS unavailable backend and adapter mismatch");return 0;}
    void* backend=gpu_sensors_create(argv[1]);assert(backend);
    for(int i=0;i<3;++i){gpu_sensors_sample(backend,&s);
        assert(std::isfinite(s.gpu_temp_c)&&s.gpu_temp_c>=0&&s.gpu_temp_c<=150);
        assert(std::isfinite(s.gpu_power_w)&&s.gpu_power_w>=0&&s.gpu_power_w<=2000);
        assert(std::isfinite(s.gpu_clock_mhz)&&s.gpu_clock_mhz>=0&&s.gpu_clock_mhz<=10000);
        assert((s.gpu_memory_clock_mhz==-1) ||
               (std::isfinite(s.gpu_memory_clock_mhz)&&s.gpu_memory_clock_mhz>=0&&s.gpu_memory_clock_mhz<=10000));
        assert((s.gpu_hotspot_temp_c==-1) ||
               (std::isfinite(s.gpu_hotspot_temp_c)&&s.gpu_hotspot_temp_c>=0&&s.gpu_hotspot_temp_c<=200));
        printf("sample=%d temperature=%.1f C power=%.1f W core=%.0f MHz memory=%.0f MHz hotspot=%.1f C\n",
               i,s.gpu_temp_c,s.gpu_power_w,s.gpu_clock_mhz,s.gpu_memory_clock_mhz,s.gpu_hotspot_temp_c);
    }
    gpu_sensors_destroy(backend);puts("PASS real backend lifecycle and samples");
}
