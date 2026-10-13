// GPT6-Astra: isolated event-clock validation, never attaches to Trainer.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cmath>
#define CHECK(x) do{hipError_t rc=(x);if(rc!=hipSuccess){fprintf(stderr,"HIP failure line %d: %d\n",__LINE__,(int)rc);return 1;}}while(0)
int main() {
    CHECK(hipInit(0));CHECK(hipSetDevice(0));
    void* data=nullptr;CHECK(hipMalloc(&data,64*1024*1024));
    CHECK(hipMemset(data,0,64*1024*1024));CHECK(hipDeviceSynchronize());
    printf("bytes,repeats,flags,invalid,mean_device_us,mean_host_us\n");
    for(unsigned flags:{0u,(unsigned)hipEventBlockingSync})for(size_t bytes:{size_t(64),size_t(64*1024*1024)}) {
        hipEvent_t start=nullptr,end=nullptr;CHECK(hipEventCreateWithFlags(&start,flags));CHECK(hipEventCreateWithFlags(&end,flags));
        int invalid=0;double device=0,host=0;LARGE_INTEGER freq;QueryPerformanceFrequency(&freq);
        for(int i=0;i<32;++i) {
            LARGE_INTEGER a,b;QueryPerformanceCounter(&a);
            CHECK(hipEventRecord(start,0));CHECK(hipMemset(data,i,bytes));CHECK(hipEventRecord(end,0));
            CHECK(hipEventSynchronize(end)); // probe only: known completed interval, never a production change
            QueryPerformanceCounter(&b);float ms=0;CHECK(hipEventElapsedTime(&ms,start,end));
            if(!std::isfinite(ms)||ms<=0)++invalid;else device+=ms*1000;
            host+=(b.QuadPart-a.QuadPart)*1e6/freq.QuadPart;
        }
        printf("%zu,32,%u,%d,%.3f,%.3f\n",bytes,flags,invalid,invalid?-1:device/32,host/32);
        CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    }
    CHECK(hipFree(data));return 0;
}
