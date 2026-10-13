// GPT6-Astra: opt-in bounded diagnostic trace. Event envelopes are NOT GPU duty.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <windows.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include "gpu_module_profile.h"
namespace gpu_profile {
inline double clock_us(){LARGE_INTEGER t,f;QueryPerformanceCounter(&t);QueryPerformanceFrequency(&f);return double(t.QuadPart)*1e6/double(f.QuadPart);}
struct Entry { hipEvent_t begin=nullptr,end=nullptr; const char* kind="",*name="",*stage=""; size_t bytes=0;double host=0,started=0;bool valid=true; };
struct Batch;
inline thread_local Batch* current=nullptr;
struct Batch {
    std::vector<Entry> entries;
    hipEvent_t origin=nullptr;
    const char* stage="setup";
    unsigned used=0,id=0;int samples=0;bool enabled=false,overflow=false,committed=false;
    double started=0;char directory[MAX_PATH]={};
    explicit Batch(int n):samples(n) {
        if(gpu_module_profile::selected())return;
        static const int limit=[](){char text[32]={};GetEnvironmentVariableA("DQN_GPU_PROFILE_BATCHES",text,sizeof(text));int n=atoi(text);return n<0?0:n>8?8:n;}();
        static std::atomic<unsigned> issued{0};
        if(!limit || current)return;
        id=issued.fetch_add(1);if(id>=(unsigned)limit)return;
        DWORD len=GetEnvironmentVariableA("DQN_GPU_PROFILE_DIR",directory,sizeof(directory));
        if(!len || len>=sizeof(directory)){fprintf(stderr,"[GPU profile] explicit output directory required\n");return;}
        DWORD attributes=GetFileAttributesA(directory);
        if(attributes==INVALID_FILE_ATTRIBUTES || !(attributes&FILE_ATTRIBUTE_DIRECTORY))return;
        try { entries.resize(2048); } catch(...) { return; } // enough for the existing batch<=32 path; truncation is flagged.
        if(hipEventCreate(&origin)!=hipSuccess)return;
        for(auto& e:entries)if(hipEventCreate(&e.begin)!=hipSuccess || hipEventCreate(&e.end)!=hipSuccess)return;
        if(hipEventRecord(origin,0)!=hipSuccess)return;
        enabled=true;started=clock_us();current=this;
    }
    Entry* begin(const char* kind,const char* name,size_t bytes,bool device=true) {
        if(!enabled)return nullptr;
        if(used==entries.size()){overflow=true;return nullptr;}
        auto& e=entries[used++];e.kind=kind;e.name=name;e.bytes=bytes;e.stage=stage;
        if(device && hipEventRecord(e.begin,0)!=hipSuccess)e.valid=false;
        e.started=clock_us();return &e;
    }
    void end(Entry* e,bool device=true,bool valid=true) {
        if(!e)return;e->host=clock_us()-e->started;e->valid=e->valid&&valid;
        if(device && hipEventRecord(e->end,0)!=hipSuccess)e->valid=false;
    }
    ~Batch() {
        if(enabled) {
            double wall=clock_us()-started;current=nullptr;
            // Diagnostic drain is outside reported batch wall time, and only in opt-in mode.
            bool drained=hipDeviceSynchronize()==hipSuccess;
            char path[MAX_PATH+80];snprintf(path,sizeof(path),"%s\\gpu_profile_%lu_%u.csv",directory,GetCurrentProcessId(),id);
            FILE* f=fopen(path,"w");
            if(f){
                fprintf(f,"# version=1,batch=%d,committed=%d,overflow=%d,drained=%d,wall_us=%.3f\n",samples,committed,overflow,drained,wall);
                fprintf(f,"kind,stage,name,bytes,host_us,start_us,end_us,duration_us,api_ok,event_ok,timeline_ok\n");
                for(unsigned i=0;i<used;++i){auto& e=entries[i];float a=0,b=0,d=0;bool device=strcmp(e.kind,"sync")!=0;
                    bool good=e.valid && drained;
                                        bool event_ok=!device || (good && hipEventElapsedTime(&d,e.begin,e.end)==hipSuccess && std::isfinite(d) && d>=0);
                    bool timeline_ok=device && good && hipEventElapsedTime(&a,origin,e.begin)==hipSuccess && hipEventElapsedTime(&b,origin,e.end)==hipSuccess && std::isfinite(a) && std::isfinite(b) && a>=0 && b>=a;
                    fprintf(f,"%s,%s,%s,%zu,%.3f,%.3f,%.3f,%.3f,%d,%d,%d\n",e.kind,e.stage,e.name,e.bytes,e.host,device?a*1000.0:-1.0,device?b*1000.0:-1.0,device?d*1000.0:-1.0,good,event_ok,timeline_ok);
                }
                fclose(f);
            }
        }
        for(auto& e:entries){if(e.begin)hipEventDestroy(e.begin);if(e.end)hipEventDestroy(e.end);}
        if(origin)hipEventDestroy(origin);
    }
};
inline void stage(const char* name){gpu_module_profile::stage(name);if(current)current->stage=name;}
inline hipError_t memset(void* p,int value,size_t bytes){
    gpu_module_profile::clear(bytes);
    auto b=current;auto e=b?b->begin("memset","hipMemset",bytes):nullptr;
    auto rc=hipMemset(p,value,bytes);if(b)b->end(e,true,rc==hipSuccess);return rc;
}
inline hipError_t copy(void* dst,const void* src,size_t bytes,hipMemcpyKind kind){
    gpu_module_profile::copy(bytes,kind==hipMemcpyHostToDevice?0:kind==hipMemcpyDeviceToHost?1:kind==hipMemcpyDeviceToDevice?2:-1);
    auto b=current;const char* name=kind==hipMemcpyHostToDevice?"H2D":kind==hipMemcpyDeviceToHost?"D2H":kind==hipMemcpyDeviceToDevice?"D2D":"other";
    auto e=b?b->begin("copy",name,bytes):nullptr;auto rc=hipMemcpy(dst,src,bytes,kind);if(b)b->end(e,true,rc==hipSuccess);return rc;
}
inline hipError_t sync(){auto b=current;auto e=b?b->begin("sync","hipDeviceSynchronize",0,false):nullptr;
    double start=gpu_module_profile::sync_begin();
    auto rc=hipDeviceSynchronize();gpu_module_profile::sync_end(start);if(b)b->end(e,false,rc==hipSuccess);return rc;}
}
#define DQN_PROFILE_LAUNCH(kernel, grid, block, shared, stream, ...) do { \
    gpu_module_profile::launch(); \
    auto* profile_batch=gpu_profile::current; \
    auto* profile_entry=profile_batch?profile_batch->begin("kernel",#kernel,0,false):nullptr; \
    if(profile_entry) { \
        hipExtLaunchKernelGGL(kernel,grid,block,shared,stream,profile_entry->begin,profile_entry->end,0,__VA_ARGS__); \
        profile_batch->end(profile_entry,false); \
    } else { hipLaunchKernelGGL(kernel,grid,block,shared,stream,__VA_ARGS__); } \
} while(0)
