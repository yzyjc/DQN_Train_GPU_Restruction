// GPT6-Astra: host module intervals, independent of deep device-event profiling.
#pragma once
#include <windows.h>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace gpu_module_profile {
inline double now() {
    static const double scale=[](){LARGE_INTEGER f;QueryPerformanceFrequency(&f);return 1e6/f.QuadPart;}();
    LARGE_INTEGER t;QueryPerformanceCounter(&t);return t.QuadPart*scale;
}
inline bool selected() {
    static const bool yes=[](){char mode[32]={};GetEnvironmentVariableA("DQN_GPU_PROFILE_MODE",mode,sizeof(mode));return strcmp(mode,"modules")==0;}();
    return yes;
}
struct Interval { const char* name=nullptr;double us=0; };
struct Slot {
    std::atomic<bool> ready{false};
    Interval intervals[512];unsigned count=0;int samples=0;
    bool committed=false,overflow=false;double wall=0;
    unsigned launches=0,clears=0,syncs=0,copies[3]={};
    size_t clear_bytes=0,copy_bytes[3]={};double sync_us=0;
};
struct Collector {
    Slot slots[8];std::atomic<unsigned> issued{0};std::atomic<bool> stop{false};
    int limit=0;char directory[MAX_PATH]={};std::thread worker;
    Collector() {
        char value[32]={};GetEnvironmentVariableA("DQN_GPU_PROFILE_BATCHES",value,sizeof(value));
        limit=atoi(value);if(limit<0)limit=0;if(limit>8)limit=8;
        DWORD len=GetEnvironmentVariableA("DQN_GPU_PROFILE_DIR",directory,sizeof(directory));
        DWORD attr=len && len<sizeof(directory)?GetFileAttributesA(directory):INVALID_FILE_ATTRIBUTES;
        if(attr==INVALID_FILE_ATTRIBUTES || !(attr&FILE_ATTRIBUTE_DIRECTORY)){limit=0;return;}
        if(limit)try{worker=std::thread([this](){run();});}catch(...){limit=0;}
    }
    void run() {
        for(;;) {
            const bool last=stop.load(std::memory_order_acquire);
            for(int i=0;i<limit;++i)if(slots[i].ready.load(std::memory_order_acquire)) {
                auto& s=slots[i];char path[MAX_PATH+80];
                snprintf(path,sizeof(path),"%s\\gpu_modules_%lu_%d.csv",directory,GetCurrentProcessId(),i);
                FILE* f=fopen(path,"w");
                if(f) {
                    fprintf(f,"# version=2,batch=%d,committed=%d,overflow=%d,wall_us=%.3f,kernel_launches=%u,memset_calls=%u,memset_bytes=%zu,sync_calls=%u,sync_us=%.3f,h2d_calls=%u,h2d_bytes=%zu,d2h_calls=%u,d2h_bytes=%zu,d2d_calls=%u,d2d_bytes=%zu\n",
                        s.samples,s.committed,s.overflow,s.wall,s.launches,s.clears,s.clear_bytes,s.syncs,s.sync_us,
                        s.copies[0],s.copy_bytes[0],s.copies[1],s.copy_bytes[1],s.copies[2],s.copy_bytes[2]);
                    fprintf(f,"module,host_us\n");
                    for(unsigned j=0;j<s.count;++j)fprintf(f,"%s,%.3f\n",s.intervals[j].name,s.intervals[j].us);
                    fclose(f);
                }
                s.ready.store(false,std::memory_order_release);
            }
            if(last)break;
            Sleep(10);
        }
    }
    ~Collector(){stop.store(true,std::memory_order_release);if(worker.joinable())worker.join();}
};
inline Collector& collector(){static Collector c;return c;}
struct Batch;
inline thread_local Batch* current=nullptr;
struct Batch {
    Slot* slot=nullptr;const char* name="setup";double started=0,began=0;
    explicit Batch(int n) {
        if(!selected() || current)return;
        auto& c=collector();unsigned id=c.issued.fetch_add(1);
        if(id>=(unsigned)c.limit)return;
        slot=&c.slots[id];slot->samples=n;current=this;began=started=now();
    }
    void end(double time) {
        if(slot->count==512){slot->overflow=true;return;}
        slot->intervals[slot->count++]={name,time-started};
    }
    void stage(const char* next) {
        const double time=now();end(time);name=next;started=now();
    }
    void commit(){if(slot)slot->committed=true;}
    ~Batch() {
        if(!slot)return;
        const double time=now();end(time);slot->wall=time-began;current=nullptr;
        slot->ready.store(true,std::memory_order_release);
    }
};
inline void stage(const char* name){if(current)current->stage(name);}
inline void launch(){if(current)++current->slot->launches;}
inline void clear(size_t bytes){if(current){++current->slot->clears;current->slot->clear_bytes+=bytes;}}
inline void copy(size_t bytes,int direction){if(current && direction>=0 && direction<3){++current->slot->copies[direction];current->slot->copy_bytes[direction]+=bytes;}}
inline double sync_begin(){return current?now():0;}
inline void sync_end(double start){if(current){double elapsed=now()-start;++current->slot->syncs;current->slot->sync_us+=elapsed;}}
}
