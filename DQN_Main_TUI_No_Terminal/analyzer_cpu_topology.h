// GPT6-Astra: CPU busy-time snapshots and OS-reported SMT topology.
#pragma once
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <intrin.h>
namespace analyzer_cpu {
struct Thread { unsigned group, number; float busy; };
struct Core { std::vector<Thread> threads; };
inline float valid_busy(double value, DWORD status) {
    if(status!=PDH_CSTATUS_VALID_DATA && status!=PDH_CSTATUS_NEW_DATA) return -1;
    if(!std::isfinite(value) || value<0 || value>100.01) return -1;
    return (float)(value>100 ? 100 : value);
}
inline SRWLOCK lock = SRWLOCK_INIT;
inline std::vector<Thread> latest;
inline std::vector<float> recent_maxima;
inline float sample(PDH_HCOUNTER counter) {
    std::vector<Thread> next; float maximum=-1;
    DWORD size=0,count=0;
    if(counter) {
        PdhGetFormattedCounterArrayA(counter,PDH_FMT_DOUBLE,&size,&count,nullptr);
        std::vector<unsigned char> buffer(size);
        auto items=reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_A*>(buffer.data());
        if(size && PdhGetFormattedCounterArrayA(counter,PDH_FMT_DOUBLE,&size,&count,items)==ERROR_SUCCESS) {
            for(DWORD i=0;i<count;++i) {
                unsigned group=0,number=0; char tail=0;
                if(!items[i].szName || sscanf_s(items[i].szName,"%u,%u%c",&group,&number,&tail,1u)!=2) continue;
                float busy=valid_busy(items[i].FmtValue.doubleValue,items[i].FmtValue.CStatus);
                if(busy<0) continue;
                next.push_back({group,number,busy}); if(busy>maximum) maximum=busy;
            }
        }
    }
        AcquireSRWLockExclusive(&lock); latest.swap(next);
    if(maximum>=0){recent_maxima.push_back(maximum);if(recent_maxima.size()>60)recent_maxima.erase(recent_maxima.begin());}
    ReleaseSRWLockExclusive(&lock);
    return maximum;
}
inline std::vector<Thread> snapshot() {
    AcquireSRWLockShared(&lock); auto copy=latest; ReleaseSRWLockShared(&lock); return copy;
}
struct BusyStats { float mean=-1,p95=-1,peak=-1;size_t count=0; };
inline BusyStats summarize(std::vector<float> values) {
    BusyStats s;if(values.empty())return s;
    double sum=0;for(float v:values){if(!std::isfinite(v)||v<0||v>100)return s;sum+=v;}
    std::sort(values.begin(),values.end());s.count=values.size();s.mean=(float)(sum/values.size());
    s.p95=values[(size_t)std::ceil(values.size()*.95)-1];s.peak=values.back();return s;
}
inline BusyStats busy_stats() {
    AcquireSRWLockShared(&lock);auto values=recent_maxima;bool fresh=!latest.empty();ReleaseSRWLockShared(&lock);
    return fresh?summarize(values):BusyStats{};
}
inline std::vector<Core> topology() {
    DWORD size=0; GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&size);
    std::vector<unsigned char> bytes(size);
    if(!size || !GetLogicalProcessorInformationEx(RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(bytes.data()),&size)) return {};
    std::vector<Core> cores;
    for(DWORD offset=0;offset<size;) {
        auto info=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(bytes.data()+offset);
        if(!info->Size || info->Size>size-offset) break;
        if(info->Relationship==RelationProcessorCore) {
            Core core;
            for(WORD g=0;g<info->Processor.GroupCount;++g) {
                const auto& mask=info->Processor.GroupMask[g];
                for(unsigned bit=0;bit<sizeof(KAFFINITY)*8;++bit)
                    if(mask.Mask & (KAFFINITY(1)<<bit)) core.threads.push_back({mask.Group,bit,-1});
            }
            cores.push_back(core);
        }
        offset+=info->Size;
    }
    return cores;
}
inline void brand(char (&text)[49]) {
    int regs[4]; __cpuid(regs,0x80000000);
    if((unsigned)regs[0]<0x80000004) { strcpy_s(text,"CPU"); return; }
    for(int i=0;i<3;++i) { __cpuid(regs,0x80000002+i); memcpy(text+16*i,regs,16); }
    text[48]=0;
}
}
