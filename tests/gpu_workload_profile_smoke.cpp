// GPT6-Astra: deterministic isolated GPU batch; no Trainer, checkpoints or replay.
#include "gpu_types.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <windows.h>
#include <string>
extern "C" {
volatile double g_bwd_fc4_us=0,g_bwd_fc3_us=0,g_bwd_fc2_us=0,g_bwd_fc1_us=0,g_bwd_pool_us=0,g_bwd_attn_us=0,g_bwd_embed_us=0;
volatile int g_bwd_count=0,g_ts_count=0;
volatile double g_ts_setup_us=0,g_ts_memset_us=0,g_ts_fwd_us=0,g_ts_hubler_us=0,g_ts_bwd_us=0,g_ts_validate_us=0;
int gpu_init();void gpu_shutdown();int gpu_nn_create(GpuNetHandle*);void gpu_nn_destroy(GpuNetHandle*);
int gpu_nn_upload(GpuNetHandle,const void*,size_t);int gpu_nn_download(GpuNetHandle,void*,size_t);
int gpu_nn_train_step_batch(GpuNetHandle,const float*,const int*,const float*,int,float,float*);
}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}}while(0)
int main(int argc,char** argv){
    CHECK(argc==2);CHECK(hipInit(0)==hipSuccess);CHECK(hipSetDevice(0)==hipSuccess);GpuNetHandle h={};CHECK(gpu_nn_create(&h)==0);
    std::vector<float> p(sizeof(GpuNetParams)/sizeof(float));
    for(size_t i=0;i<p.size();++i)p[i]=((int)(i%23)-11)*0.0001f;
    CHECK(gpu_nn_upload(h,p.data(),sizeof(GpuNetParams))==0);
    float states[32*GPU_STATE_DIM]={},targets[32];int actions[32];
    for(int i=0;i<32;++i){actions[i]=i%25;targets[i]=i%2?.2f:-.2f;states[i*GPU_STATE_DIM+25]=(float)(i%2);}
    LARGE_INTEGER start,end,freq;QueryPerformanceFrequency(&freq);QueryPerformanceCounter(&start);
    float loss=-1;CHECK(gpu_nn_train_step_batch(h,states,actions,targets,32,.0002f,&loss)==0);
    QueryPerformanceCounter(&end);
    CHECK(std::isfinite(loss));CHECK(gpu_nn_download(h,p.data(),sizeof(GpuNetParams))==0);
    FILE* f=nullptr;fopen_s(&f,argv[1],"wb");CHECK(f);CHECK(fwrite(p.data(),sizeof(GpuNetParams),1,f)==1);fclose(f);
    const std::string timing=std::string(argv[1])+".host.csv";
    fopen_s(&f,timing.c_str(),"w");CHECK(f);
    fprintf(f,"module,host_us\nsetup,%.3f\ninput_clear,%.3f\nforward,%.3f\nhuber,%.3f\nbackward,%.3f\nvalidation,%.3f\n",
        (double)g_ts_setup_us,(double)g_ts_memset_us,(double)g_ts_fwd_us,(double)g_ts_hubler_us,(double)g_ts_bwd_us,(double)g_ts_validate_us);
    fprintf(f,"wall,%.3f\n",(end.QuadPart-start.QuadPart)*1e6/freq.QuadPart);fclose(f);
    char hold_text[32]={};DWORD hold_len=GetEnvironmentVariableA("DQN_RGP_HOLD_MS",hold_text,sizeof(hold_text));
    if(hold_len && hold_len<sizeof(hold_text)){int hold=atoi(hold_text);if(hold<0)hold=0;if(hold>60000)hold=60000;if(hold)Sleep((DWORD)hold);}
    printf("PASS batch=32 loss=%.9g\n",loss);gpu_nn_destroy(&h);return 0;
}
