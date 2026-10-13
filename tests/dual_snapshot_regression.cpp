// GPT6-Astra: real IPC publication -> package -> Playground CPU/GPU side routing.
#include "../DQN_Main_TUI_No_Terminal/playground.cpp"
#include <vector>
extern "C" int gpu_init();
using namespace independent_training;
#define NEED(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
static std::vector<unsigned char> read_bytes(const char* path){
    FILE* f=fopen(path,"rb");if(!f)return {};fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
    std::vector<unsigned char> b(n);fread(b.data(),1,b.size(),f);fclose(f);return b;
}
int main(){
    auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);g_current_agent=a.get();
    NEED(indep_ensure(a.get()));a->indep->set_active(1);
    g_hot_config.memory_scale=0;g_ctx.total_games_played=7;
    for(int side=0;side<2;++side){
        auto& net=a->indep->online((Side)side);memset(&net,0,sizeof(net));net.b4[side+1]=1.f;
    }
    int board[CELL_COUNT]={};char reply[512];
    for(int gpu=0;gpu<2;++gpu){
        if(gpu){NEED(gpu_init()==0 && gpu_net_activate(a.get())==0);
            a->indep->online(Side::White).b4[2]=0; // stale shadow must not be published
        }
        process_ipc_command("SAVE_SNAPSHOT",reply,sizeof(reply));service_snapshot_requests(a.get());
        NEED(playground_load_snapshot()==0 && g_play_independent==1);
        board[0]=0;NEED(playground_ai_move(board,0)==1);
        board[0]=1;NEED(playground_ai_move(board,1)==2);
        NEED(playground_analyze(board,2,nullptr,nullptr,0,nullptr,nullptr)==-1);
        auto frozen=std::make_unique<DQNAgent>();NEED(playground_copy_agent(frozen.get()));
        NEED(indep_active(frozen.get()) && frozen->indep->online(Side::White).b4[2]==1.f);
        indep_free(frozen.get());
        if(gpu) {NEED(gpu_net_download_sides_pair(a.get())==0);gpu_net_deactivate();}
    }
    auto old=read_bytes("current_model_package.bin");NEED(!old.empty());
    HANDLE hold=CreateFileA("current_model_package.bin",GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);
    NEED(hold!=INVALID_HANDLE_VALUE);
    NEED(!memory_save_model_package_pair(&a->indep->online(Side::Black),&a->indep->online(Side::White),"current_model_package.bin",8));
    CloseHandle(hold);NEED(read_bytes("current_model_package.bin")==old);
    old.back()^=1;FILE* f=fopen("current_model_package.bin","wb");NEED(f);
    NEED(fwrite(old.data(),1,old.size(),f)==old.size());fclose(f);
    NEED(playground_load_snapshot()==-1); // raw marker exists, but no unsafe fallback
    board[0]=1;NEED(playground_ai_move(board,1)==2);
    // Legacy V2 single-network package is explicitly shared by both colours.
    NEED(memory_save_model_package(&a->indep->online(Side::Black),"current_model_package.bin",9));
    NEED(playground_load_snapshot()==0 && !g_play_independent);
    NEED(playground_ai_move(board,1)==1);
    free_dqn(a.get());g_current_agent=nullptr;
    puts("PASS IPC CPU/GPU dual snapshot -> Playground physical side / frozen skill agent / atomic failure / corrupt no fallback / V2");
}
