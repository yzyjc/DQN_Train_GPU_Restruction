// GPT6-Astra: atomic two-network model export, CRC, rollback, explicit side load.
#include "independent_networks.h"
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
using namespace independent_training;
#define NEED(x) do{if(!(x)){printf("FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
static std::vector<unsigned char> bytes(const char* name){
 FILE* f=fopen(name,"rb");if(!f)return {};fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
 std::vector<unsigned char> b(n);fread(b.data(),1,b.size(),f);fclose(f);return b;
}
int main(){
 auto a=std::make_unique<DQNAgent>();init_dqn_ex(a.get(),0);NEED(indep_ensure(a.get()));a->indep->set_active(1);
 a->indep->online(Side::Black).b4[0]=.12f;a->indep->online(Side::White).b4[0]=.34f;
 NEED(save_model_pair(a.get(),"pair.bin"));auto original=bytes("pair.bin");NEED(!original.empty());
 auto b=std::make_unique<NeuralNet>();auto w=std::make_unique<NeuralNet>();
 NEED(load_model_pair_weights("pair.bin",b.get(),w.get()));NEED(b->b4[0]==.12f && w->b4[0]==.34f);
 a->indep->online(Side::Black).b4[0]=.56f;a->indep->online(Side::White).b4[0]=.78f;
 HANDLE hold=CreateFileA("pair.bin",GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);
 NEED(hold!=INVALID_HANDLE_VALUE);NEED(!save_model_pair(a.get(),"pair.bin"));CloseHandle(hold);
 NEED(bytes("pair.bin")==original);NEED(GetFileAttributesA("pair.bin.tmp")==INVALID_FILE_ATTRIBUTES);
 NEED(save_model_pair(a.get(),"pair.bin"));
 NEED(load_model_for_side(a.get(),0,"pair.bin"));NEED(load_model_for_side(a.get(),1,"pair.bin"));
 NEED(a->indep->online(Side::Black).b4[0]==.56f && a->indep->online(Side::White).b4[0]==.78f);
 // Corrupt a finite payload byte: CRC catches more than NaN/length errors.
 auto corrupt=bytes("pair.bin");corrupt[100]^=1;FILE* f=fopen("pair.bin","wb");NEED(f);
 NEED(fwrite(corrupt.data(),1,corrupt.size(),f)==corrupt.size());fclose(f);
 b->b4[0]=8.f;w->b4[0]=9.f;NEED(!load_model_pair_weights("pair.bin",b.get(),w.get()));
 NEED(b->b4[0]==8.f && w->b4[0]==9.f);NEED(!load_model_for_side(a.get(),1,"pair.bin"));
 auto shared=std::make_unique<DQNAgent>();init_dqn_ex(shared.get(),0);shared->q_net.b4[0]=.9f;
 NEED(save_model_pair(shared.get(),"legacy.bin"));NEED(bytes("legacy.bin").size()==sizeof(NeuralNet));
 NEED(load_model_for_side(a.get(),1,"legacy.bin"));NEED(a->indep->online(Side::White).b4[0]==.9f);
 NEED(a->indep->online(Side::Black).b4[0]==.56f);
 free_dqn(shared.get());free_dqn(a.get());
 puts("PASS atomic pair / blocked replacement preserves both / CRC / side selection / legacy raw");
}
