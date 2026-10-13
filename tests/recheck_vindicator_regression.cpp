#define main original_main
#include "../Log_Vindicator/Log_Vindicator.cpp"
#undef main
static int failures=0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n",__LINE__,#x);++failures; } } while(0)
static void checkpoint_test(const char* path, int fields) {
    // shared missing middle mode; missing mode0 at next boundary; ambiguous group
    int ep[]={1000,-1,1000,-1,2000,2000,3000,-1,4000,-1,-1,-1};
    FILE* f=fopen(path,"wb"); Header h={MAGIC,(uint32_t)fields,1,{}};fwrite(&h,sizeof(h),1,f);
    for(int i=0;i<12;++i){
        Prefix p={ep[i],i%3};float v[7]={1,2,.5f,0,50,10,40};
        if(fields==4){v[0]=(float)(i%3);v[1]=50;v[2]=10;v[3]=40;}
        else v[3]=(float)(i%3);
        fwrite(&p,sizeof(p),1,f);fwrite(v,4,fields,f);
    }
    fclose(f);
    CHECK(repair_file(path,false)==2);
    f=fopen(path,"rb");fseek(f,sizeof(h),SEEK_SET);
    int expected[]={1000,1000,1000,2000,2000,2000,3000,-1,4000,-1,-1,-1};
    for(int i=0;i<12;++i){Prefix p;float v[7];fread(&p,sizeof(p),1,f);fread(v,4,fields,f);CHECK(p.episode==expected[i]);}
    fclose(f);CHECK(repair_file(path,false)==0);
}
int main(){
    CreateDirectoryA("fixture",nullptr);
    checkpoint_test("fixture/metric_win.bin",4);checkpoint_test("fixture/metric_eval.bin",7);
    const char* path="fixture/metric_loss.bin";FILE* f=fopen(path,"wb");
    Header h={MAGIC,3,1,{}};Prefix p={-1,-1};float v[3]={NAN,NAN,NAN};
    fwrite(&h,sizeof(h),1,f);fwrite(&p,sizeof(p),1,f);fwrite(v,sizeof(v),1,f);fclose(f);
    CHECK(repair_file(path,true)==0 && repair_file(path,false)==0);
    printf("RECHECK VINDICATOR: %s (%d failures)\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}
