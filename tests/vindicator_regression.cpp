#define main vindicator_original_main
#include "../Log_Vindicator/Log_Vindicator.cpp"
#undef main

static std::vector<char> bytes(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f)return {};
    std::vector<char> result; char b[4096]; size_t n;
    while((n=fread(b,1,sizeof(b),f)))result.insert(result.end(),b,b+n);
    fclose(f);return result;
}
static void fixture(const char* path, uint32_t magic, float v) {
    FILE* f=fopen(path,"wb");
    Header h={magic,3,1,{0,0,0}}; Prefix p={0,0}; float values[3]={v,.2f,.3f};
    fwrite(&h,sizeof(h),1,f);fwrite(&p,sizeof(p),1,f);fwrite(values,sizeof(values),1,f);fclose(f);
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    CreateDirectoryA("bad_header",nullptr);CreateDirectoryA("placeholder",nullptr);
    const char* bad="bad_header/metric_loss.bin";
    fixture(bad,0,.1f); auto original=bytes(bad);
    int failures=0;
    if(repair_file(fs::path(bad),true)!=1 || repair_file(fs::path(bad),false)!=1 || bytes(bad)!=original)++failures;
    const char* path="placeholder/metric_loss.bin";
    fixture(path,MAGIC,NAN); original=bytes(path);
    if(repair_file(fs::path(path),false)!=0 || bytes(path)!=original)++failures;
    fixture(path,MAGIC,INFINITY); original=bytes(path);
    if(repair_file(fs::path(path),false)!=1)++failures;
    if(bytes("placeholder/metric_loss.bin.vindicator.bak")!=original)++failures;
    auto repaired=bytes(path);
    if(repair_file(fs::path(path),false)!=0 || bytes(path)!=repaired)++failures;
    // A new corruption must not overwrite the first original backup.
    fixture(path,MAGIC,-INFINITY);
    if(repair_file(fs::path(path),false)!=1 || bytes("placeholder/metric_loss.bin.vindicator.bak")!=original)++failures;
    // Partial tail becomes one canonical missing record and stays stable.
    FILE* f=fopen(path,"ab");fputc(42,f);fclose(f);
    if(repair_file(fs::path(path),false)!=1)++failures;
    repaired=bytes(path);
    if(repair_file(fs::path(path),false)!=0 || bytes(path)!=repaired)++failures;
    printf("VINDICATOR REGRESSION: %s (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
