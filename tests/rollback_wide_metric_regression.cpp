// GPT6-Astra: the existing 14-field memory evaluation stream exceeds 40 bytes.
#include "metric_log_writer.h"
#include "training_log_rollback.h"
#include <cstring>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    NEED(CreateDirectoryA("wide_metric_case",nullptr));
    FILE* f=metric_log_open_append("wide_metric_case/metric_memory_eval.bin",14);NEED(f);
    float row[14];for(int i=0;i<14;++i)row[i]=(float)i+.25f;
    for(int ep=1;ep<=3;++ep){MetricLogPrefix p={ep,17};NEED(fwrite(&p,sizeof(p),1,f)==1);NEED(fwrite(row,sizeof(row),1,f)==1);}
    NEED(fclose(f)==0);NEED(rollback_training_logs("wide_metric_case",2,0));
    f=fopen("wide_metric_case/metric_memory_eval.bin","rb");NEED(f);MetricLogHeader h;
    NEED(fread(&h,sizeof(h),1,f)==1 && metric_log_header_valid(h,14));
    for(int ep=1;ep<=2;++ep){MetricLogPrefix p;float actual[14];
        NEED(fread(&p,sizeof(p),1,f)==1 && p.episode==ep && p.step==17);
        NEED(fread(actual,sizeof(actual),1,f)==1 && !memcmp(row,actual,sizeof(row)));}
    NEED(fgetc(f)==EOF);fclose(f);puts("PASS 14-field rollback retains complete records");return 0;
}
