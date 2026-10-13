#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <share.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <filesystem>
#include <limits>
#include <tlhelp32.h>
#include <io.h>

namespace fs = std::filesystem;
#include "../DQN_Core/include/training_log_timeline.h"
#include "../DQN_Core/include/metric_log_format.h"
static constexpr uint32_t MAGIC = METRIC_LOG_MAGIC;
using Header = MetricLogHeader;
using Prefix = MetricLogPrefix;
using Schema = MetricLogSchema;

static const Schema* schema_for(const fs::path& p) {
    return metric_log_schema(p.filename().string().c_str());
}

static bool invalid_value(const Schema& s, int field, float v) {
    if (std::isnan(v)) return false; // Documented missing-value placeholder, not a new repair.
    if (!std::isfinite(v) || fabs(v) > 1e30f) return true;
    if ((strcmp(s.name,"metric_epsilon.bin") == 0 || strcmp(s.name,"metric_epsilon_episode.bin") == 0) && (v < 0 || v > 1)) return true;
    if (strcmp(s.name,"metric_episode_time.bin") == 0 && v < 0) return true;
    if (strcmp(s.name,"metric_win.bin") == 0) {
        if (field == 0) return v < 0 || v > 2 || floorf(v) != v;
        return v < 0 || v > 100;
    }
    if (strcmp(s.name,"metric_eval.bin") == 0 && field == 3) return v < 0 || v > 2 || floorf(v) != v;
    return false;
}

static bool is_missing_row(const Prefix& p, const float* values, uint32_t fields) {
    if (p.episode != -1 || p.step != -1) return false;
    for (uint32_t j = 0; j < fields; ++j) if (!std::isnan(values[j])) return false;
    return true;
}
static int checkpoint_mode(const Schema& s, const unsigned char* row) {
    int field = strcmp(s.name, "metric_win.bin") == 0 ? 0 :
        strcmp(s.name, "metric_eval.bin") == 0 ? 3 : -1;
    if (field < 0) return -1;
    float v = ((const float*)(row + sizeof(Prefix)))[field];
    return std::isfinite(v) && v >= 0 && v <= 2 && floorf(v) == v ? (int)v : -1;
}
// A checkpoint consists of ascending modes 0/1/2, not three episodes. Only
// borrow an episode from the same unambiguous group; never guess the cadence.
static int32_t checkpoint_episode(const Schema& s, const std::vector<unsigned char>& data,
    size_t stride, int64_t count, int64_t index) {
    int mode = checkpoint_mode(s, data.data() + index * stride);
    if (mode < 0) return -1;
    int32_t inferred = -1;
    for (int direction : {-1, 1}) {
        int previous = mode;
        for (int64_t j = index + direction; j >= 0 && j < count; j += direction) {
            const unsigned char* row = data.data() + j * stride;
            int next = checkpoint_mode(s, row);
            if (next < 0 || (direction < 0 ? next >= previous : next <= previous)) break;
            int32_t ep = ((const Prefix*)row)->episode;
            if (ep >= 0) {
                if (inferred >= 0 && inferred != ep) return -1;
                inferred = ep;
            }
            previous = next;
        }
    }
    return inferred;
}

static int repair_file(const fs::path& path, bool audit_only) {
    const Schema* schema = schema_for(path); if (!schema) return 0;
    FILE* f = _fsopen(path.string().c_str(), audit_only ? "rb" : "rb+", audit_only ? _SH_DENYNO : _SH_DENYRW);
    if (!f) { printf("[SKIP] %-30s 正在使用或无权限\n", path.filename().string().c_str()); return 0; }
    _fseeki64(f,0,SEEK_END); int64_t size=_ftelli64(f); rewind(f);
    Header h{}; bool bad_header = fread(&h,sizeof(h),1,f)!=1 || !metric_log_header_valid(h, schema->fields);
    if (bad_header) {
        fclose(f);
        printf("[WARN] %s: invalid header; layout unknown, original file preserved\n", path.filename().string().c_str());
        return 1;
    }
    const size_t rec_size=sizeof(Prefix)+schema->fields*sizeof(float);
    int64_t payload=size-(int64_t)sizeof(Header);
    int64_t complete=payload>0 ? payload/(int64_t)rec_size : 0;
    bool bad_tail=payload<0 || payload%(int64_t)rec_size!=0;
    if (audit_only && !bad_header) {
        int fixes=bad_tail?1:0; int32_t last_ep=-1,last_step=-1;
        std::vector<unsigned char> row(rec_size);
        fseek(f,sizeof(Header),SEEK_SET);
        int64_t read_count=0;
        while(read_count<complete && fread(row.data(),rec_size,1,f)==1) {
            Prefix* p=(Prefix*)row.data(); float* vals=(float*)(row.data()+sizeof(Prefix));
            if (is_missing_row(*p, vals, schema->fields)) { ++read_count; continue; }
            /* Trainer 的 game/steps 使用 0-based episode，0 是合法第一局。 */
            if(p->episode<0 || p->step<0) fixes++;
            for(uint32_t j=0;j<schema->fields;j++) if(invalid_value(*schema,(int)j,vals[j])) fixes++;
            if(p->episode>0) last_ep=p->episode; if(p->step>=0) last_step=p->step;
            read_count++;
        }
        fclose(f);
        printf("[%s] %-30s records=%lld fixes=%d\n",fixes?"WARN":" OK ",path.filename().string().c_str(),(long long)read_count,fixes);
        fflush(stdout);
        return fixes;
    }
    std::vector<unsigned char> data;
    if (!bad_header && complete>0) {
        try { data.resize((size_t)complete*rec_size); }
        catch (const std::bad_alloc&) {
            fclose(f);
            printf("[SKIP] %-30s 文件过大，内存不足，保留原文件\n", path.filename().string().c_str());
            return 0;
        }
        fseek(f,sizeof(Header),SEEK_SET);
        if (fread(data.data(),rec_size,(size_t)complete,f)!=(size_t)complete) { fclose(f); return 0; }
    }
    int fixes=bad_header?1:0; if(bad_tail) fixes++;
    int32_t last_ep=-1,last_step=-1;
    const float placeholder=std::numeric_limits<float>::quiet_NaN();
    for(int64_t i=0;i<complete;i++) {
        unsigned char* row=data.data()+(size_t)i*rec_size;
        Prefix* p=(Prefix*)row; float* vals=(float*)(row+sizeof(Prefix));
        if (is_missing_row(*p, vals, schema->fields)) continue;
        bool checkpoint = strcmp(schema->name,"metric_win.bin") == 0 || strcmp(schema->name,"metric_eval.bin") == 0;
        int32_t previous_ep=last_ep;
        if(p->episode<0) {
            int32_t inferred=(last_ep>=0 && last_ep<INT32_MAX)?(schema->per_episode?last_ep+1:last_ep):-1;
            if (checkpoint) inferred = checkpoint_episode(*schema, data, rec_size, complete, i);
            if(p->episode!=inferred) {p->episode=inferred; fixes++;}
        }
        /* step 在新 episode 必须从 0 重新开始，不能沿用上一局的末步继续加。 */
        if(p->step<0) {
            int32_t inferred=(p->episode>=0 && p->episode!=previous_ep)?0:
                ((last_step>=0 && last_step<INT32_MAX)?last_step+1:-1);
            if (checkpoint) inferred = checkpoint_mode(*schema, row);
            if(p->step!=inferred) {p->step=inferred; fixes++;}
        }
        for(uint32_t j=0;j<schema->fields;j++) if(invalid_value(*schema,(int)j,vals[j])) { vals[j]=placeholder; fixes++; }
        if(p->episode>=0) last_ep=p->episode; if(p->step>=0) last_step=p->step;
    }
    fclose(f);
    printf("[%s] %-30s records=%lld fixes=%d%s\n", fixes?"WARN":" OK ",path.filename().string().c_str(),(long long)complete,fixes,audit_only?"":" repaired");
    if(audit_only || fixes==0) return fixes;
    fs::path backup=path; backup += ".vindicator.bak";
    std::error_code ec;
    // Preserve the very first backup. Repeated repairs can share a clock tick;
    // retry exclusive creation with a suffix instead of overwriting or failing.
    const fs::path original_backup=backup;
    for (unsigned attempt=0; ; ++attempt) {
        ec.clear();
        fs::copy_file(path,backup,fs::copy_options::none,ec);
        if (ec != std::errc::file_exists || attempt >= 10000) break;
        backup=original_backup;
        backup += "." + std::to_string(GetCurrentProcessId()) + "." +
            std::to_string(GetTickCount64()) + "." + std::to_string(attempt);
    }
    if(ec) { printf("[FAIL] 备份失败: %s\n",ec.message().c_str()); return fixes; }
    fs::path temp=path; temp += ".vindicator.tmp";
    FILE* out=fopen(temp.string().c_str(),"wb"); if(!out) return fixes;
    Header clean{MAGIC,schema->fields,1,{0,0,0}};
    bool ok=fwrite(&clean,sizeof(clean),1,out)==1;
    if(ok && !data.empty()) ok=fwrite(data.data(),rec_size,(size_t)complete,out)==(size_t)complete;
    if(ok && bad_tail) {
        Prefix missing{-1,-1}; ok=fwrite(&missing,sizeof(missing),1,out)==1;
        for(uint32_t j=0;j<schema->fields && ok;j++) ok=fwrite(&placeholder,sizeof(placeholder),1,out)==1;
    }
    if(fflush(out)!=0 || _commit(_fileno(out))!=0) ok=false;
    if(fclose(out)!=0) ok=false;
    if(!ok) { fs::remove(temp,ec); printf("[FAIL] Write failed; original preserved\n"); return fixes; }
    /* Windows 上直接 rename 到已存在目标会失败；旧回退先删除原文件，第二次
       rename 再失败时活动日志会消失。使用原子替换，失败则原文件与备份都保留。 */
    if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) {
        ec=std::error_code((int)GetLastError(),std::system_category());
        printf("[FAIL] 替换失败，原文件与备份均保留: %s (%s)\n",
            backup.string().c_str(),ec.message().c_str());
        fs::remove(temp,ec);
    }
    return fixes;
}

static bool process_exists(const wchar_t* name) {
    HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0); if(snap==INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W pe{}; pe.dwSize=sizeof(pe); bool found=false;
    if(Process32FirstW(snap,&pe)) do { if(_wcsicmp(pe.szExeFile,name)==0){found=true;break;} } while(Process32NextW(snap,&pe));
    CloseHandle(snap); return found;
}

// Directory disappearance/access denial is a retryable scan failure, not an
// exception that kills the daemon. Check construction, entry status AND advance.
template<class Visitor>
static int visit_log_files(const fs::path& dir, Visitor visit) {
    std::error_code error;
    fs::directory_iterator it(dir, error), end;
    int issues = 0;
    while (!error && it != end) {
        std::error_code entry_error;
        const bool regular = it->is_regular_file(entry_error);
        if (entry_error) ++issues;
        else if (regular) issues += visit(it->path());
        it.increment(error);
    }
    if (error) {
        printf("[WARN] Cannot scan %s: %s; will retry next pass\n",
            dir.string().c_str(), error.message().c_str());
        ++issues;
    }
    return issues;
}
static int run_once(const fs::path& dir, bool audit) {
    return visit_log_files(dir, [audit](const fs::path& path) { return repair_file(path, audit); });
}

/* Trainer 写入期间只做不会遍历活动记录的结构检查，避免与持续追加竞争。 */
static int audit_structure_only(const fs::path& dir) {
    return visit_log_files(dir, [](const fs::path& path) {
        const Schema* s=schema_for(path); if(!s) return 0;
        FILE* f=_fsopen(path.string().c_str(),"rb",_SH_DENYNO); if(!f) return 1;
        Header h{}; bool bad=fread(&h,sizeof(h),1,f)!=1 || !metric_log_header_valid(h, s->fields);
        _fseeki64(f,0,SEEK_END); int64_t size=_ftelli64(f); fclose(f);
        int64_t rec=(int64_t)sizeof(Prefix)+(int64_t)s->fields*4;
        return (bad || size<(int64_t)sizeof(Header) || (size-(int64_t)sizeof(Header))%rec!=0) ? 1 : 0;
    });
}

int main(int argc,char** argv) {
    fs::path dir=(argc>1 && argv[1][0]!='-')?fs::path(argv[1]):fs::path("logs");
    bool audit=false, repair=false, daemon=true;
    for(int i=1;i<argc;i++) {
        if(strcmp(argv[i],"--audit")==0) { audit=true; daemon=false; }
        else if(strcmp(argv[i],"--repair")==0) { repair=true; daemon=false; }
        else if(strcmp(argv[i],"--daemon")==0) daemon=true;
    }
    std::error_code dir_error;
    if(!daemon && !fs::is_directory(dir, dir_error)) {
        printf("日志目录不可用: %s\n",dir.string().c_str()); return 2;
    }
    if(daemon) {
        HANDLE mutex=CreateMutexW(NULL,FALSE,L"Local\\DQNLogVindicatorSingleton");
        if(!mutex || GetLastError()==ERROR_ALREADY_EXISTS) return 0;
        for(;;) {
            bool trainer=process_exists(L"DQN_Train_Trainer_NoTerminal.exe");
            fs::path active = active_training_log_dir(dir.string().c_str());
            int issues=trainer ? audit_structure_only(active) : run_once(active,false);
            FILE* st=fopen((dir/"log_vindicator.status").string().c_str(),"w");
            if(st){SYSTEMTIME t;GetLocalTime(&t);fprintf(st,"running=1\ntrainer=%d\nissues=%d\nupdated=%04u-%02u-%02u %02u:%02u:%02u\n",trainer?1:0,issues,t.wYear,t.wMonth,t.wDay,t.wHour,t.wMinute,t.wSecond);fclose(st);}
            Sleep(30000);
        }
    }
    printf("Log Vindicator | %s | %s\n",audit?"只检查":"备份并修复",dir.string().c_str());
    int total=run_once(active_training_log_dir(dir.string().c_str()),audit && !repair);
    printf("完成：发现/处理 %d 个异常字段或结构问题。占位符：整数=-1，浮点=NaN。\n",total);
    return total?1:0;
}
