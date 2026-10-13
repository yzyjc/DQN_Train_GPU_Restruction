// 存档安全回归：正常同名保存 / 失败保存不得破坏已有健康档
//
// 覆盖三件事：
//   1. 正常同名保存：覆盖已有健康档后，新档可加载且内容正确；目录里不残留 .tmp。
//   2. 失败保存不得破坏已有健康档：用**可触发的真实失败路径**让 save_checkpoint
//      返回 0，然后断言旧文件逐字节不变、目录里没有残留 .tmp/.mem.bak。
//      可触发的失败路径（不需要任何测试钩子）：
//        (a) 目标目录不存在 -> fopen(tmp) 失败；
//        (b) 回放缓冲缺失   -> 入口早退；
//        (c) 回放计数非法   -> 元数据校验失败。
//   3. 明确记录 GPU 不可信路径的证据类型：本节只覆盖**可注入的**失败路径；
//      「g_gpu_untrusted 之后 save 必然拒绝」是代码级论证，不是注入实验。
//      要做注入需要给 neural_net.cpp 加一个 test-only setter；本仓库刻意不加，
//      以免生产接口出现可被误用的开关。因此这里**不声称**做过该注入。
#include "independent_networks.h"
#include <memory>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace independent_training;

#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); fflush(stdout); return 1; } } while(0)

namespace {

std::string file_hash(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return std::string("(missing)");
    // 简单 FNV-1a 64：只用于「逐字节不变」的比较，不用于安全用途。
    unsigned long long h = 1469598103934665603ULL;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    fclose(f);
    char out[32];
    snprintf(out, sizeof(out), "%016llx", h);
    return std::string(out);
}

bool exists(const char* path) { return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES; }

/* 目录里是否有残留的临时/备份文件。 */
int residue_count(const char* dir) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int n = 0;
    do {
        const char* name = fd.cFileName;
        const size_t len = strlen(name);
        if (len >= 4 && (strcmp(name + len - 4, ".tmp") == 0 ||
                         (len >= 8 && strcmp(name + len - 8, ".mem.bak") == 0))) ++n;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return n;
}

} // namespace

int main() {
    puts("=== checkpoint save safety ==="); fflush(stdout);
    CreateDirectoryA("save_cases", nullptr);
    NEED(GetLastError() == 0 || GetLastError() == ERROR_ALREADY_EXISTS);

    auto agent = std::make_unique<DQNAgent>();
    init_dqn_ex(agent.get(), 0);
    g_current_agent = agent.get();
    NEED(agent->replay_black && agent->replay_white);

    const char* healthy = "save_cases/checkpoint_phase1_1000.bin";
    NEED(save_checkpoint(agent.get(), 1, 1000, 2, healthy));
    const std::string healthy_hash = file_hash(healthy);
    printf("[case1] healthy archive written, hash=%s size_ok=%d\n",
        healthy_hash.c_str(), (int)(file_hash(healthy) != "(missing)"));
    NEED(healthy_hash != "(missing)");

    /* ---- 1. 正常同名保存：覆盖后仍是可加载的合法档，且无残留 ---- */
    {
        NEED(save_checkpoint(agent.get(), 1, 1000, 2, healthy));
        const std::string after = file_hash(healthy);
        printf("[case1] same-name overwrite ok; hash=%s (changed=%d)\n",
            after.c_str(), (int)(after != healthy_hash));
        auto restored = std::make_unique<DQNAgent>();
        init_dqn_ex(restored.get(), 0);
        NEED(indep_ensure(restored.get()) != nullptr);
        int ph = 0, ep = 0, dp = 0;
        NEED(load_checkpoint(restored.get(), &ph, &ep, &dp, healthy, 1));
        NEED(ep == 1000);
        NEED(residue_count("save_cases") == 0);
        printf("[case1] overwritten archive reloads; tmp/mem.bak residue=0\n");
        free_dqn(restored.get());
    }

    /* ---- 2a. 目标目录不存在：必须返回 0，且不动任何已有档，不留残留 ---- */
    {
        const char* bad = "save_cases_no_such_dir/checkpoint_phase1_2000.bin";
        const std::string before = file_hash(healthy);
        const int rc = save_checkpoint(agent.get(), 1, 2000, 2, bad);
        const std::string after = file_hash(healthy);
        printf("[case2a] missing target directory -> save=%d (must be 0); "
            "healthy unchanged=%d; residue=%d\n", rc, (int)(before == after),
            residue_count("save_cases"));
        NEED(rc == 0);
        NEED(before == after);
        NEED(!exists(bad));
        NEED(residue_count("save_cases") == 0);
    }

    /* ---- 2b. 回放缓冲缺失：入口早退，不动已有档 ---- */
    {
        Experience* saved_black = agent->replay_black;
        agent->replay_black = nullptr;
        const std::string before = file_hash(healthy);
        const int rc = save_checkpoint(agent.get(), 1, 3000, 2, healthy);
        agent->replay_black = saved_black;
        const std::string after = file_hash(healthy);
        printf("[case2b] missing replay buffer -> save=%d (must be 0); healthy unchanged=%d\n",
            rc, (int)(before == after));
        NEED(rc == 0);
        NEED(before == after);
        NEED(residue_count("save_cases") == 0);
    }

    /* ---- 2c. 回放计数非法：元数据校验失败，不动已有档 ---- */
    {
        const int saved_count = agent->replay_black_count;
        agent->replay_black_count = REPLAY_CAPACITY + 1;
        const std::string before = file_hash(healthy);
        const int rc = save_checkpoint(agent.get(), 1, 4000, 2, healthy);
        agent->replay_black_count = saved_count;
        const std::string after = file_hash(healthy);
        printf("[case2c] invalid replay count -> save=%d (must be 0); healthy unchanged=%d\n",
            rc, (int)(before == after));
        NEED(rc == 0);
        NEED(before == after);
        NEED(residue_count("save_cases") == 0);
    }

    /* ---- 3. 证据类型声明 ---- */
    puts("[evidence-type] cases 1/2a/2b/2c are RUN experiments on real failure paths.");
    puts("[evidence-type] 'after g_gpu_untrusted the save is refused' is a CODE-LEVEL "
         "ARGUMENT (bind sets untrusted -> download_side requires !untrusted -> "
         "save_checkpoint's first gate returns 0). No injection was performed: doing so "
         "would need a test-only setter for g_gpu_untrusted, which this repo deliberately "
         "does not add to production interfaces.");

    free_dqn(agent.get());
    g_current_agent = nullptr;
    puts("PASS checkpoint save safety: normal overwrite is atomic and reloadable; three "
         "reachable failure paths return 0 and leave the existing healthy archive "
         "byte-identical with no .tmp/.mem.bak residue");
    return 0;
}
