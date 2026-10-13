// trajectory_analyzer.cpp - 训练轨迹分析报告实现。
//
// 只读。所有统计口径在输出中显式声明单位、窗口定义、样本数与有效样本下限。
// 无法计算时输出 null + 原因；不伪造有限数值；不输出非法 JSON。
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
// windows.h 的 min/max 宏会破坏 std::min/std::max，必须在包含 windows.h 之前禁用。
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../include/trajectory_analyzer.h"
#include "../include/metric_log_reader.h"
#include "../include/metric_log_format.h"
#include "../include/metric_eval_reader.h"
#include "../include/training_log_timeline.h"

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace traj {

/* ============================ 小工具 ============================ */

const char* unit_name(int unit) {
    switch (unit) {
    case UNIT_RECORDS:  return "records";
    case UNIT_EPISODES: return "episodes";
    case UNIT_STEPS:    return "steps";
    case UNIT_SECONDS:  return "seconds";
    case UNIT_PERCENT:  return "percent";
    case UNIT_RATIO:    return "ratio";
    case UNIT_LOSS:     return "loss";
    case UNIT_QVALUE:   return "q_value";
    default:            return "none";
    }
}

const char* aggregation_name(int agg) {
    switch (agg) {
    case AGG_RAW:                return "raw_per_record";
    case AGG_BATCH_MEAN:         return "batch_mean";
    case AGG_EPISODE_400_GAMES:  return "episode_aggregate_400_games_black_white_averaged";
    case AGG_EPISODE_SCALAR:     return "episode_scalar";
    case AGG_CHECKPOINT_MODE:    return "checkpoint_per_mode";
    default:                     return "raw";
    }
}

static bool finite_d(double v) { return v == v && v > -1e308 && v < 1e308; }

static std::string fmt_d(double v, int prec = 6) {
    if (!finite_d(v)) return "null";
    char b[64];
    snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}

static std::string fmt_i(int64_t v) {
    char b[32];
    snprintf(b, sizeof(b), "%lld", (long long)v);
    return b;
}

// JSON 数值：非有限一律 null（绝不输出 NaN/Inf，否则 JSON 非法）
static std::string jnum(double v, int prec = 6) {
    if (!finite_d(v)) return "null";
    char b[64];
    snprintf(b, sizeof(b), "%.*g", prec, v);
    // 防御：某些 libc 会打印 inf/nan
    if (!strpbrk(b, "0123456789")) return "null";
    if (strstr(b, "inf") || strstr(b, "nan") || strstr(b, "INF") || strstr(b, "NAN")) return "null";
    return b;
}

static std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
            else o += (char)c;
        }
    }
    o += "\"";
    return o;
}

static std::string now_string() {
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_s(&tmv, &t);
    char b[32];
    snprintf(b, sizeof(b), "%04d-%02d-%02d %02d:%02d:%02d",
        tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return b;
}

// 指标家族：同家族优先合并，跨家族不轻易合成一个事件。
static const char* metric_family(const std::string& k) {
    if (k.rfind("win_rate", 0) == 0 || k.rfind("lose_rate", 0) == 0 || k.rfind("draw_rate", 0) == 0)
        return "PERFORMANCE";
    if (k.rfind("q_value", 0) == 0 || k.rfind("target_q", 0) == 0 || k.rfind("delta_d", 0) == 0)
        return "VALUE_NETWORK";
    if (k.rfind("loss", 0) == 0) return "OPTIMIZATION";
    if (k.rfind("reward", 0) == 0) return "REWARD";
    if (k.rfind("epsilon", 0) == 0 || k.rfind("eval_epsilon", 0) == 0 ||
        k.rfind("depth", 0) == 0 || k.rfind("phase", 0) == 0) return "SCHEDULE";
    if (k.rfind("episode_seconds", 0) == 0 || k.rfind("episode_length", 0) == 0) return "SYSTEM";
    return "OTHER";
}

/* ============================ 统计口径 ============================ */

// 精确流式统计（Welford），对全部记录计算，不受抽样影响。
struct StreamAcc {
    int64_t n = 0;
    double mean = 0, m2 = 0, min = 0, max = 0;
    void add(double v) {
        if (!finite_d(v)) return;
        if (n == 0) { min = max = v; }
        else { if (v < min) min = v; if (v > max) max = v; }
        ++n;
        double d = v - mean;
        mean += d / (double)n;
        m2 += d * (v - mean);
    }
    // 样本标准差（n-1）。n<2 时无定义。
    bool stddev(double* out, std::string* reason) const {
        if (n < 2) { if (reason) *reason = "sample_count<2: sample standard deviation undefined"; return false; }
        double var = m2 / (double)(n - 1);
        if (!(var >= 0)) { if (reason) *reason = "negative variance (numerical)"; return false; }
        *out = sqrt(var);
        return true;
    }
    // 总体标准差（n）。
    bool stddev_pop(double* out, std::string* reason) const {
        if (n < 1) { if (reason) *reason = "no valid samples"; return false; }
        double var = m2 / (double)n;
        if (!(var >= 0)) { if (reason) *reason = "negative variance (numerical)"; return false; }
        *out = sqrt(var);
        return true;
    }
};

static double quantile_sorted(const std::vector<double>& s, double q) {
    if (s.empty()) return NAN;
    if (s.size() == 1) return s[0];
    double pos = q * (double)(s.size() - 1);
    size_t lo = (size_t)floor(pos);
    size_t hi = (size_t)ceil(pos);
    if (hi >= s.size()) hi = s.size() - 1;
    double frac = pos - (double)lo;
    return s[lo] + frac * (s[hi] - s[lo]);   // 线性插值分位数
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return quantile_sorted(v, 0.5);
}

// MAD：中位数绝对偏差，再乘 1.4826 使其与标准差可比（声明口径）。
static double mad_scaled(std::vector<double> v) {
    if (v.empty()) return NAN;
    double med = median_of(v);
    std::vector<double> dev;
    dev.reserve(v.size());
    for (double x : v) dev.push_back(fabs(x - med));
    return median_of(dev) * 1.4826;
}

// 百分位排名：严格小于的比例 + 等于的一半（声明口径）。
static double percentile_rank(const std::vector<double>& sorted, double v) {
    if (sorted.empty()) return NAN;
    size_t lo = (size_t)(std::lower_bound(sorted.begin(), sorted.end(), v) - sorted.begin());
    size_t hi = (size_t)(std::upper_bound(sorted.begin(), sorted.end(), v) - sorted.begin());
    return 100.0 * ((double)lo + 0.5 * (double)(hi - lo)) / (double)sorted.size();
}

// 线性回归（最小二乘）。返回 R²。分母为 0 时判为无定义。
static bool linreg(const std::vector<double>& x, const std::vector<double>& y,
    double* slope, double* intercept, double* r2, std::string* reason) {
    size_t n = x.size();
    if (n < 3) { if (reason) *reason = "n<3: regression undefined"; return false; }
    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (size_t i = 0; i < n; ++i) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; syy += y[i] * y[i]; }
    double denom = (double)n * sxx - sx * sx;
    if (fabs(denom) < 1e-12) { if (reason) *reason = "zero variance in x: slope undefined"; return false; }
    *slope = ((double)n * sxy - sx * sy) / denom;
    *intercept = (sy - (*slope) * sx) / (double)n;
    double ss_tot = syy - sy * sy / (double)n;
    double ss_res = 0;
    for (size_t i = 0; i < n; ++i) {
        double p = (*slope) * x[i] + (*intercept);
        ss_res += (y[i] - p) * (y[i] - p);
    }
    if (ss_tot < 1e-12) { *r2 = 1.0; if (reason) *reason = "constant y: R2 reported as 1 by definition"; }
    else *r2 = 1.0 - ss_res / ss_tot;
    return true;
}

// Pearson 相关。任一侧零方差 → 无定义（不给假 0）。
static bool pearson(const std::vector<double>& a, const std::vector<double>& b,
    double* r, std::string* reason) {
    size_t n = a.size();
    if (n < 3) { if (reason) *reason = "pairs<3: correlation undefined"; return false; }
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (size_t i = 0; i < n; ++i) { sa += a[i]; sb += b[i]; saa += a[i] * a[i]; sbb += b[i] * b[i]; sab += a[i] * b[i]; }
    double va = saa - sa * sa / (double)n, vb = sbb - sb * sb / (double)n;
    if (va <= 1e-15 || vb <= 1e-15) {
        if (reason) *reason = (va <= 1e-15 ? "series A has zero variance" : "series B has zero variance");
        return false;
    }
    double cov = sab - sa * sb / (double)n;
    *r = cov / sqrt(va * vb);
    if (*r > 1.0) *r = 1.0;
    if (*r < -1.0) *r = -1.0;
    return true;
}

/* ============================ 校验（复用既有口径） ============================ */

static const MetricLogSchema* schema_of(const char* name) { return metric_log_schema(name); }

// 与 Log_Vindicator.cpp 的 invalid_value 同口径；NaN 是文档化的缺失占位，不算非法。
static bool field_invalid(const MetricLogSchema& s, int field, float v) {
    if (std::isnan(v)) return false;
    if (!std::isfinite(v) || fabs(v) > 1e30f) return true;
    if ((strcmp(s.name, "metric_epsilon.bin") == 0 || strcmp(s.name, "metric_epsilon_episode.bin") == 0)
        && (v < 0 || v > 1)) return true;
    if (strcmp(s.name, "metric_episode_time.bin") == 0 && v < 0) return true;
    if (strcmp(s.name, "metric_win.bin") == 0) {
        if (field == 0) return v < 0 || v > 2 || floorf(v) != v;
        return v < 0 || v > 100;
    }
    if (strcmp(s.name, "metric_eval.bin") == 0 && field == 3) return v < 0 || v > 2 || floorf(v) != v;
    return false;
}

static bool is_placeholder_row(const BRRecord& r, int fields) {
    if (r.episode != -1 || r.step != -1) return false;
    for (int j = 0; j < fields; ++j) if (!std::isnan(r.fields[j])) return false;
    return true;
}

/* ============================ 读取层 ============================ */

struct RawFile {
    std::string name;
    std::string path;
    int fields = 0;
    int64_t size = 0;
    int64_t complete_records = 0;
    bool    header_ok = false;
    int64_t truncated_bytes = 0;
};

static bool stat_file(const std::string& path, int64_t* size, uint64_t* ctime_id) {
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fad)) return false;
    *size = ((int64_t)fad.nFileSizeHigh << 32) | (int64_t)fad.nFileSizeLow;
    if (ctime_id)
        *ctime_id = ((uint64_t)fad.ftCreationTime.dwHighDateTime << 32) | (uint64_t)fad.ftCreationTime.dwLowDateTime;
    return true;
}

// 打开一个 metric_*.bin，校验头，并把读取边界固定在当前文件大小。
static bool open_raw(const std::string& dir, const char* name, RawFile* out, std::string* err) {
    out->name = name;
    out->path = dir + name;
    FILE* f = fopen(out->path.c_str(), "rb");
    if (!f) { if (err) *err = "open failed"; return false; }
    MetricLogHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); if (err) *err = "short header"; return false; }
    if (!metric_log_header_valid(h)) { fclose(f); if (err) *err = "invalid header/magic/field_count"; return false; }
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); if (err) *err = "seek failed"; return false; }
    int64_t size = _ftelli64(f);
    fclose(f);
    out->fields = (int)h.field_count;
    out->size = size;
    const int64_t stride = (int64_t)metric_log_record_size(h.field_count);
    const int64_t payload = size - (int64_t)sizeof(MetricLogHeader);
    out->complete_records = payload > 0 ? payload / stride : 0;
    out->truncated_bytes = payload > 0 ? payload % stride : 0;
    out->header_ok = true;
    return true;
}

// 流式读取全部记录（分块），边界固定为打开时的大小。
template <class Fn>
static int64_t stream_raw(const RawFile& rf, Fn fn, volatile LONG* cancel = nullptr) {
    FILE* f = fopen(rf.path.c_str(), "rb");
    if (!f) return 0;
    const int stride = 8 + rf.fields * 4;
    const int64_t limit = (int64_t)sizeof(MetricLogHeader) + rf.complete_records * stride;
    std::vector<BRRecord> chunk(8192);
    int64_t index = 0, total = 0;
    if (_fseeki64(f, sizeof(MetricLogHeader), SEEK_SET) != 0) { fclose(f); return 0; }
    while (true) {
        if (cancel && InterlockedCompareExchange(cancel, 0, 0)) break;
        int made = 0;
        while (made < (int)chunk.size() && _ftelli64(f) + stride <= limit) {
            if (fread(&chunk[made], stride, 1, f) != 1) break;
            ++made;
        }
        if (made == 0) break;
        for (int i = 0; i < made; ++i) { fn(chunk[i], index); ++index; ++total; }
        if (made < (int)chunk.size()) break;
    }
    fclose(f);
    return total;
}

/* ============================ 序列构建 ============================ */

struct SeriesBuilder {
    Series s;
    StreamAcc acc;
    // 抽样：超过 cap 时按等距确定性子集保留，保证可重复
    int64_t cap = 0;
    int64_t seen = 0;
    int64_t next_keep = 0;
    bool sampling = false;

    SeriesBuilder() = default;   // std::map::operator[] 需要；未预注册的键会在落地阶段被丢弃
    SeriesBuilder(Series proto, int64_t cap_) : s(proto), cap(cap_) {}
    void add(int32_t ep, int32_t step, int64_t index, double v) {
        acc.add(v);
        ++seen;
        if (cap <= 0 || seen <= cap) {
            Sample smp; smp.episode = ep; smp.step = step; smp.record_index = index;
            smp.value = v; smp.finite = finite_d(v);
            s.samples.push_back(smp);
        } else {
            if (!sampling) sampling = true;
            // 渐进抽稀：stride 随已见条数增长（前密后疏，非等距）
            int64_t stride = (seen + cap - 1) / cap;
            if (stride < 1) stride = 1;
            if (index % stride == 0) {
                Sample smp; smp.episode = ep; smp.step = step; smp.record_index = index;
                smp.value = v; smp.finite = finite_d(v);
                s.samples.push_back(smp);
            }
        }
    }
    Series finish() {
        s.sampling_interval_known = false;
        const int64_t retained = (int64_t)s.samples.size();
        // 把流式累加器的**全量精确**统计带出去（此前被丢弃，导致报告里的均值/标准差
        // 实际是在抽样后的保留子集上算的）。
        s.total_records = acc.n;
        s.exact_mean = acc.mean;
        s.exact_min = acc.min;
        s.exact_max = acc.max;
        {
            std::string why;
            s.exact_std_valid = acc.stddev(&s.exact_stddev, &why);
            if (!s.exact_std_valid) { s.exact_stddev = NAN; s.exact_std_null_reason = why; }
        }
        if (retained < seen) {
            // 实际是「渐进抽稀」：前 cap 条原样保留，之后 stride 随已见条数增长而变大，
            // 因此保留点**前密后疏**，不是等距子集。这里如实声明，避免误称“等距”。
            s.sampling_note =
                "progressive decimation: retained " + fmt_i(retained) + " of " + fmt_i(seen) +
                " records (first " + fmt_i(cap) + " verbatim, then 1/stride decimation with stride growing "
                "as more records are seen => retained points are denser early, sparser late; NOT a uniform "
                "even-spaced subset). Streaming statistics (n/mean/std/min/max) still cover ALL " +
                fmt_i(seen) + " records exactly.";
        } else {
            s.sampling_note = "full history retained verbatim (no sampling): " + fmt_i(seen) + " records";
        }
        return s;
    }
};

/* ============================ 分析主体 ============================ */

static void add_limitation(AnalyzeResult& r, const std::string& s) {
    for (const auto& x : r.limitations) if (x == s) return;
    r.limitations.push_back(s);
}

static void mark_unavailable(AnalyzeResult& r, const char* item, const char* reason) {
    UnavailableItem u; u.item = item; u.reason = reason;
    r.unavailable.push_back(u);
}

// 扫描回档分支：<name>.rollback_<pid>_<tick>.bak
struct RollbackGroup {
    std::string id;
    uint32_t pid = 0;
    uint64_t tick = 0;
    std::vector<std::string> files;
};

static void scan_rollback_branches(const std::string& dir, std::vector<RollbackGroup>& out) {
    std::map<std::string, RollbackGroup> m;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string name = fd.cFileName;
        size_t pos = name.find(".rollback_");
        if (pos == std::string::npos) continue;
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bak") != 0) continue;
        std::string suffix = name.substr(pos + 10, name.size() - 4 - (pos + 10));
        size_t us = suffix.find('_');
        if (us == std::string::npos) continue;
        RollbackGroup g;
        g.pid = (uint32_t)strtoul(suffix.substr(0, us).c_str(), nullptr, 10);
        g.tick = (uint64_t)_strtoui64(suffix.substr(us + 1).c_str(), nullptr, 10);
        g.id = "rollback:" + suffix;
        g.files.push_back(name);
        auto it = m.find(g.id);
        if (it == m.end()) m[g.id] = g;
        else it->second.files.push_back(name);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    for (auto& kv : m) out.push_back(kv.second);
    std::sort(out.begin(), out.end(),
        [](const RollbackGroup& a, const RollbackGroup& b) { return a.tick < b.tick; });
}

// 读取 latest_side_win.txt（仅最新快照）
struct SideWinSnapshot {
    bool present = false;
    int episode = -1;
    double black[3] = { NAN, NAN, NAN };
    double white[3] = { NAN, NAN, NAN };
    std::string raw;
};
static SideWinSnapshot read_side_win(const std::string& dir) {
    SideWinSnapshot s;
    FILE* f = fopen((dir + "latest_side_win.txt").c_str(), "rb");
    if (!f) return s;
    char line[256] = {};
    if (fgets(line, sizeof(line), f)) {
        s.raw = line;
        int ep = -1; double v[6] = { 0,0,0,0,0,0 };
        if (sscanf(line, "side_win_v1 %d %lf %lf %lf %lf %lf %lf",
            &ep, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 7) {
            s.present = true; s.episode = ep;
            for (int i = 0; i < 3; ++i) { s.black[i] = v[i]; s.white[i] = v[3 + i]; }
        }
    }
    fclose(f);
    return s;
}

/* ============================ 入口 ============================ */

AnalyzeResult Analyze(const Options& options, ProgressFn progress, void* user) {
    AnalyzeResult r;
    r.options = options;
    r.generated_at = now_string();
    r.read_start_unix = (int64_t)time(nullptr);
    auto stage = [&](const char* name, const char* detail) {
        StageLog s; s.stage = name; s.detail = detail ? detail : "";
        r.stage_log.push_back(s);
        if (progress) progress(name, 0, 0, user);
    };

    DWORD t0 = GetTickCount();
    const std::string dir = training_log_root(options.log_dir.c_str());
    r.source_dir = dir;

    // ---- 时间线 ----
    stage("timeline", "resolving active timeline");
    std::string active = active_training_log_dir(options.log_dir.c_str());
    r.timeline_id = (active.size() > dir.size() && active.find("pretrain") != std::string::npos)
        ? "pretrain" : "formal";
    {
        Timeline t;
        t.id = r.timeline_id; t.kind = "active"; t.dir = active; t.is_active = true;
        FILE* f = fopen((dir + "active_timeline.txt").c_str(), "rb");
        if (f) {
            char b[64] = {}; size_t n = fread(b, 1, sizeof(b) - 1, f);
            (void)n;
            int ph = -1;
            if (sscanf(b, "timeline_v1 %d", &ph) == 1) t.marker_phase = ph;
            fclose(f);
        } else {
            t.note = "active_timeline.txt missing; assuming formal timeline dir";
        }
        r.timelines.push_back(t);
    }

    // ---- 回档分支 ----
    if (options.include_rollback_branches) {
        stage("branches", "scanning rollback branches");
        std::vector<RollbackGroup> groups;
        scan_rollback_branches(dir, groups);
        for (const auto& g : groups) {
            Timeline t;
            t.id = g.id; t.kind = "rollback"; t.dir = dir;
            t.rollback_pid = g.pid; t.rollback_tick = g.tick;
            t.note = fmt_i((int64_t)g.files.size()) + " backup file(s); pre-rollback snapshot, "
                     "must not be concatenated with the active timeline";
            r.timelines.push_back(t);
        }
        add_limitation(r,
            "回档分支以 <name>.rollback_<pid>_<GetTickCount64()>.bak 为单位识别；后缀第二段是系统 tick（非 episode），"
            "同一次回档事件的所有文件共享同一后缀。各分支不会被拼成连续轨迹。");
    }

    // ---- 读取活动时间线 ----
    stage("read", "streaming full history of the active timeline");
    struct FilePlan {
        const char* name;
        int   expect_fields;
        bool  zero_based;
        bool  per_episode;
    };
    const FilePlan plans[] = {
        { "metric_loss.bin", 3, false, false },
        { "metric_epsilon.bin", 1, false, false },
        { "metric_epsilon_episode.bin", 1, false, true },
        { "metric_rewards.bin", 8, false, true },
        { "metric_episode_time.bin", 1, false, true },
        { "metric_game.bin", 1, true,  true },
        { "metric_win.bin", 4, false, false },
        { "metric_eval.bin", 7, false, false },
    };

    // 各序列构造器。
    // 每训练步写入的序列（loss/q/target_q/D/epsilon_step）在长期训练下可达数百万条，
    // 全量驻留会耗尽内存。因此：**流式统计对全部记录精确**，点级分析使用确定性的
    // 等距子集，并在 sampling_note 中明确声明为近似。每局序列基本无上限。
    std::map<std::string, SeriesBuilder> builders;
    auto mk = [&](const char* key, const char* file, const char* field, int fidx,
                  const char* meaning, int unit, int agg, const char* epsem, const char* stepsem,
                  bool zb, bool pe) {
        int64_t cap = options.max_records_per_series;
        if (cap <= 0) cap = pe ? 1000000 : 300000;
        Series proto;
        proto.key = key; proto.file = file; proto.field = field; proto.field_index = fidx;
        proto.meaning = meaning; proto.unit = unit; proto.aggregation = agg;
        proto.episode_semantics = epsem; proto.step_semantics = stepsem;
        proto.zero_based = zb; proto.per_episode = pe; proto.timeline_grouped = true;
        return builders.emplace(key, SeriesBuilder(proto, cap)).first;
    };

    mk("loss", "metric_loss.bin", "loss", 0, "训练损失（同一 batch 均值）", UNIT_LOSS, AGG_BATCH_MEAN,
       "chart_ep = 已完成局数+1（每局多次更新共享同一 ep）", "replay_type: 0=黑方回放缓冲,1=白方回放缓冲",
       false, false);
    mk("q_value", "metric_loss.bin", "q_value", 1, "Q 值（同一 batch 均值）", UNIT_QVALUE, AGG_BATCH_MEAN,
       "同 loss", "replay_type", false, false);
    mk("target_q", "metric_loss.bin", "target_q", 2, "Target-Q（同一 batch 均值）", UNIT_QVALUE, AGG_BATCH_MEAN,
       "同 loss", "replay_type", false, false);
    mk("delta_d", "metric_loss.bin", "q_value - target_q", -1,
       "差值 D = Q − Target-Q（同一记录内配对，同 batch）", UNIT_QVALUE, AGG_BATCH_MEAN,
       "同 loss", "replay_type", false, false);
    mk("epsilon_step", "metric_epsilon.bin", "epsilon", 0, "探索率（每训练步）", UNIT_RATIO, AGG_RAW,
       "chart_ep = 已完成局数+1", "恒为 0（注意：本文件的 step 不是 replay_type；"
       "replay_type 只在 metric_loss.bin 的 step 中）", false, false);
    mk("epsilon_episode", "metric_epsilon_episode.bin", "epsilon", 0, "探索率（每局）", UNIT_RATIO, AGG_EPISODE_SCALAR,
       "已完成局号", "恒为 0", false, true);
    mk("episode_length", "metric_game.bin", "game_length_steps", 0, "局长（决策步数）", UNIT_STEPS, AGG_EPISODE_SCALAR,
       "0 基局号（首局为 0）", "恒为 0", true, true);
    mk("episode_seconds", "metric_episode_time.bin", "seconds", 0, "单局墙钟耗时", UNIT_SECONDS, AGG_EPISODE_SCALAR,
       "已完成局号", "恒为 0", false, true);
    mk("reward_total", "metric_rewards.bin", "total", 7, "该局 7 项奖励总分", UNIT_NONE, AGG_EPISODE_SCALAR,
       "已完成局号", "恒为 0", false, true);
    for (int i = 0; i < 7; ++i) {
        char k[32], f[32];
        snprintf(k, sizeof(k), "reward_%d", i);
        snprintf(f, sizeof(f), "actual[%d]", i);
        mk(k, "metric_rewards.bin", f, i, "该局第 i 项实际奖励分量", UNIT_NONE, AGG_EPISODE_SCALAR,
           "已完成局号", "恒为 0", false, true);
    }
    for (int m = 0; m < 3; ++m) {
        char k[32];
        snprintf(k, sizeof(k), "win_rate_mode%d", m);
        mk(k, "metric_win.bin", "win%", 1, "胜率（每 mode 400 盘，已黑白平均）", UNIT_PERCENT,
           AGG_EPISODE_400_GAMES, "评估 episode", "mode(0..2)", false, false);
        snprintf(k, sizeof(k), "draw_rate_mode%d", m);
        mk(k, "metric_win.bin", "draw%", 2, "平率（每 mode 400 盘，已黑白平均）", UNIT_PERCENT,
           AGG_EPISODE_400_GAMES, "评估 episode", "mode(0..2)", false, false);
        snprintf(k, sizeof(k), "lose_rate_mode%d", m);
        mk(k, "metric_win.bin", "lose%", 3, "败率（每 mode 400 盘，已黑白平均）", UNIT_PERCENT,
           AGG_EPISODE_400_GAMES, "评估 episode", "mode(0..2)", false, false);
        snprintf(k, sizeof(k), "depth_mode%d", m);
        mk(k, "metric_eval.bin", "minimax_depth", 1, "对手 Minimax 深度", UNIT_NONE, AGG_CHECKPOINT_MODE,
           "评估 episode", "mode(0..2)", false, false);
        snprintf(k, sizeof(k), "eval_epsilon_mode%d", m);
        mk(k, "metric_eval.bin", "epsilon", 2, "评估用探索率", UNIT_RATIO, AGG_CHECKPOINT_MODE,
           "评估 episode", "mode(0..2)", false, false);
        snprintf(k, sizeof(k), "phase_mode%d", m);
        mk(k, "metric_eval.bin", "phase", 0, "训练阶段", UNIT_NONE, AGG_CHECKPOINT_MODE,
           "评估 episode", "mode(0..2)", false, false);
    }

    // 类别序列：replay_type
    const std::string category_key = "replay_type";
    std::vector<std::pair<int32_t, int>> category_seq;  // (episode, value)

    int64_t bytes_read = 0;
    for (const auto& plan : plans) {
        std::string err;
        RawFile rf;
        if (!open_raw(dir, plan.name, &rf, &err)) {
            add_limitation(r, std::string("缺少或不可读: ") + plan.name + " (" + err + ")");
            continue;
        }
        if (plan.expect_fields && rf.fields != plan.expect_fields) {
            add_limitation(r, std::string(plan.name) + " 字段数=" + fmt_i(rf.fields) +
                "，预期 " + fmt_i(plan.expect_fields) + "；按文件头实际值解读，可能语义不符");
        }
        bytes_read += rf.size;
        r.quality.records_seen += rf.complete_records;
        r.quality.boundary_size += rf.size;
        if (rf.truncated_bytes > 0) {
            r.quality.truncated_tail = true;
            r.quality.truncated_bytes += rf.truncated_bytes;
        }
        const MetricLogSchema* sc = schema_of(plan.name);
        int32_t last_ep = INT32_MIN;
        // 重复检测使用 O(1) 内存：只比较**紧邻记录**的**全部字段**。
        // 定义：紧邻的两条记录 (episode, step) 相同且所有字段逐位相同 = 完全重复记录。
        // 但只有当该流的 (episode, step) 本应唯一（schema per_episode=true）时才算作重复；
        // 其它流的 step 是 replay_type 之类的类别/常量，重复是正常现象，另行单独计数。
        bool have_prev_key = false;
        int32_t prev_key_ep = 0, prev_key_step = 0;
        std::vector<float> prev_fields;
        stream_raw(rf, [&](const BRRecord& rec, int64_t index) {
            if (sc && is_placeholder_row(rec, rf.fields)) { r.quality.records_placeholder++; return; }
            bool any_invalid = false, all_invalid = true;
            for (int j = 0; j < rf.fields; ++j) {
                if (std::isnan(rec.fields[j])) { r.quality.records_nonfinite++; }
                else if (sc && field_invalid(*sc, j, rec.fields[j])) { any_invalid = true; }
                else all_invalid = false;
            }
            if (any_invalid) { r.quality.records_invalid++; return; }
            (void)all_invalid;
            if (rec.episode < 0) r.quality.unknown_episode++;
            if (rec.episode >= 0) {
                if (last_ep != INT32_MIN && rec.episode < last_ep) {
                    r.quality.reorder_events++;
                    GapRange g;
                    g.from_episode = rec.episode; g.to_episode = last_ep;
                    g.record_index_lo = index; g.record_index_hi = index;
                    g.kind = "suspect"; g.reason = "episode 回跳：可能是回档/重启或乱序记录";
                    r.quality.gaps.push_back(g);
                }
                last_ep = rec.episode;
            }
            // 同 (ep,step) 多条 = 合法；只有紧邻且**全字段完全相同**才可能算重复
            {
                bool same_key = have_prev_key && rec.episode == prev_key_ep && rec.step == prev_key_step;
                bool same_all = same_key && (int)prev_fields.size() == rf.fields;
                if (same_all)
                    for (int j = 0; j < rf.fields; ++j)
                        if (prev_fields[j] != rec.fields[j]) { same_all = false; break; }
                if (same_key) {
                    if (same_all) {
                        if (plan.per_episode) r.quality.records_duplicate++;
                        else r.quality.identical_adjacent_on_nonunique_key++;
                    } else {
                        r.quality.same_key_multi++;
                    }
                }
                prev_fields.assign(rec.fields, rec.fields + rf.fields);
                prev_key_ep = rec.episode; prev_key_step = rec.step;
                have_prev_key = true;
            }

            if (strcmp(plan.name, "metric_loss.bin") == 0) {
                builders["loss"].add(rec.episode, rec.step, index, rec.fields[0]);
                builders["q_value"].add(rec.episode, rec.step, index, rec.fields[1]);
                builders["target_q"].add(rec.episode, rec.step, index, rec.fields[2]);
                // D 必须先逐记录配对，再聚合（绝不用两个均值相减）
                builders["delta_d"].add(rec.episode, rec.step, index,
                    (double)rec.fields[1] - (double)rec.fields[2]);
                category_seq.push_back({ rec.episode, rec.step });
            }
            else if (strcmp(plan.name, "metric_epsilon.bin") == 0)
                builders["epsilon_step"].add(rec.episode, rec.step, index, rec.fields[0]);
            else if (strcmp(plan.name, "metric_epsilon_episode.bin") == 0)
                builders["epsilon_episode"].add(rec.episode, 0, index, rec.fields[0]);
            else if (strcmp(plan.name, "metric_game.bin") == 0)
                builders["episode_length"].add(rec.episode, 0, index, rec.fields[0]);
            else if (strcmp(plan.name, "metric_episode_time.bin") == 0)
                builders["episode_seconds"].add(rec.episode, 0, index, rec.fields[0]);
            else if (strcmp(plan.name, "metric_rewards.bin") == 0) {
                for (int i = 0; i < 7; ++i) {
                    char k[32]; snprintf(k, sizeof(k), "reward_%d", i);
                    builders[k].add(rec.episode, 0, index, rec.fields[i]);
                }
                builders["reward_total"].add(rec.episode, 0, index, rec.fields[7]);
            }
            else if (strcmp(plan.name, "metric_win.bin") == 0) {
                int mode = (int)rec.fields[0];
                if (mode >= 0 && mode < 3) {
                    char k[32];
                    snprintf(k, sizeof(k), "win_rate_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[1]);
                    snprintf(k, sizeof(k), "draw_rate_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[2]);
                    snprintf(k, sizeof(k), "lose_rate_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[3]);
                }
            }
            else if (strcmp(plan.name, "metric_eval.bin") == 0) {
                int mode = (int)rec.fields[3];
                if (mode >= 0 && mode < 3) {
                    char k[32];
                    snprintf(k, sizeof(k), "depth_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[1]);
                    snprintf(k, sizeof(k), "eval_epsilon_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[2]);
                    snprintf(k, sizeof(k), "phase_mode%d", mode);
                    builders[k].add(rec.episode, mode, index, rec.fields[0]);
                }
            }
        });
        r.quality.records_used += rf.complete_records;
    }
    r.bytes_read = bytes_read;
    r.quality.read_boundary_fixed = true;
    r.quality.boundary_note = "读取边界在打开时固定为该文件当时的字节数；读取期间新追加的记录被忽略并计为不确定尾部。"
        "半条记录（非整 stride）已排除。源文件未被修改。";

    // ---- 序列落地 ----
    stage("series", "assembling series and exact streaming statistics");
    long long max_ep = 0;
    for (auto& kv : builders) {
        if (kv.second.s.key.empty()) continue;   // 未预注册的键（拼写错误）不落地
        Series s = kv.second.finish();
        for (const auto& smp : s.samples) if (smp.episode > max_ep) max_ep = smp.episode;
        r.series[kv.first] = s;
    }

    // ---- 高频序列：先做 EP 分布聚合，再对聚合轨迹做压缩 ----
    // 分层：RAW 分布 → EP 聚合（mean/median/p95/max）→ 轨迹压缩。
    // 直接对逐 update 原始流做 RDP 等于压缩噪声（实测 103 万输入点要留 84.6 万点）。
    {
        stage("ep_aggregate", "EP distribution aggregation for noise-dominated series");
        static const char* fields4[4] = { "mean", "median", "p95", "max" };
        static const char* high_rate[5] = { "loss", "q_value", "target_q", "delta_d", "epsilon_step" };
        std::vector<Series> derived;
        for (const char* key : high_rate) {
            auto it = r.series.find(key);
            if (it == r.series.end()) continue;
            const Series& src = it->second;
            if (src.samples.size() < 40) continue;
            struct Agg { int32_t ep; double mean, median, p95, vmax; };
            std::vector<Agg> aggs;
            std::vector<double> buf;
            size_t i = 0;
            while (i < src.samples.size()) {
                const int32_t ep = src.samples[i].episode;
                buf.clear();
                size_t j = i;
                while (j < src.samples.size() && src.samples[j].episode == ep) {
                    if (finite_d(src.samples[j].value)) buf.push_back(src.samples[j].value);
                    ++j;
                }
                if (!buf.empty()) {
                    std::sort(buf.begin(), buf.end());
                    Agg a{};
                    a.ep = ep;
                    double s = 0;
                    for (double v : buf) s += v;
                    a.mean = s / (double)buf.size();
                    a.median = quantile_sorted(buf, 0.5);
                    a.p95 = quantile_sorted(buf, 0.95);
                    a.vmax = buf.back();
                    aggs.push_back(a);
                }
                i = (j > i) ? j : i + 1;
            }
            if (aggs.size() < 8) continue;
            // 平均每 EP 记录数 > 1 → 逐点重建是在拟合噪声
            const bool noise_dominated = (double)src.total_records / (double)aggs.size() > 1.0;
            r.series[key].noise_dominated = noise_dominated;
            if (!noise_dominated) continue;
            for (int f = 0; f < 4; ++f) {
                Series tr;
                tr.key = std::string(key) + "__ep_" + fields4[f];
                tr.file = src.file;
                tr.field = std::string("EP-") + fields4[f] + "(" + src.field + ")";
                tr.field_index = -1;
                tr.meaning = std::string("该 EP 内全部原始记录的 ") + fields4[f] +
                    "（先聚合再压缩，避免对逐 update 噪声做逐点重建）";
                tr.unit = src.unit;
                tr.aggregation = AGG_EP_DISTRIBUTION;
                tr.episode_semantics = "EP 号（每 EP 恰一个聚合点）";
                tr.step_semantics = "恒为 0";
                tr.zero_based = src.zero_based;
                tr.per_episode = true;
                tr.timeline_grouped = true;
                tr.sampling_interval_known = false;
                tr.sampling_note = std::string("每 EP 一个聚合点：由 ") + key + " 的 " +
                    fmt_i(src.total_records) + " 条原始记录聚合为 " + fmt_i((int64_t)aggs.size()) + " 点";
                for (const auto& a : aggs) {
                    const double v = (f == 0) ? a.mean : (f == 1) ? a.median : (f == 2) ? a.p95 : a.vmax;
                    Sample smp;
                    smp.episode = a.ep;
                    smp.step = 0;
                    smp.record_index = (int64_t)tr.samples.size();
                    smp.value = v;
                    smp.finite = finite_d(v);
                    tr.samples.push_back(smp);
                }
                tr.total_records = (int64_t)aggs.size();
                {
                    double s = 0, mn = 0, mx = 0, var = 0;
                    int64_t n = 0;
                    for (const auto& smp : tr.samples) {
                        if (!finite_d(smp.value)) continue;
                        if (n == 0) { mn = mx = smp.value; }
                        else { if (smp.value < mn) mn = smp.value; if (smp.value > mx) mx = smp.value; }
                        s += smp.value; ++n;
                    }
                    if (n > 0) {
                        tr.exact_mean = s / (double)n;
                        tr.exact_min = mn;
                        tr.exact_max = mx;
                        if (n > 1) {
                            for (const auto& smp : tr.samples)
                                if (finite_d(smp.value)) var += (smp.value - tr.exact_mean) * (smp.value - tr.exact_mean);
                            tr.exact_stddev = sqrt(var / (double)(n - 1));
                            tr.exact_std_valid = true;
                        } else {
                            tr.exact_std_valid = false;
                            tr.exact_std_null_reason = "n<2";
                        }
                    }
                }
                r.series[key].derived_ep_series.push_back(tr.key);
                derived.push_back(tr);
            }
        }
        for (auto& d : derived) r.series[d.key] = d;
    }

    // ---- 标量统计：全量精确（流式）与保留子集并列 ----
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        std::vector<double> vals;
        vals.reserve(s.samples.size());
        for (const auto& x : s.samples) if (finite_d(x.value)) vals.push_back(x.value);
        if (vals.empty()) continue;
        std::sort(vals.begin(), vals.end());
        AnalyzeResult::ScalarStat st{};
        // 全量精确
        st.exact_n = s.total_records;
        st.exact_mean = s.exact_mean;
        st.exact_min = s.exact_min;
        st.exact_max = s.exact_max;
        st.exact_stddev = s.exact_stddev;
        st.exact_std_valid = s.exact_std_valid;
        st.exact_std_null_reason = s.exact_std_null_reason;
        // 保留子集（分位数只能在此计算）
        st.n = (int64_t)vals.size();
        double sum = 0;
        for (double v : vals) sum += v;
        st.mean = sum / (double)st.n;
        double var = 0;
        for (double v : vals) var += (v - st.mean) * (v - st.mean);
        st.min = vals.front(); st.max = vals.back();
        if (st.n < 2) {
            st.std_valid = false;
            st.stddev = NAN;
            st.std_null_reason = "valid samples < 2: sample standard deviation undefined (not reported as 0)";
        } else {
            st.stddev = sqrt(var / (double)(st.n - 1));
        }
        st.p50 = quantile_sorted(vals, 0.50);
        st.p90 = quantile_sorted(vals, 0.90);
        st.p99 = quantile_sorted(vals, 0.99);
        // 抽样是否影响结论：全量与保留子集的均值/极值出现可见差异时标记
        st.sampling_affected =
            (st.exact_n != st.n) &&
            (fabs(st.exact_mean - st.mean) > 1e-9 * (1.0 + fabs(st.exact_mean)) ||
             st.exact_min != st.min || st.exact_max != st.max);
        r.scalars[kv.first] = st;
        r.quantiles[kv.first]["p01"] = quantile_sorted(vals, 0.01);
        r.quantiles[kv.first]["p05"] = quantile_sorted(vals, 0.05);
        r.quantiles[kv.first]["p25"] = quantile_sorted(vals, 0.25);
        r.quantiles[kv.first]["p50"] = st.p50;
        r.quantiles[kv.first]["p75"] = quantile_sorted(vals, 0.75);
        r.quantiles[kv.first]["p90"] = st.p90;
        r.quantiles[kv.first]["p95"] = quantile_sorted(vals, 0.95);
        r.quantiles[kv.first]["p99"] = st.p99;

        // ---- Top X%（按数值）暴露切片 ----
        if (options.top_percentile > 0.0 && options.top_percentile < 100.0 && st.n >= 20) {
            AnalyzeResult::TopSlice ts;
            ts.series_key = kv.first;
            ts.percent = options.top_percentile;
            ts.basis = "保留子集（点级分析受渐进抽稀影响；阈值与切片统计均基于保留点，"
                       "全量精确均值/极值见并列的全量统计）";
            ts.definition = "切片 = 数值 >= 阈值 的样本；并列值全部保留，故实际占比可能略大于请求的 X%";
            ts.threshold = quantile_sorted(vals, 1.0 - options.top_percentile / 100.0);
            std::vector<double> slice;
            for (double v : vals) if (v >= ts.threshold) slice.push_back(v);
            ts.n = (int64_t)slice.size();
            ts.actual_percent = 100.0 * (double)ts.n / (double)st.n;
            // EP / record_index 覆盖范围
            {
                bool first = true;
                for (const auto& smp : s.samples) {
                    if (!finite_d(smp.value) || smp.value < ts.threshold) continue;
                    if (first) {
                        ts.ep_lo = ts.ep_hi = smp.episode;
                        ts.record_lo = ts.record_hi = smp.record_index;
                        first = false;
                    } else {
                        if (smp.episode < ts.ep_lo) ts.ep_lo = smp.episode;
                        if (smp.episode > ts.ep_hi) ts.ep_hi = smp.episode;
                        if (smp.record_index < ts.record_lo) ts.record_lo = smp.record_index;
                        if (smp.record_index > ts.record_hi) ts.record_hi = smp.record_index;
                    }
                }
            }
            if (ts.n >= 2) {
                double ssum = 0;
                for (double v : slice) ssum += v;
                ts.mean = ssum / (double)ts.n;
                double svar = 0;
                for (double v : slice) svar += (v - ts.mean) * (v - ts.mean);
                ts.stddev = sqrt(svar / (double)(ts.n - 1));
                ts.std_valid = true;
                ts.min = slice.front(); ts.max = slice.back();   // slice 已随 vals 排序
                ts.p50 = quantile_sorted(slice, 0.50);
                ts.p90 = quantile_sorted(slice, 0.90);
                ts.p99 = quantile_sorted(slice, 0.99);
                double total = 0;
                for (double v : vals) total += v;
                if (fabs(total) > 1e-12) {
                    ts.sum_share = ssum / total;
                    ts.sum_share_valid = true;
                    // 有符号序列上该比值可能落在 [0,1] 之外，也可能因总和接近 0 而失去意义。
                    if (total < 0 || ts.sum_share < 0.0 || ts.sum_share > 1.0) {
                        ts.sum_share_valid = false;
                        ts.sum_share_null_reason =
                            "序列含正负值或总和接近 0：切片和/总和 不是「贡献占比」，该比值无意义";
                    }
                } else {
                    ts.sum_share_valid = false;
                    ts.sum_share_null_reason = "保留子集总和 ≈ 0：占比无定义（不除零）";
                }
                ts.valid = true;
            } else {
                ts.std_valid = false;
                ts.null_reason = "slice samples < 2: slice statistics undefined";
                ts.std_null_reason = ts.null_reason;
            }
            r.top_slices.push_back(ts);
        }
    }

    // ---- 类别指标（Replay Type）：频数/比例/转移/连续区间 ----
    if (!category_seq.empty()) {
        AnalyzeResult::CategoryStat cs;
        cs.series_key = category_key;
        cs.definition = "metric_loss.bin 的 prefix.step = replay_type；0=黑方回放缓冲、1=白方回放缓冲。"
            "类别量：只做频数/比例/转移/连续区间，不做斜率或线性回归。";
        int prev = INT32_MIN;
        AnalyzeResult::CategoryStat::Run cur{};
        for (size_t i = 0; i < category_seq.size(); ++i) {
            int v = category_seq[i].second;
            if (v != 0 && v != 1) continue;
            cs.frequency[v]++;
            cs.total++;
            if (prev != INT32_MIN) {
                if (v != prev) {
                    cs.transitions++;
                    cs.transition_matrix[{prev, v}]++;
                    cur.length = (int64_t)i - cur.rec_lo;
                    cs.runs.push_back(cur);
                    cur = AnalyzeResult::CategoryStat::Run{};
                    cur.value = v; cur.rec_lo = (int64_t)i; cur.ep_lo = category_seq[i].first;
                }
            } else {
                cur.value = v; cur.rec_lo = 0; cur.ep_lo = category_seq[i].first;
            }
            cur.rec_hi = (int64_t)i; cur.ep_hi = category_seq[i].first;
            prev = v;
        }
        if (cs.total > 0 && cur.rec_hi >= cur.rec_lo) {
            cur.length = (int64_t)category_seq.size() - cur.rec_lo;
            cs.runs.push_back(cur);
        }
        for (auto& kv : cs.frequency) cs.proportion[kv.first] = (double)kv.second / (double)cs.total;
        cs.note = "不做斜率/线性回归；转移矩阵元素为 (前一类别 → 后一类别) 的切换次数。";
        r.category_stats.push_back(cs);
    }

    // ---- 窗口统计（记录窗口 + EP 窗口，两种口径都输出）----
    stage("windows", "computing record and episode windows");
    const size_t WINDOW_CAP = 8;    // 主 JSON 里每个(序列,宽度)最多内联多少个窗口
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        std::vector<WindowStat> out;
        auto push = [&](const WindowStat& w) {
            if (out.size() < WINDOW_CAP) { out.push_back(w); ++r.quality.windows_computed; }
        };

        for (int width : options.record_windows) {
            if (width <= 0) continue;
            // 有界化：只对确定性的等距窗口子集计算并输出，避免百万级记录 × 多宽度导致耗时/输出爆炸。
            r.quality.windows_total_possible +=
                (int64_t)((s.samples.size() + (size_t)width - 1) / (size_t)width);
            const size_t wstride = (size_t)std::max<int64_t>(1,
                (int64_t)s.samples.size() / ((int64_t)WINDOW_CAP * (int64_t)std::max(1, width)));
            for (size_t start = 0; start < s.samples.size(); start += (size_t)width * wstride) {
                size_t end = std::min(s.samples.size(), start + (size_t)width);
                WindowStat w;
                w.id = "records:" + fmt_i((int64_t)width) + "@" + fmt_i((int64_t)start);
                w.by_records = true; w.width = width; w.min_valid = 2;
                std::vector<double> v;
                for (size_t i = start; i < end; ++i) {
                    if (finite_d(s.samples[i].value)) v.push_back(s.samples[i].value);
                    if (s.samples[i].episode > w.ep_hi) w.ep_hi = s.samples[i].episode;
                }
                w.ep_lo = s.samples[start].episode;
                w.ep_hi = s.samples[end - 1].episode;
                w.sample_count = (int64_t)v.size();
                w.episode_span = (double)(w.ep_hi - w.ep_lo);
                w.std_definition = "样本标准差（n-1 分母）";
                if ((int64_t)v.size() < w.min_valid) {
                    w.valid = false; w.null_reason = "valid samples < min_valid(2)";
                } else {
                    w.valid = true;
                    double sum = 0; for (double x : v) sum += x;
                    w.mean = sum / (double)v.size();
                    double var = 0; for (double x : v) var += (x - w.mean) * (x - w.mean);
                    w.stddev = v.size() > 1 ? sqrt(var / (double)(v.size() - 1)) : 0;
                    std::sort(v.begin(), v.end());
                    w.min = v.front(); w.max = v.back();
                    w.p25 = quantile_sorted(v, 0.25); w.p50 = quantile_sorted(v, 0.5); w.p75 = quantile_sorted(v, 0.75);
                }
                push(w);
            }
        }
        for (int width : options.episode_windows) {
            if (width <= 0) continue;
            // 先收集所有 EP 分组起点，再按等距子集计算（同样有界化）。
            std::vector<size_t> starts;
            {
                size_t i = 0;
                while (i < s.samples.size()) {
                    starts.push_back(i);
                    int32_t base = s.samples[i].episode;
                    size_t j = i;
                    while (j < s.samples.size() && s.samples[j].episode < base + width) ++j;
                    i = (j > i) ? j : i + 1;
                }
            }
            r.quality.windows_total_possible += (int64_t)starts.size();
            const size_t wstride = (size_t)std::max<int64_t>(1,
                (int64_t)starts.size() / (int64_t)WINDOW_CAP);
            for (size_t si = 0; si < starts.size(); si += wstride) {
                size_t i = starts[si];
                size_t j = (si + 1 < starts.size()) ? starts[si + 1] : s.samples.size();
                std::vector<double> v;
                for (size_t k = i; k < j; ++k)
                    if (finite_d(s.samples[k].value)) v.push_back(s.samples[k].value);
                WindowStat w;
                w.id = "episodes:" + fmt_i((int64_t)width) + "@" + fmt_i((int64_t)s.samples[i].episode);
                w.by_records = false; w.width = width; w.min_valid = 2;
                w.ep_lo = s.samples[i].episode;
                w.ep_hi = s.samples[j - 1].episode;
                w.sample_count = (int64_t)v.size();
                w.episode_span = (double)(w.ep_hi - w.ep_lo);
                w.std_definition = "样本标准差（n-1 分母）";
                if ((int64_t)v.size() < w.min_valid) {
                    w.valid = false; w.null_reason = "valid samples < min_valid(2)";
                } else {
                    w.valid = true;
                    double sum = 0; for (double x : v) sum += x;
                    w.mean = sum / (double)v.size();
                    double var = 0; for (double x : v) var += (x - w.mean) * (x - w.mean);
                    w.stddev = v.size() > 1 ? sqrt(var / (double)(v.size() - 1)) : 0;
                    std::sort(v.begin(), v.end());
                    w.min = v.front(); w.max = v.back();
                    w.p25 = quantile_sorted(v, 0.25); w.p50 = quantile_sorted(v, 0.5); w.p75 = quantile_sorted(v, 0.75);
                }
                push(w);
            }
        }
        r.windows[kv.first] = out;
    }

    // ---- 差分（原始 + 按 EP 距离归一化）----
    stage("diffs", "first and second order differences");
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        DiffSeries ds;
        ds.key = kv.first;
        ds.first_order_note = "一阶差分 = 后一条有效记录 − 前一条有效记录（原始单位）";
        ds.second_order_note = "二阶差分 = 相邻两个一阶差分之差，单位同原始差分；"
            "采样不均匀（相邻记录的 EP 距离不同）时，该定义不代表时间上的加速度，"
            "仅在 EP 距离相等时才有连续含义。";
        const Sample* prev = nullptr;
        DiffPoint prev_dp{};
        bool has_prev_dp = false;
        for (const auto& smp : s.samples) {
            if (!finite_d(smp.value)) continue;
            if (prev) {
                DiffPoint dp;
                dp.record_index = smp.record_index;
                dp.episode = smp.episode;
                dp.raw_diff = smp.value - prev->value;
                int d_ep = smp.episode - prev->episode;
                if (d_ep == 0) {
                    dp.rate_valid = false;
                    dp.rate_null_reason = "相邻记录属于同一 Episode（EP 距离=0），不除零";
                } else if (d_ep < 0) {
                    dp.rate_valid = false;
                    dp.rate_null_reason = "EP 回跳（回档/重启边界），不跨边界归一化";
                } else {
                    dp.rate_valid = true;
                    dp.rate_per_episode = dp.raw_diff / (double)d_ep;
                }
                if (has_prev_dp && prev_dp.rate_valid && dp.rate_valid) {
                    dp.second_diff = dp.rate_per_episode - prev_dp.rate_per_episode;
                    dp.second_valid = true;
                } else if (has_prev_dp) {
                    dp.second_valid = true;
                    dp.second_diff = dp.raw_diff - prev_dp.raw_diff;
                }
                ds.points.push_back(dp);
                prev_dp = dp; has_prev_dp = true;
            }
            prev = &smp;
        }
        // 存储有界化：百万级序列的差分点全量驻留会占用数百 MB。
        // 只保存确定性的等距子集，并在说明里显式声明（非静默丢弃）。
        const size_t DIFF_STORE_LIMIT = 20000;
        if (ds.points.size() > DIFF_STORE_LIMIT) {
            const size_t stride = (ds.points.size() + DIFF_STORE_LIMIT - 1) / DIFF_STORE_LIMIT;
            std::vector<DiffPoint> sub;
            for (size_t i = 0; i < ds.points.size(); i += stride) sub.push_back(ds.points[i]);
            if (!sub.empty() && sub.back().record_index != ds.points.back().record_index)
                sub.push_back(ds.points.back());
            ds.first_order_note += "（存储上限 " + fmt_i((int64_t)DIFF_STORE_LIMIT) +
                "：共 " + fmt_i((int64_t)ds.points.size()) + " 个差分点，此处保存等距子集 " +
                fmt_i((int64_t)sub.size()) + " 个）";
            ds.points.swap(sub);
        }
        r.diffs[kv.first] = ds;
    }

    // ---- 异常检测（基线只用此前有效样本）----
    stage("anomalies", "multi-rule anomaly detection with causal baseline");
    struct Rule { const char* name; const char* kind; double threshold; int min_prior; };
    const Rule rules[] = {
        { "zscore_prior_window", "abs_z", options.z_threshold > 0 ? options.z_threshold : 4.0, 20 },
        { "mad_z_prior_window", "abs_mad_z", options.mad_z_threshold > 0 ? options.mad_z_threshold : 6.0, 20 },
        { "level_shift_vs_prior", "effect_size", 0.6, 40 },
        { "spike_vs_neighbors", "spike_ratio", 0.0, 3 },
    };
    // 基线严格只用**此前**的有效样本，绝不把待检测点混入自己的基线。
    // 采用**滚动先验窗口**而不是全历史：全历史基线在非平稳序列（如 epsilon 衰减）上会把
    // 正常趋势误报成海量异常。窗口长度与刷新节奏在输出中显式声明。
    // 同时施加「统计显著 + 实际幅度」双门限，避免大样本下阈值失效。
    const size_t BASE_WINDOW = 512;
    const int64_t MAD_REFRESH = 256;
    // 异常**不因输出预算丢弃**：默认全部保留，只设内存保护上限。
    // 全部异常写入 trajectory_anomalies.jsonl；主 JSON 只放汇总/区域/Top 代表点。
    const int64_t ANOMALY_STORE_LIMIT = options.anomaly_keep_all ? 2000000 : 50000;
    r.anomaly_store_limit = ANOMALY_STORE_LIMIT;
    std::set<std::string> z_null_emitted;   // 每个序列只报一次「先验 std==0」
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        if (s.samples.size() < 5) continue;
        std::deque<double> win;
        double wsum = 0, wsumsq = 0;
        double med = 0, mad = 0;
        bool robust_ready = false;
        int64_t since_refresh = MAD_REFRESH;
        for (size_t i = 0; i < s.samples.size(); ++i) {
            double v = s.samples[i].value;
            if (!finite_d(v)) continue;
            if (win.size() >= (size_t)rules[0].min_prior) {
                const double n = (double)win.size();
                const double mean = wsum / n;
                double var = n > 1 ? (wsumsq - wsum * wsum / n) / (n - 1) : 0.0;
                if (!(var > 0)) var = 0;
                const double sd = sqrt(var);
                if (since_refresh >= MAD_REFRESH) {
                    std::vector<double> w(win.begin(), win.end());
                    med = median_of(w);
                    std::vector<double> dev;
                    dev.reserve(w.size());
                    for (double x : w) dev.push_back(fabs(x - med));
                    mad = median_of(dev) * 1.4826;
                    robust_ready = true;
                    since_refresh = 0;
                }
                ++since_refresh;
                const double dev_abs = fabs(v - mean);
                const double effect_floor = (robust_ready && mad > 0) ? 0.5 * mad : 0.0;
                auto store = [&](const Anomaly& a) {
                    ++r.anomalies_detected;
                    if ((int64_t)r.anomalies.size() < ANOMALY_STORE_LIMIT) r.anomalies.push_back(a);
                    else ++r.anomalies_omitted;
                };
                if (sd > 0) {
                    double z = dev_abs / sd;
                    if (z > rules[0].threshold && dev_abs > effect_floor) {
                        Anomaly a;
                        a.rule = rules[0].name; a.threshold = rules[0].threshold; a.threshold_kind = rules[0].kind;
                        a.episode = s.samples[i].episode; a.record_index = s.samples[i].record_index;
                        a.value = v; a.baseline = mean; a.score = z; a.baseline_n = (int64_t)win.size();
                        a.baseline_definition = "此前最近 512 个有效样本（不含当前点）的均值与样本标准差(n-1)（滚动窗口）"
                            "；另需 |偏差| > 0.5×MAD 才算触发（双门限）";
                        a.series_key = kv.first;
                        a.trigger_mask = 0x01;   // local_z
                        store(a);
                    }
                } else if (win.size() == (size_t)rules[0].min_prior && z_null_emitted.insert(kv.first).second) {
                    Anomaly a;
                    a.rule = rules[0].name; a.threshold = rules[0].threshold; a.threshold_kind = rules[0].kind;
                    a.episode = s.samples[i].episode; a.record_index = s.samples[i].record_index;
                    a.value = v; a.baseline = mean; a.score = NAN; a.baseline_n = (int64_t)win.size();
                    a.null_reason = "prior std == 0（先验窗口为常数）：z 分数无定义，不伪造有限分数";
                    a.baseline_definition = "此前最近 512 个有效样本（不含当前点）";
                    a.series_key = kv.first;
                    a.trigger_mask = 0x01;   // local_z（因先验 std==0 无法给分数，仍标记触发位）
                    store(a);
                }
                if (robust_ready && mad > 0) {
                    double mz = fabs(v - med) / mad;
                    if (mz > rules[1].threshold) {
                        Anomaly a;
                        a.rule = rules[1].name; a.threshold = rules[1].threshold; a.threshold_kind = rules[1].kind;
                        a.episode = s.samples[i].episode; a.record_index = s.samples[i].record_index;
                        a.value = v; a.baseline = med; a.score = mz; a.baseline_n = (int64_t)win.size();
                        a.baseline_definition = "此前最近 512 个有效样本（不含当前点）的中位数与 MAD(×1.4826)；"
                            "基线每 256 条记录刷新一次（近似，非逐点重算）";
                        a.series_key = kv.first;
                        a.trigger_mask = 0x02;   // robust_z
                        store(a);
                    }
                }
            }
            win.push_back(v);
            wsum += v; wsumsq += v * v;
            if (win.size() > BASE_WINDOW) {
                double old = win.front();
                win.pop_front();
                wsum -= old; wsumsq -= old * old;
            }
        }
    }

    // 排序：稳定、可重复（分数降序，其次 record_index 升序）
    std::stable_sort(r.anomalies.begin(), r.anomalies.end(),
        [](const Anomaly& a, const Anomaly& b) {
            double sa = finite_d(a.score) ? a.score : -1;
            double sb = finite_d(b.score) ? b.score : -1;
            if (sa != sb) return sa > sb;
            return a.record_index < b.record_index;
        });
    for (size_t i = 0; i < r.anomalies.size(); ++i) r.anomalies[i].rank = (int64_t)i + 1;

    // ---- 方案 B：按分位目标定阈（每条序列只留分数最高的前 keep_rate）----
    // 让「异常」在定义上就是罕见事件，而不是靠拍脑袋的绝对 z 阈值。
    if (options.anomaly_keep_rate > 0.0 && options.anomaly_keep_rate < 1.0 && !r.anomalies.empty()) {
        std::map<std::string, std::vector<double>> by_series;
        for (const auto& a : r.anomalies)
            if (finite_d(a.score)) by_series[a.series_key].push_back(a.score);
        std::map<std::string, double> cut;
        for (auto& kv : by_series) {
            std::sort(kv.second.begin(), kv.second.end());
            cut[kv.first] = quantile_sorted(kv.second, 1.0 - options.anomaly_keep_rate);
        }
        std::vector<Anomaly> kept;
        kept.reserve(r.anomalies.size());
        int64_t dropped = 0;
        for (const auto& a : r.anomalies) {
            auto it2 = cut.find(a.series_key);
            if (it2 == cut.end() || !finite_d(a.score)) { kept.push_back(a); continue; }
            if (a.score >= it2->second) kept.push_back(a);
            else ++dropped;
        }
        r.anomalies.swap(kept);
        r.anomalies_filtered_by_quantile = dropped;
        r.anomaly_keep_rate_used = options.anomaly_keep_rate;
        for (size_t i = 0; i < r.anomalies.size(); ++i) r.anomalies[i].rank = (int64_t)i + 1;
    }

    // 异常区间：同一序列同一规则且在记录序号上连续（间隔 <= 1 视为连续区间）
    {
        std::map<std::string, std::vector<size_t>> by_key;
        for (size_t i = 0; i < r.anomalies.size(); ++i)
            by_key[r.anomalies[i].series_key + "|" + r.anomalies[i].rule].push_back(i);
        for (auto& kv : by_key) {
            std::vector<size_t> idx = kv.second;
            std::sort(idx.begin(), idx.end(),
                [&](size_t a, size_t b) { return r.anomalies[a].record_index < r.anomalies[b].record_index; });
            size_t i = 0;
            while (i < idx.size()) {
                size_t j = i;
                while (j + 1 < idx.size() &&
                       r.anomalies[idx[j + 1]].record_index - r.anomalies[idx[j]].record_index <= 1) ++j;
                AnomalyInterval iv;
                iv.rule = r.anomalies[idx[i]].rule;
                iv.series_key = r.anomalies[idx[i]].series_key;
                iv.record_lo = r.anomalies[idx[i]].record_index;
                iv.record_hi = r.anomalies[idx[j]].record_index;
                iv.ep_lo = r.anomalies[idx[i]].episode;
                iv.ep_hi = r.anomalies[idx[j]].episode;
                iv.count = (int64_t)(j - i + 1);
                for (size_t k = i; k <= j; ++k) {
                    double sc = r.anomalies[idx[k]].score;
                    if (finite_d(sc) && sc > iv.max_score) iv.max_score = sc;
                    if (r.anomalies[idx[k]].episode < iv.ep_lo) iv.ep_lo = r.anomalies[idx[k]].episode;
                    if (r.anomalies[idx[k]].episode > iv.ep_hi) iv.ep_hi = r.anomalies[idx[k]].episode;
                }
                r.anomaly_intervals.push_back(iv);
                i = j + 1;
            }
        }
    }

    // ---- 异常区域聚类：把上万条独立异常聚成「持续异常事件」 ----
    // AI 真正需要的是「这里发生了一场持续 N 局的异常」，而不是第 13841/13842/13843 条异常。
    stage("anomaly_regions", "clustering anomalies into contiguous regions");
    {
        // severity 分位阈值：critical = 分数最高的 1% 异常，major = 之后 9%，其余 minor。
        double critical_cut = 0, major_cut = 0;
        std::string severity_rule;
        {
            std::vector<double> sc;
            sc.reserve(r.anomalies.size());
            for (const auto& a : r.anomalies) if (finite_d(a.score)) sc.push_back(a.score);
            if (!sc.empty()) {
                std::sort(sc.begin(), sc.end());
                critical_cut = quantile_sorted(sc, 0.99);
                major_cut = quantile_sorted(sc, 0.90);
                severity_rule = "按全部分数分位：critical = max_score >= p99(" +
                    fmt_d(critical_cut, 4) + "), major = >= p90(" + fmt_d(major_cut, 4) +
                    "), minor = 其余；保证 critical 只占极少数";
            } else {
                severity_rule = "无有效分数：全部记为 minor";
            }
        }
        std::vector<size_t> ord(r.anomalies.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        // **必须按 episode 排序**：record_index 是每个指标文件各自的物理序号，
        // 跨指标按它排序会交织错乱，使 nxt_ep - cur_ep 变成负数而不触发 gap 退出，
        // 区域链因此无限延伸（这正是之前出现 EP 5..32k 超级区域的根因）。
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
            if (r.anomalies[a].episode != r.anomalies[b].episode)
                return r.anomalies[a].episode < r.anomalies[b].episode;
            if (r.anomalies[a].series_key != r.anomalies[b].series_key)
                return r.anomalies[a].series_key < r.anomalies[b].series_key;
            return r.anomalies[a].record_index < r.anomalies[b].record_index;
        });
        const int64_t gap = options.anomaly_region_gap > 0 ? options.anomaly_region_gap : 200;
        size_t i = 0;
        while (i < ord.size()) {
            size_t j = i;
            // 合并条件 = 时间接近 **且** 区域跨度不超上限。
            // 仅按时间接近会串出横跨数万局的超级区域；按家族硬切又会过碎（实测 9.5k → 44k）。
            // 因此以「跨度上限」作为唯一硬约束，家族构成在区内单独统计输出。
            const int32_t span_ep_lo = r.anomalies[ord[i]].episode;
            const int64_t max_span = options.anomaly_region_max_span > 0
                ? options.anomaly_region_max_span : 2000;
            while (j + 1 < ord.size()) {
                const int64_t cur_ep = r.anomalies[ord[j]].episode;
                const int64_t nxt_ep = r.anomalies[ord[j + 1]].episode;
                if (nxt_ep - cur_ep > gap) break;
                if (nxt_ep - (int64_t)span_ep_lo > max_span) break;
                if (nxt_ep < cur_ep) break;   // 防御：EP 非单调时不得继续延伸
                ++j;
            }
            AnomalyRegion rg;
            rg.index = (int64_t)r.anomaly_regions.size();
            rg.ep_lo = rg.ep_hi = r.anomalies[ord[i]].episode;
            rg.record_lo = rg.record_hi = r.anomalies[ord[i]].record_index;
            rg.peak_ep = r.anomalies[ord[i]].episode;
            rg.peak_record = r.anomalies[ord[i]].record_index;
            rg.peak_series = r.anomalies[ord[i]].series_key;
            double ssum = 0;
            int64_t scored = 0;
            std::map<std::string, AnomalyRegion::MetricPeak> peaks;
            std::set<std::string> sset;
            std::vector<size_t> members;
            for (size_t k = i; k <= j; ++k) {
                const Anomaly& a = r.anomalies[ord[k]];
                members.push_back(ord[k]);
                rg.count++;
                if (a.episode < rg.ep_lo) rg.ep_lo = a.episode;
                if (a.episode > rg.ep_hi) rg.ep_hi = a.episode;
                if (a.record_index < rg.record_lo) rg.record_lo = a.record_index;
                if (a.record_index > rg.record_hi) rg.record_hi = a.record_index;
                if (finite_d(a.score)) { ssum += a.score; ++scored; }
                sset.insert(a.series_key);
                rg.family_counts[metric_family(a.series_key)]++;
                auto& pk = peaks[a.series_key];
                pk.series = a.series_key;
                const double av = fabs(a.value);
                if (av > pk.max_abs_value) { pk.max_abs_value = av; pk.at_ep = a.episode; }
                if (finite_d(a.score) && a.score > rg.max_score) {
                    rg.max_score = a.score;
                    rg.peak_ep = a.episode;
                    rg.peak_record = a.record_index;
                    rg.peak_series = a.series_key;
                }
            }
            // 无有限分数的区域（如「先验 std==0」）不得把 max_score 伪装成 0，输出 null。
            if (scored == 0) { rg.max_score = NAN; rg.mean_score = NAN; }
            else rg.mean_score = ssum / (double)scored;
            rg.span_limited = ((int64_t)rg.ep_hi - (int64_t)rg.ep_lo) >= max_span;
            {
                int64_t best = -1;
                for (const auto& kvf : rg.family_counts)
                    if (kvf.second > best) { best = kvf.second; rg.dominant_family = kvf.first; }
            }
            for (const auto& sk : sset) rg.series.push_back(sk);
            for (const auto& kvp : peaks) rg.metric_peaks.push_back(kvp.second);
            std::sort(rg.metric_peaks.begin(), rg.metric_peaks.end(),
                [](const AnomalyRegion::MetricPeak& a, const AnomalyRegion::MetricPeak& b) {
                    return a.max_abs_value > b.max_abs_value;
                });
            // 代表点：按 score 降序、record_index 升序（稳定可重复）
            std::stable_sort(members.begin(), members.end(), [&](size_t a, size_t b) {
                const double sa = finite_d(r.anomalies[a].score) ? r.anomalies[a].score : -1;
                const double sb = finite_d(r.anomalies[b].score) ? r.anomalies[b].score : -1;
                if (sa != sb) return sa > sb;
                return r.anomalies[a].record_index < r.anomalies[b].record_index;
            });
            const int64_t rep = options.anomaly_representatives > 0 ? options.anomaly_representatives : 5;
            for (size_t k = 0; k < members.size() && (int64_t)k < rep; ++k)
                rg.representatives.push_back(r.anomalies[members[k]]);
            // 分层：按区域 max_score 在**全部分数分布**中的分位决定，保证 critical 是极少数。
            // 无有限分数的区域一律 minor（不给它伪造 severity）。
            if (scored > 0 && critical_cut > 0 && rg.max_score >= critical_cut) rg.severity = "critical";
            else if (scored > 0 && major_cut > 0 && rg.max_score >= major_cut) rg.severity = "major";
            else rg.severity = "minor";
            rg.severity_rule = severity_rule;
            // 前 / 中 / 后 均值对比（区域内最主要的指标，最多 4 个）
            // 用二分定位 EP 区间后只累加该区间（样本按 EP 有序），避免 O(区域数×样本数) 的全量扫描。
            {
                const int64_t winsz = std::max<int64_t>(20, (int64_t)(rg.ep_hi - rg.ep_lo));
                for (size_t pi = 0; pi < rg.metric_peaks.size() && pi < 4; ++pi) {
                    const std::string& sk = rg.metric_peaks[pi].series;
                    auto si = r.series.find(sk);
                    if (si == r.series.end()) continue;
                    const std::vector<Sample>& sv = si->second.samples;
                    AnomalyRegion::WindowDelta wd;
                    wd.series = sk;
                    auto sum_range = [&](int32_t a, int32_t b, double* out, int64_t* n,
                                         int32_t* got_lo, int32_t* got_hi) {
                        auto lo = std::lower_bound(sv.begin(), sv.end(), a,
                            [](const Sample& s, int32_t v) { return s.episode < v; });
                        for (auto it2 = lo; it2 != sv.end() && it2->episode <= b; ++it2) {
                            if (!finite_d(it2->value)) continue;
                            if (*n == 0) *got_lo = it2->episode;
                            *got_hi = it2->episode;
                            *out += it2->value;
                            ++(*n);
                        }
                    };
                    double bs = 0, ds = 0, as = 0;
                    sum_range((int32_t)std::max<int64_t>(INT32_MIN, (int64_t)rg.ep_lo - winsz),
                        rg.ep_lo - 1, &bs, &wd.before_n, &wd.before_ep_lo, &wd.before_ep_hi);
                    sum_range(rg.ep_lo, rg.ep_hi, &ds, &wd.during_n,
                        &wd.during_ep_lo, &wd.during_ep_hi);
                    sum_range(rg.ep_hi + 1, (int32_t)std::min<int64_t>(INT32_MAX, (int64_t)rg.ep_hi + winsz),
                        &as, &wd.after_n, &wd.after_ep_lo, &wd.after_ep_hi);
                    if (wd.before_n > 0 && wd.during_n > 0) {
                        wd.before_mean = bs / (double)wd.before_n;
                        wd.during_mean = ds / (double)wd.during_n;
                        wd.after_mean = wd.after_n > 0 ? as / (double)wd.after_n : NAN;
                        wd.valid = true;
                    } else {
                        wd.null_reason = "前窗或中窗无有效样本：均值对比无定义";
                    }
                    rg.deltas.push_back(wd);
                }
            }
            r.anomaly_regions.push_back(rg);
            i = j + 1;
        }
        for (const auto& a : r.anomalies) r.anomaly_count_by_series[a.series_key]++;
        for (const auto& rg : r.anomaly_regions) {
            if (rg.severity == "critical") { r.anomaly_count_critical += rg.count; r.region_count_critical++; }
            else if (rg.severity == "major") { r.anomaly_count_major += rg.count; r.region_count_major++; }
            else { r.anomaly_count_minor += rg.count; r.region_count_minor++; }
        }
        r.anomaly_rule_legend =
            "trigger_mask bits: 0x01=local_z (prior 512-window z > 4, also |dev| > 0.5*MAD); "
            "0x02=robust_z (prior median/MAD z > 6); 0x04=delta; 0x08=delta2; 0x10=residual; "
            "0x20=level_shift (bits 0x04-0x20 reserved, not currently emitted)";
        r.anomaly_full_data_file = "trajectory_anomalies.jsonl";
    }

    // ---- 全局/局部极值 ----
    stage("extrema", "global and local extrema");
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        std::vector<Extremum> ex;
        if (!s.samples.empty()) {
            size_t imin = 0, imax = 0;
            for (size_t i = 1; i < s.samples.size(); ++i) {
                if (!finite_d(s.samples[i].value)) continue;
                if (!finite_d(s.samples[imin].value) || s.samples[i].value < s.samples[imin].value) imin = i;
                if (!finite_d(s.samples[imax].value) || s.samples[i].value > s.samples[imax].value) imax = i;
            }
            if (finite_d(s.samples[imax].value)) {
                Extremum e; e.kind = "global_max"; e.episode = s.samples[imax].episode;
                e.record_index = s.samples[imax].record_index; e.value = s.samples[imax].value;
                e.series_key = kv.first; ex.push_back(e);
            }
            if (finite_d(s.samples[imin].value)) {
                Extremum e; e.kind = "global_min"; e.episode = s.samples[imin].episode;
                e.record_index = s.samples[imin].record_index; e.value = s.samples[imin].value;
                e.series_key = kv.first; ex.push_back(e);
            }
            // 局部极值：需与相邻两点比较，且幅度超过全序列标准差的 1 倍（避免报出大量微小极值）
            double sd = 0;
            auto it = r.scalars.find(kv.first);
            if (it != r.scalars.end()) sd = it->second.stddev;
            for (size_t i = 1; i + 1 < s.samples.size(); ++i) {
                double a = s.samples[i - 1].value, b = s.samples[i].value, c = s.samples[i + 1].value;
                if (!finite_d(a) || !finite_d(b) || !finite_d(c)) continue;
                if (b > a && b > c && (sd <= 0 || (b - std::max(a, c)) > sd)) {
                    Extremum e; e.kind = "local_max"; e.episode = s.samples[i].episode;
                    e.record_index = s.samples[i].record_index; e.value = b; e.series_key = kv.first;
                    ex.push_back(e);
                }
                if (b < a && b < c && (sd <= 0 || (std::min(a, c) - b) > sd)) {
                    Extremum e; e.kind = "local_min"; e.episode = s.samples[i].episode;
                    e.record_index = s.samples[i].record_index; e.value = b; e.series_key = kv.first;
                    ex.push_back(e);
                }
            }
        }
        r.extrema[kv.first] = ex;
    }

    // ---- 当前历史位置 ----
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        std::vector<double> vals;
        for (const auto& x : s.samples) if (finite_d(x.value)) vals.push_back(x.value);
        AnalyzeResult::CurrentPosition cp;
        cp.series_key = kv.first;
        cp.percentile_definition = "百分位排名 = (严格小于当前值的样本数 + 等于当前值样本数的一半) / 总有效样本数 × 100";
        if (vals.size() < 3) {
            cp.valid = false; cp.null_reason = "valid samples < 3";
        } else {
            double cur = vals.back();
            std::sort(vals.begin(), vals.end());
            cp.current = cur;
            cp.percentile_rank = percentile_rank(vals, cur);
            cp.valid = true;
        }
        r.current_positions.push_back(cp);
    }

    // ---- 压缩摘要 ----
    stage("compression", "adaptive keypoint compression");
    auto rdp = [](const std::vector<size_t>& idx, const Series& s, double tol,
                  std::vector<size_t>* keep) {
        // 迭代式垂距法：始终保留首尾；对超阈值段二分。
        std::vector<std::pair<size_t, size_t>> stack;
        if (idx.size() < 2) { *keep = idx; return; }
        keep->clear();
        keep->push_back(idx.front());
        stack.push_back({ 0, idx.size() - 1 });
        while (!stack.empty()) {
            auto seg = stack.back(); stack.pop_back();
            size_t a = seg.first, b = seg.second;
            if (b <= a + 1) continue;
            double x0 = (double)s.samples[idx[a]].record_index, y0 = s.samples[idx[a]].value;
            double x1 = (double)s.samples[idx[b]].record_index, y1 = s.samples[idx[b]].value;
            double maxd = -1; size_t maxi = a;
            for (size_t i = a + 1; i < b; ++i) {
                double x = (double)s.samples[idx[i]].record_index, y = s.samples[idx[i]].value;
                double d;
                if (x1 == x0) d = fabs(y - y0);
                else {
                    double t = (x - x0) / (x1 - x0);
                    double proj = y0 + t * (y1 - y0);
                    d = fabs(y - proj);
                }
                if (d > maxd) { maxd = d; maxi = i; }
            }
            if (maxd > tol) {
                keep->push_back(idx[maxi]);
                stack.push_back({ a, maxi });
                stack.push_back({ maxi, b });
            }
        }
        keep->push_back(idx.back());
        std::sort(keep->begin(), keep->end());
    };
    auto eval_error = [](const std::vector<size_t>& keep, const Series& s,
                         double* max_err, double* rmse) {
        *max_err = 0; *rmse = 0;
        if (keep.size() < 2) return;
        double sum_sq = 0; int64_t n = 0;
        size_t k = 0;
        for (size_t i = 0; i < s.samples.size(); ++i) {
            while (k + 1 < keep.size() && keep[k + 1] < i) ++k;
            if (k + 1 >= keep.size()) break;
            size_t a = keep[k], b = keep[k + 1];
            if (i <= a || i >= b) continue;
            double x0 = (double)s.samples[a].record_index, y0 = s.samples[a].value;
            double x1 = (double)s.samples[b].record_index, y1 = s.samples[b].value;
            double x = (double)s.samples[i].record_index, y = s.samples[i].value;
            double proj = (x1 == x0) ? y0 : y0 + (x - x0) / (x1 - x0) * (y1 - y0);
            double e = fabs(y - proj);
            if (e > *max_err) *max_err = e;
            sum_sq += e * e; ++n;
        }
        if (n > 0) *rmse = sqrt(sum_sq / (double)n);
    };
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        if (s.samples.size() < 3) continue;
        CompressionResult c;
        c.series_key = kv.first;
        c.method = "adaptive binary segmentation (perpendicular distance / RDP) with linear reconstruction";
        c.error_unit = unit_name(s.unit);
        c.original_points = (int64_t)s.samples.size();
        std::vector<size_t> all;
        for (size_t i = 0; i < s.samples.size(); ++i) if (finite_d(s.samples[i].value)) all.push_back(i);
        if (all.size() < 3) continue;

        double p05 = 0, p95 = 0;
        {
            std::vector<double> v;
            for (size_t i : all) v.push_back(s.samples[i].value);
            std::sort(v.begin(), v.end());
            p05 = quantile_sorted(v, 0.05); p95 = quantile_sorted(v, 0.95);
        }
        double span = p95 - p05;
        double tol = options.error_tolerance > 0 ? options.error_tolerance : (span > 0 ? span * 0.05 : 0);

        if (tol <= 0) {
            c.mode = options.compression_mode;
            c.tolerance = 0; c.tolerance_is_hard = false;
            c.target_met = false;
            c.note = "序列近似常数（p95-p05 == 0）：误差阈值无定义，不声称可重建趋势；"
                "仅保留首尾点与极值点。";
            c.points.push_back({ s.samples[all.front()].record_index, s.samples[all.front()].episode, s.samples[all.front()].value, true, "series_start" });
            c.points.push_back({ s.samples[all.back()].record_index, s.samples[all.back()].episode, s.samples[all.back()].value, true, "series_end" });
            c.kept_points = (int64_t)c.points.size();
            eval_error({ all.front(), all.back() }, s, &c.max_error, &c.rmse);
            r.compression[kv.first] = c;
            continue;
        }

        std::vector<size_t> keep;
        // 模式选择：显式给了 tol/budget 就照办；否则**长轨迹自动改用点数预算**。
        // 理由：默认容差 = 区间 5%，对噪声轨迹几乎压不动（实测 20.4 万 EP 点仍留 17 万点），
        // 等于没压缩；轨迹压缩的目标是有限摘要，误差如实报告即可。
        std::string eff_mode = options.compression_mode;
        int64_t eff_budget = options.length_budget;
        bool auto_budget = false;
        if (options.error_tolerance <= 0 && options.length_budget <= 0 &&
            !s.noise_dominated && (int64_t)all.size() > 10000) {
            eff_mode = "length_priority";
            eff_budget = options.trajectory_budget > 0 ? options.trajectory_budget : 2000;
            auto_budget = true;
        }
        if (eff_mode == "length_priority" && eff_budget > 0) {
            c.mode = "length_priority";
            // 耗时控制：二分容差只在**等距子样本**上做，再用得到的容差对全集跑一次 RDP。
            // 直接对 20 万点全集二分 40 次会让单次分析从 30s 涨到 300s。
            std::vector<size_t> probe;
            {
                const int64_t PROBE_MAX = 40000;
                if ((int64_t)all.size() > PROBE_MAX) {
                    const size_t stp = (size_t)((int64_t)all.size() / PROBE_MAX);
                    for (size_t i = 0; i < all.size(); i += (stp ? stp : 1)) probe.push_back(all[i]);
                    if (!probe.empty() && probe.back() != all.back()) probe.push_back(all.back());
                } else {
                    probe = all;
                }
            }
            const int64_t probe_budget = std::max<int64_t>(16,
                (int64_t)eff_budget * (int64_t)probe.size() / (int64_t)all.size());
            double lo = 0, hi = span;
            for (int it = 0; it < 14; ++it) {
                double mid = (lo + hi) * 0.5;
                std::vector<size_t> k;
                rdp(probe, s, mid, &k);
                if ((int64_t)k.size() > probe_budget) lo = mid; else hi = mid;
            }
            rdp(all, s, hi, &keep);
            c.tolerance = hi;
            c.tolerance_is_hard = false;
            c.note = auto_budget
                ? ("长轨迹自动点数预算 目标=" + fmt_i(eff_budget) +
                   "，实测 RDP 保留=" + fmt_i((int64_t)keep.size()) +
                   "（容差在等距子样本上求解后对全集应用一次，点数与预算不必严格相等）："
                   "容差由预算反解，误差非硬保证，实测误差见下。")
                : "篇幅优先：容差由点数预算反解（子样本求解 + 全集单次应用），误差非硬保证。";
        } else {
            c.mode = "error_priority";
            c.tolerance = tol;
            c.tolerance_is_hard = true;
            rdp(all, s, tol, &keep);
            c.note = "误差优先：容差为硬阈值；点数由数据噪声决定，可能远多于预算。";
        }
        // record_index → 下标 映射（避免 O(异常数×样本数)）
        std::map<int64_t, size_t> idx_of_record;
        for (size_t i = 0; i < s.samples.size(); ++i) idx_of_record[s.samples[i].record_index] = i;

        // 噪声主导序列：逐 update 的原始流本身就是随机噪声，对其逐点重建等于拟合噪声
        // （实测 103 万输入点要留 84.6 万点）。这类序列不做 raw RDP，
        // 轨迹压缩改在 EP 分布聚合序列上做（见 derived_ep_series）。
        if (s.noise_dominated) {
            c.method = "no raw RDP (per-update stream is noise-dominated); trajectory compression applied to "
                       "EP-aggregated distribution series instead (see derived_ep_series)";
            c.tolerance = 0;
            c.tolerance_is_hard = false;
            c.note = "噪声主导：不对逐 update 原始点做重建，本序列只保留不可删除点"
                     "（异常/事件/极值/变点/分段边界）；轨迹形态见 EP 聚合序列。";
            keep.clear();
        }

        // ---- 关键点预算：按优先级保留，**绝不**对最终集合做等距抽样 ----
        // Priority = Anomaly > ConfigEvent > Extrema > ChangePoint > SegmentBoundary > Ordinary
        {
            std::map<size_t, int> cls_of;
            std::map<size_t, double> rank_of;
            auto put = [&](size_t idx, int cls, double rank) {
                auto it2 = cls_of.find(idx);
                if (it2 == cls_of.end() || cls < it2->second) { cls_of[idx] = cls; rank_of[idx] = rank; }
                else if (cls == it2->second && rank > rank_of[idx]) rank_of[idx] = rank;
            };
            // 0) 异常
            for (const auto& a : r.anomalies) {
                if (a.series_key != kv.first) continue;
                auto f = idx_of_record.find(a.record_index);
                if (f != idx_of_record.end()) put(f->second, 0, finite_d(a.score) ? a.score : 0.0);
            }
            // 1) 配置事件附近的记录（depth 等字段的变化点，证据来自记录本身）
            {
                std::set<int64_t> ev;
                for (int m = 0; m < 3; ++m) {
                    char ek[32];
                    snprintf(ek, sizeof(ek), "depth_mode%d", m);
                    auto ei = r.series.find(ek);
                    if (ei == r.series.end()) continue;
                    const Series& es = ei->second;
                    for (size_t k = 1; k < es.samples.size(); ++k)
                        if (es.samples[k].value != es.samples[k - 1].value)
                            ev.insert(es.samples[k].record_index);
                }
                for (size_t k = 0; k < s.samples.size(); ++k)
                    if (ev.count(s.samples[k].record_index)) put(k, 1, 1.0);
            }
            // 2) 全局极值 —— MANDATORY
            {
                auto ei = r.extrema.find(kv.first);
                if (ei != r.extrema.end())
                    for (const auto& e : ei->second) {
                        auto f = idx_of_record.find(e.record_index);
                        if (f == idx_of_record.end()) continue;
                        if (e.kind == "global_max" || e.kind == "global_min") put(f->second, 2, 2.0);
                        else put(f->second, 4, 1.0);   // 局部极值 = PRIORITIZED
                    }
            }
            // 3) 分段边界 —— MANDATORY
            {
                const size_t seg_len = std::max<size_t>(8, all.size() / 8);
                for (size_t st = 0; st < all.size(); st += seg_len) put(all[st], 3, 1.0);
            }
            // 4) 变点 —— PRIORITIZED，且对**低复杂度序列强抑制**。
            // epsilon 这类平滑单调曲线 RDP 只需极少数点；若仍把上万个局部 Δ² 升级为变点，
            // 说明变点定义过敏感。规则：RDP 点数远小于样本数时，不再判变点。
            {
                const bool low_complexity = !keep.empty() &&
                    ((int64_t)keep.size() <= std::max<int64_t>(8, (int64_t)all.size() / 1000));
                if (low_complexity) {
                    c.note += " 低复杂度序列（RDP 只需 " + fmt_i((int64_t)keep.size()) +
                              " 点 / " + fmt_i((int64_t)all.size()) +
                              " 样本）：抑制伪变点，Δ/Δ² 不升级为 change_point。";
                } else {
                    auto di = r.diffs.find(kv.first);
                    if (di != r.diffs.end())
                        for (const auto& dp : di->second.points) {
                            auto f = idx_of_record.find(dp.record_index);
                            if (f == idx_of_record.end()) continue;
                            double rk = fabs(dp.raw_diff);
                            if (dp.second_valid) rk = std::max(rk, fabs(dp.second_diff));
                            put(f->second, 5, rk);
                        }
                }
            }
            // 5) 普通重建点 —— PRIORITIZED
            for (size_t i : keep) put(i, 6, 0.0);

            const int64_t budget = options.keypoint_budget > 0 ? options.keypoint_budget : 20000;
            std::set<size_t> stored;
            static const char* cnames[7] = { "anomaly", "config_event", "global_extremum",
                                             "segment_boundary", "local_extremum", "change_point",
                                             "ordinary" };
            // Mandatory != Priority：前 4 类**不受预算限制**（配置变化可能只有 1 个点，
            // 却可能解释后面几万局，其价值高于数百个普通异常点）。
            const int MANDATORY_CLASSES = 4;
            for (int cls = 0; cls < 7; ++cls) {
                std::vector<size_t> v;
                for (const auto& kv2 : cls_of) if (kv2.second == cls) v.push_back(kv2.first);
                // 类内排序：rank 降序、下标升序 → 稳定可重复
                std::stable_sort(v.begin(), v.end(), [&](size_t a, size_t b) {
                    double ra = rank_of[a], rb = rank_of[b];
                    if (ra != rb) return ra > rb;
                    return a < b;
                });
                CompressionResult::ClassBudget cb;
                cb.name = cnames[cls];
                cb.priority = cls;
                cb.available = (int64_t)v.size();
                // 异常类不受预算限制：异常点一个都不能被压缩掉。
                cb.budget_exempt = (cls < MANDATORY_CLASSES);
                for (size_t i : v) {
                    if (!cb.budget_exempt && (int64_t)stored.size() >= budget) { cb.omitted++; continue; }
                    stored.insert(i);
                    cb.kept++;
                }
                c.budget_classes.push_back(cb);
            }
            c.retention_policy =
                "MANDATORY (budget-exempt): anomaly > config_event > global_extremum > segment_boundary; "
                "PRIORITIZED (budget-limited): local_extremum > change_point > ordinary; "
                "equispaced sampling of the final keypoint set is NOT used";
            c.equispaced_sampling_used = false;
            c.kept_points = (int64_t)keep.size();   // RDP 决策出的保留点数（真实值）

            if (s.noise_dominated) {
                c.max_error = NAN;
                c.rmse = NAN;
                c.target_met = false;
                c.note += " 未计算重建误差：不对噪声主导的原始流声称可重建趋势。";
            } else {
                // 误差按 RDP 的完整保留集评估（不受预算截断影响）
                eval_error(keep, s, &c.max_error, &c.rmse);
                c.target_met = (c.tolerance > 0) ? (c.max_error <= c.tolerance * 1.0000001) : false;
                if (!c.target_met && c.tolerance > 0)
                    c.note += " 实测最大误差超过容差：噪声大的序列无法同时满足「很短」与「最大误差很小」，"
                              "此处以实际误差为准。";
            }

            for (size_t i : stored) {
                KeyPoint kp;
                kp.record_index = s.samples[i].record_index;
                kp.episode = s.samples[i].episode;
                kp.value = s.samples[i].value;
                kp.must_keep = (i == all.front() || i == all.back());
                kp.reason = kp.must_keep ? "series_boundary" : cnames[cls_of[i]];
                c.points.push_back(kp);
            }
            // 异常被预算挤掉时必须显式告警（异常优先原则不得被预算破坏）
            for (const auto& cb : c.budget_classes) {
                if (cb.name == "anomaly" && cb.omitted > 0)
                    c.note += " [警告] 异常类关键点因预算不足被省略 " + fmt_i(cb.omitted) +
                              " 个（异常数已超过预算 " + fmt_i(budget) + "），建议提高 keypoint_budget。";
            }
        }
        r.compression[kv.first] = c;
    }

    // ---- 相关与滞后相关 ----
    stage("correlation", "correlation and lagged correlation");
    struct Pair { const char* a; const char* b; };
    const Pair pairs[] = {
        { "loss", "q_value" }, { "loss", "delta_d" }, { "epsilon_episode", "episode_length" },
        { "q_value", "target_q" }, { "delta_d", "loss" },
    };
    for (const auto& p : pairs) {
        auto ia = r.series.find(p.a), ib = r.series.find(p.b);
        if (ia == r.series.end() || ib == r.series.end()) continue;
        // 按 EP 聚合（同一 EP 取均值）后再配对；只配同一 EP 都存在的位置
        auto agg = [](const Series& s) {
            std::map<int32_t, std::pair<double, int64_t>> m;
            for (const auto& x : s.samples) {
                if (!finite_d(x.value)) continue;
                auto& e = m[x.episode];
                e.first += x.value; e.second++;
            }
            std::map<int32_t, double> out;
            for (auto& kv : m) out[kv.first] = kv.second.first / (double)kv.second.second;
            return out;
        };
        auto A = agg(ia->second), B = agg(ib->second);
        for (int transform = 0; transform < 3; ++transform) {
            std::vector<double> av, bv;
            std::vector<int32_t> eps;
            for (auto& kv : A) {
                auto it = B.find(kv.first);
                if (it == B.end()) continue;
                eps.push_back(kv.first); av.push_back(kv.second); bv.push_back(it->second);
            }
            if (eps.size() < 3) continue;
            std::string tname = transform == 0 ? "raw" : (transform == 1 ? "diff" : "detrended");
            std::vector<double> x = av, y = bv;
            if (transform == 1) {
                std::vector<double> dx, dy;
                for (size_t i = 1; i < x.size(); ++i) { dx.push_back(x[i] - x[i - 1]); dy.push_back(y[i] - y[i - 1]); }
                x = dx; y = dy;
            } else if (transform == 2) {
                // 去趋势：各自减去对序号的线性拟合（声明口径）
                for (size_t s2 = 0; s2 < 2; ++s2) {
                    std::vector<double>& src = (s2 == 0) ? x : y;
                    std::vector<double> xs(src.size());
                    for (size_t i = 0; i < src.size(); ++i) xs[i] = (double)i;
                    double a2, b2, rr; std::string w2;
                    if (linreg(xs, src, &a2, &b2, &rr, &w2)) {
                        for (size_t i = 0; i < src.size(); ++i) src[i] -= (a2 * (double)i + b2);
                    }
                }
            }
            for (int lag : { 0, 1, 2, 5, -1, -2, -5 }) {
                std::vector<double> ca, cb;
                int32_t lo = 0, hi = 0;
                if (lag >= 0) {
                    if ((size_t)lag >= x.size()) continue;
                    for (size_t i = 0; i + (size_t)lag < x.size(); ++i) { ca.push_back(x[i]); cb.push_back(y[i + lag]); }
                } else {
                    size_t l = (size_t)(-lag);
                    if (l >= y.size()) continue;
                    for (size_t i = 0; i + l < y.size(); ++i) { ca.push_back(x[i + l]); cb.push_back(y[i]); }
                }
                Correlation c;
                c.series_a = p.a; c.series_b = p.b; c.lag = lag;
                c.lag_sign_convention = "corr(a(EP), b(EP+lag))；lag 单位为「局(episode)」。正 lag 表示 b 滞后 a。";
                c.alignment = "同一时间线内按 episode 精确配对；不做前向填充、不用最近邻制造配对；"
                    "每个 EP 先在该序列内取均值再配对。";
                c.transform = tname;
                c.pairs = (int64_t)ca.size();
                if (ca.empty()) { lo = hi = 0; } else { c.ep_lo = eps.front(); c.ep_hi = eps.back(); }
                (void)lo; (void)hi;
                std::string why;
                if (!pearson(ca, cb, &c.r, &why)) {
                    c.valid = false; c.null_reason = why;
                } else {
                    c.valid = true; c.abs_r = fabs(c.r);
                }
                r.correlations.push_back(c);
            }
        }
    }

    // ---- 配置事件（必须有证据）----
    stage("events", "configuration events from recorded evidence");
    for (int m = 0; m < 3; ++m) {
        char key[32];
        snprintf(key, sizeof(key), "depth_mode%d", m);
        auto it = r.series.find(key);
        if (it == r.series.end()) continue;
        const Series& s = it->second;
        for (size_t i = 1; i < s.samples.size(); ++i) {
            if (s.samples[i].value == s.samples[i - 1].value) continue;
            ConfigEvent e;
            e.field = std::string("minimax_depth(mode ") + char('0' + m) + ")";
            e.kind = "numeric";
            e.episode = s.samples[i].episode;
            e.record_index = s.samples[i].record_index;
            e.old_value = s.samples[i - 1].value;
            e.new_value = s.samples[i].value;
            e.evidence = std::string("metric_eval.bin fields[1] @ episode ") + fmt_i(e.episode);
            e.evidence_is_log = false;
            if (fabs(e.old_value) > 1e-9) { e.change_ratio = e.new_value / e.old_value; e.ratio_valid = true; }
            else { e.ratio_valid = false; e.ratio_null_reason = "分母(旧值)≈0：倍数无定义"; }
            // 前后窗口
            {
                ConfigEventWindow w;
                size_t lo = i > 20 ? i - 20 : 0;
                for (size_t k = lo; k < i; ++k) {
                    if (!finite_d(s.samples[k].value)) { w.before_missing++; continue; }
                    w.before_mean += s.samples[k].value; w.before_n++;
                }
                size_t hi = std::min(s.samples.size(), i + 20);
                for (size_t k = i; k < hi; ++k) {
                    if (!finite_d(s.samples[k].value)) { w.after_missing++; continue; }
                    w.after_mean += s.samples[k].value; w.after_n++;
                }
                if (w.before_n > 0) { w.before_mean /= (double)w.before_n; w.before_valid = true; }
                if (w.after_n > 0) { w.after_mean /= (double)w.after_n; w.after_valid = true; }
                w.before_ep_lo = s.samples[lo].episode; w.before_ep_hi = s.samples[i - 1].episode;
                w.after_ep_lo = s.samples[i].episode; w.after_ep_hi = s.samples[hi - 1].episode;
                if (!w.before_valid || !w.after_valid)
                    w.null_reason = "前后窗口之一无有效样本";
                e.window = w;
            }
            e.note = "事件依据记录字段变化确定；未记录过的配置修改不会被推断。";
            r.config_events.push_back(e);
        }
    }
    // 日志证据事件
    {
        FILE* f = fopen((dir + "training_fault.log").c_str(), "rb");
        if (f) {
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                ConfigEvent e;
                e.field = "training_fault";
                e.kind = "categorical";
                int ep = -1;
                const char* p = strstr(line, "ep=");
                if (p) ep = atoi(p + 3);
                e.episode = ep;
                e.old_value = 0; e.new_value = 1;
                e.evidence = std::string("training_fault.log: ") + line;
                e.evidence_is_log = true;
                e.ratio_valid = false; e.ratio_null_reason = "类别事件：不计算倍数，保留原始记录";
                e.note = "故障/重启事件；由日志直接确认，非推断。";
                r.config_events.push_back(e);
            }
            fclose(f);
        } else {
            add_limitation(r, "training_fault.log 不存在：无法确认重启/故障事件");
        }
    }
    // 事件重叠标注（同时发生）
    for (size_t i = 0; i < r.config_events.size(); ++i)
        for (size_t j = 0; j < r.config_events.size(); ++j)
            if (i != j && r.config_events[i].episode >= 0 &&
                r.config_events[i].episode == r.config_events[j].episode)
                r.config_events[i].overlapping.push_back(r.config_events[j].field);
    for (auto& e : r.config_events) {
        std::sort(e.overlapping.begin(), e.overlapping.end());
        e.overlapping.erase(std::unique(e.overlapping.begin(), e.overlapping.end()), e.overlapping.end());
        if (!e.overlapping.empty())
            e.note += " 与同时发生的其它事件重叠列出；不声称因果关系。";
    }

    // ---- D 专项与长尾 Loss ----
    stage("special", "delta-D and long-tail loss specials");
    {
        auto it = r.series.find("delta_d");
        r.delta_d.definition = "D = q_value − target_q，在**同一条记录内**配对（二者是同一次更新、同一被接受批的均值）。";
        r.delta_d.pairing = "逐记录配对：先算 D，再对 D 聚合。";
        r.delta_d.is_batch_mean_td_error = true;
        r.delta_d.confirmation_evidence =
            "已由代码确认 target_q 即相应 TD 目标：dqn_agent.cpp:598 用 "
            "batch_qvals[b][batch_actions[b]] - batch_targets[b] 作为 Huber 损失的误差项；"
            "dqn_agent.cpp:656-662 对**同一个被接受批** b=0..sample_count-1 分别累加 "
            "batch_qvals[b][actions[b]] 与 batch_targets[b] 后各除以 sample_count。"
            "因此 mean(Q) - mean(target) == mean(Q - target)，即**该 batch 的平均 TD 误差**。";
        r.delta_d.forbidden_note =
            "仍然禁止：(1) 称为「逐样本 TD 误差」——逐样本 Q/target 值未落盘，只有 batch 均值；"
            "(2) 把来自不同总体（不同被接受批、不同 replay_type、不同时间线）的聚合均值相减后冒充误差统计。"
            "另注意被拒样本（delta 门/Huber 门）已从该批剔除，D 只描述**被接受样本**的批均值。";
        if (it != r.series.end() && !it->second.samples.empty()) {
            std::vector<double> d;
            for (const auto& x : it->second.samples) if (finite_d(x.value)) d.push_back(x.value);
            r.delta_d.raw_records = (int64_t)it->second.samples.size();
            r.delta_d.pairs = (int64_t)d.size();
            if (d.size() >= 2) {
                std::sort(d.begin(), d.end());
                double sum = 0; for (double v : d) sum += v;
                r.delta_d.mean = sum / (double)d.size();
                double var = 0; for (double v : d) var += (v - r.delta_d.mean) * (v - r.delta_d.mean);
                r.delta_d.stddev = sqrt(var / (double)(d.size() - 1));
                r.delta_d.min = d.front(); r.delta_d.max = d.back();
                r.delta_d.p50 = quantile_sorted(d, 0.5);
                r.delta_d.p90 = quantile_sorted(d, 0.9);
                r.delta_d.p99 = quantile_sorted(d, 0.99);
                r.delta_d.valid = true;
            } else {
                r.delta_d.null_reason = "valid paired samples < 2";
            }
        } else {
            r.delta_d.null_reason = "metric_loss.bin 不可用或为空";
        }
    }
    {
        auto it = r.series.find("loss");
        r.long_tail_loss.series_key = "loss";
        r.long_tail_loss.long_tail_ratio_definition =
            "Long-tail ratio = 尾部均值(>p95) / 全序列中位数。无量纲比值；1 = 无长尾。"
            "取值不是 [-1,1]，**不是效应量**，不要与 Cliff's delta 混用。";
        r.long_tail_loss.cliff_delta_available = false;
        r.long_tail_loss.cliff_delta_null_reason =
            "Cliff's delta = N/A：以分位数划分的两组（>p95 与 [p25,p75]）其 delta 恒等于 1，"
            "是构造成立而非观测效应，故不报告该字段。";
        if (it != r.series.end()) {
            std::vector<double> v;
            for (const auto& x : it->second.samples) if (finite_d(x.value)) v.push_back(x.value);
            r.long_tail_loss.n = (int64_t)v.size();
            if (v.size() >= 10) {
                std::vector<double> sv = v;
                std::sort(sv.begin(), sv.end());
                r.long_tail_loss.median = quantile_sorted(sv, 0.5);
                std::vector<double> dev;
                for (double x : sv) dev.push_back(fabs(x - r.long_tail_loss.median));
                r.long_tail_loss.mad = median_of(dev) * 1.4826;
                r.long_tail_loss.p90 = quantile_sorted(sv, 0.90);
                r.long_tail_loss.p95 = quantile_sorted(sv, 0.95);
                r.long_tail_loss.p99 = quantile_sorted(sv, 0.99);
                double total = 0; for (double x : sv) total += x;
                const double p25 = quantile_sorted(sv, 0.25);
                const double p75 = quantile_sorted(sv, 0.75);
                std::vector<double> tail, body;
                // 尾部 = 超过 p95；主体 = 四分位区间 [p25,p75]。
                // 若主体取"全部非尾部样本"，Cliff's delta 会因构造成立而恒为 1，失去意义。
                for (double x : sv) {
                    if (x > r.long_tail_loss.p95) tail.push_back(x);
                    else if (x >= p25 && x <= p75) body.push_back(x);
                }
                r.long_tail_loss.tail_n = (int64_t)tail.size();
                if (!tail.empty()) {
                    double ts = 0; for (double x : tail) ts += x;
                    r.long_tail_loss.tail_mean = ts / (double)tail.size();
                    if (total > 0) r.long_tail_loss.tail_share_of_sum = ts / total;
                }
                if (!tail.empty() && fabs(r.long_tail_loss.median) > 1e-12) {
                    // 只导出值定义的长尾严重度；分位切分的 Cliff's delta 恒为 1，不报告
                    r.long_tail_loss.long_tail_ratio = r.long_tail_loss.tail_mean / r.long_tail_loss.median;
                    r.long_tail_loss.valid = true;
                } else {
                    r.long_tail_loss.valid = false;
                    r.long_tail_loss.null_reason = tail.empty()
                        ? "empty tail group" : "median ≈ 0: tail/median ratio undefined (no division by zero)";
                }
                int64_t shown = 0;
                for (const auto& a : r.anomalies) {
                    if (a.series_key != "loss") continue;
                    if (shown++ >= 20) break;
                    r.long_tail_loss.tail_points.push_back(a);
                }
            } else {
                r.long_tail_loss.null_reason = "valid samples < 10";
            }
        } else {
            r.long_tail_loss.null_reason = "metric_loss.bin 不可用";
        }
    }

    // ---- 分段拟合（自适应）----
    stage("segments", "adaptive segmentation and fit statistics");
    for (auto& kv : r.series) {
        const Series& s = kv.second;
        std::vector<Segment> segs;
        std::vector<double> v;
        for (const auto& x : s.samples) if (finite_d(x.value)) v.push_back(x.value);
        if (v.size() >= 12) {
            // 简单自适应：先按固定最大段长切分，再按斜率差合并（声明式、可重复）
            const size_t max_len = std::max<size_t>(8, v.size() / 8);
            size_t start = 0;
            while (start < v.size()) {
                size_t end = std::min(v.size(), start + max_len);
                Segment sg;
                sg.index = (int64_t)segs.size();
                sg.lo_record = s.samples[start].record_index;
                sg.hi_record = s.samples[end - 1].record_index;
                sg.ep_lo = s.samples[start].episode;
                sg.ep_hi = s.samples[end - 1].episode;
                sg.sample_count = (int64_t)(end - start);
                std::vector<double> xs, ys;
                for (size_t i = start; i < end; ++i) {
                    if (!finite_d(s.samples[i].value)) continue;
                    xs.push_back((double)s.samples[i].episode);
                    ys.push_back(s.samples[i].value);
                }
                std::string why;
                if (linreg(xs, ys, &sg.slope, &sg.intercept, &sg.r2, &why)) {
                    sg.slope_per_episode = sg.slope;
                    sg.fit_note = "最小二乘线性拟合；slope 单位=值/局；R²=决定系数；残差=实际值−拟合值";
                    // 残差统计（与旧报告 print_regression 对齐；样本标准差 n-1 分母）
                    double rsum = 0, rmin = 0, rmax = 0, rvar = 0;
                    int64_t rn = 0;
                    for (size_t k = 0; k < xs.size(); ++k) {
                        double res = ys[k] - (sg.slope * xs[k] + sg.intercept);
                        if (rn == 0) { rmin = rmax = res; }
                        else { if (res < rmin) rmin = res; if (res > rmax) rmax = res; }
                        rsum += res; ++rn;
                    }
                    sg.residual_n = rn;
                    if (rn > 0) {
                        sg.residual_mean = rsum / (double)rn;
                        sg.residual_min = rmin; sg.residual_max = rmax;
                        for (size_t k = 0; k < xs.size(); ++k) {
                            double res = ys[k] - (sg.slope * xs[k] + sg.intercept) - sg.residual_mean;
                            rvar += res * res;
                        }
                        if (rn > 1) {
                            sg.residual_stddev = sqrt(rvar / (double)(rn - 1));
                            sg.residual_valid = true;
                        } else {
                            sg.residual_valid = false;
                            sg.residual_null_reason = "residual samples < 2: residual std undefined";
                        }
                    } else {
                        sg.residual_valid = false;
                        sg.residual_null_reason = "no valid residual samples";
                    }
                } else {
                    sg.fit_note = why;
                    sg.residual_valid = false;
                    sg.residual_null_reason = why;
                }
                segs.push_back(sg);
                start = end;
            }
        }
        r.segments[kv.first] = segs;
    }

    // 样本预览：前若干条有效训练记录**原样**（不聚合、不抽样），与旧报告的 Sample 行对齐。
    {
        auto itl = r.series.find("loss");
        auto itq = r.series.find("q_value");
        auto itt = r.series.find("target_q");
        if (itl != r.series.end() && itq != r.series.end() && itt != r.series.end()) {
            std::map<int64_t, const Sample*> qi, ti;
            for (const auto& x : itq->second.samples) qi[x.record_index] = &x;
            for (const auto& x : itt->second.samples) ti[x.record_index] = &x;
            for (const auto& x : itl->second.samples) {
                if ((int64_t)r.preview_records.size() >= r.preview_limit) break;
                auto a = qi.find(x.record_index);
                auto b = ti.find(x.record_index);
                if (a == qi.end() || b == ti.end()) continue;
                AnalyzeResult::PreviewRecord pv;
                pv.episode = x.episode;
                pv.replay_type = x.step;      // metric_loss.bin 的 step = replay_type
                pv.record_index = x.record_index;
                pv.loss = x.value;
                pv.q_value = a->second->value;
                pv.target_q = b->second->value;
                pv.delta_d = pv.q_value - pv.target_q;
                r.preview_records.push_back(pv);
            }
        }
    }

    // ---- 侧别（黑白）历史 ----
    {
        SideWinSnapshot sw = read_side_win(dir);
        if (sw.present) {
            mark_unavailable(r, "黑/白方分边历史胜率",
                "历史流中未记录分边数据；metric_win.bin 存的是黑白平均。仅 latest_side_win.txt 提供最新一次快照，已读入。"
                "因此本报告不提供分边时间序列、分边趋势或分边异常。");
            r.timelines[0].note += " latest_side_win: " + sw.raw;
        } else {
            mark_unavailable(r, "黑/白方分边胜率", "latest_side_win.txt 不存在或格式不可识别。");
        }
    }

    // ---- 未记录信息清单 ----
    mark_unavailable(r, "逐样本 TD 误差（per-sample）",
        "逐样本 Q/target 值未落盘，仅记录同一被接受批的均值；因此只能给出 batch 平均 TD 误差（见差值 D 专项）。");
    mark_unavailable(r, "采样周期 / 评估周期配置",
        "没有记录周期配置字段；因此不得把 Episode 间隔当作缺失记录。");
    mark_unavailable(r, "配置变更完整历史",
        "没有专门的配置日志；只报告能从 metric_eval.bin 字段变化与 watchdog.log/training_fault.log 取得证据的事件。");
    mark_unavailable(r, "每次评估的对手模型版本/身份",
        "未记录。评估只记录 minimax 深度。");
    add_limitation(r,
        "metric_steps.bin（每决策步 4 字段：player/action/reward/done）未纳入逐记录序列分析，"
        "以控制内存与耗时（该流规模约为局数的 12 倍）。本报告以每局聚合视图"
        "（metric_game.bin 局长、metric_rewards.bin 奖励分量）代替；如需逐步明细请另行导出。");
    add_limitation(r,
        "metric_skill_v4.bin（技能测试，29 字段）与 metric_memory_eval.bin（记忆调度）使用各自的"
        "魔数/schema，不在本次 metric_* 统一契约分析范围内，故未展开。"
        "注意 metric_skill_v4 的行会有缺失（写入门有 1000 局与 64 快照完整性条件），"
        "且行内 15 个字段是跨快照统计、14 个是单模型结果，min/max margin 不可与 mean margin 同列比较。");
    add_limitation(r,
        "metric_eval.bin 的 phase 字段是训练器的局部 phase 变量，与 get_curriculum_phase(ep) 的"
        "课程阶段不一定一致（已实测存在差异）；本报告把它当作训练器标签原样呈现，不做推断。");

    r.quality.excluded_total = r.quality.records_invalid + r.quality.records_nonfinite +
        r.quality.records_placeholder;
    r.quality.records_used = r.quality.records_seen - r.quality.excluded_total;
    if (r.quality.records_used < 0) r.quality.records_used = 0;

    r.read_end_unix = (int64_t)time(nullptr);
    r.elapsed_seconds = (double)(GetTickCount() - t0) / 1000.0;
    stage("done", "analysis complete");
    // 输出体积由调用方按需测量：在 Analyze 内构造数百 MB 字符串会让峰值内存翻倍。
    r.output_bytes_text = 0;
    r.output_bytes_json = 0;
    if (r.quality.windows_total_possible > 0 && r.quality.windows_computed > 0)
        r.quality.window_stride =
            std::max<int64_t>(1, r.quality.windows_total_possible / r.quality.windows_computed);
    return r;
}

/* ============================ 序列化 ============================ */

std::string ToCompactText(const AnalyzeResult& r) {
    std::ostringstream o;
    o << "========== 训练轨迹分析（紧凑文本）==========\n";
    o << "schema_version: " << r.schema_version << "\n";
    o << "生成时间: " << r.generated_at << "\n";
    o << "数据来源: " << r.source_dir << "  时间线: " << r.timeline_id << "\n";
    o << "读取字节: " << fmt_i(r.bytes_read) << "  记录(完整): " << fmt_i(r.quality.records_seen)
      << "  使用: " << fmt_i(r.quality.records_used) << "\n";
    o << "读取边界: 已固定=" << (r.quality.read_boundary_fixed ? "yes" : "no")
      << "  截断尾部=" << (r.quality.truncated_tail ? "yes" : "no")
      << " (" << fmt_i(r.quality.truncated_bytes) << " bytes)\n";
    o << "质量: 非法=" << fmt_i(r.quality.records_invalid)
      << " 非有限=" << fmt_i(r.quality.records_nonfinite)
      << " 占位=" << fmt_i(r.quality.records_placeholder)
      << " 重复(键唯一流)=" << fmt_i(r.quality.records_duplicate)
      << " 同键多条(合法)=" << fmt_i(r.quality.same_key_multi)
      << " 相邻全同(键非唯一,正常)=" << fmt_i(r.quality.identical_adjacent_on_nonunique_key)
      << " EP回跳=" << fmt_i(r.quality.reorder_events)
      << " EP未知=" << fmt_i(r.quality.unknown_episode)
      << " 合计排除=" << fmt_i(r.quality.excluded_total) << "\n";
    o << "窗口有界化: 理论窗口=" << fmt_i(r.quality.windows_total_possible)
      << " 实际计算=" << fmt_i(r.quality.windows_computed)
      << " 等距步长=" << fmt_i(r.quality.window_stride)
      << "（滚动统计为等距子集，已声明；非全量逐窗口）\n";
    o << "边界说明: " << r.quality.boundary_note << "\n";

    o << "\n-- 时间线/分支 --\n";
    for (const auto& t : r.timelines) {
        o << "  [" << (t.is_active ? "活动" : "分支") << "] " << t.id;
        if (t.kind == "rollback")
            o << "  pid=" << t.rollback_pid << " tick=" << (unsigned long long)t.rollback_tick;
        o << "\n";
        if (!t.note.empty()) o << "      " << t.note << "\n";
    }

    o << "\n-- 指标口径（名称/含义/单位/聚合/EP 语义/step 语义）--\n";
    for (const auto& kv : r.series) {
        const Series& s = kv.second;
        o << "  " << s.key << " | " << s.file << "#" << s.field
          << " | " << s.meaning
          << " | unit=" << unit_name(s.unit)
          << " | agg=" << aggregation_name(s.aggregation)
          << " | n=" << fmt_i((int64_t)s.samples.size())
          << " | zero_based=" << (s.zero_based ? 1 : 0)
          << " | per_episode=" << (s.per_episode ? 1 : 0) << "\n";
        o << "      EP 语义: " << s.episode_semantics << "\n";
        o << "      step 语义: " << s.step_semantics << "\n";
        o << "      采样: " << s.sampling_note << "\n";
        if (s.noise_dominated) {
            o << "      [噪声主导] 逐 update 流不做逐点重建；轨迹压缩在 EP 聚合序列上进行:";
            for (const auto& d : s.derived_ep_series) o << " " << d;
            o << "\n";
        }
    }

    o << "\n-- 统计量：exact_global（全量精确） vs sampled_point（保留子集）——两个总体，勿混用 --\n";
    for (const auto& kv : r.scalars) {
        const auto& s = kv.second;
        o << "  " << kv.first << "\n";
        o << "      exact_global : n=" << fmt_i(s.exact_n)
          << " mean=" << fmt_d(s.exact_mean)
          << " std=" << (s.exact_std_valid ? fmt_d(s.exact_stddev) : std::string("null"))
          << (s.exact_std_valid ? "" : (" (" + s.exact_std_null_reason + ")"))
          << " min=" << fmt_d(s.exact_min) << " max=" << fmt_d(s.exact_max) << "\n";
        o << "      sampled_point: n=" << fmt_i(s.n)
          << " mean=" << fmt_d(s.mean)
          << " std=" << (s.std_valid ? fmt_d(s.stddev) : std::string("null"))
          << " min=" << fmt_d(s.min) << " max=" << fmt_d(s.max)
          << " p50=" << fmt_d(s.p50) << " p90=" << fmt_d(s.p90) << " p99=" << fmt_d(s.p99) << "\n";
        if (s.sampling_affected)
            o << "      [注意] 抽样已影响该序列：exact_global 与 sampled_point 的均值/极值不一致；"
                 "分位数与点级分析以 sampled_point 为准。\n";
    }

    o << "\n-- Top X% 暴露切片（按数值取最大 X%，与该序列统计并列）--\n";
    if (r.top_slices.empty()) o << "  (无)\n";
    for (const auto& ts : r.top_slices) {
        o << "  " << ts.series_key << ": ";
        if (!ts.valid) { o << "null(" << ts.null_reason << ")\n"; continue; }
        o << "阈值=" << fmt_d(ts.threshold) << " 实际占比=" << fmt_d(ts.actual_percent, 3) << "%"
          << " n=" << fmt_i(ts.n) << "（基于保留子集）\n";
        o << "      切片统计: mean=" << fmt_d(ts.mean)
          << " std=" << (ts.std_valid ? fmt_d(ts.stddev) : std::string("null"))
          << " min=" << fmt_d(ts.min) << " max=" << fmt_d(ts.max)
          << " p50=" << fmt_d(ts.p50) << " p90=" << fmt_d(ts.p90) << " p99=" << fmt_d(ts.p99) << "\n";
        o << "      覆盖范围: EP " << ts.ep_lo << ".." << ts.ep_hi
          << "  record_index " << fmt_i(ts.record_lo) << ".." << fmt_i(ts.record_hi) << "\n";
        if (ts.sum_share_valid)
            o << "      占保留子集总和 = " << fmt_d(ts.sum_share, 4) << "\n";
        else
            o << "      占保留子集总和 = null(" << ts.sum_share_null_reason << ")\n";
        o << "      口径: " << ts.definition << "；" << ts.basis << "\n";
    }

    o << "\n-- 样本预览（前 " << fmt_i(r.preview_limit) << " 条有效训练记录，原样未聚合、未抽样）--\n";
    if (r.preview_records.empty()) o << "  (无)\n";
    for (const auto& pv : r.preview_records)
        o << "  record_index=" << fmt_i(pv.record_index) << " episode=" << pv.episode
          << " replay_type=" << pv.replay_type
          << " loss=" << fmt_d(pv.loss) << " q=" << fmt_d(pv.q_value)
          << " target_q=" << fmt_d(pv.target_q) << " D=" << fmt_d(pv.delta_d) << "\n";

    o << "\n-- 分段拟合与残差（最小二乘；slope 单位=值/局）--\n";
    if (r.segments.empty()) o << "  (无)\n";
    for (const auto& kv : r.segments) {
        const auto& segs = kv.second;
        if (segs.empty()) continue;
        o << "  " << kv.first << ": " << segs.size() << " 段（自适应切分）\n";
        int shown_seg = 0;
        for (const auto& sg : segs) {
            if (shown_seg++ >= 6) { o << "      ... 其余 " << (segs.size() - 6) << " 段省略\n"; break; }
            o << "      段" << fmt_i(sg.index) << " EP " << sg.ep_lo << ".." << sg.ep_hi
              << " n=" << fmt_i(sg.sample_count)
              << " slope=" << fmt_d(sg.slope_per_episode, 8)
              << " R²=" << fmt_d(sg.r2, 6);
            if (sg.residual_valid)
                o << " 残差: 均值=" << fmt_d(sg.residual_mean)
                  << " 标准差=" << fmt_d(sg.residual_stddev)
                  << " 最小=" << fmt_d(sg.residual_min)
                  << " 最大=" << fmt_d(sg.residual_max);
            else
                o << " 残差=null(" << sg.residual_null_reason << ")";
            o << "\n";
        }
    }

    o << "\n-- 当前历史位置（百分位排名）--\n";
    for (const auto& cp : r.current_positions) {
        o << "  " << cp.series_key << ": ";
        if (cp.valid) o << "current=" << fmt_d(cp.current) << " rank=" << fmt_d(cp.percentile_rank, 2) << "%\n";
        else o << "null (" << cp.null_reason << ")\n";
    }

    o << "\n-- 类别指标：Replay Type（频数/比例/转移/连续区间；不做斜率或回归）--\n";
    for (const auto& cs : r.category_stats) {
        o << "  " << cs.series_key << ": total=" << fmt_i(cs.total)
          << " transitions=" << fmt_i(cs.transitions) << "\n";
        o << "      定义: " << cs.definition << "\n";
        for (auto& kv : cs.frequency)
            o << "      类别 " << kv.first << ": 频数=" << fmt_i(kv.second)
              << " 比例=" << fmt_d(cs.proportion.count(kv.first) ? cs.proportion.at(kv.first) : NAN, 4) << "\n";
        for (auto& kv : cs.transition_matrix)
            o << "      转移 " << kv.first.first << "->" << kv.first.second << ": " << fmt_i(kv.second) << "\n";
        int shown = 0;
        for (const auto& run : cs.runs) {
            if (shown++ >= 8) { o << "      ... 其余连续区间省略 " << (cs.runs.size() - 8) << " 段\n"; break; }
            o << "      区间 类别" << run.value << ": 长度=" << fmt_i(run.length)
              << " EP " << run.ep_lo << ".." << run.ep_hi << "\n";
        }
    }

    o << "\n-- 窗口统计（记录窗口与 EP 窗口两种口径分别声明）--\n";
    for (const auto& kv : r.windows) {
        const auto& ws = kv.second;
        if (ws.empty()) continue;
        o << "  " << kv.first << ": " << ws.size() << " 个窗口（上限截断）\n";
        int shown = 0;
        for (const auto& w : ws) {
            if (shown++ >= 4) break;
            o << "      " << w.id << " 口径=" << (w.by_records ? "记录窗口" : "EP窗口")
              << " 宽度=" << w.width << " n=" << fmt_i(w.sample_count)
              << " EP跨度=" << fmt_d(w.episode_span, 1) << " EP " << w.ep_lo << ".." << w.ep_hi;
            if (w.valid)
                o << " mean=" << fmt_d(w.mean) << " std=" << fmt_d(w.stddev)
                  << " min=" << fmt_d(w.min) << " max=" << fmt_d(w.max) << "\n";
            else
                o << " null(" << w.null_reason << ")\n";
        }
        if (ws.size() > 4) o << "      ... 其余窗口省略\n";
    }

    o << "\n-- 异常区域（聚类后的持续异常事件；全部原始异常见 " << r.anomaly_full_data_file << "）--\n";
    o << "  区域数=" << fmt_i((int64_t)r.anomaly_regions.size())
      << " 异常总数=" << fmt_i((int64_t)r.anomalies.size()) << "\n";
    o << "  按异常数分层: critical=" << fmt_i(r.anomaly_count_critical)
      << " major=" << fmt_i(r.anomaly_count_major)
      << " minor=" << fmt_i(r.anomaly_count_minor) << "\n";
    o << "  按区域数分层: critical=" << fmt_i(r.region_count_critical)
      << " major=" << fmt_i(r.region_count_major)
      << " minor=" << fmt_i(r.region_count_minor) << "\n";
    if (r.anomaly_regions.empty()) o << "  (无区域)\n";
    {
        int region_shown = 0;
        for (const auto& rg : r.anomaly_regions) {
            if (region_shown++ >= 12) {
                o << "  ... 其余 " << fmt_i((int64_t)r.anomaly_regions.size() - 12)
                  << " 个区域省略（完整内容见 JSON，未静默丢弃）\n";
                break;
            }
            o << "  [" << rg.severity << "] EP " << rg.ep_lo << ".." << rg.ep_hi
              << " n=" << fmt_i(rg.count)
              << " family=" << (rg.dominant_family.empty() ? "?" : rg.dominant_family)
              << (rg.span_limited ? "(跨度上限切分)" : "")
              << " peak_ep=" << rg.peak_ep << " peak=" << rg.peak_series
              << " peak_score=" << fmt_d(rg.max_score, 3)
              << " mean_score=" << fmt_d(rg.mean_score, 3) << "\n";
            o << "      受影响指标(" << rg.series.size() << "): ";
            for (size_t k = 0; k < rg.series.size() && k < 6; ++k) o << (k ? "," : "") << rg.series[k];
            if (rg.series.size() > 6) o << ",...";
            o << "\n";
            for (size_t k = 0; k < rg.metric_peaks.size() && k < 3; ++k)
                o << "      max|" << rg.metric_peaks[k].series << "| = "
                  << fmt_d(rg.metric_peaks[k].max_abs_value) << " @EP " << rg.metric_peaks[k].at_ep << "\n";
            for (const auto& wd : rg.deltas) {
                if (!wd.valid) { o << "      " << wd.series << " 前后对比: null(" << wd.null_reason << ")\n"; continue; }
                o << "      " << wd.series << " 前/中/后均值 = " << fmt_d(wd.before_mean)
                  << " / " << fmt_d(wd.during_mean) << " / "
                  << (wd.after_n > 0 ? fmt_d(wd.after_mean) : std::string("null"))
                  << "  (n=" << fmt_i(wd.before_n) << "/" << fmt_i(wd.during_n) << "/" << fmt_i(wd.after_n) << ")\n";
            }
        }
    }

    o << "\n-- 异常排行榜（Top " << r.options.top_n
      << "；规则触发≠训练崩溃/恢复/成功）--\n";
    o << "  触发总数=" << fmt_i(r.anomalies_detected)
      << " 已存=" << fmt_i((int64_t)r.anomalies.size())
      << " 省略=" << fmt_i(r.anomalies_omitted)
      << "（存储上限 " << fmt_i(r.anomaly_store_limit) << "，省略数已显式报告，未静默丢弃）\n";
    int shown = 0;
    for (const auto& a : r.anomalies) {
        if (shown++ >= r.options.top_n) break;
        o << "  #" << a.rank << " " << a.series_key << " " << a.rule
          << " EP=" << a.episode << " value=" << fmt_d(a.value)
          << " baseline=" << fmt_d(a.baseline)
          << " score=" << (finite_d(a.score) ? fmt_d(a.score, 3) : std::string("null"))
          << " n_base=" << fmt_i(a.baseline_n) << "\n";
        if (!a.null_reason.empty()) o << "      注: " << a.null_reason << "\n";
    }
    if ((int64_t)r.anomalies.size() > (int64_t)r.options.top_n)
        o << "  ... 省略 " << ((int64_t)r.anomalies.size() - (int64_t)r.options.top_n)
          << " 条（完整结果见 JSON/CSV，未静默丢弃）\n";

    o << "\n-- 异常区间 --\n";
    shown = 0;
    for (const auto& iv : r.anomaly_intervals) {
        if (shown++ >= 10) { o << "  ... 其余区间省略 " << ((int64_t)r.anomaly_intervals.size() - 10) << " 段\n"; break; }
        o << "  " << iv.series_key << " " << iv.rule << " EP " << iv.ep_lo << ".." << iv.ep_hi
          << " count=" << fmt_i(iv.count) << " max_score=" << fmt_d(iv.max_score, 3) << "\n";
    }

    o << "\n-- 极值 --\n";
    for (const auto& kv : r.extrema) {
        int n_local = 0;
        for (const auto& e : kv.second) if (e.kind == "local_max" || e.kind == "local_min") ++n_local;
        for (const auto& e : kv.second) {
            if (e.kind != "global_max" && e.kind != "global_min") continue;
            o << "  " << kv.first << " " << e.kind << " value=" << fmt_d(e.value) << " EP=" << e.episode << "\n";
        }
        o << "  " << kv.first << " 局部极值数=" << n_local << "（阈值=全序列标准差，非全部微小极值）\n";
    }

    o << "\n-- 压缩摘要（重建方式/误差单位/点数/实测误差）--\n";
    int comp_shown = 0;
    for (const auto& kv : r.compression) {
        if (comp_shown++ >= 14) {
            o << "  ... 其余 " << fmt_i((int64_t)r.compression.size() - 14)
              << " 条序列的压缩摘要在此省略（完整内容见 JSON，未静默丢弃）\n";
            break;
        }
        const auto& c = kv.second;
        o << "  " << c.series_key << " mode=" << c.mode << "\n";
        o << "      重建=" << c.method << "  误差单位=" << c.error_unit << "\n";
        o << "      原始点数=" << fmt_i(c.original_points) << " 保留点数=" << fmt_i(c.kept_points)
          << "  容差=" << fmt_d(c.tolerance) << "(硬=" << (c.tolerance_is_hard ? "yes" : "no") << ")\n";
        o << "      实测最大误差=" << fmt_d(c.max_error) << " 实测RMSE=" << fmt_d(c.rmse)
          << " 达标=" << (c.target_met ? "yes" : "no") << "\n";
        o << "      预算保留策略: " << c.retention_policy << "\n";
        o << "      等距抽样最终关键点: " << (c.equispaced_sampling_used ? "是（不应发生）" : "否")
          << "  points 实存=" << fmt_i((int64_t)c.points.size()) << "\n";
        for (const auto& cb : c.budget_classes) {
            if (cb.available == 0) continue;
            o << "        [" << fmt_i(cb.priority) << "] " << cb.name
              << ": 可用=" << fmt_i(cb.available) << " 保留=" << fmt_i(cb.kept);
            if (cb.budget_exempt) o << "（预算豁免：必须保留）";
            if (cb.omitted > 0) o << " 省略=" << fmt_i(cb.omitted);
            o << "\n";
        }
        if (!c.note.empty()) o << "      说明: " << c.note << "\n";
    }

    o << "\n-- 配置事件（必须有记录证据；不推断未记录的修改）--\n";
    shown = 0;
    for (const auto& e : r.config_events) {
        if (shown++ >= 15) { o << "  ... 其余事件省略 " << ((int64_t)r.config_events.size() - 15) << " 条\n"; break; }
        o << "  EP=" << e.episode << " " << e.field << " " << fmt_d(e.old_value) << " -> " << fmt_d(e.new_value);
        if (e.ratio_valid) o << " 倍数=" << fmt_d(e.change_ratio, 4);
        else o << " 倍数=null(" << e.ratio_null_reason << ")";
        o << "\n      证据: " << e.evidence << "\n";
        o << "      前窗 n=" << fmt_i(e.window.before_n) << " mean=" << fmt_d(e.window.before_mean)
          << " EP " << e.window.before_ep_lo << ".." << e.window.before_ep_hi
          << " 缺失=" << fmt_i(e.window.before_missing) << "\n";
        o << "      后窗 n=" << fmt_i(e.window.after_n) << " mean=" << fmt_d(e.window.after_mean)
          << " EP " << e.window.after_ep_lo << ".." << e.window.after_ep_hi
          << " 缺失=" << fmt_i(e.window.after_missing) << "\n";
        if (!e.overlapping.empty()) {
            o << "      重叠事件: ";
            for (size_t i = 0; i < e.overlapping.size(); ++i) o << (i ? ", " : "") << e.overlapping[i];
            o << "（不声称因果）\n";
        }
    }

    o << "\n-- 相关与滞后相关（r 与 |r| 分别输出）--\n";
    shown = 0;
    for (const auto& c : r.correlations) {
        if (c.lag != 0 && !c.valid) continue;
        if (shown++ >= 20) break;
        o << "  " << c.series_a << " ~ " << c.series_b << " lag=" << c.lag
          << " transform=" << c.transform << " pairs=" << fmt_i(c.pairs);
        if (c.valid) o << " r=" << fmt_d(c.r, 4) << " |r|=" << fmt_d(c.abs_r, 4) << "\n";
        else o << " null(" << c.null_reason << ")\n";
    }
    if (!r.correlations.empty())
        o << "  口径: " << r.correlations.front().lag_sign_convention << "\n"
          << "        " << r.correlations.front().alignment << "\n";

    o << "\n-- 差值 D 专项（Q − Target-Q，逐记录配对）--\n";
    o << "  定义: " << r.delta_d.definition << "\n";
    o << "  配对: " << r.delta_d.pairing << "\n";
    if (r.delta_d.valid)
        o << "  pairs=" << fmt_i(r.delta_d.pairs) << " mean=" << fmt_d(r.delta_d.mean)
          << " std=" << fmt_d(r.delta_d.stddev) << " min=" << fmt_d(r.delta_d.min)
          << " max=" << fmt_d(r.delta_d.max) << " p50=" << fmt_d(r.delta_d.p50)
          << " p90=" << fmt_d(r.delta_d.p90) << " p99=" << fmt_d(r.delta_d.p99) << "\n";
    else
        o << "  null(" << r.delta_d.null_reason << ")\n";
    o << "  禁止: " << r.delta_d.forbidden_note << "\n";

    o << "\n-- 长尾 Loss 专项 --\n";
    if (r.long_tail_loss.valid)
        o << "  n=" << fmt_i(r.long_tail_loss.n) << " median=" << fmt_d(r.long_tail_loss.median)
          << " MAD(×1.4826)=" << fmt_d(r.long_tail_loss.mad)
          << " p90=" << fmt_d(r.long_tail_loss.p90) << " p95=" << fmt_d(r.long_tail_loss.p95)
          << " p99=" << fmt_d(r.long_tail_loss.p99) << "\n"
          << "  尾部(>p95): n=" << fmt_i(r.long_tail_loss.tail_n)
          << " mean=" << fmt_d(r.long_tail_loss.tail_mean)
          << " 占总和=" << fmt_d(r.long_tail_loss.tail_share_of_sum, 4) << "\n"
          << "  长尾严重度: " << r.long_tail_loss.long_tail_ratio_definition << "\n"
          << "      Long-tail ratio = " << fmt_d(r.long_tail_loss.long_tail_ratio, 4) << "\n"
          << "      Cliff's delta = N/A (" << r.long_tail_loss.cliff_delta_null_reason << ")\n";
    else
        o << "  null(" << r.long_tail_loss.null_reason << ")\n";

    o << "\n-- 数据缺口（不把 EP 间隔直接当缺失）--\n";
    if (r.quality.gaps.empty()) o << "  未发现可确认缺口\n";
    else {
        shown = 0;
        for (const auto& g : r.quality.gaps) {
            if (shown++ >= 10) { o << "  ... 其余省略 " << ((int64_t)r.quality.gaps.size() - 10) << " 条\n"; break; }
            o << "  " << g.kind << ": EP " << g.from_episode << ".." << g.to_episode
              << " 估计缺失=" << fmt_i(g.missing_estimate) << " 原因=" << g.reason << "\n";
        }
    }

    o << "\n-- 受现有日志限制（不可用信息）--\n";
    for (const auto& u : r.unavailable) o << "  " << u.item << ": " << u.reason << "\n";

    o << "\n-- 其它限制 --\n";
    for (const auto& l : r.limitations) o << "  - " << l << "\n";

    o << "\n-- 运行开销与压缩比 --\n";
    o << "  分析耗时=" << fmt_d(r.elapsed_seconds, 3) << " s\n";
    o << "  原始输入=" << fmt_d((double)r.raw_input_bytes / 1048576.0, 2) << " MB\n";
    for (const auto& of : r.output_files)
        o << "  " << of.name << " = " << fmt_d((double)of.bytes / 1024.0, 1) << " KB"
          << " (" << fmt_d((double)of.bytes / 1048576.0, 3) << " MB)\n";
    o << "  主报告压缩比(core)  = " << fmt_d(r.core_compression_ratio, 3) << "×  (原始输入 / report.json)\n";
    o << "  结构化压缩比(total) = " << fmt_d(r.total_compression_ratio, 3) << "×  (原始输入 / 全部结构化输出)\n";
    o << "  说明: core 与 total 是两个不同口径，不要混用；total 含全部异常与关键点明细。\n";
    o << "==========================================\n";
    return o.str();
}

// JSON 主体：写到任意 ostream，既可返回字符串，也可直接流式落盘（避免构造数百 MB 字符串）。
static void JsonInto(const AnalyzeResult& r, std::ostream& o) {
    o << "{\n";
    o << "  \"schema_version\": " << jstr(r.schema_version) << ",\n";
    o << "  \"generated_at\": " << jstr(r.generated_at) << ",\n";
    o << "  \"source\": { \"dir\": " << jstr(r.source_dir)
      << ", \"timeline\": " << jstr(r.timeline_id)
      << ", \"bytes_read\": " << fmt_i(r.bytes_read) << " },\n";
    o << "  \"options\": { \"log_dir\": " << jstr(r.options.log_dir)
      << ", \"compression_mode\": " << jstr(r.options.compression_mode)
      << ", \"error_tolerance\": " << jnum(r.options.error_tolerance)
      << ", \"length_budget\": " << fmt_i(r.options.length_budget)
      << ", \"top_n\": " << fmt_i(r.options.top_n) << " },\n";

    o << "  \"timelines\": [\n";
    for (size_t i = 0; i < r.timelines.size(); ++i) {
        const auto& t = r.timelines[i];
        o << "    { \"id\": " << jstr(t.id) << ", \"kind\": " << jstr(t.kind)
          << ", \"dir\": " << jstr(t.dir) << ", \"is_active\": " << (t.is_active ? "true" : "false")
          << ", \"marker_phase\": " << fmt_i(t.marker_phase)
          << ", \"rollback_pid\": " << fmt_i((int64_t)t.rollback_pid)
          << ", \"rollback_tick\": " << fmt_i((int64_t)t.rollback_tick)
          << ", \"record_count\": " << fmt_i(t.record_count)
          << ", \"note\": " << jstr(t.note) << " }" << (i + 1 < r.timelines.size() ? "," : "") << "\n";
    }
    o << "  ],\n";

    o << "  \"preview_records\": [";
    for (size_t i = 0; i < r.preview_records.size(); ++i) {
        const auto& pv = r.preview_records[i];
        o << (i ? "," : "") << "\n    { \"record_index\": " << fmt_i(pv.record_index)
          << ", \"episode\": " << fmt_i(pv.episode)
          << ", \"replay_type\": " << fmt_i(pv.replay_type)
          << ", \"loss\": " << jnum(pv.loss)
          << ", \"q_value\": " << jnum(pv.q_value)
          << ", \"target_q\": " << jnum(pv.target_q)
          << ", \"delta_d\": " << jnum(pv.delta_d) << " }";
    }
    o << "\n  ],\n";

    o << "  \"quality\": {\n";
    o << "    \"records_seen\": " << fmt_i(r.quality.records_seen) << ",\n";
    o << "    \"records_used\": " << fmt_i(r.quality.records_used) << ",\n";
    o << "    \"records_invalid\": " << fmt_i(r.quality.records_invalid) << ",\n";
    o << "    \"records_nonfinite\": " << fmt_i(r.quality.records_nonfinite) << ",\n";
    o << "    \"records_placeholder\": " << fmt_i(r.quality.records_placeholder) << ",\n";
    o << "    \"records_duplicate\": " << fmt_i(r.quality.records_duplicate) << ",\n";
    o << "    \"same_key_multi\": " << fmt_i(r.quality.same_key_multi) << ",\n";
    o << "    \"reorder_events\": " << fmt_i(r.quality.reorder_events) << ",\n";
    o << "    \"unknown_episode\": " << fmt_i(r.quality.unknown_episode) << ",\n";
    o << "    \"excluded_total\": " << fmt_i(r.quality.excluded_total) << ",\n";
    o << "    \"truncated_tail\": " << (r.quality.truncated_tail ? "true" : "false") << ",\n";
    o << "    \"truncated_bytes\": " << fmt_i(r.quality.truncated_bytes) << ",\n";
    o << "    \"read_boundary_fixed\": " << (r.quality.read_boundary_fixed ? "true" : "false") << ",\n";
    o << "    \"boundary_size\": " << fmt_i(r.quality.boundary_size) << ",\n";
    o << "    \"boundary_note\": " << jstr(r.quality.boundary_note) << ",\n";
    o << "    \"gaps\": [";
    for (size_t i = 0; i < r.quality.gaps.size(); ++i) {
        const auto& g = r.quality.gaps[i];
        o << (i ? "," : "") << " { \"from_episode\": " << fmt_i(g.from_episode)
          << ", \"to_episode\": " << fmt_i(g.to_episode)
          << ", \"missing_estimate\": " << fmt_i(g.missing_estimate)
          << ", \"record_index_lo\": " << fmt_i(g.record_index_lo)
          << ", \"record_index_hi\": " << fmt_i(g.record_index_hi)
          << ", \"kind\": " << jstr(g.kind) << ", \"reason\": " << jstr(g.reason) << " }";
    }
    o << "]\n  },\n";

    o << "  \"series\": {\n";
    size_t si = 0;
    for (const auto& kv : r.series) {
        const Series& s = kv.second;
        o << "    " << jstr(kv.first) << ": { \"file\": " << jstr(s.file)
          << ", \"field\": " << jstr(s.field)
          << ", \"field_index\": " << fmt_i(s.field_index)
          << ", \"meaning\": " << jstr(s.meaning)
          << ", \"unit\": " << jstr(unit_name(s.unit))
          << ", \"aggregation\": " << jstr(aggregation_name(s.aggregation))
          << ", \"episode_semantics\": " << jstr(s.episode_semantics)
          << ", \"step_semantics\": " << jstr(s.step_semantics)
          << ", \"zero_based\": " << (s.zero_based ? "true" : "false")
          << ", \"per_episode\": " << (s.per_episode ? "true" : "false")
          << ", \"sampling_interval_known\": " << (s.sampling_interval_known ? "true" : "false")
          << ", \"sampling_note\": " << jstr(s.sampling_note)
          << ", \"sample_count\": " << fmt_i((int64_t)s.samples.size())
          << ", \"noise_dominated\": " << (s.noise_dominated ? "true" : "false")
          << ", \"derived_ep_series\": [";
        for (size_t j = 0; j < s.derived_ep_series.size(); ++j)
            o << (j ? "," : "") << jstr(s.derived_ep_series[j]);
        o << "] }"
          << (++si < r.series.size() ? "," : "") << "\n";
    }
    o << "  },\n";

    o << "  \"scalars\": {";
    si = 0;
    for (const auto& kv : r.scalars) {
        const auto& s = kv.second;
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": { "
          << "\"exact_global\": { \"n\": " << fmt_i(s.exact_n)
          << ", \"mean\": " << jnum(s.exact_mean)
          << ", \"stddev\": " << (s.exact_std_valid ? jnum(s.exact_stddev) : std::string("null"))
          << ", \"std_null_reason\": " << (s.exact_std_valid ? std::string("null") : jstr(s.exact_std_null_reason))
          << ", \"min\": " << jnum(s.exact_min) << ", \"max\": " << jnum(s.exact_max)
          << ", \"basis\": \"ALL records, streaming exact (unaffected by sampling)\" }, "
          << "\"sampled_point\": { \"n\": " << fmt_i(s.n)
          << ", \"mean\": " << jnum(s.mean)
          << ", \"stddev\": " << (s.std_valid ? jnum(s.stddev) : std::string("null"))
          << ", \"std_null_reason\": " << (s.std_valid ? std::string("null") : jstr(s.std_null_reason))
          << ", \"min\": " << jnum(s.min) << ", \"max\": " << jnum(s.max)
          << ", \"p50\": " << jnum(s.p50) << ", \"p90\": " << jnum(s.p90)
          << ", \"p99\": " << jnum(s.p99)
          << ", \"basis\": \"retained points only (point-level statistics; quantiles live here)\""
          << ", \"std_definition\": \"sample standard deviation (n-1)\""
          << ", \"quantile_definition\": \"linear interpolation on sorted values\" }, "
          << "\"sampling_affected\": " << (s.sampling_affected ? "true" : "false")
          << ", \"note\": \"exact_global and sampled_point describe DIFFERENT populations; do not mix\" }";
    }
    o << "\n  },\n";

    o << "  \"top_slices\": [";
    for (size_t i = 0; i < r.top_slices.size(); ++i) {
        const auto& ts = r.top_slices[i];
        o << (i ? "," : "") << "\n    { \"series\": " << jstr(ts.series_key)
          << ", \"requested_percent\": " << jnum(ts.percent)
          << ", \"threshold\": " << jnum(ts.threshold)
          << ", \"definition\": " << jstr(ts.definition)
          << ", \"basis\": " << jstr(ts.basis)
          << ", \"valid\": " << (ts.valid ? "true" : "false")
          << ", \"null_reason\": " << (ts.null_reason.empty() ? std::string("null") : jstr(ts.null_reason))
          << ", \"n\": " << fmt_i(ts.n)
          << ", \"actual_percent\": " << jnum(ts.actual_percent)
          << ", \"ep_lo\": " << fmt_i(ts.ep_lo) << ", \"ep_hi\": " << fmt_i(ts.ep_hi)
          << ", \"record_lo\": " << fmt_i(ts.record_lo) << ", \"record_hi\": " << fmt_i(ts.record_hi)
          << ", \"mean\": " << jnum(ts.mean)
          << ", \"stddev\": " << (ts.std_valid ? jnum(ts.stddev) : std::string("null"))
          << ", \"std_null_reason\": " << (ts.std_valid ? std::string("null") : jstr(ts.std_null_reason))
          << ", \"min\": " << jnum(ts.min) << ", \"max\": " << jnum(ts.max)
          << ", \"p50\": " << jnum(ts.p50) << ", \"p90\": " << jnum(ts.p90) << ", \"p99\": " << jnum(ts.p99)
          << ", \"sum_share\": " << (ts.sum_share_valid ? jnum(ts.sum_share) : std::string("null"))
          << ", \"sum_share_null_reason\": " << (ts.sum_share_valid ? std::string("null") : jstr(ts.sum_share_null_reason))
          << " }";
    }
    o << "\n  ],\n";

    o << "  \"quantiles\": {";
    si = 0;
    for (const auto& kv : r.quantiles) {
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": {";
        size_t qi = 0;
        for (const auto& q : kv.second)
            o << (qi++ ? "," : "") << " " << jstr(q.first) << ": " << jnum(q.second);
        o << " }";
    }
    o << "\n  },\n";

    o << "  \"current_positions\": [";
    for (size_t i = 0; i < r.current_positions.size(); ++i) {
        const auto& cp = r.current_positions[i];
        o << (i ? "," : "") << "\n    { \"series\": " << jstr(cp.series_key)
          << ", \"current\": " << jnum(cp.current)
          << ", \"percentile_rank\": " << (cp.valid ? jnum(cp.percentile_rank) : std::string("null"))
          << ", \"definition\": " << jstr(cp.percentile_definition)
          << ", \"valid\": " << (cp.valid ? "true" : "false")
          << ", \"null_reason\": " << (cp.null_reason.empty() ? std::string("null") : jstr(cp.null_reason)) << " }";
    }
    o << "\n  ],\n";

    o << "  \"category_stats\": [";
    for (size_t i = 0; i < r.category_stats.size(); ++i) {
        const auto& cs = r.category_stats[i];
        o << (i ? "," : "") << "\n    { \"series\": " << jstr(cs.series_key)
          << ", \"total\": " << fmt_i(cs.total)
          << ", \"transitions\": " << fmt_i(cs.transitions)
          << ", \"definition\": " << jstr(cs.definition)
          << ", \"note\": " << jstr(cs.note)
          << ", \"frequency\": {";
        size_t k = 0;
        for (auto& kv : cs.frequency) o << (k++ ? "," : "") << " " << jstr(fmt_i(kv.first)) << ": " << fmt_i(kv.second);
        o << " }, \"proportion\": {";
        k = 0;
        for (auto& kv : cs.proportion) o << (k++ ? "," : "") << " " << jstr(fmt_i(kv.first)) << ": " << jnum(kv.second);
        o << " }, \"transition_matrix\": [";
        k = 0;
        for (auto& kv : cs.transition_matrix)
            o << (k++ ? "," : "") << " { \"from\": " << kv.first.first << ", \"to\": " << kv.first.second
              << ", \"count\": " << fmt_i(kv.second) << " }";
        o << "], \"runs\": [";
        for (size_t j = 0; j < cs.runs.size(); ++j) {
            const auto& run = cs.runs[j];
            if (j >= 200) { o << (j ? "," : "") << " { \"omitted_rest\": " << fmt_i((int64_t)(cs.runs.size() - j)) << " }"; break; }
            o << (j ? "," : "") << " { \"value\": " << run.value << ", \"length\": " << fmt_i(run.length)
              << ", \"ep_lo\": " << run.ep_lo << ", \"ep_hi\": " << run.ep_hi << " }";
        }
        o << "] }";
    }
    o << "\n  ],\n";

    o << "  \"windows\": {";
    si = 0;
    for (const auto& kv : r.windows) {
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            const auto& w = kv.second[i];
            o << (i ? "," : "") << "\n      { \"id\": " << jstr(w.id)
              << ", \"by_records\": " << (w.by_records ? "true" : "false")
              << ", \"width\": " << fmt_i(w.width)
              << ", \"min_valid\": " << fmt_i(w.min_valid)
              << ", \"sample_count\": " << fmt_i(w.sample_count)
              << ", \"episode_span\": " << jnum(w.episode_span)
              << ", \"ep_lo\": " << fmt_i(w.ep_lo) << ", \"ep_hi\": " << fmt_i(w.ep_hi)
              << ", \"mean\": " << (w.valid ? jnum(w.mean) : std::string("null"))
              << ", \"stddev\": " << (w.valid ? jnum(w.stddev) : std::string("null"))
              << ", \"min\": " << (w.valid ? jnum(w.min) : std::string("null"))
              << ", \"max\": " << (w.valid ? jnum(w.max) : std::string("null"))
              << ", \"p50\": " << (w.valid ? jnum(w.p50) : std::string("null"))
              << ", \"std_definition\": " << jstr(w.std_definition)
              << ", \"valid\": " << (w.valid ? "true" : "false")
              << ", \"null_reason\": " << (w.null_reason.empty() ? std::string("null") : jstr(w.null_reason))
              << ", \"crossed_boundary\": " << (w.crossed_boundary ? "true" : "false") << " }";
        }
        o << "\n    ]";
    }
    o << "\n  },\n";

    o << "  \"diffs\": {";
    si = 0;
    for (const auto& kv : r.diffs) {
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": { \"first_order_note\": "
          << jstr(kv.second.first_order_note)
          << ", \"second_order_note\": " << jstr(kv.second.second_order_note) << ", \"points\": [";
        const auto& pts = kv.second.points;
        // 主 JSON 只放代表性差分点；完整数据由明细文件承载（避免主报告膨胀）
        size_t lim = pts.size() > 400 ? 400 : pts.size();
        for (size_t i = 0; i < lim; ++i) {
            const auto& d = pts[i];
            o << (i ? "," : "") << "\n      { \"record_index\": " << fmt_i(d.record_index)
              << ", \"episode\": " << fmt_i(d.episode)
              << ", \"raw_diff\": " << jnum(d.raw_diff)
              << ", \"rate_valid\": " << (d.rate_valid ? "true" : "false")
              << ", \"rate_per_episode\": " << (d.rate_valid ? jnum(d.rate_per_episode) : std::string("null"))
              << ", \"rate_null_reason\": " << (d.rate_null_reason.empty() ? std::string("null") : jstr(d.rate_null_reason))
              << ", \"second_diff\": " << (d.second_valid ? jnum(d.second_diff) : std::string("null"))
              << " }";
        }
        if (pts.size() > lim) o << ",\n      { \"omitted_rest\": " << fmt_i((int64_t)(pts.size() - lim)) << " }";
        o << "\n    ] }";
    }
    o << "\n  },\n";

    o << "  \"segments\": {";
    si = 0;
    for (const auto& kv : r.segments) {
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            const auto& sg = kv.second[i];
            o << (i ? "," : "") << "\n      { \"index\": " << fmt_i(sg.index)
              << ", \"record_lo\": " << fmt_i(sg.lo_record) << ", \"record_hi\": " << fmt_i(sg.hi_record)
              << ", \"ep_lo\": " << fmt_i(sg.ep_lo) << ", \"ep_hi\": " << fmt_i(sg.ep_hi)
              << ", \"sample_count\": " << fmt_i(sg.sample_count)
              << ", \"slope_per_episode\": " << jnum(sg.slope_per_episode)
              << ", \"intercept\": " << jnum(sg.intercept)
              << ", \"r2\": " << jnum(sg.r2)
              << ", \"residual_n\": " << fmt_i(sg.residual_n)
              << ", \"residual_mean\": " << (sg.residual_valid ? jnum(sg.residual_mean) : std::string("null"))
              << ", \"residual_stddev\": " << (sg.residual_valid ? jnum(sg.residual_stddev) : std::string("null"))
              << ", \"residual_min\": " << (sg.residual_valid ? jnum(sg.residual_min) : std::string("null"))
              << ", \"residual_max\": " << (sg.residual_valid ? jnum(sg.residual_max) : std::string("null"))
              << ", \"residual_null_reason\": " << (sg.residual_valid ? std::string("null") : jstr(sg.residual_null_reason))
              << ", \"fit_note\": " << jstr(sg.fit_note) << " }";
        }
        o << "\n    ]";
    }
    o << "\n  },\n";

    // 分层异常存储（Level 1）：主 JSON 只放汇总 + 区域 + Top 代表点 + 索引。
    // 全部异常（默认不限条数）在 trajectory_anomalies.jsonl；主报告不再内联数万条对象。
    auto emit_anomaly = [&](const Anomaly& a) {
        o << "{ \"rank\": " << fmt_i(a.rank)
          << ", \"series\": " << jstr(a.series_key)
          << ", \"rule\": " << jstr(a.rule)
          << ", \"trigger_mask\": " << fmt_i((int64_t)a.trigger_mask)
          << ", \"threshold\": " << jnum(a.threshold)
          << ", \"threshold_kind\": " << jstr(a.threshold_kind)
          << ", \"episode\": " << fmt_i(a.episode)
          << ", \"record_index\": " << fmt_i(a.record_index)
          << ", \"value\": " << jnum(a.value)
          << ", \"baseline\": " << jnum(a.baseline)
          << ", \"score\": " << jnum(a.score)
          << ", \"baseline_n\": " << fmt_i(a.baseline_n)
          << ", \"baseline_definition\": " << jstr(a.baseline_definition)
          << ", \"null_reason\": " << (a.null_reason.empty() ? std::string("null") : jstr(a.null_reason))
          << " }";
    };
    o << "  \"anomalies\": {\n";
    o << "    \"count\": " << fmt_i((int64_t)r.anomalies.size()) << ",\n";
    o << "    \"detected\": " << fmt_i(r.anomalies_detected)
      << ", \"omitted\": " << fmt_i(r.anomalies_omitted)
      << ", \"store_limit\": " << fmt_i(r.anomaly_store_limit)
      << ", \"thresholds\": { \"z\": " << jnum(r.options.z_threshold)
      << ", \"mad_z\": " << jnum(r.options.mad_z_threshold)
      << ", \"keep_rate\": " << (r.anomaly_keep_rate_used > 0 ? jnum(r.anomaly_keep_rate_used) : std::string("null"))
      << ", \"filtered_by_quantile\": " << fmt_i(r.anomalies_filtered_by_quantile) << " },\n";
    o << "    \"severity\": { \"anomalies\": { \"critical\": " << fmt_i(r.anomaly_count_critical)
      << ", \"major\": " << fmt_i(r.anomaly_count_major)
      << ", \"minor\": " << fmt_i(r.anomaly_count_minor) << " }"
      << ", \"regions\": { \"critical\": " << fmt_i(r.region_count_critical)
      << ", \"major\": " << fmt_i(r.region_count_major)
      << ", \"minor\": " << fmt_i(r.region_count_minor) << " } },\n";
    o << "    \"severity_rule\": \"critical: max_score>=8 or count>=500; major: max_score>=6 or count>=50; minor: otherwise\",\n";
    o << "    \"by_series\": {";
    {
        size_t k = 0;
        for (const auto& kv : r.anomaly_count_by_series)
            o << (k++ ? "," : "") << " " << jstr(kv.first) << ": " << fmt_i(kv.second);
    }
    o << " },\n";
    o << "    \"rule_legend\": " << jstr(r.anomaly_rule_legend) << ",\n";
    o << "    \"note\": \"规则触发仅表示统计规则被触发，不代表训练崩溃/恢复/成功\",\n";
    o << "    \"region_clustering\": \"时间连续(EP 间隔 <= " << fmt_i(r.options.anomaly_region_gap)
      << ") 且 区域 EP 跨度 <= " << fmt_i(r.options.anomaly_region_max_span)
      << "；家族构成作为区内明细输出，不作为切分依据\",\n";
    o << "    \"full_data\": " << jstr(r.anomaly_full_data_file)
      << ", \"full_data_contains\": \"ALL anomalies, compact keys, unbounded count\",\n";
    o << "    \"regions\": [";
    {
    const size_t REGION_JSON_LIMIT = 300;
    const size_t region_n = r.anomaly_regions.size() > REGION_JSON_LIMIT ? REGION_JSON_LIMIT : r.anomaly_regions.size();
    for (size_t i = 0; i < region_n; ++i) {
        const auto& rg = r.anomaly_regions[i];
        o << (i ? "," : "") << "\n      { \"index\": " << fmt_i(rg.index)
          << ", \"ep\": [" << fmt_i(rg.ep_lo) << ", " << fmt_i(rg.ep_hi) << "]"
          << ", \"record\": [" << fmt_i(rg.record_lo) << ", " << fmt_i(rg.record_hi) << "]"
          << ", \"n\": " << fmt_i(rg.count)
          << ", \"severity\": " << jstr(rg.severity)
          << ", \"dominant_family\": " << jstr(rg.dominant_family)
          << ", \"span_limited\": " << (rg.span_limited ? "true" : "false")
          << ", \"family_counts\": {";
        {
            size_t kf = 0;
            for (const auto& kvf : rg.family_counts)
                o << (kf++ ? "," : "") << " " << jstr(kvf.first) << ": " << fmt_i(kvf.second);
        }
        o << " }"
          << ", \"peak_ep\": " << fmt_i(rg.peak_ep)
          << ", \"peak_record\": " << fmt_i(rg.peak_record)
          << ", \"peak_series\": " << jstr(rg.peak_series)
          << ", \"peak_score\": " << jnum(rg.max_score)
          << ", \"mean_score\": " << jnum(rg.mean_score)
          << ", \"metrics\": [";
        for (size_t k = 0; k < rg.series.size(); ++k) o << (k ? "," : "") << jstr(rg.series[k]);
        o << "], \"metric_peaks\": [";
        {
            // 区域内涉及指标可能很多；JSON 只内联最大的 8 个（完整列表见 JSONL 回查）
            const size_t mp_lim = rg.metric_peaks.size() > 8 ? 8 : rg.metric_peaks.size();
            for (size_t k = 0; k < mp_lim; ++k)
                o << (k ? "," : "") << " { \"series\": " << jstr(rg.metric_peaks[k].series)
                  << ", \"max_abs_value\": " << jnum(rg.metric_peaks[k].max_abs_value)
                  << ", \"at_ep\": " << fmt_i(rg.metric_peaks[k].at_ep) << " }";
            if (rg.metric_peaks.size() > mp_lim)
                o << (mp_lim ? "," : "") << " { \"omitted_in_json\": "
                  << fmt_i((int64_t)(rg.metric_peaks.size() - mp_lim)) << " }";
        }
        o << "], \"deltas\": [";
        for (size_t k = 0; k < rg.deltas.size(); ++k) {
            const auto& wd = rg.deltas[k];
            o << (k ? "," : "") << "\n        { \"series\": " << jstr(wd.series)
              << ", \"valid\": " << (wd.valid ? "true" : "false")
              << ", \"before_mean\": " << (wd.valid ? jnum(wd.before_mean) : std::string("null"))
              << ", \"during_mean\": " << (wd.valid ? jnum(wd.during_mean) : std::string("null"))
              << ", \"after_mean\": " << (wd.valid && wd.after_n > 0 ? jnum(wd.after_mean) : std::string("null"))
              << ", \"before_n\": " << fmt_i(wd.before_n)
              << ", \"during_n\": " << fmt_i(wd.during_n)
              << ", \"after_n\": " << fmt_i(wd.after_n)
              << ", \"before_ep\": [" << fmt_i(wd.before_ep_lo) << ", " << fmt_i(wd.before_ep_hi) << "]"
              << ", \"during_ep\": [" << fmt_i(wd.during_ep_lo) << ", " << fmt_i(wd.during_ep_hi) << "]"
              << ", \"after_ep\": [" << fmt_i(wd.after_ep_lo) << ", " << fmt_i(wd.after_ep_hi) << "]"
              << ", \"null_reason\": " << (wd.null_reason.empty() ? std::string("null") : jstr(wd.null_reason))
              << " }";
        }
        o << "], \"representatives\": [";
        // 代表点用**紧凑形式**（不重复长字段串）；完整字段在 Top 列表与 JSONL 里
        for (size_t k = 0; k < rg.representatives.size(); ++k) {
            const Anomaly& a = rg.representatives[k];
            o << (k ? "," : "") << " { \"ep\": " << fmt_i(a.episode)
              << ", \"ri\": " << fmt_i(a.record_index)
              << ", \"s\": " << jstr(a.series_key)
              << ", \"v\": " << jnum(a.value)
              << ", \"z\": " << jnum(a.score)
              << ", \"m\": " << fmt_i((int64_t)a.trigger_mask) << " }";
        }
        o << "] }";
    }
    if (r.anomaly_regions.size() > region_n)
        o << (region_n ? "," : "") << "\n      { \"omitted_in_json\": "
          << fmt_i((int64_t)(r.anomaly_regions.size() - region_n))
          << ", \"regions_total\": " << fmt_i((int64_t)r.anomaly_regions.size())
          << ", \"note\": \"regions beyond the inline limit; every anomaly is recoverable from "
          << r.anomaly_full_data_file << " by episode/record_index\" }";
    }
    o << "\n    ],\n";
    o << "    \"top\": [";
    {
        const int64_t top_n = r.options.anomaly_top_in_json > 0 ? r.options.anomaly_top_in_json : 20;
        for (size_t i = 0; i < r.anomalies.size() && (int64_t)i < top_n; ++i) {
            o << (i ? "," : "") << "\n      ";
            emit_anomaly(r.anomalies[i]);
        }
        if ((int64_t)r.anomalies.size() > top_n)
            o << (top_n ? "," : "") << "\n      { \"note\": \"remaining "
              << fmt_i((int64_t)r.anomalies.size() - top_n)
              << " anomalies are in " << r.anomaly_full_data_file << "\" }";
    }
    o << "\n    ]\n  },\n";

    // 注：原先的 anomaly_intervals 数组已由 anomalies.regions 取代（同一信息且更丰富），
    // 不再重复内联，避免主报告膨胀二十余 MB。

    o << "  \"extrema\": {";
    si = 0;
    for (const auto& kv : r.extrema) {
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": [";
        const size_t ex_lim = kv.second.size() > 200 ? 200 : kv.second.size();
        for (size_t i = 0; i < ex_lim; ++i) {
            const auto& e = kv.second[i];
            o << (i ? "," : "") << " { \"kind\": " << jstr(e.kind) << ", \"episode\": " << fmt_i(e.episode)
              << ", \"record_index\": " << fmt_i(e.record_index) << ", \"value\": " << jnum(e.value) << " }";
        }
        if (kv.second.size() > ex_lim)
            o << (ex_lim ? "," : "") << " { \"omitted_in_json\": "
              << fmt_i((int64_t)(kv.second.size() - ex_lim)) << " }";
        o << " ]";
    }
    o << "\n  },\n";

    o << "  \"compression\": {";
    si = 0;
    for (const auto& kv : r.compression) {
        const auto& c = kv.second;
        o << (si++ ? "," : "") << "\n    " << jstr(kv.first) << ": { \"mode\": " << jstr(c.mode)
          << ", \"method\": " << jstr(c.method)
          << ", \"error_unit\": " << jstr(c.error_unit)
          << ", \"tolerance\": " << jnum(c.tolerance)
          << ", \"tolerance_is_hard\": " << (c.tolerance_is_hard ? "true" : "false")
          << ", \"original_points\": " << fmt_i(c.original_points)
          << ", \"kept_points\": " << fmt_i(c.kept_points)
          << ", \"max_error\": " << jnum(c.max_error)
          << ", \"rmse\": " << jnum(c.rmse)
          << ", \"target_met\": " << (c.target_met ? "true" : "false")
          << ", \"retention_policy\": " << jstr(c.retention_policy)
          << ", \"equispaced_sampling_used\": " << (c.equispaced_sampling_used ? "true" : "false")
          << ", \"budget_classes\": [";
        for (size_t i = 0; i < c.budget_classes.size(); ++i) {
            const auto& cb = c.budget_classes[i];
            o << (i ? "," : "") << " { \"class\": " << jstr(cb.name)
              << ", \"priority\": " << fmt_i(cb.priority)
              << ", \"available\": " << fmt_i(cb.available)
              << ", \"kept\": " << fmt_i(cb.kept)
              << ", \"omitted\": " << fmt_i(cb.omitted)
              << ", \"budget_exempt\": " << (cb.budget_exempt ? "true" : "false") << " }";
        }
        o << "], \"note\": " << jstr(c.note)
          << ", \"points\": [";
        // JSON 里的关键点也设上限（完整点集在 CSV/明细文件里），超出显式注明省略数
        {
            const size_t JSON_POINTS_LIMIT = 800;
            const size_t n_pts = c.points.size();
            const size_t lim = n_pts > JSON_POINTS_LIMIT ? JSON_POINTS_LIMIT : n_pts;
            for (size_t i = 0; i < lim; ++i) {
                const auto& kp = c.points[i];
                o << (i ? "," : "") << " { \"record_index\": " << fmt_i(kp.record_index)
                  << ", \"episode\": " << fmt_i(kp.episode) << ", \"value\": " << jnum(kp.value)
                  << ", \"must_keep\": " << (kp.must_keep ? "true" : "false")
                  << ", \"reason\": " << jstr(kp.reason) << " }";
            }
            if (n_pts > lim)
                o << (lim ? "," : "") << " { \"omitted_in_json\": " << fmt_i((int64_t)(n_pts - lim))
                  << ", \"points_total\": " << fmt_i((int64_t)n_pts)
                  << ", \"note\": \"full point set is in trajectory_keypoints.csv\" }";
        }
        o << "] }";
    }
    o << "\n  },\n";

    o << "  \"correlations\": [";
    for (size_t i = 0; i < r.correlations.size(); ++i) {
        const auto& c = r.correlations[i];
        o << (i ? "," : "") << "\n    { \"a\": " << jstr(c.series_a) << ", \"b\": " << jstr(c.series_b)
          << ", \"lag\": " << fmt_i(c.lag)
          << ", \"lag_sign_convention\": " << jstr(c.lag_sign_convention)
          << ", \"alignment\": " << jstr(c.alignment)
          << ", \"transform\": " << jstr(c.transform)
          << ", \"pairs\": " << fmt_i(c.pairs)
          << ", \"ep_lo\": " << fmt_i(c.ep_lo) << ", \"ep_hi\": " << fmt_i(c.ep_hi)
          << ", \"r\": " << (c.valid ? jnum(c.r) : std::string("null"))
          << ", \"abs_r\": " << (c.valid ? jnum(c.abs_r) : std::string("null"))
          << ", \"valid\": " << (c.valid ? "true" : "false")
          << ", \"null_reason\": " << (c.null_reason.empty() ? std::string("null") : jstr(c.null_reason)) << " }";
    }
    o << "\n  ],\n";

    o << "  \"config_events\": [";
    for (size_t i = 0; i < r.config_events.size(); ++i) {
        const auto& e = r.config_events[i];
        o << (i ? "," : "") << "\n    { \"field\": " << jstr(e.field)
          << ", \"kind\": " << jstr(e.kind)
          << ", \"episode\": " << fmt_i(e.episode)
          << ", \"record_index\": " << fmt_i(e.record_index)
          << ", \"old_value\": " << jnum(e.old_value)
          << ", \"new_value\": " << jnum(e.new_value)
          << ", \"evidence\": " << jstr(e.evidence)
          << ", \"evidence_is_log\": " << (e.evidence_is_log ? "true" : "false")
          << ", \"change_ratio\": " << (e.ratio_valid ? jnum(e.change_ratio) : std::string("null"))
          << ", \"ratio_null_reason\": " << (e.ratio_null_reason.empty() ? std::string("null") : jstr(e.ratio_null_reason))
          << ", \"window_before\": { \"n\": " << fmt_i(e.window.before_n)
          << ", \"mean\": " << (e.window.before_valid ? jnum(e.window.before_mean) : std::string("null"))
          << ", \"ep_lo\": " << fmt_i(e.window.before_ep_lo) << ", \"ep_hi\": " << fmt_i(e.window.before_ep_hi)
          << ", \"missing\": " << fmt_i(e.window.before_missing) << " }"
          << ", \"window_after\": { \"n\": " << fmt_i(e.window.after_n)
          << ", \"mean\": " << (e.window.after_valid ? jnum(e.window.after_mean) : std::string("null"))
          << ", \"ep_lo\": " << fmt_i(e.window.after_ep_lo) << ", \"ep_hi\": " << fmt_i(e.window.after_ep_hi)
          << ", \"missing\": " << fmt_i(e.window.after_missing) << " }"
          << ", \"overlapping\": [";
        for (size_t j = 0; j < e.overlapping.size(); ++j)
            o << (j ? "," : "") << jstr(e.overlapping[j]);
        o << "], \"note\": " << jstr(e.note) << " }";
    }
    o << "\n  ],\n";

    o << "  \"delta_d\": { \"definition\": " << jstr(r.delta_d.definition)
      << ", \"pairing\": " << jstr(r.delta_d.pairing)
      << ", \"is_batch_mean_td_error\": " << (r.delta_d.is_batch_mean_td_error ? "true" : "false")
      << ", \"confirmation_evidence\": " << jstr(r.delta_d.confirmation_evidence)
      << ", \"forbidden_note\": " << jstr(r.delta_d.forbidden_note)
      << ", \"pairs\": " << fmt_i(r.delta_d.pairs)
      << ", \"raw_records\": " << fmt_i(r.delta_d.raw_records)
      << ", \"valid\": " << (r.delta_d.valid ? "true" : "false")
      << ", \"null_reason\": " << (r.delta_d.null_reason.empty() ? std::string("null") : jstr(r.delta_d.null_reason))
      << ", \"mean\": " << (r.delta_d.valid ? jnum(r.delta_d.mean) : std::string("null"))
      << ", \"stddev\": " << (r.delta_d.valid ? jnum(r.delta_d.stddev) : std::string("null"))
      << ", \"p50\": " << (r.delta_d.valid ? jnum(r.delta_d.p50) : std::string("null"))
      << ", \"p90\": " << (r.delta_d.valid ? jnum(r.delta_d.p90) : std::string("null"))
      << ", \"p99\": " << (r.delta_d.valid ? jnum(r.delta_d.p99) : std::string("null")) << " },\n";

    o << "  \"long_tail_loss\": { \"valid\": " << (r.long_tail_loss.valid ? "true" : "false")
      << ", \"null_reason\": " << (r.long_tail_loss.null_reason.empty() ? std::string("null") : jstr(r.long_tail_loss.null_reason))
      << ", \"n\": " << fmt_i(r.long_tail_loss.n)
      << ", \"median\": " << jnum(r.long_tail_loss.median)
      << ", \"mad_scaled\": " << jnum(r.long_tail_loss.mad)
      << ", \"p90\": " << jnum(r.long_tail_loss.p90)
      << ", \"p95\": " << jnum(r.long_tail_loss.p95)
      << ", \"p99\": " << jnum(r.long_tail_loss.p99)
      << ", \"tail_n\": " << fmt_i(r.long_tail_loss.tail_n)
      << ", \"tail_mean\": " << jnum(r.long_tail_loss.tail_mean)
      << ", \"tail_share_of_sum\": " << jnum(r.long_tail_loss.tail_share_of_sum)
      << ", \"long_tail_ratio\": " << jnum(r.long_tail_loss.long_tail_ratio)
      << ", \"long_tail_ratio_definition\": " << jstr(r.long_tail_loss.long_tail_ratio_definition)
      << ", \"cliff_delta\": null"
      << ", \"cliff_delta_available\": false"
      << ", \"cliff_delta_null_reason\": " << jstr(r.long_tail_loss.cliff_delta_null_reason) << " },\n";

    o << "  \"unavailable\": [";
    for (size_t i = 0; i < r.unavailable.size(); ++i)
        o << (i ? "," : "") << "\n    { \"item\": " << jstr(r.unavailable[i].item)
          << ", \"reason\": " << jstr(r.unavailable[i].reason) << " }";
    o << "\n  ],\n";

    o << "  \"limitations\": [";
    for (size_t i = 0; i < r.limitations.size(); ++i)
        o << (i ? "," : "") << "\n    " << jstr(r.limitations[i]);
    o << "\n  ],\n";

    o << "  \"runtime\": { \"elapsed_seconds\": " << jnum(r.elapsed_seconds)
      << ", \"peak_memory_kb\": " << fmt_i(r.peak_memory_kb)
      << ", \"output_bytes_text\": " << fmt_i(r.output_bytes_text)
      << ", \"output_bytes_json\": " << fmt_i(r.output_bytes_json) << " },\n";

    o << "  \"stages\": [";
    for (size_t i = 0; i < r.stage_log.size(); ++i)
        o << (i ? "," : "") << "\n    { \"stage\": " << jstr(r.stage_log[i].stage)
          << ", \"detail\": " << jstr(r.stage_log[i].detail) << " }";
    o << "\n  ]\n}\n";
}

std::string ToJson(const AnalyzeResult& r) {
    std::ostringstream o;
    JsonInto(r, o);
    return o.str();
}

std::vector<std::string> WriteDetailFiles(AnalyzeResult& r) {
    std::vector<std::string> written;
    if (r.options.detail_dir.empty()) return written;
    std::string dir = r.options.detail_dir;
    if (!dir.empty() && dir.back() != '\\' && dir.back() != '/') dir += '\\';
    CreateDirectoryA(dir.c_str(), nullptr);

    // CSV：异常明细（流式写盘）
    {
        std::string p = dir + "trajectory_anomalies.csv";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fprintf(f, "rank,series,rule,threshold,threshold_kind,episode,record_index,value,baseline,score,baseline_n,null_reason\n");
            for (const auto& a : r.anomalies) {
                fprintf(f, "%lld,%s,%s,%.9g,%s,%d,%lld,%.9g,%.9g,",
                    (long long)a.rank, a.series_key.c_str(), a.rule.c_str(),
                    a.threshold, a.threshold_kind.c_str(), a.episode,
                    (long long)a.record_index, a.value, a.baseline);
                if (finite_d(a.score)) fprintf(f, "%.9g", a.score);
                fprintf(f, ",%lld,\"%s\"\n", (long long)a.baseline_n, a.null_reason.c_str());
            }
            fprintf(f, "# detected=%lld stored=%lld omitted=%lld store_limit=%lld\n",
                (long long)r.anomalies_detected, (long long)r.anomalies.size(),
                (long long)r.anomalies_omitted, (long long)r.anomaly_store_limit);
            fclose(f);
            written.push_back(p);
        }
    }
    // CSV：压缩关键点（逐行流式写盘；每个序列有写入上限并显式注明省略量）
    {
        const int64_t KEYPOINT_WRITE_LIMIT = 20000;
        std::string p = dir + "trajectory_keypoints.csv";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fprintf(f, "series,mode,kept_index,record_index,episode,value,must_keep,reason,original_points,max_error,rmse,target_met\n");
            for (const auto& kv : r.compression) {
                const auto& comp = kv.second;
                int64_t k = 0;
                for (const auto& kp : comp.points) {
                    if (k >= KEYPOINT_WRITE_LIMIT) {
                        fprintf(f, "# %s: omitted %lld of %lld key points (write limit %lld); "
                                   "full count in trajectory_report.json\n",
                            kv.first.c_str(), (long long)(comp.points.size() - KEYPOINT_WRITE_LIMIT),
                            (long long)comp.points.size(), (long long)KEYPOINT_WRITE_LIMIT);
                        break;
                    }
                    fprintf(f, "%s,%s,%lld,%lld,%d,%.9g,%d,%s,%lld,%.9g,%.9g,%d\n",
                        kv.first.c_str(), comp.mode.c_str(), (long long)k++,
                        (long long)kp.record_index, kp.episode, kp.value,
                        kp.must_keep ? 1 : 0, kp.reason.c_str(),
                        (long long)comp.original_points, comp.max_error, comp.rmse,
                        comp.target_met ? 1 : 0);
                }
            }
            fclose(f);
            written.push_back(p);
        }
    }
    // Level 2：全部异常，紧凑 JSONL（短键 + 触发位图），条数不设上限。
    // 主 JSON 只放汇总/区域/Top；需要查细节时按 episode / record_index 从这里无损回查。
    {
        std::string p = dir + "trajectory_anomalies.jsonl";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fprintf(f, "# trajectory_anomalies.jsonl | schema=1\n");
            fprintf(f, "# keys: ri=record_index ep=episode s=series m=trigger_mask v=value b=baseline z=score t=threshold n=baseline_n\n");
            fprintf(f, "# %s\n", r.anomaly_rule_legend.c_str());
            for (const auto& a : r.anomalies) {
                fprintf(f, "{\"ri\":%lld,\"ep\":%d,\"s\":\"%s\",\"m\":%u,\"v\":%.9g,\"b\":%.9g,",
                    (long long)a.record_index, a.episode, a.series_key.c_str(), a.trigger_mask,
                    a.value, a.baseline);
                if (finite_d(a.score)) fprintf(f, "\"z\":%.9g,", a.score);
                else fprintf(f, "\"z\":null,");
                fprintf(f, "\"t\":%.9g,\"n\":%lld}\n", a.threshold, (long long)a.baseline_n);
            }
            fclose(f);
            written.push_back(p);
        }
    }
    // Level 2：完整关键点 JSONL（不设条数上限；CSV 另带每序列写入上限）
    {
        std::string p = dir + "trajectory_keypoints.jsonl";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fprintf(f, "# keys: s=series ri=record_index ep=episode v=value k=class/reason\n");
            for (const auto& kv : r.compression) {
                for (const auto& kp : kv.second.points)
                    fprintf(f, "{\"s\":\"%s\",\"ri\":%lld,\"ep\":%d,\"v\":%.9g,\"k\":\"%s\"}\n",
                        kv.first.c_str(), (long long)kp.record_index, kp.episode, kp.value,
                        kp.reason.c_str());
            }
            fclose(f);
            written.push_back(p);
        }
    }
    // CSV：Top X% 暴露切片（与全量精确统计并列）
    {
        std::string p = dir + "trajectory_top_slices.csv";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fprintf(f, "series,requested_percent,threshold,actual_percent,n,ep_lo,ep_hi,record_lo,record_hi,"
                       "slice_mean,slice_stddev,slice_min,slice_max,slice_p50,slice_p90,slice_p99,sum_share,"
                       "exact_n,exact_mean,exact_stddev,exact_min,exact_max\n");
            for (const auto& ts : r.top_slices) {
                auto sc = r.scalars.find(ts.series_key);
                fprintf(f, "%s,%.6g,%.9g,%.6g,%lld,%d,%d,%lld,%lld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,",
                    ts.series_key.c_str(), ts.percent, ts.threshold, ts.actual_percent,
                    (long long)ts.n, ts.ep_lo, ts.ep_hi,
                    (long long)ts.record_lo, (long long)ts.record_hi,
                    ts.mean, ts.stddev, ts.min, ts.max, ts.p50, ts.p90, ts.p99);
                if (ts.sum_share_valid) fprintf(f, "%.9g,", ts.sum_share);
                else fprintf(f, ",");
                if (sc != r.scalars.end())
                    fprintf(f, "%lld,%.9g,%.9g,%.9g,%.9g\n", (long long)sc->second.exact_n,
                        sc->second.exact_mean,
                        sc->second.exact_std_valid ? sc->second.exact_stddev : 0.0,
                        sc->second.exact_min, sc->second.exact_max);
                else fprintf(f, ",,,,\n");
            }
            fclose(f);
            written.push_back(p);
        }
    }
    // JSONL：逐记录明细（含配对的 D）。流式写出，不在内存里拼整份字符串。
    // 默认关闭：内容与 metric_loss.bin 重复且体积约为其 6 倍；需要时用 records 参数开启。
    if (r.options.write_record_detail_jsonl) {
        std::string p = dir + "trajectory_metric_loss.jsonl";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            auto itq = r.series.find("q_value");
            auto itt = r.series.find("target_q");
            auto itl = r.series.find("loss");
            if (itl != r.series.end() && itq != r.series.end() && itt != r.series.end()) {
                const Series& ls = itl->second;
                const Series& q = itq->second;
                const Series& t = itt->second;
                // 以 record_index 对齐三条序列（同一记录的三列），不做下标假设
                std::map<int64_t, const Sample*> qi, ti;
                for (const auto& x : q.samples) qi[x.record_index] = &x;
                for (const auto& x : t.samples) ti[x.record_index] = &x;
                int64_t written_n = 0;
                for (const auto& x : ls.samples) {
                    auto a = qi.find(x.record_index);
                    auto b = ti.find(x.record_index);
                    if (a == qi.end() || b == ti.end()) continue;
                    fprintf(f,
                        "{\"timeline\":\"%s\",\"series\":\"metric_loss.bin\",\"record_index\":%lld,"
                        "\"episode\":%d,\"step\":%d,\"replay_type\":%d,\"loss\":%.9g,\"q_value\":%.9g,"
                        "\"target_q\":%.9g,\"delta_d\":%.9g}\n",
                        r.timeline_id.c_str(), (long long)x.record_index, x.episode, x.step,
                        a->second->step, x.value, a->second->value, b->second->value,
                        a->second->value - b->second->value);
                    ++written_n;
                }
                if (written_n < (int64_t)ls.samples.size())
                    fprintf(f, "{\"note\":\"omitted %lld records without a paired q/target by record_index\"}\n",
                        (long long)((int64_t)ls.samples.size() - written_n));
            }
            fclose(f);
            written.push_back(p);
        }
    }
    // 完整 JSON（流式落盘，不在内存里构造整份字符串）
    {
        std::string p = dir + "trajectory_report.json";
        std::ofstream ofs(p, std::ios::binary);
        if (ofs) {
            JsonInto(r, ofs);
            ofs.close();
            written.push_back(p);
        }
    }
    // 回填真实输出体积与压缩比（此前 runtime 里是占位 0）。
    // 在写 .txt 之前回填，这样紧凑报告本身也能带上体积表。
    {
        r.output_files.clear();
        int64_t total_struct = 0;
        for (const auto& f : written) {
            FILE* fp = fopen(f.c_str(), "rb");
            int64_t sz = 0;
            if (fp) { _fseeki64(fp, 0, SEEK_END); sz = _ftelli64(fp); fclose(fp); }
            AnalyzeResult::OutputFile of;
            const size_t slash = f.find_last_of("\\/");
            of.name = (slash == std::string::npos) ? f : f.substr(slash + 1);
            of.bytes = sz;
            r.output_files.push_back(of);
            total_struct += sz;
        }
        r.raw_input_bytes = r.bytes_read;
        for (const auto& of : r.output_files) {
            if (of.name == "trajectory_report.json") r.output_bytes_json = of.bytes;
        }
        if (r.raw_input_bytes > 0) {
            r.core_compression_ratio = (double)r.raw_input_bytes /
                (double)std::max<int64_t>(1, r.output_bytes_json);
            r.total_compression_ratio = (double)r.raw_input_bytes /
                (double)std::max<int64_t>(1, total_struct);
        }
    }
    // 紧凑文本
    {
        std::string tx = ToCompactText(r);
        std::string p = dir + "trajectory_report.txt";
        FILE* f = fopen(p.c_str(), "wb");
        if (f) {
            fwrite(tx.data(), 1, tx.size(), f);
            fclose(f);
            written.push_back(p);
            r.output_bytes_text = (int64_t)tx.size();
            AnalyzeResult::OutputFile of;
            of.name = "trajectory_report.txt";
            of.bytes = (int64_t)tx.size();
            r.output_files.push_back(of);
        }
    }
    return written;
}

} // namespace traj
