// trajectory_analyzer_regression.cpp - 训练轨迹分析器回归测试
//
// 覆盖要求中列出的边界场景：常数、线性、阶跃、尖峰、方差突变、已知滞后双序列、
// 不均匀采样、同局多记录、缺口/重复/乱序/回档分支、NaN/Inf、空数据、小样本、零 MAD、
// 多评估模式、类别指标、压缩误差与预算冲突、文件尾部只写了一半。
//
// 断言对象：异常位置、分段/窗口范围、单位、差分、相关对齐、重建误差、JSON 合法性。
#define _CRT_SECURE_NO_WARNINGS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "../DQN_Core/include/trajectory_analyzer.h"
#include "../DQN_Core/include/metric_log_format.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static const char* g_case = "";

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; printf("  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)
#define CHECK_MSG(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; printf("  [FAIL] %s:%d  %s  (%s)\n", __FILE__, __LINE__, #cond, msg); } \
} while (0)
#define NEAR(a, b, tol) (fabs((a) - (b)) <= (tol))

struct Rec { int32_t ep; int32_t step; std::vector<float> fields; };

static std::string dir_of(const char* name) {
    std::string d = "diag\\traj_test\\";
    CreateDirectoryA("diag", nullptr);
    CreateDirectoryA("diag\\traj_test", nullptr);
    CreateDirectoryA(("diag\\traj_test\\" + std::string(name)).c_str(), nullptr);
    return "diag\\traj_test\\" + std::string(name) + "\\";
}

static bool write_metric(const std::string& dir, const char* name, int fields,
    const std::vector<Rec>& recs, bool truncate_tail = false) {
    std::string path = dir + name;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    MetricLogHeader h = metric_log_header((uint32_t)fields);
    fwrite(&h, sizeof(h), 1, f);
    for (const auto& r : recs) {
        int32_t ep = r.ep, st = r.step;
        fwrite(&ep, 4, 1, f);
        fwrite(&st, 4, 1, f);
        for (int i = 0; i < fields; ++i) {
            // 缺省字段补 0：避免越界读到垃圾值（那会被分析器正确地判为非法记录）
            float v = (i < (int)r.fields.size()) ? r.fields[i] : 0.0f;
            fwrite(&v, 4, 1, f);
        }
    }
    if (truncate_tail) { unsigned char junk[3] = { 1, 2, 3 }; fwrite(junk, 1, 3, f); }
    fclose(f);
    return true;
}

/* ---------- 极简 JSON 语法校验器（结构 + 禁止 NaN/Inf 字面量） ---------- */
// 只把**裸 token**（字符串字面量之外、且两侧不是标识符字符）的 nan/inf 判为非法。
// 之前用朴素 substring 检查，会把 "domiNANt_family" 这类字段名误判。
static bool json_valid(const std::string& s, std::string* why) {
    const char* bad[] = { "nan", "inf", "NaN", "Inf" };
    for (const char* t : bad) {
        size_t p = 0;
        while ((p = s.find(t, p)) != std::string::npos) {
            // 是否在字符串字面量内？统计 p 之前的未转义引号数量。
            size_t quotes = 0;
            for (size_t k = 0; k < p; ++k) {
                if (s[k] == '"' && (k == 0 || s[k - 1] != '\\')) ++quotes;
            }
            const bool in_str = (quotes % 2) != 0;
            auto ident = [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_';
            };
            const bool left_ok = (p == 0) || !ident(s[p - 1]);
            const size_t after = p + strlen(t);
            const bool right_ok = (after >= s.size()) || !ident(s[after]);
            if (!in_str && left_ok && right_ok) {
                size_t a = p > 60 ? p - 60 : 0;
                *why = std::string("bare '") + t + "' near: ..." + s.substr(a, 130) + "...";
                return false;
            }
            p += strlen(t);
        }
    }
    size_t i = 0;
    struct P { int type; char close; };
    std::vector<char> stack;
    bool in_str = false, esc = false;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{' || c == '[') stack.push_back(c);
        else if (c == '}' || c == ']') {
            if (stack.empty()) { *why = "unbalanced close"; return false; }
            char open = stack.back(); stack.pop_back();
            if ((open == '{' && c != '}') || (open == '[' && c != ']')) { *why = "mismatched brackets"; return false; }
        }
    }
    if (in_str) { *why = "unterminated string"; return false; }
    if (!stack.empty()) { *why = "unclosed brackets"; return false; }
    return true;
}

/* ============================== 各场景 ============================== */

static void case_constant_zero_mad() {
    g_case = "constant / zero MAD";
    std::string d = dir_of("constant");
    std::vector<Rec> eps;
    for (int i = 1; i <= 60; ++i) eps.push_back({ i, 0, { 0.5f } });
    write_metric(d, "metric_epsilon_episode.bin", 1, eps);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false; o.detail_dir = d;
    traj::AnalyzeResult r = traj::Analyze(o);
    CHECK(r.error.empty());
    auto it = r.scalars.find("epsilon_episode");
    CHECK(it != r.scalars.end());
    if (it != r.scalars.end()) CHECK(it->second.stddev == 0.0);
    // 常数序列不得输出伪造的 z 分数
    bool saw_null = false;
    for (const auto& a : r.anomalies)
        if (!a.null_reason.empty() && a.null_reason.find("std == 0") != std::string::npos) saw_null = true;
    CHECK_MSG(saw_null, "constant series must report z-score null with a reason");
    std::string why;
    CHECK_MSG(json_valid(traj::ToJson(r), &why), why.c_str());
}

static void case_linear_and_segments() {
    g_case = "linear / segments / slope unit";
    std::string d = dir_of("linear");
    std::vector<Rec> loss;
    for (int i = 1; i <= 200; ++i) loss.push_back({ i, 0, { (float)(i * 0.01), 0.0f, 0.0f } });
    write_metric(d, "metric_loss.bin", 3, loss);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    auto seg = r.segments.find("loss");
    CHECK(seg != r.segments.end());
    if (seg != r.segments.end() && !seg->second.empty()) {
        // slope 单位 = 值/局；本数据每局 0.01
        CHECK_MSG(NEAR(seg->second[0].slope_per_episode, 0.01, 1e-4), "slope unit must be value-per-episode");
        CHECK_MSG(seg->second[0].r2 > 0.99, "linear data must fit with R2 near 1");
        // 残差统计（与旧报告 print_regression 对齐）
        CHECK_MSG(seg->second[0].residual_valid, "residual stats must be available for a valid fit");
        if (seg->second[0].residual_valid)
            CHECK_MSG(NEAR(seg->second[0].residual_mean, 0.0, 1e-6),
                "least-squares residual mean must be ~0");
    }
    // 全量精确统计 vs 保留子集：无抽样时必须完全一致
    {
        auto sc = r.scalars.find("loss");
        CHECK(sc != r.scalars.end());
        if (sc != r.scalars.end()) {
            CHECK_MSG(sc->second.exact_n == sc->second.n,
                "without sampling, exact record count must equal retained count");
            CHECK_MSG(NEAR(sc->second.exact_mean, sc->second.mean, 1e-9),
                "without sampling, exact mean must equal retained mean");
            CHECK_MSG(NEAR(sc->second.exact_stddev, sc->second.stddev, 1e-9),
                "without sampling, exact std must equal retained std");
            CHECK_MSG(NEAR(sc->second.exact_min, sc->second.min, 1e-9) &&
                NEAR(sc->second.exact_max, sc->second.max, 1e-9), "exact min/max must match");
            CHECK_MSG(!sc->second.sampling_affected, "no sampling must not be flagged as sampling-affected");
        }
    }
    // Top X%（按数值）暴露切片：阈值 + 子集统计 + 与全量并列
    {
        CHECK_MSG(!r.top_slices.empty(), "top percentile slice must be produced");
        bool found = false;
        for (const auto& ts : r.top_slices) {
            if (ts.series_key != "loss") continue;
            found = true;
            CHECK(ts.valid);
            CHECK_MSG(ts.n > 0 && ts.n < r.scalars.at("loss").n,
                "top slice must be a strict, non-empty subset");
            CHECK_MSG(ts.actual_percent >= r.options.top_percentile - 1e-6,
                "actual percent (with ties) must be >= requested percent");
            CHECK_MSG(ts.mean >= r.scalars.at("loss").mean,
                "top slice mean must be >= full mean");
            CHECK_MSG(NEAR(ts.max, r.scalars.at("loss").max, 1e-9),
                "top slice max must equal the series max");
            CHECK(ts.sum_share_valid);
            CHECK_MSG(ts.std_valid, "slice std must be defined for a >=2 sample slice");
            CHECK_MSG(!ts.basis.empty(), "slice must declare its basis");
        }
        CHECK_MSG(found, "loss series must have a top slice");
    }
    auto it = r.diffs.find("loss");
    CHECK(it != r.diffs.end());
    if (it != r.diffs.end() && !it->second.points.empty()) {
        CHECK_MSG(NEAR(it->second.points[0].raw_diff, 0.01, 1e-5), "raw diff");
        CHECK_MSG(it->second.points[0].rate_valid, "rate must be valid when EP distance is 1");
        CHECK_MSG(NEAR(it->second.points[0].rate_per_episode, 0.01, 1e-5), "normalized rate per episode");
    }
}

static void case_step_and_variance_change() {
    g_case = "step / variance change";
    std::string d = dir_of("step");
    std::vector<Rec> g;
    for (int i = 1; i <= 100; ++i) {
        float v = (i <= 50) ? 1.0f : 5.0f;   // 阶跃
        g.push_back({ i, 0, { v } });
    }
    write_metric(d, "metric_game.bin", 1, g);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    auto ex = r.extrema.find("episode_length");
    CHECK(ex != r.extrema.end());
    if (ex != r.extrema.end()) {
        bool has_max = false, has_min = false;
        for (const auto& e : ex->second) { if (e.kind == "global_max") has_max = true; if (e.kind == "global_min") has_min = true; }
        CHECK(has_max && has_min);
    }
    // 阶跃必须在差分上体现：第 50→51 条的差分 = +4
    auto it = r.diffs.find("episode_length");
    if (it != r.diffs.end()) {
        double best = 0; int64_t at = -1;
        for (const auto& p : it->second.points)
            if (fabs(p.raw_diff) > fabs(best)) { best = p.raw_diff; at = p.record_index; }
        CHECK_MSG(NEAR(best, 4.0, 1e-5), "step magnitude 5-1=4");
        CHECK_MSG(at == 50, "step must be located at record_index 50");
    }
    // 方差突变：窗口标准差前后不同
    auto w = r.windows.find("episode_length");
    CHECK(w != r.windows.end());
}

static void case_spike_anomaly_position() {
    g_case = "spike anomaly position";
    std::string d = dir_of("spike");
    std::vector<Rec> eps;
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 16) & 0x7fff) / 32767.0; };
    for (int i = 1; i <= 200; ++i) eps.push_back({ i, 0, { (float)(0.5 + 0.02 * (rnd() - 0.5)) } });
    eps[150].fields[0] = 0.99f;   // 尖峰位于 record_index 150 (ep=151)
    write_metric(d, "metric_epsilon_episode.bin", 1, eps);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    bool found = false;
    for (const auto& a : r.anomalies) {
        if (a.series_key != "epsilon_episode") continue;
        if (a.record_index == 150) { found = true; CHECK(a.episode == 151); CHECK(a.score > 4.0); }
    }
    CHECK_MSG(found, "spike at record_index 150 must be reported");
    // 基线不得包含待检测点本身
    for (const auto& a : r.anomalies)
        if (a.record_index == 150)
            CHECK_MSG(a.baseline_definition.find("不含当前点") != std::string::npos, "baseline must exclude the point");
}

static void case_known_lag_pair() {
    g_case = "known-lag pair correlation";
    std::string d = dir_of("lag");
    std::vector<Rec> eps, len;
    for (int i = 1; i <= 120; ++i) {
        float e = (float)(0.1 + 0.8 * (i - 1) / 119.0);
        eps.push_back({ i, 0, { e } });
        // 局长 = 上一局的 epsilon × 100 → corr(eps(EP), len(EP+1)) 应≈1
        float prev = (i == 1) ? 0.1f : (float)(0.1 + 0.8 * (i - 2) / 119.0);
        len.push_back({ i, 0, { prev * 100.0f } });
    }
    write_metric(d, "metric_epsilon_episode.bin", 1, eps);
    write_metric(d, "metric_game.bin", 1, len);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    bool checked = false;
    for (const auto& c : r.correlations) {
        if (c.series_a != "epsilon_episode" || c.series_b != "episode_length") continue;
        if (c.transform != "raw") continue;
        if (c.lag == 1) {
            checked = true;
            // len(EP+1) == eps(EP) 由构造保证，故 lag=+1 时有 r≈1
            CHECK_MSG(c.valid, "lagged correlation should be valid");
            CHECK_MSG(NEAR(c.r, 1.0, 0.02), "corr(eps(EP), len(EP+1)) must be ~1 for the known-lag pair");
        }
        if (c.valid) CHECK_MSG(c.abs_r >= fabs(c.r) - 1e-12, "abs_r must be |r|");
        if (c.valid && c.r < 0) CHECK_MSG(NEAR(c.abs_r, -c.r, 1e-9), "negative r must not be written as |r|");
    }
    CHECK(checked);
    // 相关必须声明对齐与正负号口径
    CHECK(!r.correlations.empty());
    if (!r.correlations.empty()) {
        CHECK(r.correlations.front().lag_sign_convention.find("corr(a(EP), b(EP+lag))") != std::string::npos);
        CHECK(r.correlations.front().alignment.find("不做前向填充") != std::string::npos);
    }
}

static void case_uneven_and_same_episode() {
    g_case = "uneven sampling / multiple records per episode";
    std::string d = dir_of("uneven");
    std::vector<Rec> loss;
    int64_t idx = 0;
    // 每局 3 次更新（同一 ep 多条记录），EP 间隔也不是固定 1（有跳号）
    int ep = 1;
    for (int g = 0; g < 40; ++g) {
        ep += (g % 3 == 0) ? 2 : 1;   // EP 间距不均匀（1 或 2）
        for (int k = 0; k < 3; ++k)
            loss.push_back({ ep, 0, { (float)(1.0 + 0.001 * idx + 0.0001 * k), 2.0f, 1.5f } });
        ++idx;
    }
    write_metric(d, "metric_loss.bin", 3, loss);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    // 同 ep 多条必须算作合法，不能判为重复
    CHECK_MSG(r.quality.same_key_multi > 0, "same (episode,step) multiple records must be counted as legitimate");
    CHECK_MSG(r.quality.records_duplicate == 0, "distinct values must not be counted as duplicates");
    // EP 距离为 0 的差分必须不除零并给出原因
    auto it = r.diffs.find("loss");
    CHECK(it != r.diffs.end());
    if (it != r.diffs.end()) {
        int zero_dist = 0, nulled = 0;
        for (const auto& p : it->second.points) if (!p.rate_valid) { ++nulled;
            if (p.rate_null_reason.find("EP 距离=0") != std::string::npos) ++zero_dist; }
        CHECK_MSG(zero_dist > 0, "same-episode neighbours must report EP-distance-0 null reason");
        CHECK(nulled > 0);
    }
    // 采样周期未知 → 每个序列都声明
    for (const auto& kv : r.series)
        CHECK_MSG(!kv.second.sampling_interval_known, "sampling interval must be declared unknown");
    // D 必须逐记录配对（q - tq = 0.5），聚合值为 0.5
    CHECK(r.delta_d.valid);
    if (r.delta_d.valid) CHECK_MSG(NEAR(r.delta_d.mean, 0.5, 1e-5), "paired D mean must be q-tq per record");
    CHECK_MSG(r.delta_d.is_batch_mean_td_error, "target_q is confirmed as the TD target for the same accepted batch");
    CHECK_MSG(r.delta_d.forbidden_note.find("逐样本") != std::string::npos,
        "must forbid calling the batch mean a per-sample TD error");
    // 样本预览（旧报告 "Sample: episode=.. loss=.. qv=.. tq=.. rtype=.." 的对应物）
    // 高频序列：EP 分布聚合（先聚合再压缩，不对逐 update 噪声做逐点重建）
    {
        auto ls = r.series.find("loss");
        CHECK(ls != r.series.end());
        if (ls != r.series.end()) {
            CHECK_MSG(ls->second.noise_dominated,
                "a series with several records per EP must be flagged noise_dominated");
            CHECK_MSG(!ls->second.derived_ep_series.empty(),
                "EP-aggregated distribution trajectories must be derived");
            for (const auto& dk : ls->second.derived_ep_series)
                CHECK_MSG(r.series.find(dk) != r.series.end(), "derived EP series must exist");
            if (!ls->second.derived_ep_series.empty()) {
                auto d0 = r.series.find(ls->second.derived_ep_series[0]);
                if (d0 != r.series.end()) {
                    CHECK_MSG((int64_t)d0->second.samples.size() < (int64_t)ls->second.samples.size(),
                        "EP aggregation must reduce the point count");
                    CHECK(d0->second.aggregation == traj::AGG_EP_DISTRIBUTION);
                }
            }
        }
    }
    // 关键点预算：按优先级保留，绝不对最终集合等距抽样
    {
        auto ci = r.compression.find("loss");
        CHECK(ci != r.compression.end());
        if (ci != r.compression.end()) {
            CHECK_MSG(!ci->second.equispaced_sampling_used,
                "the final keypoint set must never be equispaced-sampled");
            CHECK_MSG(!ci->second.budget_classes.empty(), "budget classes must be reported");
            for (const auto& cb : ci->second.budget_classes)
                if (cb.name == "anomaly")
                    CHECK_MSG(cb.omitted == 0, "anomalies must not be dropped while under budget");
            CHECK_MSG(ci->second.retention_policy.find("MANDATORY") != std::string::npos,
                "retention policy must separate MANDATORY from PRIORITIZED");
            // config_event 必须与 anomaly 一样预算豁免（MANDATORY != Priority）
            for (const auto& cb : ci->second.budget_classes) {
                if (cb.name == "config_event")
                    CHECK_MSG(cb.budget_exempt, "config_event must be budget-exempt (mandatory)");
                if (cb.name == "segment_boundary")
                    CHECK_MSG(cb.budget_exempt, "segment_boundary must be budget-exempt (mandatory)");
            }
        }
    }
    // 分层异常存储：全部异常不得因预算丢弃；主 JSON 只放区域/Top，不内联全部
    {
        CHECK_MSG(r.anomalies_omitted == 0,
            "anomalies must not be dropped because of an output budget");
        // 默认启用分位定阈：detected = stored + filtered（不因输出预算丢失，只按分位筛）
        CHECK_MSG((int64_t)r.anomalies.size() + r.anomalies_filtered_by_quantile == r.anomalies_detected,
            "stored + quantile-filtered must equal detected");
        if (r.anomaly_keep_rate_used > 0)
            CHECK_MSG((int64_t)r.anomalies.size() <= r.anomalies_detected,
                "quantile mode must not increase the anomaly count");
        // 关闭分位定阈时必须全部保留
        {
            traj::Options o2 = r.options;
            o2.anomaly_keep_rate = 0.0;
            traj::AnalyzeResult r2 = traj::Analyze(o2);
            CHECK_MSG((int64_t)r2.anomalies.size() == r2.anomalies_detected,
                "keeprate=0 must keep every detected anomaly");
            CHECK(r2.anomalies_filtered_by_quantile == 0);
            CHECK(r2.anomalies_omitted == 0);
        }
        CHECK(!r.anomaly_rule_legend.empty());
        CHECK_MSG(r.anomaly_full_data_file == "trajectory_anomalies.jsonl",
            "full anomaly data must point at the sidecar JSONL");
        // 区域：计数必须等于异常总数（聚类不丢点）
        int64_t region_total = 0;
        for (const auto& rg : r.anomaly_regions) region_total += rg.count;
        CHECK_MSG(region_total == (int64_t)r.anomalies.size(),
            "region clustering must account for every anomaly");
        for (const auto& rg : r.anomaly_regions) {
            CHECK(!rg.severity.empty());
            CHECK(rg.ep_lo <= rg.ep_hi);
            CHECK_MSG((rg.ep_hi - rg.ep_lo) <= r.options.anomaly_region_max_span,
                "region EP span must respect the span limit (no multi-10k mega regions)");
            CHECK_MSG(!rg.family_counts.empty(), "region must report its metric family composition");
            CHECK_MSG(!rg.dominant_family.empty(), "region must report its dominant family");
            CHECK_MSG((int64_t)rg.representatives.size() <= 5,
                "representatives must be bounded");
            CHECK_MSG(!rg.severity_rule.empty(), "severity thresholds must be declared");
        }
        // 过碎检查：区域数不应接近异常数（否则等于没聚类）
        CHECK_MSG((int64_t)r.anomaly_regions.size() * 4 <= (int64_t)r.anomalies.size(),
            "regions must actually cluster (avg anomalies per region must be > 4)");
        // 主 JSON 不得内联全部异常（否则就是 113MB 那种表示膨胀）
        std::string js = traj::ToJson(r);
        CHECK_MSG(js.find("\"full_data\"") != std::string::npos,
            "main JSON must point at the sidecar file");
        CHECK_MSG(js.find("\"regions\"") != std::string::npos, "main JSON must carry regions");
        CHECK_MSG(js.find("\"trigger_mask\"") != std::string::npos,
            "anomalies must use the compact trigger bitmask");
    }
    // Loss 长尾：字段名与取值范围必须自洽（Cliff's delta 不适用）
    {
        CHECK_MSG(!r.long_tail_loss.cliff_delta_available,
            "Cliff's delta must be N/A for quantile-defined groups");
        CHECK_MSG(r.long_tail_loss.long_tail_ratio_definition.find("不是效应量") != std::string::npos,
            "long-tail ratio must be declared as NOT an effect size");
        if (r.long_tail_loss.valid)
            CHECK_MSG(r.long_tail_loss.long_tail_ratio >= 0.0,
                "long-tail ratio is a non-negative ratio, not a [-1,1] effect size");
        std::string why2;
        std::string js = traj::ToJson(r);
        CHECK_MSG(json_valid(js, &why2), why2.c_str());
        CHECK_MSG(js.find("\"cliff_delta\": null") != std::string::npos,
            "JSON must carry cliff_delta = null, not a ratio value");
        CHECK_MSG(js.find("\"long_tail_ratio\"") != std::string::npos, "JSON must carry long_tail_ratio");
    }
    // 样本预览（旧报告 "Sample: episode=.. loss=.. qv=.. tq=.. rtype=.." 的对应物）
    {
        CHECK_MSG(!r.preview_records.empty(), "preview records must exist when metric_loss is readable");
        if (!r.preview_records.empty()) {
            CHECK(r.preview_records[0].replay_type == 0 || r.preview_records[0].replay_type == 1);
            CHECK_MSG(NEAR(r.preview_records[0].delta_d,
                r.preview_records[0].q_value - r.preview_records[0].target_q, 1e-9),
                "preview D must equal q - target_q within the same record");
        }
    }
}

static void case_gap_duplicate_reorder() {
    g_case = "gap / duplicate / reorder";
    std::string d = dir_of("gapdup");
    std::vector<Rec> g;
    for (int i = 1; i <= 30; ++i) g.push_back({ i, 0, { (float)i } });
    g.push_back({ 30, 0, { 30.0f } });   // 紧邻完全重复
    g.push_back({ 5, 0, { 5.0f } });     // 乱序/回跳（回档或重启）
    write_metric(d, "metric_game.bin", 1, g);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    CHECK_MSG(r.quality.records_duplicate >= 1, "adjacent identical record must count as duplicate");
    CHECK_MSG(r.quality.reorder_events >= 1, "episode rewind must be counted");
    bool suspect = false;
    for (const auto& gp : r.quality.gaps)
        if (gp.kind == "suspect") suspect = true;
    CHECK_MSG(suspect, "rewind must be reported as a suspect gap, not silently stitched");
}

static void case_rollback_branches() {
    g_case = "rollback branch enumeration";
    std::string d = dir_of("rollback");
    std::vector<Rec> g; for (int i = 1; i <= 5; ++i) g.push_back({ i, 0, { (float)i } });
    write_metric(d, "metric_game.bin", 1, g);
    FILE* f = fopen((d + "active_timeline.txt").c_str(), "wb");
    if (f) { fprintf(f, "timeline_v1 1\n"); fclose(f); }
    // 两个不同的回档事件（不同 pid/tick），各含多个文件
    const char* names[] = { "metric_game.bin.rollback_1111_5000.bak",
                            "metric_loss.bin.rollback_1111_5000.bak",
                            "metric_game.bin.rollback_2222_9000.bak" };
    for (const char* n : names) { FILE* b = fopen((d + n).c_str(), "wb"); if (b) { fwrite("x", 1, 1, b); fclose(b); } }
    traj::Options o; o.log_dir = d; o.include_rollback_branches = true;
    traj::AnalyzeResult r = traj::Analyze(o);
    CHECK_MSG(r.timeline_id == "formal", "active_timeline marker must select the formal timeline");
    int branches = 0;
    for (const auto& t : r.timelines) if (t.kind == "rollback") ++branches;
    CHECK_MSG(branches == 2, "two distinct rollback events must yield two branches");
    // tick 排序：5000 在前
    std::vector<uint64_t> ticks;
    for (const auto& t : r.timelines) if (t.kind == "rollback") ticks.push_back(t.rollback_tick);
    CHECK(ticks.size() == 2 && ticks[0] == 5000 && ticks[1] == 9000);
    // 分支不得并入活动轨迹
    for (const auto& t : r.timelines)
        if (t.kind == "rollback")
            CHECK_MSG(t.note.find("must not be concatenated") != std::string::npos,
                "branch must declare it is not concatenated");
}

static void case_nonfinite_and_empty() {
    g_case = "NaN/Inf and empty data";
    std::string d = dir_of("nonfinite");
    std::vector<Rec> eps;
    eps.push_back({ 1, 0, { NAN } });
    eps.push_back({ 2, 0, { INFINITY } });
    eps.push_back({ 3, 0, { 0.3f } });
    eps.push_back({ 4, 0, { 5.0f } });     // 越界（epsilon 必须 [0,1]）
    write_metric(d, "metric_epsilon_episode.bin", 1, eps);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    CHECK_MSG(r.quality.records_nonfinite >= 1, "NaN must be counted");
    CHECK_MSG(r.quality.records_invalid >= 1, "out-of-range value must be counted as invalid");
    CHECK(r.quality.excluded_total >= 1);
    std::string why;
    CHECK_MSG(json_valid(traj::ToJson(r), &why), why.c_str());

    // 空目录：不得崩溃，必须给出限制说明
    std::string e = dir_of("empty");
    traj::Options o2; o2.log_dir = e; o2.include_rollback_branches = false;
    traj::AnalyzeResult r2 = traj::Analyze(o2);
    CHECK(r2.error.empty());
    CHECK_MSG(!r2.limitations.empty(), "empty dir must report limitations");
    CHECK_MSG(json_valid(traj::ToJson(r2), &why), why.c_str());
}

static void case_small_sample() {
    g_case = "small sample / min_valid";
    std::string d = dir_of("small");
    std::vector<Rec> g; g.push_back({ 1, 0, { 1.0f } });
    write_metric(d, "metric_game.bin", 1, g);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    // n=1：标量可给出 mean/min/max，但样本标准差必须为 null + 原因，不得伪造 0
    auto sc = r.scalars.find("episode_length");
    CHECK(sc != r.scalars.end());
    if (sc != r.scalars.end()) {
        CHECK(sc->second.n == 1);
        CHECK_MSG(!sc->second.std_valid, "n=1 std must be invalid, not fabricated as 0");
        CHECK_MSG(!sc->second.std_null_reason.empty(), "n=1 std must carry a reason");
    }
    for (const auto& cp : r.current_positions)
        if (cp.series_key == "episode_length") CHECK_MSG(!cp.valid, "n<3 percentile rank must be null");
    for (const auto& kv : r.windows)
        for (const auto& w : kv.second)
            if (!w.valid) CHECK(!w.null_reason.empty());
}

static void case_eval_modes_and_category() {
    g_case = "multi eval mode / category metric";
    std::string d = dir_of("modes");
    std::vector<Rec> win; std::vector<Rec> loss;
    for (int ep = 100; ep <= 500; ep += 100) {
        for (int m = 0; m < 3; ++m) {
            float w = 20.0f + 10.0f * m + (ep / 100);
            win.push_back({ ep, m, { (float)m, w, 10.0f, 100.0f - w - 10.0f } });
        }
    }
    write_metric(d, "metric_win.bin", 4, win);
    // Replay Type 交替 0/1，含一段连续区间
    int ep = 1;
    for (int i = 0; i < 60; ++i) {
        if (i % 20 == 0) ++ep;
        loss.push_back({ ep, i % 2, { 1.0f, 1.0f, 1.0f } });
    }
    write_metric(d, "metric_loss.bin", 3, loss);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    // 三个 mode 必须是彼此独立的序列
    for (int m = 0; m < 3; ++m) {
        char k[32]; snprintf(k, sizeof(k), "win_rate_mode%d", m);
        CHECK_MSG(r.series.find(k) != r.series.end(), k);
    }
    // 类别指标：频数/比例/转移/连续区间
    CHECK(!r.category_stats.empty());
    if (!r.category_stats.empty()) {
        const auto& cs = r.category_stats[0];
        CHECK_MSG(cs.total == 60, "category total must equal record count");
        int64_t fsum = 0;
        for (auto& kv : cs.frequency) fsum += kv.second;
        CHECK(fsum == cs.total);
        CHECK_MSG(cs.transitions > 0, "alternating categories must produce transitions");
        CHECK_MSG(!cs.runs.empty(), "contiguous runs must be reported");
        CHECK_MSG(cs.note.find("不做斜率") != std::string::npos, "category must declare no slope/regression");
        double psum = 0;
        for (auto& kv : cs.proportion) psum += kv.second;
        CHECK_MSG(NEAR(psum, 1.0, 1e-9), "proportions must sum to 1");
    }
    // 评估比率序列单位必须是百分数（0..100）
    for (int m = 0; m < 3; ++m) {
        char k[32]; snprintf(k, sizeof(k), "win_rate_mode%d", m);
        auto it = r.series.find(k);
        if (it != r.series.end()) {
            CHECK(it->second.unit == traj::UNIT_PERCENT);
            for (const auto& s : it->second.samples) CHECK(s.value >= 0 && s.value <= 100);
        }
    }
}

static void case_compression_budget_conflict() {
    g_case = "compression error vs length budget";
    std::string d = dir_of("compress");
    std::vector<Rec> loss;
    unsigned seed = 999;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 16) & 0x7fff) / 32767.0; };
    for (int i = 1; i <= 400; ++i)
        loss.push_back({ i, 0, { (float)(1.0 + (rnd() - 0.5) * 2.0) } });   // 高噪声
    write_metric(d, "metric_loss.bin", 3, loss);

    // 误差优先：严格容差
    {
        traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
        o.compression_mode = "error_priority"; o.error_tolerance = 0.01;
        traj::AnalyzeResult r = traj::Analyze(o);
        auto it = r.compression.find("loss");
        CHECK(it != r.compression.end());
        if (it != r.compression.end()) {
            CHECK(it->second.mode == "error_priority");
            CHECK(it->second.tolerance_is_hard);
            CHECK_MSG(it->second.original_points == 400, "original point count must be exact");
            CHECK_MSG(it->second.kept_points > 2, "tight tolerance must keep extra points");
            CHECK_MSG(it->second.max_error <= it->second.tolerance * 1.0000001 + 1e-9,
                "error-priority mode must meet its hard tolerance");
            CHECK_MSG(!it->second.error_unit.empty(), "error unit must be declared");
        }
    }
    // 篇幅优先：极小预算
    {
        traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
        o.compression_mode = "length_priority"; o.length_budget = 12;
        traj::AnalyzeResult r = traj::Analyze(o);
        auto it = r.compression.find("loss");
        CHECK(it != r.compression.end());
        if (it != r.compression.end()) {
            CHECK(it->second.mode == "length_priority");
            CHECK_MSG(!it->second.tolerance_is_hard, "length-priority tolerance is not a hard guarantee");
            CHECK_MSG(it->second.kept_points <= 12, "budget must be respected (plus forced anomaly points)");
            // 噪声大时不能同时保证「很短」和「最大误差很小」→ 必须如实报告
            CHECK_MSG(it->second.max_error > 0, "actual max error must be reported, not zero");
            CHECK_MSG(it->second.note.find("篇幅优先") != std::string::npos, "must state the tradeoff");
        }
    }
}

static void case_half_written_tail() {
    g_case = "half-written tail";
    std::string d = dir_of("tail");
    std::vector<Rec> g; for (int i = 1; i <= 20; ++i) g.push_back({ i, 0, { (float)i } });
    write_metric(d, "metric_game.bin", 1, g, /*truncate_tail=*/true);
    traj::Options o; o.log_dir = d; o.include_rollback_branches = false;
    traj::AnalyzeResult r = traj::Analyze(o);
    CHECK_MSG(r.quality.truncated_tail, "incomplete tail must be flagged");
    CHECK_MSG(r.quality.truncated_bytes == 3, "truncated byte count must be exact");
    auto it = r.series.find("episode_length");
    CHECK(it != r.series.end());
    if (it != r.series.end()) CHECK_MSG(it->second.samples.size() == 20, "only complete records may be used");
    CHECK(r.quality.read_boundary_fixed);
    CHECK_MSG(r.quality.boundary_note.find("未被修改") != std::string::npos, "must state the source is unmodified");
}

/* ============================== 主流程 ============================== */

int main() {
    printf("=== trajectory analyzer regression ===\n");
    case_constant_zero_mad();
    case_linear_and_segments();
    case_step_and_variance_change();
    case_spike_anomaly_position();
    case_known_lag_pair();
    case_uneven_and_same_episode();
    case_gap_duplicate_reorder();
    case_rollback_branches();
    case_nonfinite_and_empty();
    case_small_sample();
    case_eval_modes_and_category();
    case_compression_budget_conflict();
    case_half_written_tail();
    printf("\n--- cases last run: %s ---\n", g_case);
    printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
