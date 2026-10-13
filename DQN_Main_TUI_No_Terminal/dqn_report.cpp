// dqn_report.cpp - logreader 分析功能移植（命令面板触发，文本写入 buffer 供 TUI 查看器显示）
// 来源：DQN_Train_Logreader.cpp 的统计/摘要/综合分析模块，所有 printf 改为 report_printf
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <map>
#include <float.h>
#define LOG_DIR "logs\\"   /* 通信兼日志文件目录（eval.dat/train.dat 等） */
#include "bin_reader.h"
#include "../DQN_Core/include/metric_eval_reader.h"
#include "../DQN_Core/include/trajectory_analyzer.h"

/* ---------- 报告输出 buffer（供 TUI 查看器显示） ---------- */
/* 轨迹报告比历史 summary/analyze 长得多；缓冲区放大，并在真正截断时**显式报告**，
   绝不静默丢弃（完整结构化结果同时落盘，路径在报告尾部给出）。 */
static char g_report_buf[1024 * 1024];
static size_t g_report_len = 0;
static size_t g_report_dropped = 0;
static char g_report_title[64] = "";
static std::string g_report_directory;
static volatile LONG g_report_cancel_requested = 0;
void tui_report_prepare(void) { InterlockedExchange(&g_report_cancel_requested, 0); }
void tui_report_request_cancel(void) { InterlockedExchange(&g_report_cancel_requested, 1); }
static int report_cancelled(void) {
    return InterlockedCompareExchange(&g_report_cancel_requested, 0, 0) != 0;
}
static std::string report_log_path(const char* name) {
    return (g_report_directory.empty() ? active_training_log_dir(LOG_DIR) : g_report_directory) + name;
}

static void report_printf(const char* fmt, ...) {
    /* 预留 REPORT_RESERVE 字节给末尾的截断说明，保证「不静默丢弃」。 */
    const size_t cap = sizeof(g_report_buf) - 1024;
    va_list ap;
    va_start(ap, fmt);
    size_t remain = cap > g_report_len ? cap - g_report_len : 0;
    if (remain > 1) {
        int n = vsnprintf(g_report_buf + g_report_len, remain, fmt, ap);
        if (n > 0) {
            if ((size_t)n >= remain) {
                g_report_dropped += (size_t)n - (remain - 1);
                g_report_len += remain - 1;
            } else {
                g_report_len += (size_t)n;
            }
            g_report_buf[g_report_len] = '\0';
        }
    } else {
        va_list ap2;
        va_start(ap2, fmt);
        int need = vsnprintf(NULL, 0, fmt, ap2);
        va_end(ap2);
        if (need > 0) g_report_dropped += (size_t)need;
    }
    va_end(ap);
}
/* 用预留区写不计入截断统计的收尾说明（完整结果位置）。 */
static void report_tail_notice(const char* text) {
    size_t remain = sizeof(g_report_buf) - g_report_len;
    if (remain <= 1) return;
    size_t n = strlen(text);
    if (n >= remain) n = remain - 1;
    memcpy(g_report_buf + g_report_len, text, n);
    g_report_len += n;
    g_report_buf[g_report_len] = '\0';
}
static void report_begin(const char* title) {
    g_report_directory = active_training_log_dir(LOG_DIR);
    g_report_len = 0;
    g_report_dropped = 0;
    g_report_buf[0] = '\0';
    strncpy(g_report_title, title, sizeof(g_report_title) - 1);
    g_report_title[sizeof(g_report_title) - 1] = '\0';
}
const char* tui_report_get_text(void) { return g_report_buf; }
const char* tui_report_get_title(void) { return g_report_title; }

/* ---------- 轨迹分析的真实阶段提示（后台线程运行时供 TUI 轮询） ---------- */
static char g_traj_stage[96] = "";
static volatile LONG g_traj_stage_seq = 0;
static volatile LONG g_traj_running = 0;
static char g_traj_error[256] = "";
static char g_traj_detail_dir[512] = "";
static void traj_progress_cb(const char* stage, int64_t done, int64_t total, void* user) {
    (void)done; (void)total; (void)user;
    strncpy(g_traj_stage, stage ? stage : "", sizeof(g_traj_stage) - 1);
    g_traj_stage[sizeof(g_traj_stage) - 1] = '\0';
    InterlockedIncrement(&g_traj_stage_seq);
}
/* 真实阶段提示：返回当前阶段名与序号；running=0 表示已结束。 */
extern "C" const char* tui_trajectory_stage(void) { return g_traj_stage; }
extern "C" int tui_trajectory_running(void) { return (int)InterlockedCompareExchange(&g_traj_running, 0, 0); }
extern "C" const char* tui_trajectory_error(void) { return g_traj_error; }
extern "C" const char* tui_trajectory_detail_dir(void) { return g_traj_detail_dir; }

/* ---------- 数据结构（与原 logreader 一致） ---------- */
#define MAX_LINE 256
#define MAX_TRAIN_RECORDS 200
#define MAX_EVAL_RECORDS 1000

typedef struct {
    long long global_step;
    float loss;
    float q_value;
    float target_q;
    float epsilon;
    int replay_type;
} TrainStep;

typedef struct {
    int episode;
    int phase;
    int minimax_depth;
    float epsilon;
    int mode;
    float win_rate;
    float draw_rate;
    float lose_rate;
} EvalRecord;

/* 大日志工作区放静态存储，避免 analyze/trainstat 在默认 1MB 线程栈上
   同时放置两组约 320KB 数组而接近栈溢出。报告一次只运行一个。 */
static BRRecord g_report_loss_buf[BR_MAX_RECORDS];
static BRRecord g_report_eps_buf[BR_MAX_RECORDS];

/* Count spaces to determine format (7-col has 6 spaces, 6-col has 5 spaces) */
static int count_spaces(const char* s) {
    int count = 0;
    while (*s) { if (*s == ' ') count++; s++; }
    return count;
}

/* ---------- 加载 metric_loss.bin 中指定 global_step 附近的记录 ---------- */
static void load_train_around(const char* filename, long long target_global, int radius,
    TrainStep* buffer, int* count) {
    *count = 0;
    (void)filename;   /* 从 metric_loss.bin 读取 */
    BRRecord buf[BR_MAX_RECORDS];
    BRFile bf = {};
    snprintf(bf.path, sizeof(bf.path), "%s", report_log_path("metric_loss.bin").c_str());
    int n = br_read_tail(&bf, buf, BR_MAX_RECORDS);
    if (n <= 0) { report_printf("metric_loss.bin unavailable.\n"); return; }
    int scanned = 0;
    for (int i = 0; i < n; i++) {
        if ((i & 1023) == 0 && report_cancelled()) return;
        /* metric_loss.bin: episode=global_step, step=replay_type, fields={loss,qv,tq} */
        long long gs = buf[i].episode;
        if (gs >= target_global - radius && gs <= target_global + radius) {
            if (*count < MAX_TRAIN_RECORDS) {
                TrainStep* v = &buffer[(*count)++];
                v->global_step = gs;
                v->loss = buf[i].fields[0];
                v->q_value = buf[i].fields[1];
                v->target_q = buf[i].fields[2];
                v->epsilon = 0;   /* epsilon 需从 metric_epsilon.bin 关联 */
                v->replay_type = (int)buf[i].step;
            }
        }
        scanned++;
    }
    /* 尝试关联 epsilon */
    BRRecord eps_buf[BR_MAX_RECORDS];
    BRFile eps_file = {};
    snprintf(eps_file.path, sizeof(eps_file.path), "%s", report_log_path("metric_epsilon.bin").c_str());
    int eps_n = br_read_tail(&eps_file, eps_buf, BR_MAX_RECORDS);
    for (int i = 0; i < *count && eps_n > 0; i++) {
        long long gs = buffer[i].global_step;
        int lo = 0, hi = eps_n - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (eps_buf[mid].episode < gs) lo = mid + 1;
            else if (eps_buf[mid].episode > gs) hi = mid - 1;
            else { buffer[i].epsilon = eps_buf[mid].fields[0]; break; }
        }
    }
    report_printf("Scanned %d training records, found %d near global step %lld\n",
        scanned, *count, target_global);
}

/* ---------- 加载评估记录（从 metric_win.bin + metric_eval.bin） ---------- */
static std::vector<BRRecord> read_eval_rows(const char* path, int fields, int limit, int episode) {
    std::vector<BRRecord> rows;
    if (limit <= 0) return rows;
    BRRecord buf[2048];
    int n = br_read_all(path, buf, 2048);
    if (n <= 0) return rows;
    for (int i = 0; i < n; ++i) {
        if ((i & 1023) == 0 && report_cancelled()) break;
        if (buf[i].episode < 0 || (episode >= 0 && buf[i].episode != episode)) continue;
        const bool valid = fields == 4 ? metric_win_row_valid(buf[i]) : metric_eval_row_valid(buf[i]);
        if (valid) rows.push_back(buf[i]);
    }
    // Trim entire oldest checkpoint groups, never the first one or two modes.
    while ((int)rows.size() > limit) {
        int ep = rows.front().episode;
        size_t cnt = 0; while (cnt < rows.size() && rows[cnt].episode == ep) ++cnt;
        rows.erase(rows.begin(), rows.begin() + cnt);
    }
    return rows;
}

static int load_eval_records(const char* filename, EvalRecord* buffer, int max_records, int target_episode = -1) {
    (void)filename;   /* 文件路径由 bin 格式确定 */
    auto win_buf = read_eval_rows(report_log_path("metric_win.bin").c_str(), 4, max_records, target_episode);
    auto eval_buf = read_eval_rows(report_log_path("metric_eval.bin").c_str(), 7, max_records, target_episode);
    int win_n = (int)win_buf.size();
    std::map<std::pair<int,int>, BRRecord> metadata;
    for (const auto& row : eval_buf) metadata[{row.episode, (int)row.fields[3]}] = row;
    int count = 0;
    for (int i = 0; i < win_n && count < max_records; i++) {
        buffer[count].episode = win_buf[i].episode;
        buffer[count].mode = (int)win_buf[i].fields[0];
        buffer[count].win_rate = win_buf[i].fields[1];
        buffer[count].draw_rate = win_buf[i].fields[2];
        buffer[count].lose_rate = win_buf[i].fields[3];
        /* 从 metric_eval.bin 查找同 episode 的 phase/depth/epsilon */
        buffer[count].phase = 0; buffer[count].minimax_depth = 0; buffer[count].epsilon = 0;
        auto meta = metadata.find({win_buf[i].episode, buffer[count].mode});
        if (meta != metadata.end()) {
            buffer[count].phase = (int)meta->second.fields[0];
            buffer[count].minimax_depth = (int)meta->second.fields[1];
            buffer[count].epsilon = meta->second.fields[2];
        }
        count++;
    }
    report_printf("从 metric_win.bin 读取了 %d 条评估记录。\n", count);
    return count;
}

/* ---------- 显示评估记录 ---------- */
static void show_eval_records(EvalRecord* records, int count, int target_episode) {
    if (count == 0) {
        if (target_episode >= 0) report_printf("Episode %d: no valid evaluation records found in the log.\n", target_episode);
        else report_printf("没有评估记录。\n");
        return;
    }

    if (target_episode == -1) {
        int last_idx = count - 1;
        int ep = records[last_idx].episode;
        report_printf("最近评估记录 (Episode %d):\n", ep);
        report_printf("Phase %d, Minimax深度 %d, ε=%.4f\n", records[last_idx].phase, records[last_idx].minimax_depth, records[last_idx].epsilon);
        report_printf("Mode\tWin%%\tDraw%%\tLose%%\n");
        for (int i = 0; i < count; i++) {
            if (records[i].episode == ep) {
                report_printf("%d\t%.1f\t%.1f\t%.1f\n", records[i].mode, records[i].win_rate, records[i].draw_rate, records[i].lose_rate);
            }
        }
    }
    else {
        int found = 0;
        for (int i = 0; i < count; i++) {
            if (records[i].episode == target_episode) {
                if (!found) {
                    report_printf("Episode %d 评估记录:\n", target_episode);
                    report_printf("Phase %d, Minimax深度 %d, ε=%.4f\n", records[i].phase, records[i].minimax_depth, records[i].epsilon);
                    report_printf("Mode\tWin%%\tDraw%%\tLose%%\n");
                    found = 1;
                }
                report_printf("%d\t%.1f\t%.1f\t%.1f\n", records[i].mode, records[i].win_rate, records[i].draw_rate, records[i].lose_rate);
            }
        }
        if (!found) {
            report_printf("未找到 Episode %d 的评估记录。\n", target_episode);
            report_printf("已检索日志中的指定局号；该检查点可能尚未评估、已删除或数据无效。\n");
            report_printf("本次读取的评估 Episode：\n");
            int last_ep = -1;
            for (int i = 0; i < count; i++) {
                if (records[i].episode != last_ep) {
                    report_printf("%d ", records[i].episode);
                    last_ep = records[i].episode;
                }
            }
            report_printf("\n");
        }
    }
}

/* ---------- 统计辅助 ---------- */
typedef struct {
    int count;
    double sum_x, sum_y, sum_x2, sum_y2, sum_xy;
    double min_val, max_val;
    float* samples;
    int sample_cap, sample_count;
} StreamStats;

static void stats_init(StreamStats* s, int sample_cap) {
    memset(s, 0, sizeof(StreamStats));
    s->min_val = 1e9;
    s->max_val = -1e9;
    s->samples = (float*)malloc(sample_cap * sizeof(float));
    s->sample_cap = sample_cap;
    s->sample_count = 0;
    if (!s->samples) { report_printf("内存分配失败\n"); s->sample_cap = 0; }
}

static void stats_add(StreamStats* s, float val) {
    s->count++;
    s->sum_x += s->count - 1;
    s->sum_y += val;
    s->sum_x2 += (double)(s->count - 1) * (s->count - 1);
    s->sum_y2 += (double)val * val;
    s->sum_xy += (double)(s->count - 1) * val;
    if (val < s->min_val) s->min_val = val;
    if (val > s->max_val) s->max_val = val;
    if (s->sample_count < s->sample_cap) {
        s->samples[s->sample_count++] = val;
    }
}

static void stats_mean_std(StreamStats* s, float* mean, float* std) {
    if (s->count == 0) { *mean = 0; *std = 0; return; }
    *mean = (float)(s->sum_y / s->count);
    double var = (s->sum_y2 / s->count) - (*mean) * (*mean);
    *std = (var > 0) ? (float)sqrt(var) : 0;
}

static void stats_linear_regression(StreamStats* s, float* slope, float* intercept, float* r2) {
    if (s->count < 2) { *slope = 0; *intercept = 0; *r2 = 0; return; }
    double n = s->count;
    double sx = s->sum_x, sy = s->sum_y;
    double sxx = s->sum_x2, syy = s->sum_y2, sxy = s->sum_xy;
    double denom = n * sxx - sx * sx;
    if (fabs(denom) < 1e-12) { *slope = 0; *intercept = 0; *r2 = 0; return; }
    *slope = (float)((n * sxy - sx * sy) / denom);
    *intercept = (float)((sy - (*slope) * sx) / n);
    double mean_y = sy / n;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < s->sample_count; i++) {
        double pred = (*slope) * i + (*intercept);
        ss_tot += (s->samples[i] - mean_y) * (s->samples[i] - mean_y);
        ss_res += (s->samples[i] - pred) * (s->samples[i] - pred);
    }
    *r2 = (ss_tot < 1e-12) ? 1.0f : (float)(1.0 - ss_res / ss_tot);
}

static void stats_free(StreamStats* s) {
    if (s->samples) free(s->samples);
    s->samples = NULL;
}

static void print_histogram_from_samples(float* samples, int count, const char* label, int max_buckets) {
    if (count == 0) { report_printf("  (无数据)\n"); return; }
    float min_val = samples[0], max_val = samples[0];
    for (int i = 1; i < count; i++) {
        if (samples[i] < min_val) min_val = samples[i];
        if (samples[i] > max_val) max_val = samples[i];
    }
    float range = max_val - min_val;
    if (range < 1e-12f) {
        report_printf("  所有值均为 %.4f\n", min_val);
        return;
    }
    int bucket_count = (count < max_buckets) ? count : max_buckets;
    int* buckets = (int*)calloc(bucket_count, sizeof(int));
    if (!buckets) return;
    float bucket_size = range / bucket_count;
    for (int i = 0; i < count; i++) {
        int idx = (int)((samples[i] - min_val) / bucket_size);
        if (idx >= bucket_count) idx = bucket_count - 1;
        buckets[idx]++;
    }
    int max_freq = 0;
    for (int i = 0; i < bucket_count; i++) if (buckets[i] > max_freq) max_freq = buckets[i];
    const int BAR_WIDTH = 40;
    report_printf("  %s 分布 (%d 个样本):\n", label, count);
    for (int i = 0; i < bucket_count; i++) {
        float bin_start = min_val + i * bucket_size;
        float bin_end = bin_start + bucket_size;
        int bar_len = (max_freq > 0) ? (int)((float)buckets[i] / max_freq * BAR_WIDTH) : 0;
        report_printf("    [%.3f-%.3f]: %5d ", bin_start, bin_end, buckets[i]);
        for (int j = 0; j < bar_len; j++) report_printf("#");
        report_printf("\n");
    }
    free(buckets);
}

static void downsample(const float* x, const float* y, int count, int max_points,
    float** out_x, float** out_y, int* out_count) {
    if (count == 0) { *out_x = NULL; *out_y = NULL; *out_count = 0; return; }
    int points = (count < max_points) ? count : max_points;
    float* sampled_x = (float*)malloc(points * sizeof(float));
    float* sampled_y = (float*)malloc(points * sizeof(float));
    if (!sampled_x || !sampled_y) { free(sampled_x); free(sampled_y); *out_count = 0; return; }
    for (int i = 0; i < points; i++) {
        int idx = (int)((float)i / points * count);
        if (idx >= count) idx = count - 1;
        sampled_x[i] = x[idx];
        sampled_y[i] = y[idx];
    }
    *out_x = sampled_x;
    *out_y = sampled_y;
    *out_count = points;
}

static void print_array(const char* name, const char* x_label,
    const float* x, const float* y, int count,
    float y_scale, int max_print) {
    if (count == 0) { report_printf("%s: (无数据)\n", name); return; }
    int print_cnt = (count < max_print) ? count : max_print;
    report_printf("\n%s 曲线 (共 %d 个采样点，显示前 %d 个):\n", name, count, print_cnt);
    report_printf("  %s: [", x_label);
    for (int i = 0; i < print_cnt; i++) {
        report_printf("%.0f%s", x[i], (i < print_cnt - 1) ? ", " : "");
        if ((i + 1) % 20 == 0 && i < print_cnt - 1) report_printf("\n       ");
    }
    if (count > max_print) report_printf(", ... (共 %d 点)", count);
    report_printf("]\n  %s: [", name);
    for (int i = 0; i < print_cnt; i++) {
        if (isfinite(y[i])) report_printf("%.3f", y[i] * y_scale);
        else report_printf("missing");
        report_printf("%s", (i < print_cnt - 1) ? ", " : "");
        if ((i + 1) % 20 == 0 && i < print_cnt - 1) report_printf("\n       ");
    }
    if (count > max_print) report_printf(", ...");
    report_printf("]\n");
}

static void print_scalar_stats(const char* label, const float* data, int n) {
    if (n == 0) { report_printf("%s: (无数据)\n", label); return; }
    double sum = 0; float min_val = FLT_MAX, max_val = -FLT_MAX;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(data[i])) continue;
        ++valid;
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    if (!valid) { report_printf("%s: no valid samples (%d missing/invalid)\n", label, n); return; }
    double mean = sum / valid;
    double var = 0;
    for (int i = 0; i < n; i++) if (isfinite(data[i])) var += (data[i] - mean) * (data[i] - mean);
    var /= valid;
    float std = sqrt(var);
    report_printf("%s: 均值 = %.4f, 标准差 = %.4f, 最小 = %.4f, 最大 = %.4f\n",
        label, mean, std, min_val, max_val);
}

static void linear_regression_full(const float* x, const float* y, int n,
    float* slope, float* intercept, float* r2) {
    if (n < 2) { *slope = 0; *intercept = 0; *r2 = 0; return; }
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i];
        sy += y[i];
        sxx += (double)x[i] * x[i];
        syy += (double)y[i] * y[i];
        sxy += (double)x[i] * y[i];
    }
    double denom = n * sxx - sx * sx;
    if (fabs(denom) < 1e-12) { *slope = 0; *intercept = 0; *r2 = 0; return; }
    *slope = (float)((n * sxy - sx * sy) / denom);
    *intercept = (float)((sy - (*slope) * sx) / n);
    double mean_y = sy / n;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        double pred = (*slope) * x[i] + (*intercept);
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        ss_res += (y[i] - pred) * (y[i] - pred);
    }
    *r2 = (ss_tot < 1e-12) ? 1.0f : (float)(1.0 - ss_res / ss_tot);
}

static void print_regression(const char* name, const float* x, const float* y, int n,
    const char* x_label, float x_scale, float y_scale) {
    if (n < 2) { report_printf("%s: 数据不足\n", name); return; }
    float slope, intercept, r2;
    linear_regression_full(x, y, n, &slope, &intercept, &r2);
    report_printf("\n--- %s 回归 ---\n", name);
    report_printf("  样本数: %d\n", n);
    report_printf("  回归公式: %s = %.6f * %s + %.6f\n", name, slope, x_label, intercept);
    if (x_scale != 1.0f)
        report_printf("  斜率解释: 每 %.0f 单位 %s 变化 %.4f\n", x_scale, x_label, slope * x_scale);
    report_printf("  R^2: %.4f (%.1f%% 的变异可由线性趋势解释)\n", r2, r2 * 100);
    float* residuals = (float*)malloc(n * sizeof(float));
    if (residuals) {
        float min_r = FLT_MAX, max_r = -FLT_MAX;
        double sum_r = 0, sum_r2 = 0;
        for (int i = 0; i < n; i++) {
            float pred = slope * x[i] + intercept;
            residuals[i] = y[i] - pred;
            if (residuals[i] < min_r) min_r = residuals[i];
            if (residuals[i] > max_r) max_r = residuals[i];
            sum_r += residuals[i];
            sum_r2 += (double)residuals[i] * residuals[i];
        }
        double mean_r = sum_r / n;
        double std_r = sqrt(fmax(0.0, (sum_r2 / n) - mean_r * mean_r));
        report_printf("  残差: 均值 = %.6f, 标准差 = %.6f, 最小 = %.6f, 最大 = %.6f\n",
            mean_r, std_r, min_r, max_r);
        free(residuals);
    }
}

/* ---------- 摘要报告 ---------- */
static void generate_summary(void) {
    report_printf("\n========== AI 训练完整分析报告 ==========\n");
    time_t now = time(NULL);
    report_printf("生成时间: %s", ctime(&now));

    StreamStats win_stats, draw_stats, lose_stats;
    stats_init(&win_stats, 100000);
    stats_init(&draw_stats, 100000);
    stats_init(&lose_stats, 100000);

    int eval_mode0 = 0, pop_eval_mode0 = 0;
    int first_episode = -1, last_episode = -1;
    int first_phase = -1, last_phase = -1;
    /* depth 变化追踪 */
    int prev_depth = -1;
    int depth_changes = 0;
    int depth_episodes[64];
    int depth_values[64];

    /* 从 bin 读取评估记录（metric_win.bin + metric_eval.bin） */
    EvalRecord eval_buf[512];
    int eval_n = load_eval_records(NULL, eval_buf, 512);
    for (int i = 0; i < eval_n; i++) {
        if (eval_buf[i].mode != 0) continue;
        eval_mode0++;
        if (eval_buf[i].episode < 4000) pop_eval_mode0++;
        if (first_episode == -1) first_episode = eval_buf[i].episode;
        last_episode = eval_buf[i].episode;
        if (first_phase == -1) first_phase = eval_buf[i].phase;
        last_phase = eval_buf[i].phase;
        stats_add(&win_stats, eval_buf[i].win_rate / 100.0f);
        stats_add(&draw_stats, eval_buf[i].draw_rate / 100.0f);
        stats_add(&lose_stats, eval_buf[i].lose_rate / 100.0f);
        /* 追踪 depth 变化 */
        int d = eval_buf[i].minimax_depth;
        if (d > 0 && d != prev_depth) {
            if (depth_changes < 64) {
                depth_episodes[depth_changes] = eval_buf[i].episode;
                depth_values[depth_changes] = d;
            }
            depth_changes++;
            prev_depth = d;
        }
    }
    if (eval_n == 0) {
        report_printf("未找到 metric_win.bin 文件。\n");
    }

    if (eval_mode0 == 0) {
        report_printf("metric_win.bin 中没有 mode=0 的记录。\n");
    }
    else {
        report_printf("\n--- 评估数据 (mode=0, 总共 %d 条) ---\n", eval_mode0);
        report_printf("  Episode 范围: %d ~ %d, Phase: %d ~ %d\n", first_episode, last_episode, first_phase, last_phase);
        if (pop_eval_mode0 > 0) {
            report_printf("  其中种群训练阶段 (ep < 4000) 有 %d 条记录，已在统计中保留（可手动过滤）\n", pop_eval_mode0);
        }
        float win_mean, win_std, draw_mean, draw_std, lose_mean, lose_std;
        stats_mean_std(&win_stats, &win_mean, &win_std);
        stats_mean_std(&draw_stats, &draw_mean, &draw_std);
        stats_mean_std(&lose_stats, &lose_mean, &lose_std);
        report_printf("  胜率: 均值 %.2f%%, 标准差 %.2f%%\n", win_mean * 100, win_std * 100);
        report_printf("  平率: 均值 %.2f%%, 标准差 %.2f%%\n", draw_mean * 100, draw_std * 100);
        report_printf("  败率: 均值 %.2f%%, 标准差 %.2f%%\n", lose_mean * 100, lose_std * 100);

        float win_slope, win_intercept, win_r2;
        stats_linear_regression(&win_stats, &win_slope, &win_intercept, &win_r2);
        report_printf("  胜率趋势: 斜率 %.6f (每评估周期变化 %.4f%%), R²=%.4f\n",
            win_slope, win_slope * 100, win_r2);
        float lose_slope, lose_intercept, lose_r2;
        stats_linear_regression(&lose_stats, &lose_slope, &lose_intercept, &lose_r2);
        report_printf("  败率趋势: 斜率 %.6f (每评估周期变化 %.4f%%), R²=%.4f\n",
            lose_slope, lose_slope * 100, lose_r2);
        if (win_stats.sample_count > 0) {
            print_histogram_from_samples(win_stats.samples, win_stats.sample_count, "胜率", 15);
        }
    }
    stats_free(&win_stats);
    stats_free(&draw_stats);
    stats_free(&lose_stats);

    /* Depth 变化走向 */
    if (depth_changes > 0) {
        report_printf("\n--- Minimax 深度变化 ---\n");
        int show = depth_changes < 64 ? depth_changes : 64;
        for (int i = 0; i < show; i++) {
            report_printf("  Episode %5d → Depth %d\n", depth_episodes[i], depth_values[i]);
        }
        if (depth_changes > 64) report_printf("  ... 共 %d 次变化（仅显示最近64条）\n", depth_changes);
    }

    report_printf("\n--- 训练数据 (metric_loss.bin) ---\n");
    StreamStats loss_stats, q_stats, target_q_stats;
    stats_init(&loss_stats, 100000);
    stats_init(&q_stats, 100000);
    stats_init(&target_q_stats, 100000);

    /* 从 metric_loss.bin 读取训练数据（loss, qv, tq） */
    BRFile loss_file = {};
    snprintf(loss_file.path, sizeof(loss_file.path), "%s", report_log_path("metric_loss.bin").c_str());
    loss_file.field_count = 3;
    int loss_n = br_read_tail(&loss_file, g_report_loss_buf, BR_MAX_RECORDS);
    int train_total = 0;
    for (int i = 0; i < loss_n; i++) {
        float loss = g_report_loss_buf[i].fields[0];
        float qv = g_report_loss_buf[i].fields[1];
        float tq = g_report_loss_buf[i].fields[2];
        if (!isfinite(loss) || loss < 0 || !isfinite(g_report_loss_buf[i].fields[1]) ||
                        !isfinite(g_report_loss_buf[i].fields[2])) continue;
        train_total++;
        stats_add(&loss_stats, loss);
        stats_add(&q_stats, qv);
        stats_add(&target_q_stats, tq);
    }
    if (train_total > 0) report_printf("  总记录数: %d\n", train_total);
    else report_printf("  未找到训练数据 (metric_loss.bin)。\n");

    if (train_total > 0) {
        float loss_mean, loss_std, q_mean, q_std, tq_mean, tq_std;
        stats_mean_std(&loss_stats, &loss_mean, &loss_std);
        stats_mean_std(&q_stats, &q_mean, &q_std);
        stats_mean_std(&target_q_stats, &tq_mean, &tq_std);
        report_printf("  Loss: 均值 %.6f, 标准差 %.6f\n", loss_mean, loss_std);
        report_printf("  Q 值: 均值 %.6f, 标准差 %.6f\n", q_mean, q_std);
        report_printf("  Target Q: 均值 %.6f, 标准差 %.6f\n", tq_mean, tq_std);
        float q_slope, q_intercept, q_r2;
        stats_linear_regression(&q_stats, &q_slope, &q_intercept, &q_r2);
        report_printf("  Q 值趋势: 斜率 %.6f, R²=%.4f\n", q_slope, q_r2);
        if (loss_stats.sample_count > 0) {
            print_histogram_from_samples(loss_stats.samples, loss_stats.sample_count, "Loss", 12);
        }
    }
    stats_free(&loss_stats);
    stats_free(&q_stats);
    stats_free(&target_q_stats);

    report_printf("\n========================================\n");
}

/* ---------- 综合分析报告 ---------- */
static void generate_analysis_report(void) {
    report_printf("\n========== 综合分析报告 ==========\n");
    time_t now = time(NULL);
    report_printf("生成时间: %s", ctime(&now));
    report_printf("说明：包含 metric_win.bin 和 metric_loss.bin 的全部字段，提供采样数组、统计量、线性回归及残差分析。\n\n");

    const int MAX_POINTS = 100;
    const int SHOW_POINTS = 20;

    EvalRecord eval_records[MAX_EVAL_RECORDS];
    int eval_count = load_eval_records(NULL, eval_records, MAX_EVAL_RECORDS);
    int mode0_count = 0;
    float* ep_vals = NULL, * win_vals = NULL, * draw_vals = NULL, * lose_vals = NULL;
    float* eps_vals = NULL, * depth_vals = NULL, * phase_vals = NULL;

    if (eval_count > 0) {
        for (int i = 0; i < eval_count; i++) if (eval_records[i].mode == 0) mode0_count++;
        if (mode0_count > 0) {
            ep_vals = (float*)malloc(mode0_count * sizeof(float));
            win_vals = (float*)malloc(mode0_count * sizeof(float));
            draw_vals = (float*)malloc(mode0_count * sizeof(float));
            lose_vals = (float*)malloc(mode0_count * sizeof(float));
            eps_vals = (float*)malloc(mode0_count * sizeof(float));
            depth_vals = (float*)malloc(mode0_count * sizeof(float));
            phase_vals = (float*)malloc(mode0_count * sizeof(float));
            int idx = 0;
            for (int i = 0; i < eval_count; i++) {
                if (eval_records[i].mode == 0) {
                    ep_vals[idx] = (float)eval_records[i].episode;
                    win_vals[idx] = eval_records[i].win_rate;
                    draw_vals[idx] = eval_records[i].draw_rate;
                    lose_vals[idx] = eval_records[i].lose_rate;
                    eps_vals[idx] = eval_records[i].epsilon;
                    depth_vals[idx] = (float)eval_records[i].minimax_depth;
                    phase_vals[idx] = (float)eval_records[i].phase;
                    idx++;
                }
            }
            report_printf("--- 评估数据 (mode=0, 共 %d 条) ---\n", mode0_count);
            float* sampled_ep, * sampled_win; int s_cnt;
            downsample(ep_vals, win_vals, mode0_count, MAX_POINTS, &sampled_ep, &sampled_win, &s_cnt);
            print_array("胜率", "Episode", sampled_ep, sampled_win, s_cnt, 1.0f, SHOW_POINTS);
            free(sampled_ep); free(sampled_win);

            downsample(ep_vals, lose_vals, mode0_count, MAX_POINTS, &sampled_ep, &sampled_win, &s_cnt);
            print_array("败率", "Episode", sampled_ep, sampled_win, s_cnt, 1.0f, SHOW_POINTS);
            free(sampled_ep); free(sampled_win);

            downsample(ep_vals, eps_vals, mode0_count, MAX_POINTS, &sampled_ep, &sampled_win, &s_cnt);
            print_array("Epsilon", "Episode", sampled_ep, sampled_win, s_cnt, 1.0f, SHOW_POINTS);
            free(sampled_ep); free(sampled_win);

            report_printf("\n--- 评估统计量 ---\n");
            print_scalar_stats("胜率", win_vals, mode0_count);
            print_scalar_stats("败率", lose_vals, mode0_count);
            print_scalar_stats("Epsilon", eps_vals, mode0_count);
            print_scalar_stats("Depth", depth_vals, mode0_count);

            /* Depth 变化走向 */
            {
                int prev_d = -1;
                int changes = 0;
                int ch_ep[64], ch_d[64];
                for (int i = 0; i < mode0_count; i++) {
                    int d = (int)depth_vals[i];
                    if (d > 0 && d != prev_d) {
                        if (changes < 64) { ch_ep[changes] = (int)ep_vals[i]; ch_d[changes] = d; }
                        changes++;
                        prev_d = d;
                    }
                }
                if (changes > 0) {
                    report_printf("\n--- Depth 变化走向 ---\n");
                    int show = changes < 64 ? changes : 64;
                    for (int i = 0; i < show; i++)
                        report_printf("  Episode %5d → Depth %d\n", ch_ep[i], ch_d[i]);
                    if (changes > 64) report_printf("  ... 共 %d 次变化\n", changes);
                }
            }

            print_regression("胜率 vs Episode", ep_vals, win_vals, mode0_count, "Episode", 1000.0f, 1.0f);
            print_regression("败率 vs Episode", ep_vals, lose_vals, mode0_count, "Episode", 1000.0f, 1.0f);

            if (mode0_count > 20) {
                int half = mode0_count / 2;
                float s1, i1, r1, s2, i2, r2;
                linear_regression_full(ep_vals, win_vals, half, &s1, &i1, &r1);
                linear_regression_full(&ep_vals[half], &win_vals[half], mode0_count - half, &s2, &i2, &r2);
                report_printf("\n--- 胜率分段回归 ---\n");
                report_printf("  前半 (ep %d~%d): 斜率=%.6f, R^2=%.4f\n",
                    (int)ep_vals[0], (int)ep_vals[half - 1], s1, r1);
                report_printf("  后半 (ep %d~%d): 斜率=%.6f, R^2=%.4f\n",
                    (int)ep_vals[half], (int)ep_vals[mode0_count - 1], s2, r2);
            }
            free(ep_vals); free(win_vals); free(draw_vals); free(lose_vals);
            free(eps_vals); free(depth_vals); free(phase_vals);
        }
        else {
            report_printf("eval.dat 中无 mode=0 记录。\n");
        }
    }
    else {
        report_printf("eval.dat 未找到或为空。\n");
    }

    /* 从 metric_loss.bin 读取训练数据（Loss, Q, Target-Q）+ metric_epsilon.bin（Epsilon） */
    {
        BRFile loss_file = {}, eps_file = {};
        snprintf(loss_file.path, sizeof(loss_file.path), "%s", report_log_path("metric_loss.bin").c_str());
        snprintf(eps_file.path, sizeof(eps_file.path), "%s", report_log_path("metric_epsilon.bin").c_str());
        loss_file.field_count = 3; eps_file.field_count = 1;
        int loss_n = br_read_tail(&loss_file, g_report_loss_buf, BR_MAX_RECORDS);
        int eps_n = br_read_tail(&eps_file, g_report_eps_buf, BR_MAX_RECORDS);
        if (loss_n > 0) {
            float* steps    = (float*)malloc(loss_n * sizeof(float));
            float* losses   = (float*)malloc(loss_n * sizeof(float));
            float* qvals    = (float*)malloc(loss_n * sizeof(float));
            float* tqvals   = (float*)malloc(loss_n * sizeof(float));
            float* epsilons = (float*)malloc(loss_n * sizeof(float));
            int* rtypes     = (int*)malloc(loss_n * sizeof(int));
            if (steps && losses && qvals && tqvals && epsilons && rtypes) {
                int n = 0, missing = 0, invalid = 0;
                // Epsilon's second field is always zero, NOT replay_type.
                // Use the last finite epsilon logged for each episode for both colours.
                std::map<int, float> epsilon_by_record;
                for (int i = 0; i < eps_n; ++i)
                    if (isfinite(g_report_eps_buf[i].fields[0]) && g_report_eps_buf[i].fields[0] >= 0 && g_report_eps_buf[i].fields[0] <= 1)
                        epsilon_by_record[g_report_eps_buf[i].episode] = g_report_eps_buf[i].fields[0];
                for (int i = 0; i < loss_n; i++) {
                    float loss = g_report_loss_buf[i].fields[0];
                    const auto& row = g_report_loss_buf[i];
                    if (isnan(loss) || isnan(row.fields[1]) || isnan(row.fields[2])) { ++missing; continue; }
                    if (!isfinite(loss) || loss < 0 || !isfinite(row.fields[1]) || !isfinite(row.fields[2]) || row.episode < 0) { ++invalid; continue; }
                    steps[n] = (float)g_report_loss_buf[i].episode;
                    losses[n] = loss;
                    qvals[n] = g_report_loss_buf[i].fields[1];
                    tqvals[n] = g_report_loss_buf[i].fields[2];
                    rtypes[n] = g_report_loss_buf[i].step;
                    /* 按同一时间线的局号关联；缺失保持 missing，不伪造为 0。 */
                    auto epsilon = epsilon_by_record.find(row.episode);
                    epsilons[n] = epsilon != epsilon_by_record.end() ? epsilon->second : NAN;
                    n++;
                }
                report_printf("Training samples: valid=%d missing=%d invalid=%d\n", n, missing, invalid);
                report_printf("Epsilon: last finite logged value per episode (shared by replay colours).\n");
                if (n > 0) {
                    report_printf("  Sample: episode=%.0f loss=%.6f qv=%.6f tq=%.6f rtype=%d\n",
                        steps[0], losses[0], qvals[0], tqvals[0], rtypes[0]);
                    if (n > 1)
                        report_printf("  Sample: episode=%.0f loss=%.6f qv=%.6f tq=%.6f rtype=%d\n",
                            steps[1], losses[1], qvals[1], tqvals[1], rtypes[1]);
                    report_printf("\n--- 训练数据 (metric_loss.bin, 共 %d 条) ---\n", n);
                    float* s_steps, * s_vals; int s_cnt;
                    downsample(steps, losses, n, MAX_POINTS, &s_steps, &s_vals, &s_cnt);
                    print_array("Loss", "Episode", s_steps, s_vals, s_cnt, 1.0f, SHOW_POINTS);
                    free(s_steps); free(s_vals);
                    downsample(steps, qvals, n, MAX_POINTS, &s_steps, &s_vals, &s_cnt);
                    print_array("Q-value", "Episode", s_steps, s_vals, s_cnt, 1.0f, SHOW_POINTS);
                    free(s_steps); free(s_vals);
                    downsample(steps, tqvals, n, MAX_POINTS, &s_steps, &s_vals, &s_cnt);
                    print_array("Target-Q", "Episode", s_steps, s_vals, s_cnt, 1.0f, SHOW_POINTS);
                    free(s_steps); free(s_vals);
                    downsample(steps, epsilons, n, MAX_POINTS, &s_steps, &s_vals, &s_cnt);
                    print_array("Epsilon", "Episode", s_steps, s_vals, s_cnt, 1.0f, SHOW_POINTS);
                    free(s_steps); free(s_vals);
                    report_printf("\n--- 训练统计量 ---\n");
                    print_scalar_stats("Loss", losses, n);
                    print_scalar_stats("Q-value", qvals, n);
                    print_scalar_stats("Target-Q", tqvals, n);
                    print_scalar_stats("Epsilon", epsilons, n);
                    print_regression("Loss vs Episode", steps, losses, n, "Episode", 1000.0f, 1.0f);
                    print_regression("Q-value vs Episode", steps, qvals, n, "Episode", 1000.0f, 1.0f);
                    if (n > 200) {
                        int half = n / 2;
                        float s1, i1, r1, s2, i2, r2;
                        linear_regression_full(steps, losses, half, &s1, &i1, &r1);
                        linear_regression_full(&steps[half], &losses[half], n - half, &s2, &i2, &r2);
                        report_printf("\n--- Loss 分段回归 ---\n");
                        report_printf("  前半: 斜率=%.6f, R^2=%.4f\n", s1, r1);
                        report_printf("  后半: 斜率=%.6f, R^2=%.4f\n", s2, r2);
                    }
                    int black_count = 0, white_count = 0;
                    for (int i = 0; i < n; i++) { if (rtypes[i] == 0) black_count++; else if (rtypes[i] == 1) white_count++; }
                    report_printf("\n--- ReplayType 分布 (0=黑, 1=白) ---\n");
                    report_printf("  黑: %d (%.1f%%), 白: %d (%.1f%%)\n",
                        black_count, 100.0f * black_count / n, white_count, 100.0f * white_count / n);
                }
            }
            free(steps); free(losses); free(qvals); free(tqvals); free(epsilons); free(rtypes);
        } else {
            report_printf("metric_loss.bin 未找到或为空。\n");
        }
    }
    report_printf("\n========================================\n");
}

/* ---------- 训练轨迹分析报告（重构版入口） ----------
   与 report/analyze 的关键差别：
     * 分析**全部可用历史**（流式），不沿用 TUI 图表的尾部条数限制；
     * 输出由**同一份 AnalyzeResult** 生成：紧凑文本 + 带 schema_version 的完整 JSON + 明细 CSV/JSONL；
     * 全程只读源日志；耗时统计与真实阶段提示一并输出。 */
static void generate_trajectory_report(const char* args) {
    traj::Options o;
    o.log_dir = LOG_DIR;

    /* 可选参数：analyze [error|length] [budget=N] [tol=X] [top=N] [nodetail]
       未知参数不再静默忽略，会在报告开头列出，避免拼错后以为已生效。 */
    std::vector<std::string> unknown_opts;
    {
        const char* p = args;
        while (p && *p) {
            while (*p == ' ') ++p;
            if (!*p) break;
            const char* end = strchr(p, ' ');
            size_t len = end ? (size_t)(end - p) : strlen(p);
            char tok[64] = {};
            if (len >= sizeof(tok)) len = sizeof(tok) - 1;
            memcpy(tok, p, len);
            if (!strcmp(tok, "length")) o.compression_mode = "length_priority";
            else if (!strcmp(tok, "error")) o.compression_mode = "error_priority";
            else if (!strncmp(tok, "budget=", 7)) o.length_budget = atoi(tok + 7);
            else if (!strncmp(tok, "tol=", 4)) o.error_tolerance = atof(tok + 4);
            else if (!strncmp(tok, "top=", 4)) o.top_n = atoi(tok + 4);
            else if (!strncmp(tok, "toppct=", 7)) o.top_percentile = atof(tok + 7);
            else if (!strncmp(tok, "z=", 2)) o.z_threshold = atof(tok + 2);
            else if (!strncmp(tok, "madz=", 5)) o.mad_z_threshold = atof(tok + 5);
            else if (!strncmp(tok, "keeprate=", 9)) o.anomaly_keep_rate = atof(tok + 9);
            else if (!strcmp(tok, "nodetail")) o.detail_dir.clear();
            else if (!strcmp(tok, "records")) o.write_record_detail_jsonl = true;
            else unknown_opts.push_back(tok);
            p = end;
        }
    }

    const std::string active = active_training_log_dir(LOG_DIR);
    const std::string detail = active + "trajectory_report\\";
    CreateDirectoryA(LOG_DIR, nullptr);
    CreateDirectoryA(active.c_str(), nullptr);
    CreateDirectoryA(detail.c_str(), nullptr);
    o.detail_dir = detail;
    snprintf(g_traj_detail_dir, sizeof(g_traj_detail_dir), "%s", detail.c_str());

    g_traj_error[0] = '\0';
    snprintf(g_traj_stage, sizeof(g_traj_stage), "starting");
    InterlockedExchange(&g_traj_running, 1);

    traj::AnalyzeResult r = traj::Analyze(o, traj_progress_cb, nullptr);

    if (!r.error.empty()) {
        snprintf(g_traj_error, sizeof(g_traj_error), "%s", r.error.c_str());
        report_printf("轨迹分析失败: %s\n", r.error.c_str());
        InterlockedExchange(&g_traj_running, 0);
        return;
    }
    std::vector<std::string> files = traj::WriteDetailFiles(r);

    /* 先在**开头**给出完整结果位置：查看器只显示前 512 行，尾部信息可能看不到。 */
    report_printf("========== 训练轨迹分析（重构版）==========\n");
    if (!unknown_opts.empty()) {
        report_printf("[参数警告] 以下参数无法识别，已忽略（本次分析仍按默认参数执行）:\n");
        for (const auto& u : unknown_opts) report_printf("    %s\n", u.c_str());
        report_printf("  可用参数: error | length | tol=X | budget=N | top=N | toppct=X | z=X | madz=X | keeprate=X | records | nodetail\n");
    }
    report_printf("完整结构化结果（JSON/CSV/JSONL）:\n");
    if (files.empty()) report_printf("  (未生成明细文件)\n");
    for (const auto& f : files) report_printf("  %s\n", f.c_str());
    report_printf("耗时 %.3f s   源目录 %s   时间线 %s\n\n",
        r.elapsed_seconds, r.source_dir.c_str(), r.timeline_id.c_str());

    report_printf("%s", traj::ToCompactText(r).c_str());

    report_printf("\n-- 真实阶段序列（分析器实际执行的阶段，不是模拟进度）--\n");
    for (const auto& s : r.stage_log)
        report_printf("  %s : %s\n", s.stage.c_str(), s.detail.c_str());

    report_printf("\n-- 本次读取范围与参数 --\n");
    report_printf("  源目录: %s\n", r.source_dir.c_str());
    report_printf("  时间线: %s   压缩模式: %s   容差: %.6g   预算: %d   Top-N: %d\n",
        r.timeline_id.c_str(), r.options.compression_mode.c_str(),
        r.options.error_tolerance, r.options.length_budget, r.options.top_n);
    report_printf("  读取字节: %lld   完整记录: %lld   使用: %lld\n",
        (long long)r.bytes_read, (long long)r.quality.records_seen,
        (long long)r.quality.records_used);

    report_printf("\n-- 结构化输出文件 --\n");
    if (files.empty()) report_printf("  (未生成明细文件)\n");
    for (const auto& f : files) report_printf("  %s\n", f.c_str());

    InterlockedExchange(&g_traj_running, 0);
    snprintf(g_traj_stage, sizeof(g_traj_stage), "done");
}

/* ---------- 命令面板入口：生成报告文本 ---------- */
void tui_report_exec(const char* cmd) {
    report_begin(cmd);
    if (report_cancelled()) return;
    /* 重构后的入口：analyze 即新的训练轨迹分析报告（全量历史 + 口径声明 + 结构化输出）。
       旧实现保留为 analyze-legacy，便于回溯对比（必须先判 legacy，否则会被 "analyze" 前缀吃掉）。 */
    if (strncmp(cmd, "trajectory", 10) == 0) {
        generate_trajectory_report(cmd + 10);
    }
    else if (strncmp(cmd, "traj", 4) == 0) {
        generate_trajectory_report(cmd + 4);
    }
    else if (strncmp(cmd, "analyze-legacy", 14) == 0) {
        generate_analysis_report();
    }
    else if (strncmp(cmd, "analyze", 7) == 0) {
        generate_trajectory_report(cmd + 7);
    }
    else if (strncmp(cmd, "report", 6) == 0) {
        generate_summary();
    }
    else if (strncmp(cmd, "analyze", 7) == 0) {
        generate_analysis_report();
    }
    else if (strncmp(cmd, "trainstat", 9) == 0) {
        long long target = atoll(cmd + 9);
        TrainStep steps[MAX_TRAIN_RECORDS];
        int cnt = 0;
        load_train_around(NULL, target, 50, steps, &cnt);
        report_printf("\n--- 训练参数 @ step %lld ---\n", target);
        for (int i = 0; i < cnt; i++) {
            report_printf("  step %lld: loss=%.6f q=%.6f target_q=%.6f eps=%.6f rtype=%d\n",
                steps[i].global_step, steps[i].loss, steps[i].q_value,
                steps[i].target_q, steps[i].epsilon, steps[i].replay_type);
        }
    }
    else if (strncmp(cmd, "eval", 4) == 0) {
        int ep = -1;
        const char* arg = cmd + 4;
        while (*arg == ' ') arg++;
        if (*arg) ep = atoi(arg);
        EvalRecord recs[MAX_EVAL_RECORDS];
        int cnt = load_eval_records(NULL, recs, MAX_EVAL_RECORDS, ep);
        show_eval_records(recs, cnt > 0 ? cnt : 0, ep);
    }
    else {
        report_printf("未知分析命令: %s\n", cmd);
    }
    /* 确保末尾有换行 */
    if (g_report_len == 0) report_printf("(无输出)\n");
    /* 截断绝不静默：显式报告省略量与完整结果位置 */
    if (g_report_dropped > 0) {
        char notice[700];
        snprintf(notice, sizeof(notice),
            "\n[显示缓冲已满] 因查看器缓冲上限，以上内容省略了约 %llu 字节文本；"
            "完整结构化结果（JSON/CSV/JSONL）见: %s\n",
            (unsigned long long)g_report_dropped,
            g_traj_detail_dir[0] ? g_traj_detail_dir : "(本次未生成明细目录)");
        report_tail_notice(notice);
    }
}
