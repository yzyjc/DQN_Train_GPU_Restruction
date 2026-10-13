// trajectory_analyzer.h - 训练轨迹分析报告的核心模型与 API。
//
// 设计约束（与《新一代训练轨迹总结要求》一致）：
//  * 只读取，绝不修改源日志；复用 metric_log_reader.h / metric_log_format.h /
//    metric_eval_reader.h / training_log_timeline.h / training_log_rollback.h 的既有契约。
//  * Episode 不是唯一主键：样本身份 = (timeline, episode, step, record_index)。
//  * 任何聚合都必须声明单位、窗口口径、样本数与有效样本下限；无法计算时输出 null + 原因，
//    绝不伪造有限数值，也不输出非法 JSON。
//  * 紧凑文本、完整 JSON、明细 CSV/JSONL 三者由**同一份 AnalyzeResult** 生成。
#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <map>

namespace traj {

// 输出 JSON 的 schema 版本。任何字段语义变化都必须提升该版本号。
#define TRAJECTORY_SCHEMA_VERSION "1.0"

/* ============================ 基础枚举 ============================ */

// 单位。所有输出项必须携带，避免把「条记录」当成「局」。
enum Unit {
    UNIT_NONE = 0,
    UNIT_RECORDS,   // 记录条数
    UNIT_EPISODES,  // 局
    UNIT_STEPS,     // 决策步
    UNIT_SECONDS,   // 秒
    UNIT_PERCENT,   // 百分数 [0,100]
    UNIT_RATIO,     // 比例 [0,1]
    UNIT_LOSS,      // 无量纲损失
    UNIT_QVALUE,    // Q 值
};

const char* unit_name(int unit);

// 聚合层级。区分原始逐记录值与已聚合值。
enum Aggregation {
    AGG_RAW = 0,             // 逐记录原始值
    AGG_BATCH_MEAN,          // 同一 batch 内均值（loss/q/target_q）
    AGG_EPISODE_400_GAMES,   // 一局 400 盘（200 黑 + 200 白）后聚合，且已黑白平均
    AGG_EPISODE_SCALAR,      // 每局一个标量
    AGG_CHECKPOINT_MODE,     // 每评估点每 mode 一条
    AGG_EP_DISTRIBUTION,     // 某 EP 内全部原始记录的分布统计（mean/median/std/p05/p95/min/max）
};

const char* aggregation_name(int agg);

/* ============================ 时间线与分支 ============================ */

// 一条时间线（pretrain / formal）或一个回档分支快照。
struct Timeline {
    std::string id;          // "formal" / "pretrain" / "rollback:<pid>_<tick>"
    std::string kind;        // "active" / "rollback"
    std::string dir;         // 目录
    bool is_active = false;
    int  marker_phase = -1;  // active_timeline.txt 中的值（0/1），非 active 为 -1
    uint64_t rollback_tick = 0;   // 回档事件的 GetTickCount64()，用于事件排序
    uint32_t rollback_pid = 0;
    int64_t  record_count = 0;    // 该分支内已读记录总数
    std::string note;             // 不可读/不完整等说明
};

/* ============================ 样本与序列 ============================ */

// 保序样本。record_index 是文件内原始物理序号，用于打破 episode/step 的重复。
struct Sample {
    int32_t episode = 0;
    int32_t step = 0;          // 原始 step（含义随序列而定）
    int64_t record_index = 0;  // 文件内原始顺序号（0 基）
    double  value = 0.0;
    bool    finite = true;     // false 表示 NaN/Inf，已排除出统计
};

// 一条指标序列及其完整口径声明。
struct Series {
    std::string key;              // 稳定标识，如 "metric_loss.bin#loss"
    std::string file;             // 来源文件名
    std::string field;            // 字段名
    int         field_index = -1; // 字段下标，-1 表示由 prefix.step 承载
    std::string meaning;          // 含义
    int         unit = UNIT_NONE;
    int         aggregation = AGG_RAW;
    std::string episode_semantics; // episode 字段的真实含义
    std::string step_semantics;    // step 字段的真实含义
    bool        zero_based = false;
    bool        per_episode = false;
    bool        timeline_grouped = false; // 该序列已按时间线隔离
    std::vector<Sample> samples;
    // 采样周期是否已知（未知时不得把 EP 间隔算作缺失）
    bool        sampling_interval_known = false;
    std::string sampling_note;
    // 全量精确统计：流式累计，覆盖该序列的**全部**记录，不受抽样影响。
    int64_t     total_records = 0;      // add() 被调用的总次数（= 全部有效记录）
    double      exact_mean = 0, exact_min = 0, exact_max = 0;
    double      exact_stddev = 0;
    bool        exact_std_valid = false;
    std::string exact_std_null_reason;
    // 该序列是否「噪声主导」：每个 EP 有多条逐更新记录，逐点重建等于拟合噪声。
    // 这类序列的关键点压缩必须建在 EP/窗口分布聚合之上（见 derived_ep_series）。
    bool        noise_dominated = false;
    // 噪声主导序列的 EP 分布轨迹键名（mean/median/p05/p95/max...）
    std::vector<std::string> derived_ep_series;

    size_t size() const { return samples.size(); }
};

/* ============================ 数据质量 ============================ */

struct GapRange {
    int32_t from_episode = 0;
    int32_t to_episode = 0;
    int64_t missing_estimate = 0;   // 仅当 per_episode 且周期已知时给出
    int64_t record_index_lo = 0;
    int64_t record_index_hi = 0;
    std::string kind;               // "confirmed_missing" / "suspected_gap" / "timeline_boundary"
    std::string reason;
};

struct QualityReport {
    int64_t records_seen = 0;          // 文件内完整记录数
    int64_t records_used = 0;          // 进入统计的记录数
    int64_t records_invalid = 0;       // 字段非法被剔除
    int64_t records_nonfinite = 0;     // NaN/Inf 被剔除
    int64_t records_placeholder = 0;   // 缺失占位行（ep=-1,step=-1,全 NaN）
    int64_t records_duplicate = 0;     // 完全重复记录（同 ep/step 且同值）
    int64_t same_key_multi = 0;        // 同 (ep,step) 多条 → 合法，非重复
    int64_t reorder_events = 0;        // episode 回跳次数
    int64_t unknown_episode = 0;       // EP 缺失且无法可靠映射
    int64_t excluded_total = 0;        // 合计排除数
    bool    truncated_tail = false;
    int64_t truncated_bytes = 0;
    bool    read_boundary_fixed = true; // 读取期间固定边界
    int64_t boundary_size = 0;          // 固定时的文件字节数
    std::string boundary_note;
    std::vector<GapRange> gaps;
    // 相邻且**全部字段**逐位相同的记录数。仅在 per_episode 流（(episode,step) 应当唯一）上
    // 才计为 records_duplicate；其它流的 step 是类别/常量，重复属正常，另行计数。
    int64_t identical_adjacent_on_nonunique_key = 0;
    // 窗口统计做有界化：仅对确定性的等距子集计算并输出，window_stride 声明步长。
    int64_t window_stride = 1;
    int64_t windows_total_possible = 0;
    int64_t windows_computed = 0;
};

/* ============================ 统计口径 ============================ */

struct WindowStat {
    std::string id;             // "W5" 等
    bool   by_records = true;   // true=样本窗口；false=EP 窗口
    int    width = 0;           // 窗口宽度（记录数或 EP 数）
    int    min_valid = 0;       // 最小有效样本要求
    bool   valid = false;
    std::string null_reason;    // 不满足时为原因；满足时为空
    int64_t sample_count = 0;
    double episode_span = 0.0;  // 窗口内真实 EP 跨度
    int32_t ep_lo = 0, ep_hi = 0;
    double mean = 0, stddev = 0, min = 0, max = 0;
    double p25 = 0, p50 = 0, p75 = 0;
    std::string std_definition; // 标准差口径
    bool crossed_boundary = false; // 是否跨时间线/大缺口（跨则不计算）
};

struct DiffPoint {
    int64_t record_index = 0;
    int32_t episode = 0;
    double raw_diff = 0;            // 原始差分（后 - 前）
    bool   rate_valid = false;      // 按 EP 距离归一化的变化率是否可用
    double rate_per_episode = 0;    // 单位：值 / 局
    std::string rate_null_reason;   // EP 距离为 0 时不除零，给原因
    double second_diff = 0;         // 二阶差分（不均匀采样下按定义给出，见 note）
    bool   second_valid = false;
};

struct DiffSeries {
    std::string key;
    std::string first_order_note;   // 一阶差分定义
    std::string second_order_note;  // 二阶差分在非均匀采样下的定义声明
    std::vector<DiffPoint> points;
};

struct Segment {
    int64_t index = 0;          // 分段序号
    int64_t lo_record = 0, hi_record = 0;
    int32_t ep_lo = 0, ep_hi = 0;
    int64_t sample_count = 0;
    double slope = 0, intercept = 0, r2 = 0;
    double slope_per_episode = 0;
    std::string fit_note;
    // 回归残差统计（与旧报告 print_regression 对齐）
    int64_t residual_n = 0;
    double residual_mean = 0, residual_stddev = 0, residual_min = 0, residual_max = 0;
    bool   residual_valid = false;
    std::string residual_null_reason;
};

/* ============================ 异常与极值 ============================ */

struct Anomaly {
    int64_t rank = 0;
    std::string rule;           // 触发的规则名
    double threshold = 0;
    std::string threshold_kind; // "abs_z" / "abs_mad_z" / "ratio" / "range"
    int32_t episode = 0;
    int64_t record_index = 0;
    double value = 0;
    double baseline = 0;
    double score = 0;
    int64_t baseline_n = 0;
    std::string baseline_definition; // 基线只用此前有效样本
    std::string null_reason;         // std/MAD 为 0 或样本不足时的原因
    std::string series_key;
    // 触发规则位图（紧凑编码，避免重复存规则名字符串）：
    // 0x01=local_z  0x02=robust_z  0x04=delta  0x08=delta2  0x10=residual  0x20=level_shift
    uint32_t trigger_mask = 0;
};

// 异常区域：把「上万条独立异常」聚类成「一批持续异常事件」，AI 可读性显著更高。
// 全部原始异常仍然无损保存在独立文件中，可通过 episode / record_index 回查。
struct AnomalyRegion {
    int64_t index = 0;
    int32_t ep_lo = 0, ep_hi = 0;
    int64_t record_lo = 0, record_hi = 0;
    int64_t count = 0;
    double mean_score = 0, max_score = 0;
    int32_t peak_ep = 0;
    int64_t peak_record = 0;
    std::string peak_series;
    std::vector<std::string> series;      // 区域内涉及的指标（去重）
    // 区内按指标家族的构成（切分依据是跨度上限，家族只做展示，避免过碎）
    std::map<std::string, int64_t> family_counts;
    std::string dominant_family;
    bool span_limited = false;            // 是否因跨度上限被切分
    struct MetricPeak {
        std::string series;
        double max_abs_value = 0;
        int32_t at_ep = 0;
    };
    std::vector<MetricPeak> metric_peaks; // 每个受影响指标的区域内最大绝对值
    std::vector<Anomaly> representatives; // Top-N 代表异常点（完整字段）
    struct WindowDelta {
        std::string series;
        bool valid = false;
        double before_mean = 0, during_mean = 0, after_mean = 0;
        int64_t before_n = 0, during_n = 0, after_n = 0;
        int32_t before_ep_lo = 0, before_ep_hi = 0, during_ep_lo = 0, during_ep_hi = 0;
        int32_t after_ep_lo = 0, after_ep_hi = 0;
        std::string null_reason;
    };
    std::vector<WindowDelta> deltas;      // 前 / 中 / 后 均值对比
    std::string severity;                 // critical / major / minor
    std::string severity_rule;            // 分层阈值声明
};

struct AnomalyInterval {
    std::string rule;
    int32_t ep_lo = 0, ep_hi = 0;
    int64_t record_lo = 0, record_hi = 0;
    int64_t count = 0;
    double max_score = 0;
    std::string series_key;
};

struct Extremum {
    std::string kind;        // "global_max" / "global_min" / "local_max" / "local_min"
    int32_t episode = 0;
    int64_t record_index = 0;
    double value = 0;
    std::string series_key;
};

/* ============================ 压缩摘要 ============================ */

struct KeyPoint {
    int64_t record_index = 0;
    int32_t episode = 0;
    double value = 0;
    bool must_keep = false;   // 异常/边界/配置事件点
    std::string reason;
};

struct CompressionResult {
    std::string series_key;
    std::string mode;            // "error_priority" / "length_priority"
    std::string method;          // 重建方式（自适应二分/垂距法）
    std::string error_unit;      // 误差单位 = 该序列单位
    double tolerance = 0;        // 目标误差阈值
    bool   tolerance_is_hard = true;
    int64_t original_points = 0;
    int64_t kept_points = 0;
    double max_error = 0;        // 实测最大误差
    double rmse = 0;             // 实测 RMSE
    bool   target_met = false;
    std::string note;            // 未达标/放宽说明；两模式不可兼得的声明
    std::vector<KeyPoint> points;

    // ---- 关键点预算的按优先级保留（绝不对比最终集合做等距抽样）----
    struct ClassBudget {
        std::string name;        // anomaly / config_event / extremum / change_point / segment_boundary / ordinary
        int    priority = 0;     // 数字越小越先保留
        int64_t available = 0;   // 该类可用点数
        int64_t kept = 0;        // 实际保留
        int64_t omitted = 0;     // 因预算被省略
        bool budget_exempt = false;  // true = 该类不受预算限制（异常类）
    };
    std::vector<ClassBudget> budget_classes;
    std::string retention_policy;
    // 恒为 false：保留顺序按优先级，不做等距抽样（保留此字段是为了让消费者能断言这一点）
    bool equispaced_sampling_used = false;
};

/* ============================ 相关与配置事件 ============================ */

struct Correlation {
    std::string series_a, series_b;
    int    lag = 0;                 // corr(a(EP), b(EP+lag))
    std::string lag_sign_convention; // 正负号定义与单位
    std::string alignment;           // 配对方式（同时间线、同一 EP）
    std::string transform;           // "raw" / "diff" / "detrended"
    int64_t pairs = 0;
    int32_t ep_lo = 0, ep_hi = 0;
    bool   valid = false;
    std::string null_reason;
    double r = 0;                   // 带符号
    double abs_r = 0;               // |r|，单独输出
};

struct ConfigEventWindow {
    int64_t before_n = 0, after_n = 0;
    double  before_mean = 0, after_mean = 0;
    int32_t before_ep_lo = 0, before_ep_hi = 0;
    int32_t after_ep_lo = 0, after_ep_hi = 0;
    int64_t before_missing = 0, after_missing = 0;
    bool    before_valid = false, after_valid = false;
    std::string null_reason;
};

struct ConfigEvent {
    std::string field;           // 变化字段（depth / epsilon / phase / mode）
    std::string kind;            // "numeric" / "categorical"
    int32_t episode = 0;
    int64_t record_index = 0;
    double old_value = 0, new_value = 0;
    std::string evidence;        // 证据来源：文件 + 字段
    bool   evidence_is_log = false; // 是否来自独立日志文件
    ConfigEventWindow window;
    double change_ratio = 0;
    bool   ratio_valid = false;
    std::string ratio_null_reason;
    std::vector<std::string> overlapping; // 同时发生的其它事件（不声称因果）
    std::string note;
};

struct UnavailableItem {
    std::string item;
    std::string reason;
};

/* ============================ 专项 ============================ */

struct DeltaDSpecial {
    // D = q_value - target_q，同一 batch 配对。
    std::string definition;
    std::string pairing;          // 同一记录内配对（同 batch）
    int64_t pairs = 0;
    int64_t raw_records = 0;
    double mean = 0, stddev = 0, min = 0, max = 0;
    double p50 = 0, p90 = 0, p99 = 0;
    bool   valid = false;
    std::string null_reason;
    // 经代码确认：target_q 就是同一批样本的 TD 目标，故 batch 均值差 = batch 平均 TD 误差。
    // 但逐样本值未落盘，因此**不得**称为逐样本 TD 误差。
    bool   is_batch_mean_td_error = false;
    std::string confirmation_evidence;
    std::string forbidden_note;
};

struct LongTailLossSpecial {
    std::string series_key;
    int64_t n = 0;
    double median = 0, mad = 0;
    double p90 = 0, p95 = 0, p99 = 0;
    double tail_mean = 0;          // 超过 p95 的样本均值
    int64_t tail_n = 0;
    double tail_share_of_sum = 0;  // 尾部样本占总和比例
    // 长尾严重度 = 尾部均值 / 中位数（无量纲比值，1 = 无长尾）。
    // 注意：这不是效应量，取值范围不是 [-1,1]。
    double long_tail_ratio = 0;
    std::string long_tail_ratio_definition;
    // Cliff's delta 在「以分位数划分两组」时恒等于 1（构造成立），故不报告。
    bool   cliff_delta_available = false;
    std::string cliff_delta_null_reason;
    bool   valid = false;
    std::string null_reason;
    std::vector<Anomaly> tail_points;
};

/* ============================ 报告主体 ============================ */

struct Options {
    std::string log_dir = "logs\\";  // 日志根目录
    bool include_rollback_branches = true;
    bool include_detail_files = true;   // 是否生成明细 CSV/JSONL
    std::string detail_dir;             // 明细输出目录（空=不落盘，仅返回内容）
    // 逐记录明细 JSONL（loss/q/target_q/D/replay_type 每条一行）。
    // **默认关闭**：该内容与 metric_loss.bin 重复，文本形式体积约为二进制的 6 倍
    // （实测 198MB vs 100MB），而按分层设计要求 Level 3 就是原始日志本身。
    // 需要时用 `records` 参数开启。
    bool write_record_detail_jsonl = false;
    // 压缩模式
    std::string compression_mode = "error_priority"; // or "length_priority"
    double error_tolerance = 0.0;       // <=0 表示按序列自动取（分位距的一部分）
    int    length_budget = 0;           // 篇幅优先模式的目标摘要点数
    int    top_n = 20;                  // 紧凑文本 Top-N
    // 关键点预算：超出时按优先级保留（异常 > 事件 > 极值 > 变点 > 分段边界 > 普通点），
    // **不做等距抽样**。<=0 用默认 20000。异常类**不受预算限制**（必须保留）。
    int64_t keypoint_budget = 20000;
    // 长轨迹的自动点数预算：当未显式给 tol/budget 且轨迹点数超过 10000 时，
    // 自动改用「篇幅优先」并按该预算压缩（否则 5% 容差对噪声轨迹几乎压不动）。
    int64_t trajectory_budget = 2000;
    // 异常区域聚类：相邻异常之间的 EP 间隔 <= 该阈值则归入同一区域。
    int64_t anomaly_region_gap = 200;
    // 单个区域的 EP 跨度上限：超过就强制切分。
    // 只按时间连续性合并会把「指标 A@1k、B@2k、C@3k…接力」串成横跨数万局的超级区域；
    // 但按指标家族硬切又会过碎（实测 9.5k → 44k 个区域）。因此用跨度上限约束，
    // 家族构成作为区内明细输出，而不是切分依据。
    int64_t anomaly_region_max_span = 2000;
    // 每个区域保留多少条完整代表异常点。
    int64_t anomaly_representatives = 5;
    // 主 JSON 里内联的 Top 异常条数（其余在 JSONL）。
    int64_t anomaly_top_in_json = 20;
    // 异常全部保留：0 表示不设上限（仅受内存保护上限约束）。
    // 全部异常写入 trajectory_anomalies.jsonl，主 JSON 只留索引/区域/代表点。
    bool anomaly_keep_all = true;
    // 方案 A：固定阈值（可调）。默认 4.0 / 6.0；抬高到 8 / 12 可显著降低误报率。
    double z_threshold = 4.0;
    double mad_z_threshold = 6.0;
    // 方案 B（默认）：按分位目标定阈——每条序列只保留分数最高的前 keep_rate 比例，
    // 使「异常」在定义上就是罕见事件。0 表示关闭（退回方案 A 的固定阈值行为）。
    //
    // 默认 0.005 的实测依据（活动正式时间线，1090 万条记录）：
    //   keeprate=0     -> 19.2 万条异常，区域退化成「2000 局一格的连续平铺」
    //   z=8/mad_z=12   -> 8.6 万条，仍然平铺（抬高阈值只减少每格条数，不改变平铺本质）
    //   keeprate=1e-3  -> 329 条 / 58 区域，无平铺
    //   keeprate=5e-3  -> 1106 条 / 96 区域，无平铺，critical 区域 2 个  ← 采用
    double anomaly_keep_rate = 0.005;
    // 按数值取最大的 X% 作为「暴露切片」，并与全量/保留统计并列输出。
    // <=0 或 >=100 表示关闭。
    double top_percentile = 5.0;
    // 窗口宽度（两种口径都输出）
    std::vector<int> record_windows = { 5, 10, 20, 50 };
    std::vector<int> episode_windows = { 5, 10, 20, 50 };
    int    max_records_per_series = 0;  // 0=不限制（全量）
};

struct StageLog { std::string stage; std::string detail; };

struct AnalyzeResult {
    std::string schema_version = TRAJECTORY_SCHEMA_VERSION;
    std::string generated_at;
    Options options;
    // 读取范围与来源
    std::string source_dir;
    std::string timeline_id;
    std::vector<Timeline> timelines;
    int64_t bytes_read = 0;
    int64_t read_start_unix = 0, read_end_unix = 0;
    std::vector<std::string> limitations;   // 受现有日志限制的说明
    std::vector<UnavailableItem> unavailable;

    std::map<std::string, Series> series;       // key → 序列（活动时间线）
    QualityReport quality;

    std::map<std::string, std::vector<WindowStat>> windows;
    std::map<std::string, DiffSeries> diffs;
    std::map<std::string, std::vector<Segment>> segments;

    std::vector<Anomaly> anomalies;
    std::vector<AnomalyInterval> anomaly_intervals;
    // 分层异常存储：
    //   Level 1 主 JSON  -> 只放汇总 + 区域 + Top 代表点（本结构）
    //   Level 2 JSONL    -> 全部异常，紧凑编码（detail 文件）
    //   Level 3 原始日志 -> 按 episode/record_index 回查
    std::vector<AnomalyRegion> anomaly_regions;
    std::map<std::string, int64_t> anomaly_count_by_series;
    int64_t anomaly_count_critical = 0;
    int64_t anomaly_count_major = 0;
    int64_t anomaly_count_minor = 0;
    int64_t region_count_critical = 0;
    int64_t region_count_major = 0;
    int64_t region_count_minor = 0;
    std::string anomaly_rule_legend;      // 位图含义说明
    std::string anomaly_full_data_file;   // 全部异常所在文件（相对明细目录）
    bool    anomaly_full_in_main_json = false;  // 主 JSON 是否内联全部异常（默认 false）
    // 触发规则的总数；若超过存储上限，anomalies_omitted 给出被省略条数（绝不静默丢弃）。
    int64_t anomalies_detected = 0;
    int64_t anomalies_omitted = 0;
    int64_t anomaly_store_limit = 0;
    // 因「按分位目标定阈」被筛掉的条数（方案 B 生效时 > 0）
    int64_t anomalies_filtered_by_quantile = 0;
    double  anomaly_keep_rate_used = 0.0;
    std::map<std::string, std::vector<Extremum>> extrema;
    std::map<std::string, std::map<std::string, double>> quantiles; // key → {"p50":..}

    std::map<std::string, CompressionResult> compression;

    std::vector<Correlation> correlations;
    std::vector<ConfigEvent> config_events;

    DeltaDSpecial delta_d;
    LongTailLossSpecial long_tail_loss;

    // 类别指标（Replay Type）专用：频数/比例/转移/连续区间
    struct CategoryStat {
        std::string series_key;
        std::string definition;
        std::map<int, int64_t> frequency;        // 类别 → 频数
        std::map<int, double>  proportion;       // 类别 → 比例 [0,1]
        int64_t total = 0;
        int64_t transitions = 0;                 // 切换次数
        std::map<std::pair<int,int>, int64_t> transition_matrix;
        struct Run { int value; int64_t length; int32_t ep_lo, ep_hi; int64_t rec_lo, rec_hi; };
        std::vector<Run> runs;                   // 连续区间
        std::string note;                        // 声明不做斜率/回归
    };
    std::vector<CategoryStat> category_stats;

    // 统计量
    // 每个序列同时给出**两套**统计，便于判断抽样是否影响结论：
    //   exact_* = 全量精确（流式累计，覆盖所有记录）
    //   其余     = 保留子集（点级统计，分位数等只能在这里算）
    struct ScalarStat {
        // 全量精确
        int64_t exact_n = 0;
        double  exact_mean = 0, exact_stddev = 0, exact_min = 0, exact_max = 0;
        bool    exact_std_valid = true;
        std::string exact_std_null_reason;
        // 保留子集
        int64_t n = 0;
        double mean = 0, min = 0, max = 0, p50 = 0, p90 = 0, p99 = 0;
        double stddev = 0;
        bool   std_valid = true;
        std::string std_null_reason;
        // 两套是否一致（无抽样时应完全一致）
        bool   sampling_affected = false;
    };
    std::map<std::string, ScalarStat> scalars;

    // Top X%（按数值）暴露切片：与全量/保留统计并列，用于判断上尾有多重并可直接核对
    struct TopSlice {
        std::string series_key;
        double percent = 0;            // 请求的 X%
        double threshold = 0;          // 阈值 = 保留子集的 p(100-X)
        std::string definition;
        bool   valid = false;
        std::string null_reason;
        int64_t n = 0;
        double  actual_percent = 0;    // 实际占比（并列值全部保留，可能略大于 X%）
        int32_t ep_lo = 0, ep_hi = 0;
        int64_t record_lo = 0, record_hi = 0;
        double  mean = 0, stddev = 0, min = 0, max = 0, p50 = 0, p90 = 0, p99 = 0;
        bool    std_valid = true;
        std::string std_null_reason;
        double  sum_share = 0;         // 切片之和 / 保留子集之和
        bool    sum_share_valid = false;
        std::string sum_share_null_reason;
        std::string basis;             // 切片基于哪一套样本
    };
    std::vector<TopSlice> top_slices;
    struct CurrentPosition {
        std::string series_key;
        double current = 0, percentile_rank = 0;  // 百分位排名口径
        std::string percentile_definition;
        bool valid = false; std::string null_reason;
    };
    std::vector<CurrentPosition> current_positions;

    // 前若干条有效训练记录的原样预览（与旧报告 "Sample: episode=.. loss=.. qv=.. tq=.. rtype=.." 对齐）
    struct PreviewRecord {
        int32_t episode = 0;
        int32_t replay_type = 0;
        int64_t record_index = 0;
        double loss = 0, q_value = 0, target_q = 0, delta_d = 0;
    };
    std::vector<PreviewRecord> preview_records;
    int64_t preview_limit = 5;

    // 明细输出（若 detail_dir 非空也会落盘）
    std::map<std::string, std::string> detail_csv;
    std::map<std::string, std::string> detail_jsonl;

    // 运行开销统计
    double elapsed_seconds = 0;
    int64_t peak_memory_kb = 0;
    int64_t output_bytes_text = 0, output_bytes_json = 0;
    // 实际输出文件与真实体积（由 WriteDetailFiles 回填），用于计算压缩比
    struct OutputFile { std::string name; int64_t bytes = 0; };
    std::vector<OutputFile> output_files;
    int64_t raw_input_bytes = 0;         // 本次实际读取的源日志字节数
    double core_compression_ratio = 0;   // 主报告相对原始输入的压缩比（越小越强）
    double total_compression_ratio = 0;  // 全部结构化输出合计相对原始输入的压缩比

    // 阶段提示（后台运行时可查询）
    std::vector<StageLog> stage_log;
    std::string error;   // 非空表示整体失败
};

// 进度回调：stage 为当前阶段名，done/total 可为 0 表示不确定进度。
using ProgressFn = void (*)(const char* stage, int64_t done, int64_t total, void* user);

// 主入口。只读；不修改任何源文件。
AnalyzeResult Analyze(const Options& options, ProgressFn progress = nullptr, void* user = nullptr);

/* ============================ 序列化 ============================ */

// 紧凑文本：适合粘贴给 AI。超预算时注明省略数量与完整结果位置，不静默丢弃。
std::string ToCompactText(const AnalyzeResult& r);

// 完整结构化 JSON，含 schema_version。保证不输出 NaN/Inf（用 null 表示）。
std::string ToJson(const AnalyzeResult& r);

// 明细输出。返回写入的文件路径列表（detail_dir 为空时返回空）。
std::vector<std::string> WriteDetailFiles(AnalyzeResult& r);

} // namespace traj
