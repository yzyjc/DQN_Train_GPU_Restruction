#pragma once
#include "metric_log_reader.h"
#include <cmath>
#include <limits.h>

// Validate before converting float metadata to int: NaN placeholders and damaged
// records are not modes, phases or chart points. They must never enter the UI.
inline bool metric_nonnegative_integer(float value) {
    return std::isfinite(value) && value >= 0 && (double)value <= INT_MAX &&
        std::floor(value) == value;
}
inline bool metric_eval_mode(float mode) {
    return metric_nonnegative_integer(mode) && mode <= 2;
}
inline bool metric_eval_rates(const float* rates) {
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(rates[i]) || rates[i] < 0 || rates[i] > 100) return false;
    return std::fabs(rates[0] + rates[1] + rates[2] - 100.0f) < 0.1f;
}
inline bool metric_win_row_valid(const BRRecord& row) {
    return row.episode > 0 && metric_eval_mode(row.fields[0]) && metric_eval_rates(row.fields + 1);
}
inline bool metric_eval_row_valid(const BRRecord& row) {
    return row.episode > 0 && metric_nonnegative_integer(row.fields[0]) && row.fields[0] <= 5 &&
        metric_nonnegative_integer(row.fields[1]) && row.fields[1] <= 25 && std::isfinite(row.fields[2]) &&
        row.fields[2] >= 0 && row.fields[2] <= 1 && metric_eval_mode(row.fields[3]) &&
        metric_eval_rates(row.fields + 4);
}
