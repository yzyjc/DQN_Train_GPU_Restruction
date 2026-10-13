#pragma once
#include "metric_log_format.h"
#include <windows.h>
#include <stdio.h>
#include <io.h>
#include <share.h>
#include <errno.h>
#include <string>

// Trainer-only lifecycle operation: call BEFORE opening the append handle, with
// the single-writer lock held. Observers must never truncate or quarantine logs.
inline bool metric_log_prepare_append(const char* path, uint32_t fields) {
    if (!path || !fields || fields > METRIC_LOG_MAX_FIELDS) return false;
    const auto* schema = metric_log_schema(path);
    if (schema && schema->fields != fields) return false;
    FILE* f = _fsopen(path, "rb+", _SH_DENYWR);
    if (!f) return errno == ENOENT; // Sharing/permission failure is NOT a new file.
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    const int64_t size = _ftelli64(f);
    if (size < 0) { fclose(f); return false; }
    if (size == 0) return fclose(f) == 0;
    MetricLogHeader h{};
    bool known_layout = false;
    if (size >= (int64_t)sizeof(h)) {
        if (_fseeki64(f, 0, SEEK_SET) != 0 || fread(&h, sizeof(h), 1, f) != 1) {
            fclose(f); return false;
        }
        // A future version is not corruption. Leave every byte/name untouched
        // and refuse append rather than mixing old records into a newer format.
        if (h.magic == METRIC_LOG_MAGIC && h.version > METRIC_LOG_VERSION) {
            fclose(f);
            printf("[LOG] Unsupported metric version %u; append refused: %s\n", h.version, path);
            return false;
        }
        known_layout = metric_log_header_valid(h, fields);
    }
    if (known_layout) {
        const int64_t stride = (int64_t)metric_log_record_size(fields);
        const int64_t aligned = sizeof(h) + (size - sizeof(h)) / stride * stride;
        bool ok = aligned == size || _chsize_s(_fileno(f), aligned) == 0;
        if (fclose(f) != 0) ok = false;
        return ok;
    }
    if (fclose(f) != 0) return false;
    // Preserve even a partial header for diagnosis. No overwrite of backups.
    FILETIME ft; GetSystemTimeAsFileTime(&ft);
    const uint64_t stamp = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    const std::string quarantine = std::string(path) + ".corrupt_" + std::to_string(stamp);
    if (!MoveFileExA(path, quarantine.c_str(), MOVEFILE_WRITE_THROUGH)) return false;
    printf("[LOG] Invalid metric layout preserved as: %s\n", quarantine.c_str());
    return true;
}

inline FILE* metric_log_open_append(const char* path, uint32_t fields) {
    if (!metric_log_prepare_append(path, fields)) return nullptr;
    FILE* f = _fsopen(path, "ab", _SH_DENYWR);
    if (!f) return nullptr;
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return nullptr; }
    const int64_t size = _ftelli64(f);
    if (size < 0) { fclose(f); return nullptr; }
    if (size == 0) {
        const auto h = metric_log_header(fields);
        if (fwrite(&h, sizeof(h), 1, f) != 1 || fflush(f) != 0) {
            fclose(f); return nullptr;
        }
    }
    return f;
}
