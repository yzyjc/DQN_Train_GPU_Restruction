// Read-only metric access. Complete-record boundaries are always relative to the
// header, never EOF (the trainer may be partway through an append).
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "metric_log_format.h"
#include <limits.h>
#include "binary_log_timeline.h"

#ifdef _WIN32
#include <windows.h>
#endif

// BR names retained for callers; the disk contract is owned by metric_log_format.
#define BR_MAGIC       METRIC_LOG_MAGIC
#define BR_VERSION     METRIC_LOG_VERSION
#define BR_MAX_FIELDS  METRIC_LOG_MAX_FIELDS
#define BR_MAX_FILES   16
#define BR_MAX_RECORDS 8192
using BRHeader = MetricLogHeader;

typedef struct {
    int32_t episode;
    int32_t step;
    float   fields[BR_MAX_FIELDS];
} BRRecord;

typedef struct {
    char name[64];
    char path[256];
    int  field_count;
    int64_t file_size;   /* BUG-024: 64 位偏移，支持超过 2GB 文件 */
    int64_t record_count;
    int64_t offset;      /* 增量读取偏移 */
    int32_t tail_episode;/* 磁盘最后一个完整记录的 episode，用于识别截断/回滚 */
    uint64_t identity;   /* Windows 创建时间；同名文件被重建时重置 reader */
} BRFile;

/* ========== 内部辅助 ========== */

static inline int _br_parse_header(FILE* f, BRFile* bf) {
    BRHeader hdr{};
    if (!f || !bf || fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        !metric_log_header_valid(hdr)) return -1;
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    const int64_t size = _ftelli64(f);
    if (size < (int64_t)sizeof(hdr)) return -1;
    const int64_t stride = (int64_t)metric_log_record_size(hdr.field_count);
    bf->field_count = (int)hdr.field_count;
    bf->file_size = size;
    bf->identity = 0;
    bf->record_count = (size - sizeof(hdr)) / stride;
    bf->tail_episode = 0;
    if (bf->record_count > 0) {
        const int64_t tail = sizeof(hdr) + (bf->record_count - 1) * stride;
        if (_fseeki64(f, tail, SEEK_SET) != 0 ||
            fread(&bf->tail_episode, sizeof(bf->tail_episode), 1, f) != 1) return -1;
    }
    bf->offset = sizeof(hdr);
    return 0;
}

/* ========== 公共 API ========== */

/* 扫描 dir\metric_*.bin，返回发现的文件数 */
static inline int br_scan(const char* dir, BRFile* files, int max_files) {
    int n = 0;
#ifdef _WIN32
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "%smetric_*.bin", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        if (n >= max_files) break;
        char path[256];
        snprintf(path, sizeof(path), "%s%s", dir, fd.cFileName);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        if (_br_parse_header(f, &files[n]) == 0) {
            files[n].identity = ((uint64_t)fd.ftCreationTime.dwHighDateTime << 32) |
                (uint64_t)fd.ftCreationTime.dwLowDateTime;
            snprintf(files[n].name, 64, "%s", fd.cFileName);
            snprintf(files[n].path, 256, "%s", path);
            n++;
        }
        fclose(f);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#endif
    return n;
}

/* 按文件名查找，返回索引，-1=未找到 */
static inline int br_find(BRFile* files, int count, const char* name) {
    for (int i = 0; i < count; i++)
        if (strcmp(files[i].name, name) == 0) return i;
    return -1;
}

/* 增量读取新记录（从 offset 到文件末尾），返回读取数 */
static inline int br_read_incremental(BRFile* bf, BRRecord* out, int max_records) {
    if (!bf || !out || max_records <= 0 || bf->field_count <= 0 || bf->field_count > BR_MAX_FIELDS) return 0;
    FILE* f = fopen(bf->path, "rb");
    if (!f) return 0;
    BRFile disk{};
    if (_br_parse_header(f, &disk) != 0 || disk.field_count != bf->field_count) {
        fclose(f); return 0; // Owner must rescan a replaced/schema-changed file.
    }
    const int64_t size = disk.file_size;
    bf->file_size = size; bf->record_count = disk.record_count;
    if (bf->offset < (int64_t)sizeof(BRHeader)) bf->offset = (int64_t)sizeof(BRHeader);
    /* BUG-028: offset > size 表示文件被截断（rollback），重置到 header 之后；
       offset == size 表示没有新数据，直接返回 0（不重置，避免重复读取历史） */
    if (bf->offset > size) {
        bf->offset = (int64_t)sizeof(BRHeader);
        if (bf->offset >= size) { fclose(f); return 0; }
    } else if (bf->offset >= size) {
        fclose(f); return 0;
    }
    if (_fseeki64(f, bf->offset, SEEK_SET) != 0) { fclose(f); return 0; }
    const int64_t read_start = bf->offset;
    int rec_size = 8 + bf->field_count * 4;
    int count = 0;
    while (count < max_records && _ftelli64(f) + rec_size <= size) {
        if (fread(&out[count], rec_size, 1, f) != 1) break;
        count++;
    }
    bf->offset = read_start + (int64_t)count * rec_size;
    if (count > 0) bf->tail_episode = out[count - 1].episode;
    fclose(f);
    return count;
}

/* 全量读取整个文件，返回记录数 */
static inline int br_read_all(const char* path, BRRecord* out, int max_records) {
    if (!path || !out || max_records <= 0) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    BRFile bf;
    if (_br_parse_header(f, &bf) != 0) { fclose(f); return 0; }
    _fseeki64(f, (int64_t)sizeof(BRHeader), SEEK_SET);
    int rec_size = 8 + bf.field_count * 4;
    int count = 0;
    while (count < max_records && _ftelli64(f) + rec_size <= bf.file_size) {
        if (fread(&out[count], rec_size, 1, f) != 1) break;
        count++;
    }
    fclose(f);
    return count;
}

/* 首次展示用：覆盖整个文件等距采样，并把增量偏移推进到文件末尾。 */
static inline int br_read_sampled_all(BRFile* bf, BRRecord* out, int max_records) {
    if (!bf || !out || max_records <= 0) return 0;
    FILE* f = fopen(bf->path, "rb");
    if (!f) return 0;
    BRFile disk;
    if (_br_parse_header(f, &disk) != 0) { fclose(f); return 0; }
    const int64_t total = disk.record_count;
    const int take = (total < max_records) ? (int)total : max_records;
    const int rec_size = 8 + disk.field_count * 4;
    bf->field_count = disk.field_count;
    bf->file_size = disk.file_size;
    bf->record_count = disk.record_count;
    for (int i = 0; i < take; i++) {
        int64_t index = (take <= 1) ? 0 : ((int64_t)i * (total - 1) / (take - 1));
        _fseeki64(f, (int64_t)sizeof(BRHeader) + index * rec_size, SEEK_SET);
        if (fread(&out[i], rec_size, 1, f) != 1) {
            bf->offset = (int64_t)sizeof(BRHeader) + index * rec_size;
            bf->tail_episode = i ? out[i-1].episode : -1;
            fclose(f); return i;
        }
    }
    bf->field_count = disk.field_count;
    bf->file_size = disk.file_size;
    bf->record_count = disk.record_count;
    bf->offset = (int64_t)sizeof(BRHeader) + (int64_t)disk.record_count * rec_size;
    bf->tail_episode = disk.tail_episode;
    fclose(f);
    return take;
}

/* 读取最后 max_records 条连续记录，用于启动时恢复近期高密度曲线。 */
// -1 means unavailable/invalid; 0 means a valid empty file. Dashboard needs this
// distinction to preserve its last valid snapshot across transient read failures.
static inline int br_read_tail_checked(BRFile* bf, BRRecord* out, int max_records, uint32_t expected_fields = 0) {
    if (!bf || !out || max_records <= 0) return -1;
    FILE* f = fopen(bf->path, "rb"); if (!f) return -1;
    BRFile disk;
    if (_br_parse_header(f, &disk) != 0 || (expected_fields && disk.field_count != expected_fields)) { fclose(f); return -1; }
    int take = disk.record_count < max_records ? (int)disk.record_count : max_records;
    int rec_size = 8 + disk.field_count * 4;
    int64_t start = (int64_t)sizeof(BRHeader) + (int64_t)(disk.record_count - take) * rec_size;
    if (_fseeki64(f, start, SEEK_SET) != 0) { fclose(f); return -1; }
    int got = 0;
    while (got < take && fread(&out[got], rec_size, 1, f) == 1) got++;
    if (got != take) { fclose(f); return -1; }
    bf->field_count = disk.field_count; bf->file_size = disk.file_size;
    bf->record_count = disk.record_count; bf->offset = start + (int64_t)got * rec_size;
    bf->tail_episode = disk.tail_episode;
    // Old logs may contain Phase0 followed by formal ep1 in the same file.
    // Report only the newest contiguous timeline, never average both phases.
    int begin = 0, previous = -1;
    for (int i = 0; i < got; ++i) if (out[i].episode >= 0) {
        if (previous > out[i].episode) begin = i;
        previous = out[i].episode;
    }
    if (begin) { memmove(out, out + begin, (got - begin) * sizeof(*out)); got -= begin; }
    fclose(f); return got;
}

// Compatibility API: historical callers treat both empty/unavailable as zero.
static inline int br_read_tail(BRFile* bf, BRRecord* out, int max_records) {
    const int n = br_read_tail_checked(bf, out, max_records);
    return n < 0 ? 0 : n;
}

static inline bool br_seek_episode_window(BRFile* bf, int min_episode) {
    if (!bf) return false;
    FILE* f = fopen(bf->path, "rb"); if (!f) return false;
    BRFile disk = {};
    if (_br_parse_header(f, &disk) != 0) { fclose(f); return false; }
    int64_t start = binary_episode_start(f, 8 + disk.field_count * 4, disk.file_size, min_episode);
    fclose(f);
    if (start < 0) return false;
    bf->field_count = disk.field_count; bf->record_count = disk.record_count;
    bf->file_size = disk.file_size; bf->tail_episode = disk.tail_episode; bf->offset = start;
    return true;
}

/* 按 episode 流式筛选读取。旧实现先按日志总记录数分配整文件缓冲，
   长期训练后仅回放一局也可能申请数 GB 内存并导致 TUI 退出。 */
static inline int br_read_episode(const char* path, int episode, BRRecord* out, int max_records) {
    if (!out || max_records <= 0) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    BRFile bf;
    if (_br_parse_header(f, &bf) != 0) { fclose(f); return 0; }
    int rec_size = 8 + bf.field_count * 4;
    _fseeki64(f, (int64_t)sizeof(BRHeader), SEEK_SET);
    int n = 0;
    BRRecord rec{};
    while (n < max_records && _ftelli64(f) + rec_size <= bf.file_size) {
        if (fread(&rec, rec_size, 1, f) != 1) break;
        if (rec.episode == episode) out[n++] = rec;
    }
    fclose(f);
    return n;
}

/* 读取文件头信息（不读记录） */
static inline int br_read_header(const char* path, int* out_field_count, int* out_record_count) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    BRFile bf;
    int r = _br_parse_header(f, &bf);
    fclose(f);
    if (r != 0) return -1;
    if (out_field_count) *out_field_count = bf.field_count;
    // Legacy int API saturates; new code uses BRFile::record_count (64 bit).
    if (out_record_count) *out_record_count = bf.record_count > INT_MAX ? INT_MAX : (int)bf.record_count;
    return 0;
}
