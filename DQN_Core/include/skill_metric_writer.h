#pragma once
#include "skill_metric_format.h"
#include <stdio.h>
#include <io.h>
#include <share.h>
#include <errno.h>
#include <process.h>
#include <time.h>

// Single writer, read sharing allowed. Never append v4 results to an older/future
// file. Partial RECORD tails are repairable. An incompatible complete file is
// preserved beside the new file instead of permanently blocking every later
// evaluation at the same checkpoint boundary.
inline bool append_skill_metric(const char* path, const SkillMetricRow& row) {
    if (!path || !skill_metric_row_valid(row)) return false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        FILE* f = _fsopen(path, "rb+", _SH_DENYWR);
        if (!f && errno == ENOENT) f = _fsopen(path, "wb+", _SH_DENYWR);
        if (!f) return false;
        setvbuf(f, nullptr, _IONBF, 0); // failed writes cannot be flushed again after rollback
        bool ok = _fseeki64(f, 0, SEEK_END) == 0;
        int64_t size = ok ? _ftelli64(f) : -1;
        SkillMetricHeader h{SKILL_MAGIC, SKILL_METRIC_FIELDS, 1, SKILL_SUITE_VERSION, {0, 0}};
        bool incompatible = false;
        bool future = false;
        if (size == 0) {
            ok = fwrite(&h, sizeof(h), 1, f) == 1;
            if (!ok) _chsize_s(_fileno(f), 0);
            size = sizeof(h);
        } else {
            ok = size >= (int64_t)sizeof(h) && _fseeki64(f, 0, SEEK_SET) == 0 &&
                fread(&h, sizeof(h), 1, f) == 1;
            future = ok && h.magic == SKILL_MAGIC &&
                (h.version > 1 || h.suite > SKILL_SUITE_VERSION);
            incompatible = ok && (!skill_metric_header_valid(h) ||
                h.suite != SKILL_SUITE_VERSION || h.fields != SKILL_METRIC_FIELDS);
            ok = ok && !incompatible;
        }
        /* A newer producer owns this pathname.  Preserve it byte-for-byte and
           refuse to rename or replace data whose schema we do not understand. */
        if (future) {
            fclose(f);
            return false;
        }
        if (incompatible && attempt == 0) {
            fclose(f);
            char backup[1024];
            snprintf(backup, sizeof(backup), "%s.incompatible_%lld_%d.bak", path,
                (long long)time(nullptr), _getpid());
            if (rename(path, backup) != 0) return false;
            continue; // Create a clean current-schema file on the second attempt.
        }
        const int64_t end = ok ? sizeof(h) + (size - sizeof(h)) / sizeof(row) * sizeof(row) : -1;
        if (ok && end != size) ok = _chsize_s(_fileno(f), end) == 0;
        if (ok) ok = _fseeki64(f, end, SEEK_SET) == 0;
        if (ok) {
            ok = fwrite(&row, sizeof(row), 1, f) == 1 && fflush(f) == 0 && _commit(_fileno(f)) == 0;
            if (!ok) { clearerr(f); _chsize_s(_fileno(f), end); }
        }
        if (fclose(f) != 0) ok = false;
        return ok;
    }
    return false;
}
