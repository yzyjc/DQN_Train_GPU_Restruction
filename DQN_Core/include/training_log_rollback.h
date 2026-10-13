#pragma once
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <io.h>
#include <string>
#include <vector>
#include <limits.h>
#include "binary_log_timeline.h"
#include "skill_metric_format.h"

// Called before any writer opens. Prepare every replacement first, keep backups,
// and never let a partial rollback be mistaken for a successful resume.
inline bool rollback_training_logs(const char* dir, int completed_episodes, long long next_step) {
    if (!dir || completed_episodes < 0 || next_step < 0) return false;
    struct Spec { const char* name; int fields; bool zero_based; bool training_step; uint32_t magic; };
    std::vector<Spec> specs = {
        {"game.dat",0,true,false,0}, {"train.dat",0,true,true,0},
        {"eval.dat",0,false,false,0}, {"minimax_games.dat",0,false,false,0},
        {"reward_episode.dat",0,false,false,0}
    };
    // Rollback and repair must agree on both record width and EP base.
    for (const auto& s : METRIC_LOG_SCHEMAS)
        specs.push_back({s.name, (int)s.fields, s.zero_based, false, METRIC_LOG_MAGIC});
    specs.push_back({"metric_skill_v2.bin",7,false,false,0x4B494C53});
    specs.push_back({"metric_skill_v3.bin",7,false,false,0x4B494C53});
    specs.push_back({SKILL_METRIC_FILENAME,SKILL_METRIC_FIELDS,false,false,SKILL_MAGIC});
    struct Replacement { std::string path, temp, backup; };
    std::vector<Replacement> pending;
    const std::string root = std::string(dir) + "/";
    const std::string journal = root + "rollback.pending";
    // An interrupted commit needs inspection/recovery, never silent training on mixed logs.
    if (GetFileAttributesA(journal.c_str()) != INVALID_FILE_ATTRIBUTES) return false;
    const std::string suffix = ".rollback_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64());
    bool ok = true;
    for (const auto& spec : specs) {
        Replacement item{root + spec.name, root + spec.name + suffix + ".tmp", root + spec.name + suffix + ".bak"};
        DWORD attrs = GetFileAttributesA(item.path.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND) continue;
        FILE* src = fopen(item.path.c_str(), "rb");
        if (!src) { ok = false; break; }
        FILE* dst = fopen(item.temp.c_str(), "wb");
        if (!dst) { fclose(src); ok = false; break; }
        if (spec.fields) {
            uint32_t h[6] = {};
            if (fread(h, sizeof(h), 1, src) != 1 || h[0] != spec.magic ||
                h[1] != (uint32_t)spec.fields || h[2] > 1 || fwrite(h, sizeof(h), 1, dst) != 1) ok = false;
            if (ok && spec.magic == SKILL_MAGIC) {
                SkillMetricHeader skill_header{};
                memcpy(&skill_header, h, sizeof(h));
                if (!skill_metric_header_valid(skill_header)) ok = false;
            }
            // GPT6-Astra: memory evaluation already has 14 fields (64 bytes).
            // Size from the schema instead of overflowing the old 8-field buffer.
            std::vector<unsigned char> row(8 + spec.fields * sizeof(float));
            const size_t stride = 8 + spec.fields * sizeof(float);
            // Legacy loss/epsilon reused ep1 after pretraining. Only the last
            // timeline belongs to the formal checkpoint being resumed.
            if (ok && (strcmp(spec.name, "metric_loss.bin") == 0 ||
                       strcmp(spec.name, "metric_epsilon.bin") == 0)) {
                _fseeki64(src, 0, SEEK_END);
                int64_t start = binary_episode_start(src, (int)stride, _ftelli64(src));
                if (start < 0 || _fseeki64(src, start, SEEK_SET) != 0) ok = false;
            }
            while (ok) {
                size_t got = fread(row.data(), 1, stride, src);
                if (got != stride) { if (ferror(src)) ok = false; break; }
                int32_t ep; memcpy(&ep, row.data(), sizeof(ep));
                bool keep = ep >= 0 && (spec.zero_based ? ep < completed_episodes : ep <= completed_episodes);
                if (keep && fwrite(row.data(), 1, stride, dst) != stride) ok = false;
            }
        } else {
            char line[4096];
            while (ok && fgets(line, sizeof(line), src)) {
                if (!strchr(line, '\n')) { // Discard an incomplete tail; reject overlong complete lines.
                    if (!feof(src)) ok = false;
                    break;
                }
                long long index;
                if (sscanf(line, "%lld", &index) != 1) {
                    if (fputs(line, dst) < 0) ok = false;
                    continue;
                }
                bool keep = index >= 0 && (spec.training_step ? index < next_step :
                    (spec.zero_based ? index < completed_episodes : index <= completed_episodes));
                if (keep && fputs(line, dst) < 0) ok = false;
            }
            if (ferror(src)) ok = false;
        }
        fclose(src);
        if (fflush(dst) != 0 || _commit(_fileno(dst)) != 0) ok = false;
        if (fclose(dst) != 0) ok = false;
        if (!ok) { remove(item.temp.c_str()); break; }
        pending.push_back(item);
    }
    if (!ok) { for (const auto& item : pending) remove(item.temp.c_str()); return false; }
    if (pending.empty()) return true;
    FILE* manifest = fopen(journal.c_str(), "wb");
    if (!manifest) { for (const auto& item : pending) remove(item.temp.c_str()); return false; }
    for (const auto& item : pending)
        if (fprintf(manifest, "%s\t%s\t%s\n", item.path.c_str(), item.backup.c_str(), item.temp.c_str()) < 0) ok = false;
    if (fflush(manifest) != 0 || _commit(_fileno(manifest)) != 0) ok = false;
    if (fclose(manifest) != 0) ok = false;
    for (const auto& item : pending) {
        if (!ok) break;
        if (!CopyFileA(item.path.c_str(), item.backup.c_str(), TRUE)) ok = false;
    }
    size_t committed = 0;
    if (ok) for (; committed < pending.size(); ++committed) {
        const auto& item = pending[committed];
        if (!MoveFileExA(item.temp.c_str(), item.path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            ok = false; break;
        }
    }
    if (!ok) {
        bool restored = true;
        for (size_t i = 0; i < committed; ++i) {
            const auto& item = pending[i];
            if (!CopyFileA(item.backup.c_str(), item.path.c_str(), FALSE)) restored = false;
        }
        for (const auto& item : pending) remove(item.temp.c_str());
        if (restored) remove(journal.c_str());
        return false;
    }
    // Backups remain recoverable and are never overwritten by the next rollback.
    return remove(journal.c_str()) == 0;
}
