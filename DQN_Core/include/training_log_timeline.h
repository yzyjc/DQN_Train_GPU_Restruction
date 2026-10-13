#pragma once
#include <windows.h>
#include <stdio.h>
#include <io.h>
#include <string>

// Checkpoints choose the writer timeline; this atomic marker only selects what
// observers display. Both timelines can start at ep1 without sharing records.
inline std::string training_log_root(const char* root) {
    std::string path = root;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path += '/';
    return path;
}
inline std::string active_training_log_dir(const char* root) {
    std::string dir = training_log_root(root);
    int phase = -1;
    HANDLE f = CreateFileA((dir + "active_timeline.txt").c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f != INVALID_HANDLE_VALUE) {
        char data[64] = {}; DWORD got = 0;
        if (ReadFile(f, data, sizeof(data)-1, &got, nullptr) &&
            sscanf(data, "timeline_v1 %d", &phase) != 1) phase = -1;
        CloseHandle(f);
    }
    return phase == 0 ? dir + "pretrain/" : dir;
}
inline std::string active_training_log_path(const char* root, const char* name) {
    return active_training_log_dir(root) + name;
}
inline std::string& training_writer_dir() {
    static std::string dir = "logs/";
    return dir;
}
inline std::string dqn_log_path(const char* name) { return training_writer_dir() + name; }
inline bool select_training_log_directory(const char* root, bool pretrain) {
    const std::string base = training_log_root(root);
    const std::string dir = pretrain ? base + "pretrain/" : base;
    if (!CreateDirectoryA(base.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) return false;
    if (!CreateDirectoryA(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) return false;
    const std::string marker = base + "active_timeline.txt", temp = marker + ".tmp";
    FILE* f = fopen(temp.c_str(), "wb");
    if (!f) return false;
    bool ok = fprintf(f, "timeline_v1 %d\n", pretrain ? 0 : 1) > 0;
    if (fflush(f) != 0 || _commit(_fileno(f)) != 0) ok = false;
    if (fclose(f) != 0) ok = false;
    if (!ok || !MoveFileExA(temp.c_str(), marker.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(temp.c_str()); return false;
    }
    training_writer_dir() = dir;
    return true;
}
