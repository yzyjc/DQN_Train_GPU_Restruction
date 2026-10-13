#pragma once
#include "core.h"

// Only report a completed exit save after both the model and enabled memory
// files have committed. Individual atomic writes remain retryable on failure.
template<class SaveCheckpoint>
bool save_checkpoint_and_memory(SaveCheckpoint save_checkpoint) {
    if (!save_checkpoint()) return false;
    if (!g_hot_config.memory_enabled) return true;
    return memory_save_cache_perspective("memory_cache_b.dat", 0) &&
        memory_save_cache_perspective("memory_cache_w.dat", 1) &&
        memory_save_kb_perspective("memory_kb_b.dat", 0) &&
        memory_save_kb_perspective("memory_kb_w.dat", 1);
}

// Only the training owner calls this at a completed-episode boundary. IPC stays
// live while saving is paused; another SAVE_AND_EXIT requests a retry. No new
// games, frees, or phase transitions may occur while a usable model is unsaved.
template<class Save, class Pump>
bool save_before_exit(Save save, Pump pump) {
    InterlockedExchange(&g_shutdown_state, 1);
    for (;;) {
        if (g_training_fault) {
            InterlockedExchange(&g_shutdown_state, 5);
            return false;
        }
        if (InterlockedCompareExchange(&g_shutdown_state, 2, 1) == 1) {
            if (save()) {
                InterlockedExchange(&g_shutdown_state, 4);
                return true;
            }
            // Also disable watchdog recovery for a failed automatic phase-end
            // save. Set the retry-needed state AFTER requesting shutdown so the
            // request callback cannot accidentally trigger an immediate retry.
            if (g_trainer_request_exit_cb) g_trainer_request_exit_cb();
            InterlockedExchange(&g_shutdown_state, 3);
            printf("[EXIT] SAVE FAILED. Training paused with model retained in memory. Fix storage and send SAVE_AND_EXIT again.\n");
            fflush(stdout);
        }
        pump();
    }
}
