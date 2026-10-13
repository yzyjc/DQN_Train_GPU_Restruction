#define UNICODE
#define _UNICODE
#define wWinMain unused_tui_entry
#include "../DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp"
#undef wWinMain
#undef printf
static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    /* Narrow windows use one function per merged page: 3 fixed + 8 functions.
       Guard the final Help tab and adjacent tab-state arrays from regression. */
    g_console_width = 130;
    g_console_height = 42;
    compute_merge_layout();
    CHECK(g_merge_n == 1);
    CHECK(g_total_tabs == 11 && g_total_tabs <= MAX_TABS);
    CHECK(strcmp(g_tab_names[10], "11 Help") == 0);
    CHECK(select_training_log_directory("logs", false));
    FILE* win = fopen("logs/metric_win.bin", "wb");
    FILE* eval = fopen("logs/metric_eval.bin", "wb");
    auto h = metric_log_header(4); fwrite(&h, sizeof(h), 1, win);
    h = metric_log_header(7); fwrite(&h, sizeof(h), 1, eval);
    BRRecord row{};
    for (int ep = 1000; ep <= 20000; ep += 1000) for (int mode = 0; mode < 3; ++mode) {
        row = {ep, mode, {(float)mode, 50, 25, 25}};
        fwrite(&row, metric_log_record_size(4), 1, win);
        row = {ep, mode, {1, 2, .5f, (float)mode, 50, 25, 25}};
        fwrite(&row, metric_log_record_size(7), 1, eval);
    }
    fclose(win); fclose(eval); read_eval();
    CHECK(g_eval_episode == 20000 && g_eval_phase == 1 && g_eval_depth == 2);
    // Different unfinished suffix lengths previously misaligned both independent parsers.
    win = fopen("logs/metric_win.bin", "ab"); fwrite(&row, 3, 1, win); fclose(win);
    eval = fopen("logs/metric_eval.bin", "ab"); fwrite(&row, 5, 1, eval); fclose(eval);
    read_eval();
    for (int mode = 0; mode < 3; ++mode) {
        CHECK(g_win_count[mode] == 20 && g_ep_history[mode][0] == 1000);
        CHECK(g_ep_history[mode][19] == 20000 && g_win_history[mode][19] == 50);
    }
    CHECK(g_eval_episode == 20000 && g_eval_phase == 1 && g_eval_depth == 2);
    CHECK(MoveFileA("logs/metric_win.bin", "logs/win.saved"));
    read_eval(); CHECK(g_win_count[0] == 20 && g_eval_episode == 20000); // unavailable != empty
    win = fopen("logs/metric_win.bin", "wb"); h = metric_log_header(4); h.version = 2;
    fwrite(&h, sizeof(h), 1, win); fclose(win); read_eval(); CHECK(g_win_count[0] == 20);
    win = fopen("logs/metric_win.bin", "wb"); h.version = 1; fwrite(&h, sizeof(h), 1, win);
    row = {30000, 0, {NAN, 50, 25, 25}}; fwrite(&row, metric_log_record_size(4), 1, win);
    row = {30000, 0, {0, INFINITY, 25, 25}}; fwrite(&row, metric_log_record_size(4), 1, win);
    row = {30000, 0, {0, 75, 10, 15}}; fwrite(&row, metric_log_record_size(4), 1, win); fclose(win);
    read_eval(); CHECK(g_win_count[0] == 1 && g_eval_episode == 30000 && g_eval_win == 75);
    CHECK(g_win_count[1] == 0 && g_win_count[2] == 0);
    win = fopen("logs/metric_win.bin", "wb"); fwrite(&h, sizeof(h), 1, win); fclose(win);
    read_eval(); CHECK(g_win_count[0] == 0 && g_eval_episode == 0); // truly empty clears stale summary
    printf("ARCHITECTURE UI: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
