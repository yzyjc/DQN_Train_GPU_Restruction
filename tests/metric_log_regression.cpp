#define _CRT_SECURE_NO_WARNINGS
#include "metric_log_writer.h"
#include "metric_eval_reader.h"
#include <vector>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)
static std::vector<char> bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    std::vector<char> out; char block[1024]; size_t n;
    while ((n = fread(block, 1, sizeof(block), f))) out.insert(out.end(), block, block + n);
    fclose(f); return out;
}
static void write_fixture(const char* path, uint32_t version, uint32_t fields = 1) {
    FILE* f = fopen(path, "wb"); auto h = metric_log_header(fields); h.version = version;
    BRRecord r = {100, 0, {.5f}};
    fwrite(&h, sizeof(h), 1, f); fwrite(&r, metric_log_record_size(fields), 1, f); fclose(f);
}
int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    static_assert(sizeof(BRFile::record_count) == 8, "Long-running logs need 64-bit row counts");
    CHECK(metric_log_schema("logs/pretrain/metric_steps.bin")->zero_based);
    CHECK(!metric_log_schema("logs/metric_skill_v2.bin"));
    for (uint32_t version : {0u, 1u}) {
        write_fixture("reader.bin", version);
        BRFile file{}; strcpy(file.path, "reader.bin"); BRRecord out[4]{};
        CHECK(br_read_tail_checked(&file, out, 4, 1) == 1 && out[0].episode == 100);
        const auto offset = file.offset;
        BRRecord r = {101, 0, {.25f}};
        FILE* f = fopen(file.path, "ab"); fwrite(&r, 3, 1, f); fclose(f);
        CHECK(br_read_tail_checked(&file, out, 4, 1) == 1 && file.offset == offset);
        CHECK(br_read_incremental(&file, out, 4) == 0 && file.offset == offset);
        f = fopen(file.path, "ab"); fwrite((char*)&r + 3, metric_log_record_size(1) - 3, 1, f); fclose(f);
        CHECK(br_read_incremental(&file, out, 4) == 1 && out[0].episode == 101);
        // Same filename, new incompatible format: do not keep using cached stride.
        write_fixture(file.path, 2);
        const auto before = file.offset;
        CHECK(br_read_incremental(&file, out, 4) == 0 && file.offset == before);
        CHECK(br_read_tail_checked(&file, out, 4, 1) == -1);
        const auto original = bytes(file.path);
        CHECK(!metric_log_open_append(file.path, 1) && bytes(file.path) == original);
    }
    write_fixture("metric_epsilon_episode.bin", 1);
    FILE* f = fopen("metric_epsilon_episode.bin", "ab"); fputc(42, f); fclose(f);
    CHECK(metric_log_prepare_append("metric_epsilon_episode.bin", 1));
    CHECK(bytes("metric_epsilon_episode.bin").size() == sizeof(MetricLogHeader) + metric_log_record_size(1));
    const auto original = bytes("metric_epsilon_episode.bin");
    f = _fsopen("metric_epsilon_episode.bin", "rb", _SH_DENYRW);
    CHECK(f && !metric_log_prepare_append("metric_epsilon_episode.bin", 1));
    if (f) fclose(f);
    CHECK(bytes("metric_epsilon_episode.bin") == original);
    CHECK(!metric_log_open_append("metric_epsilon_episode.bin", 3)); // registry guards field drift
    f = metric_log_open_append("new.bin", 1); CHECK(f);
    BRFile file{}; strcpy(file.path, "new.bin"); BRRecord out[4]{};
    CHECK(br_read_tail_checked(&file, out, 4, 1) == 0); // reader can coexist with writer
    if (f) fclose(f);
    strcpy(file.path, "absent.bin"); CHECK(br_read_tail_checked(&file, out, 4) == -1);
    f = fopen("partial_header.bin", "wb"); fputs("partial", f); fclose(f);
    CHECK(metric_log_prepare_append("partial_header.bin", 1));
    CHECK(GetFileAttributesA("partial_header.bin") == INVALID_FILE_ATTRIBUTES);
    WIN32_FIND_DATAA found{}; HANDLE find = FindFirstFileA("partial_header.bin.corrupt_*", &found);
    CHECK(find != INVALID_HANDLE_VALUE);
    if (find != INVALID_HANDLE_VALUE) { CHECK(bytes(found.cFileName).size() == 7); FindClose(find); }
    BRRecord win = {1000, 0, {0, 50, 25, 25}};
    CHECK(metric_win_row_valid(win));
    for (float mode : {NAN, INFINITY, 1.5f, 3.0f}) { win.fields[0] = mode; CHECK(!metric_win_row_valid(win)); }
    printf("METRIC LOG: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
