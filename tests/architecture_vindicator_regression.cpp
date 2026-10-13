#define main unused_vindicator_entry
#include "../Log_Vindicator/Log_Vindicator.cpp"
#undef main
static int failures = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)
int main() {
    // A daemon pass must return a diagnostic, survive and retry successfully.
    CHECK(run_once("missing", true) == 1);
    CHECK(run_once("missing", false) == 1);
    CHECK(audit_structure_only("missing") == 1);
    CreateDirectoryA("missing", nullptr);
    CHECK(run_once("missing", true) == 0 && audit_structure_only("missing") == 0);
    FILE* f = fopen("not_a_directory", "wb"); fclose(f);
    CHECK(run_once("not_a_directory", true) == 1 && audit_structure_only("not_a_directory") == 1);
    CHECK(select_training_log_directory("logs", true));
    CHECK(RemoveDirectoryA("logs/pretrain"));
    CHECK(run_once(active_training_log_dir("logs"), true) == 1);
    CreateDirectoryA("logs/pretrain", nullptr);
    CHECK(run_once(active_training_log_dir("logs"), true) == 0);
    printf("ARCHITECTURE VINDICATOR: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
