// Author: GPT6-Astra. Real save/load paths; all fixtures are generated locally.
#include "core.h"
#include <memory>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
using Bytes = std::vector<unsigned char>;

static Bytes read_bytes(const char* path) {
    FILE* f = fopen(path, "rb"); REQUIRE(f);
    REQUIRE(_fseeki64(f, 0, SEEK_END) == 0);
    const auto size = _ftelli64(f); REQUIRE(size >= 0); rewind(f);
    Bytes bytes((size_t)size);
    REQUIRE(fread(bytes.data(), 1, bytes.size(), f) == bytes.size()); fclose(f);
    return bytes;
}
static void write_bytes(const char* path, const Bytes& bytes) {
    FILE* f = fopen(path, "wb"); REQUIRE(f);
    REQUIRE(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size()); REQUIRE(fclose(f) == 0);
}
static uint32_t fixture_crc(const Bytes& bytes) {
    uint32_t crc = ~0u;
    for (unsigned char b : bytes) {
        crc ^= b;
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
static void seed_memory() {
    memory_system_init();
    KBEntry& kb = g_knowledge_base[0]; memset(&kb, 0, sizeof(kb));
    for (int i = 0; i < PATCH_COUNT; ++i) kb.patch_attention[i] = 1;
    kb.sample_count = kb.total = kb.wins = 100;
    kb.win_rate = 1; kb.is_positive = 1; kb.timeout_threshold = 10;
    kb.avg_step = 6; kb.avg_piece_count = 7; g_kb_count = 1;
}

// Reconstruct historical wire layouts, including nonempty replay and pool data.
// No dependency on ignored diagnostic files or a particular developer's checkpoints.
static void legacy_pair(const char* path, int checkpoint_version, int memory_version) {
    Bytes body = read_bytes("current.bin"); REQUIRE(body.size() > 8); body.resize(body.size() - 8);
    memcpy(body.data(), &checkpoint_version, sizeof(checkpoint_version));
    if (checkpoint_version == 8) {
        const size_t offset = sizeof(CheckpointMeta) + sizeof(CheckpointTrainingStateV6);
        body.erase(body.begin() + offset, body.begin() + offset +
            sizeof(CheckpointTrainingState) - sizeof(CheckpointTrainingStateV6));
    }
    /* GPT5-Sol: V11 added SideBalanceController between training state and
       replay.  Historical V8/V9 fixtures must remove it after first shrinking
       the V8 training-state block, otherwise replay begins at a false offset. */
    if (checkpoint_version < 11) {
        const size_t training_size = checkpoint_version >= 9
            ? sizeof(CheckpointTrainingState) : sizeof(CheckpointTrainingStateV6);
        const size_t side_offset = sizeof(CheckpointMeta) + training_size;
        body.erase(body.begin() + side_offset,
            body.begin() + side_offset + sizeof(SideBalanceController));
    }
    const uint32_t crc = fixture_crc(body);
    const Bytes original = read_bytes("current.bin");
    body.insert(body.end(), original.end() - 8, original.end());
    memcpy(body.data() + body.size() - 4, &crc, 4);
    write_bytes(path, body);
    Bytes mem = read_bytes("current.bin.mem");
    memcpy(mem.data() + 4, &memory_version, 4);
    if (memory_version == 10) mem.erase(mem.begin() + 12, mem.begin() + 16);
    else memcpy(mem.data() + 12, &crc, 4);
    char side[256]; snprintf(side, sizeof(side), "%s.mem", path); write_bytes(side, mem);
}

static void reject_unchanged(DQNAgent* live, const char* path) {
    live->q_net.b4[0] = .125f;
    const auto before = std::make_unique<DQNAgent>(*live);
    const KBEntry kb = g_knowledge_base[0];
    const int kb_count = g_kb_count, cache_count = g_cache_count;
    const float lr = g_learning_rate; const time_t start = g_start_time;
    int p = -7, ep = -8, depth = -9;
    REQUIRE(!load_checkpoint(live, &p, &ep, &depth, path, 1));
    REQUIRE(memcmp(live, before.get(), sizeof(*live)) == 0);
    REQUIRE(g_kb_count == kb_count && g_cache_count == cache_count);
    REQUIRE(memcmp(&kb, &g_knowledge_base[0], sizeof(kb)) == 0);
    REQUIRE(g_learning_rate == lr && g_start_time == start);
    REQUIRE(p == -7 && ep == -8 && depth == -9);
}
static void restore(DQNAgent* live, const char* path, int version, int kb_count, float weight) {
    int p = 0, ep = 0, depth = 0;
    REQUIRE(load_checkpoint(live, &p, &ep, &depth, path, 1));
    REQUIRE(p == 2 && ep == 31000 && depth == 2 && live->checkpoint_version == version);
    REQUIRE(g_kb_count == kb_count && live->q_net.b4[0] == weight);
    REQUIRE(live->replay_black_count == 1 && live->replay_black[0].reward == 1);
    REQUIRE(live->opponent_pool_count == 1 && live->opponent_pool[0].b4[0] == .375f);
}

int main() {
    // Refuse to reuse fixtures so a previous run cannot hide missing-directory/file cases.
    REQUIRE(CreateDirectoryA("checkpoint_memory_cases", nullptr));
    REQUIRE(SetCurrentDirectoryA("checkpoint_memory_cases"));
    auto source = std::make_unique<DQNAgent>(); init_dqn_ex(source.get(), 0);
    auto live = std::make_unique<DQNAgent>(); init_dqn_ex(live.get(), 0);
    source->q_net.b4[0] = .25f;
    source->replay_black_count = source->replay_black_idx = 1;
    Experience& e = source->replay_black[0]; e = {};
    e.state[CELL_COUNT] = e.next_state[CELL_COUNT] = 1;
    e.action = 3; e.reward = 1; e.next_state[3] = -1; e.done = 0;
    REQUIRE(validate_experience(&e));
    source->opponent_pool_count = 1; source->opponent_pool_next = 0;
    source->opponent_pool[0] = source->q_net; source->opponent_pool[0].b4[0] = .375f;
    g_learning_rate = .0001f; memory_system_init();
    REQUIRE(save_checkpoint(source.get(), 2, 31000, 2, "current.bin"));
    restore(live.get(), "current.bin", 11, 0, .25f);
    puts("PASS current V11 checkpoint/memory roundtrip");

    legacy_pair("v8.bin", 8, 10); legacy_pair("v9.bin", 9, 10); legacy_pair("v9_v11.bin", 9, 11);
    restore(live.get(), "v8.bin", 8, 0, .25f);
    restore(live.get(), "v9.bin", 9, 0, .25f);
    restore(live.get(), "v9_v11.bin", 9, 0, .25f);
    puts("PASS historical V8/V10, V9/V10 and V9/V11 pairs");

    // Missing/mismatched current memory cannot alter the live agent or memory bank.
    REQUIRE(CopyFileA("current.bin", "missing.bin", TRUE)); seed_memory();
    reject_unchanged(live.get(), "missing.bin");
    REQUIRE(CopyFileA("current.bin", "mismatch.bin", TRUE));
    Bytes bad_mem = read_bytes("current.bin.mem"); bad_mem[12] ^= 1;
    write_bytes("mismatch.bin.mem", bad_mem); reject_unchanged(live.get(), "mismatch.bin");
    Bytes old_tagged = read_bytes("v9_v11.bin.mem"); old_tagged[12] ^= 1;
    write_bytes("v9_v11.bin.mem", old_tagged); reject_unchanged(live.get(), "v9_v11.bin");
    puts("PASS missing/mismatched memory preserves live state, including tagged historical pairs");

    // The reported failure: new checkpoint commits but occupied legacy memory cannot be replaced.
    REQUIRE(CopyFileA("v9.bin", "upgrade.bin", TRUE));
    REQUIRE(CopyFileA("v9.bin.mem", "upgrade.bin.mem", TRUE));
    HANDLE held = CreateFileA("upgrade.bin.mem", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    REQUIRE(held != INVALID_HANDLE_VALUE);
    source->q_net.b4[0] = .75f; source->step_counter = source->target_hard_counter = 1990;
    REQUIRE(!save_checkpoint(source.get(), 2, 31000, 2, "upgrade.bin"));
    CloseHandle(held);
    /* GPT5-Sol: save aborts before replacing either file when the old sidecar
       cannot be backed up.  The complete V9 pair must remain loadable. */
    restore(live.get(), "upgrade.bin", 9, 0, .25f);
    seed_memory();
    REQUIRE(save_checkpoint(source.get(), 2, 31000, 2, "upgrade.bin"));
    restore(live.get(), "upgrade.bin", 11, 1, .75f);
    REQUIRE(live->step_counter == 1990 && live->target_hard_counter == 1990);
    puts("PASS interrupted upgrade preserves old pair; successful retry restores complete pair");

    // A zero CRC is a value, not permission to disable strict matching.
    REQUIRE(memory_save_full("zero.mem", 31000, 0));
    int mem_ep = -1;
    REQUIRE(memory_load_full("zero.mem", &mem_ep, 0, true));
    REQUIRE(mem_ep == 31000 && g_kb_count == 1);
    REQUIRE(memory_save_full("nonzero.mem", 31000, 1));
    REQUIRE(!memory_load_full("nonzero.mem", &mem_ep, 0, true));
    REQUIRE(!memory_load_full("v9.bin.mem", &mem_ep, 0, true));
    REQUIRE(g_kb_count == 1);
    puts("PASS strict zero generation and rejection of untagged memory");

    // GPT6-Astra: interruption before/after main-file commit, followed by a
    // successful retry or a denied main replacement. Never lose the committed pair.
    const Bytes main_a = read_bytes("current.bin"), mem_a = read_bytes("current.bin.mem");
    const Bytes main_b = read_bytes("upgrade.bin"), mem_b = read_bytes("upgrade.bin.mem");
    for (int committed_b = 0; committed_b < 2; ++committed_b) {
        for (int deny_commit = 0; deny_commit < 2; ++deny_commit) {
            write_bytes("retry.bin", committed_b ? main_b : main_a);
            write_bytes("retry.bin.mem", mem_b);
            write_bytes("retry.bin.mem.bak", mem_a);
            HANDLE lock = INVALID_HANDLE_VALUE;
            if (deny_commit) {
                lock = CreateFileA("retry.bin", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
                REQUIRE(lock != INVALID_HANDLE_VALUE);
            }
            seed_memory(); source->q_net.b4[0] = .625f;
            REQUIRE(save_checkpoint(source.get(), 2, 31000, 2, "retry.bin") == !deny_commit);
            if (deny_commit) CloseHandle(lock);
            restore(live.get(), "retry.bin", 11,
                deny_commit ? committed_b : 1,
                deny_commit ? (committed_b ? .75f : .25f) : .625f);
            REQUIRE(GetFileAttributesA("retry.bin.mem.bak") == INVALID_FILE_ATTRIBUTES);
        }
    }
    puts("PASS both interruption windows, successful retries and denied commits");

    // A matching generation header is insufficient: a truncated active memory
    // must not cause the only complete matching backup to be discarded.
    write_bytes("retry.bin", main_a);
    Bytes truncated = mem_a; truncated.resize(16);
    write_bytes("retry.bin.mem", truncated); write_bytes("retry.bin.mem.bak", mem_a);
    seed_memory();
    REQUIRE(save_checkpoint(source.get(), 2, 31000, 2, "retry.bin"));
    restore(live.get(), "retry.bin", 11, 1, .625f);
    write_bytes("retry.bin", main_b);
    write_bytes("retry.bin.mem", mem_a); write_bytes("retry.bin.mem.bak", truncated);
    REQUIRE(!save_checkpoint(source.get(), 2, 31000, 2, "retry.bin"));
    REQUIRE(read_bytes("retry.bin") == main_b && read_bytes("retry.bin.mem") == mem_a);
    REQUIRE(read_bytes("retry.bin.mem.bak") == truncated);
    puts("PASS full memory validation and preservation of ambiguous recovery files");

    free_dqn(source.get()); free_dqn(live.get());
    puts("PASS checkpoint memory regression");
    return 0;
}
