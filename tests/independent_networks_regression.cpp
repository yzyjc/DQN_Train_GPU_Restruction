// GPT6-Astra: CPU foundation isolation, not a production routing/GPU test.
#include "independent_networks.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
using namespace independent_training;
#define NEED(x) do { if (!(x)) { printf("FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    auto seed = std::make_unique<NeuralNet>();
    init_net(seed.get());
    IndependentNetworks nets(*seed, *seed);
    NEED(&nets.at(Side::Black).online != &nets.at(Side::White).online);
    NEED(&nets.at(Side::Black).target != &nets.at(Side::White).target);
    NEED(memcmp(&nets.at(Side::Black).online, seed.get(), sizeof(*seed)) == 0);
    NEED(memcmp(&nets.at(Side::White).online, seed.get(), sizeof(*seed)) == 0);
    bool invalid_rejected = false;
    try { nets.at(static_cast<Side>(2)); }
    catch (const std::invalid_argument&) { invalid_rejected = true; }
    NEED(invalid_rejected);
    for (int side = 0; side < 2; ++side) {
        auto& own = nets.at(static_cast<Side>(side));
        auto& other = nets.at(static_cast<Side>(1-side));
        auto other_before = std::make_unique<SideNetworks>(other);
        auto own_before = std::make_unique<NeuralNet>(own.online);
        auto target_before = std::make_unique<NeuralNet>(own.target);
        float state[CELL_COUNT+1] = {};
        state[CELL_COUNT] = (float)side;
        if (side) state[0] = 1.f; // legal white-to-move board
        float q[CELL_COUNT];
        forward_cpu_only(&own.online, state, q);
        const int action = side ? 1 : 0;
        float loss = train_step_cpu_only(&own.online, state, action, q[action]+.1f, .00016f);
        NEED(std::isfinite(loss) && loss > 0.f);
        // Check actual output weights, not forward caches, changed.
        NEED(memcmp(own.online.b4, own_before->b4, sizeof(own.online.b4)) != 0);
        own.updates.committed_batches++;
        own.updates.delta_ema = .001;
        NEED(memcmp(&own.target, target_before.get(), sizeof(NeuralNet)) == 0);
        NEED(memcmp(&other.online, &other_before->online, sizeof(NeuralNet)) == 0);
        NEED(memcmp(&other.target, &other_before->target, sizeof(NeuralNet)) == 0);
        NEED(other.updates.committed_batches == other_before->updates.committed_batches);
        NEED(other.updates.delta_ema == other_before->updates.delta_ema);
        NEED(other.updates.delta_warmup == other_before->updates.delta_warmup);
        NEED(other.updates.rejected_batches == other_before->updates.rejected_batches);
        NEED(other.updates.target_update_counter == other_before->updates.target_update_counter);
    }
    puts("PASS foundation: both directions CPU updates isolated; target/other-side state unchanged");
    return 0;
}
