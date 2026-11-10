#include "history_bank_alias.hpp"
#include <cassert>
#include <thread>
#include <stdexcept>

namespace policy = deep_c4_history_bank_alias;
int main() {
    std::size_t yes = 0, no = 0;
    for (std::size_t past = 0; past < 32768; ++past) {
        const std::size_t before = past / 4, after = (past + 1) / 4;
        const bool accepted = policy::eligible(1, 1, past, after, after > before);
        const bool unchanged_full_bank = before > 0 && before == after;
        assert(accepted == unchanged_full_bank);
        if (accepted) ++yes;
        else ++no;
    }
    assert(!policy::eligible(2, 1, 1252, 313, false));
    assert(!policy::eligible(1, 2, 1252, 313, false));
    assert(!policy::eligible(1, 1, 1252, 314, false));
    assert(!policy::eligible(1, 1, 1252, 313, true));
    assert(!policy::eligible(1, 1, 262148, 65537, false));
    std::uint64_t out[4]{};
    assert(policy::read(-1, out, 4) == 1);
    assert(policy::read(8, out, 4) == 1);
    assert(policy::read(0, nullptr, 4) == 1);
    assert(policy::read(0, out, 3) == 1);
    assert(policy::set(-1) == 1 && policy::set(2) == 1);
    {
        policy::Use use(0);
        assert(!use.enabled);
        assert(policy::set(1) == 2);
        assert(policy::read(0, out, 4) == 2);
        std::thread other([] { assert(policy::set(1) == 2); });
        other.join();
        use.complete(true, false, 313 * 1024);
    }
    assert(policy::read(0, out, 4) == 0);
    assert(out[0] == 1 && out[1] == 1 && out[2] == 0 && out[3] == 0);
    assert(policy::set(1) == 0);
    try { policy::Use failing(1); throw std::runtime_error("drain"); }
    catch (const std::runtime_error &) {}
    assert(policy::set(1) == 0);
    {
        policy::Use use(0);
        assert(use.enabled);
        use.complete(true, true, 313 * 1024);
    }
    assert(policy::read(0, out, 4) == 0);
    assert(out[0] == 2 && out[1] == 2 && out[2] == 1 && out[3] == 313 * 1024);
    assert(policy::set(0) == 0);
    assert(yes == 24573 && no == 8195);
}
