#include "commit_trace.hpp"

#include <cassert>
#include <iostream>

namespace trace = deep_csa_commit_trace;

int main() {
    int owner = 1, foreign = 2;
    std::array<std::uint64_t, trace::width> row{};
    assert(trace::arm(nullptr, 1) != 0);
    assert(trace::arm(&owner, 0) != 0);
    assert(trace::arm(&owner, 5) == 0);
    assert(trace::arm(&owner, 6) != 0);
    {
        trace::Forward ignored(&foreign, 1, 1);
        ignored.direct(1);
        ignored.complete_after_drain();
    }
    assert(trace::read(&owner, row.data(), row.size()) != 0);
    {
        trace::Forward forward(&owner, 129, 2);
        forward.direct(129);
        forward.store();
        forward.complete_after_drain();
        assert(trace::clear(&owner) != 0);
    }
    assert(trace::read(&foreign, row.data(), row.size()) != 0);
    assert(trace::read(&owner, row.data(), row.size() - 1) != 0);
    assert(trace::read(&owner, row.data(), row.size()) == 0);
    assert((row == std::array<std::uint64_t, trace::width>{5, 129, 2, 129, 1, 0, 1, 1}));
    assert(trace::read(&owner, row.data(), row.size()) != 0);

    assert(trace::arm(&owner, 11) == 0);
    {
        trace::Forward forward(&owner, 129, 1);
        forward.index_select(128);
        forward.store();
        forward.complete_after_drain();
    }
    assert(trace::read(&owner, row.data(), row.size()) == 0);
    assert((row == std::array<std::uint64_t, trace::width>{11, 129, 1, 128, 0, 1, 1, 1}));

    assert(trace::arm(&owner, 12) == 0);
    {
        trace::Forward failed(&owner, 1, 1);
        failed.index_select(1);
    }
    assert(trace::read(&owner, row.data(), row.size()) == 0);
    assert((row == std::array<std::uint64_t, trace::width>{12, 1, 1, 1, 0, 1, 0, 0}));

    assert(trace::arm(&owner, 13) == 0);
    {
        trace::Forward first(&owner, 1, 1);
        first.direct(1);
    }
    {
        trace::Forward duplicate(&owner, 1, 1);
        duplicate.direct(1);
    }
    assert(trace::read(&owner, row.data(), row.size()) != 0);
    assert(trace::clear(&foreign) != 0);
    assert(trace::clear(&owner) == 0);
    std::cout << "Trace ownership, direct, IndexSelect, failure and duplicate boundaries passed\n";
}
