#include "identity_swa_commit.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <utility>

constexpr std::size_t D = 512;
namespace rt {
enum class DataType { kBFloat16 };
using Shape = std::vector<std::size_t>;
struct Values {
    Shape dimensions;
    bool contiguous = true;
    std::vector<std::uint16_t> words;
    int device() const { return 0; }
    DataType dtype() const { return DataType::kBFloat16; }
    const Shape& shape() const { return dimensions; }
    bool is_contiguous() const { return contiguous; }
};
using Tensor = std::shared_ptr<Values>;
}

struct Device {
    operator int() const { return 0; }
    int type() const { return 0; }
};
struct Runtime {
    Device device() const { return {}; }
    unsigned selects = 0;
};
namespace infini::ops {
struct IndexSelect {
    static int DefaultConfig(int) { return 0; }
};
}
namespace ex {
template<typename T>
void submit(Runtime& runtime, int, const rt::Tensor& input, const std::vector<std::int64_t>& rows,
            std::int64_t axis, const rt::Tensor& output) {
    assert(axis == 0);
    ++runtime.selects;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::copy_n(input->words.begin() + rows[i] * D, D, output->words.begin() + i * D);
    }
}
}

struct Work {
    unsigned allocations = 0;
    rt::Tensor alloc(rt::Shape dims, rt::DataType) {
        ++allocations;
        auto t = std::make_shared<rt::Values>();
        t->words.resize(dims[0] * dims[1]);
        t->dimensions = std::move(dims);
        return t;
    }
};

struct Cache {
    std::map<std::int64_t, std::vector<std::uint16_t>> stored;
    const rt::Values* input_seen = nullptr;
    void store(Runtime&, const std::vector<std::int64_t>& slots, const rt::Tensor& values) {
        assert(values->shape() == (rt::Shape{slots.size(), D}) && values->is_contiguous());
        input_seen = values.get();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            const auto first = values->words.begin() + i * D;
            assert(stored.emplace(slots[i], std::vector<std::uint16_t>(first, first + D)).second);
        }
    }
};

struct Inputs {
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> slots;
    std::size_t count = 0;
};

Inputs schedule(const std::vector<std::pair<std::int64_t, std::size_t>>& requests, std::size_t past) {
    Inputs result;
    for (auto request : requests) {
        const auto span = std::min<std::size_t>(128, request.second);
        for (std::size_t t = request.second - span; t < request.second; ++t) {
            result.rows.push_back(static_cast<std::int64_t>(result.count + t));
            result.slots.push_back(request.first * 128 + (past + t) % 128);
        }
        result.count += request.second;
    }
    return result;
}

bool one_case(const Inputs& input, bool exact_identity, bool contiguous, bool opt_in) {
    const auto& commit_rows = input.rows;
    const auto& commit_slots = input.slots;
    const auto tokens = input.count;
    Work work;
    struct { rt::Tensor kv; } prep{work.alloc({tokens, D}, rt::DataType::kBFloat16)};
    prep.kv->contiguous = contiguous;
    for (std::size_t i = 0; i < prep.kv->words.size(); ++i) {
        prep.kv->words[i] = static_cast<std::uint16_t>(i * 709 + i / D);
    }
    const auto raw_before = prep.kv->words;
    unsigned id_uploads = 0;
    const auto device_ids = [&](const auto& host) {
        ++id_uploads;
        return host;
    };
    Runtime r;
    Cache cache;
    auto* cache_ = &cache;
#include "derived_commit.inc"
    const bool direct = opt_in && exact_identity && contiguous;
    assert((cache.input_seen == prep.kv.get()) == direct);
    assert(r.selects == (direct ? 0 : 1));
    assert(id_uploads == (direct ? 1 : 2));
    assert(work.allocations == (direct ? 1 : 2));
    assert(raw_before == prep.kv->words);
    assert(cache.stored.size() == commit_rows.size());
    for (std::size_t i = 0; i < commit_slots.size(); ++i) {
        const auto begin = raw_before.begin() + commit_rows[i] * D;
        const auto& stored = cache.stored.at(commit_slots[i]);
        assert(std::equal(stored.begin(), stored.end(), begin, begin + D));
    }
    return direct;
}

int main() {
    assert(!deep_identity_swa_commit::enabled(nullptr));
    assert(!deep_identity_swa_commit::enabled(""));
    assert(!deep_identity_swa_commit::enabled("0"));
    assert(!deep_identity_swa_commit::enabled("10"));
    assert(!deep_identity_swa_commit::enabled("yes"));
    assert(deep_identity_swa_commit::enabled("1"));
    assert(!deep_identity_swa_commit::can_use_prepared_kv(true, true, {}, 0, 0));
    assert(!deep_identity_swa_commit::can_use_prepared_kv(true, true, {0}, 2, 1));
    assert(!deep_identity_swa_commit::can_use_prepared_kv(true, true, {-1}, 1, 1));
    const bool enabled = deep_identity_swa_commit::enabled(std::getenv("INFINILM_DEEP_CSA_IDENTITY_SWA_COMMIT"));
    unsigned direct = 0;
    direct += one_case(schedule({{0, 1}}, 8496), true, true, enabled);
    direct += one_case(schedule({{3, 1}}, 8575), true, true, enabled);
    direct += one_case(schedule({{6, 128}}, 127), true, true, enabled);
    direct += one_case(schedule({{4, 8}, {1, 12}, {9, 1}}, 127), true, true, enabled);
    direct += one_case(schedule({{0, 129}}, 0), false, true, enabled);
    direct += one_case(schedule({{0, 1}, {2, 129}, {1, 1}}, 11), false, true, enabled);
    direct += one_case(schedule({{4, 1}}, 8496), true, false, enabled);
    direct += one_case({{1, 0}, {4, 7}, 2}, false, true, enabled);
    direct += one_case({{0, 0}, {4, 7}, 2}, false, true, enabled);
    std::cout << "{\"direct\":" << direct << ",\"legacy\":" << 9 - direct << "}\n";
}
