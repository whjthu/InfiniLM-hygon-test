#pragma once
// Diagnostic-only consumer of real r482 Pending values, after original checkpoint load.
#include "deepseek_v4_c4_indexer.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deep483_native_guard {
namespace rt = infinilm::runtime;
using Indexer = infinilm::models::deepseek_v4::C4Indexer;

struct Snapshot {
    std::vector<std::uint8_t> state;
    std::vector<std::uint8_t> cache;
};

inline void copy(rt::Runtime& runtime, const rt::Tensor& tensor, rt::DataType dtype,
                 const rt::Shape& shape, std::vector<std::uint8_t>& bytes) {
    if (!tensor || tensor->dtype() != dtype || tensor->shape() != shape ||
        tensor->nbytes() != bytes.size() || !tensor->is_contiguous()) {
        throw std::runtime_error("C4 guard snapshot geometry differs");
    }
    try {
        runtime.memcpyD2H(bytes.data(), tensor->data(), bytes.size());
        runtime.syncStream();
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

inline Snapshot snapshot(rt::Runtime& runtime, const Indexer& indexer) {
    Snapshot result{std::vector<std::uint8_t>(8192), std::vector<std::uint8_t>(1081344)};
    copy(runtime, indexer.state(), rt::DataType::kBFloat16, {8, 512}, result.state);
    copy(runtime, indexer.raw_cache(), rt::DataType::kUInt8, {128, 8448}, result.cache);
    return result;
}

inline void unchanged(const Snapshot& before, const Snapshot& after) {
    if (before.state != after.state || before.cache != after.cache) {
        throw std::runtime_error("rejected C4 operation changed committed state or cache");
    }
}

template<class Operation>
void reject(Operation operation, std::string_view expected) {
    try {
        operation();
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()) == expected) return;
        throw;
    }
    throw std::runtime_error("C4 guard accepted the rejected operation");
}

// Both indexers must have loaded the same seven ranges, be fresh and owned by this
// runtime. positions is [0, 1, 2, 3] and wrong_positions differs in one value.
// The caller keeps guarded inputs alive and owns floor/deadline checks and cleanup.
template<class Boundary>
void run(rt::Runtime& runtime, Indexer& primary, Indexer& foreign,
         const rt::Tensor& hidden, const rt::Tensor& query,
         const rt::Tensor& positions, const rt::Tensor& wrong_positions,
         Boundary boundary) {
    if (&primary == &foreign || primary.local_heads() != 64 || foreign.local_heads() != 64) {
        throw std::invalid_argument("distinct TP1 C4 indexers required");
    }
    const std::vector<Indexer::Request> valid{{0, 0, 4}};
    const auto start = snapshot(runtime, primary);
    boundary("invalid_row");
    reject([&] { (void)primary.prepare(runtime, hidden, query, positions, {{1, 0, 4}}); },
           "DeepSeek C4 indexer row/past/capacity differs");
    unchanged(start, snapshot(runtime, primary));
    boundary("invalid_positions");
    reject([&] { (void)primary.prepare(runtime, hidden, query, wrong_positions, valid); },
           "DeepSeek C4 indexer logical positions differ");
    unchanged(start, snapshot(runtime, primary));
    boundary("foreign_prepare");
    auto pending = primary.prepare(runtime, hidden, query, positions, valid);
    const auto foreign_start = snapshot(runtime, foreign);
    boundary("foreign_commit");
    reject([&] { foreign.commit(runtime, pending); },
           "DeepSeek C4 indexer stale/foreign transaction");
    unchanged(start, snapshot(runtime, primary));
    unchanged(foreign_start, snapshot(runtime, foreign));
    boundary("reset_with_pending");
    primary.reset(runtime);
    runtime.syncStream();
    const auto reset = snapshot(runtime, primary);
    boundary("stale_commit");
    reject([&] { primary.commit(runtime, pending); },
           "DeepSeek C4 indexer stale/foreign transaction");
    unchanged(reset, snapshot(runtime, primary));
    boundary("valid_prepare_commit");
    auto accepted = primary.prepare(runtime, hidden, query, positions, valid);
    if (accepted.slots != std::vector<std::int64_t>{0}) {
        throw std::runtime_error("valid C4 prepare did not produce slot zero");
    }
    std::vector<std::uint8_t> key(128), scale(4);
    copy(runtime, accepted.new_keys.values, rt::DataType::kUInt8, {1, 128}, key);
    copy(runtime, accepted.new_keys.scales, rt::DataType::kFloat32, {1, 1}, scale);
    boundary("valid_commit");
    primary.commit(runtime, accepted);
    std::vector<std::uint8_t> committed_cache(reset.cache.size());
    copy(runtime, primary.raw_cache(), rt::DataType::kUInt8, {128, 8448}, committed_cache);
    auto expected_cache = reset.cache;
    std::copy(key.begin(), key.end(), expected_cache.begin());
    std::copy(scale.begin(), scale.end(), expected_cache.begin() + 8192);
    if (committed_cache != expected_cache) {
        throw std::runtime_error("valid C4 commit differs from the expected cache image");
    }
    boundary("complete");
}
}
