"""Derive the opt-in C1 compact compressed bank from the audited C4 source."""

import argparse
from hashlib import sha256
from pathlib import Path

ORIGINAL = "9e350f77fbe264dd05e76bb1b719ed6dd0a6f3de8da1ae9198c673eb89350005"


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("missing or ambiguous C4 source anchor")
    return text.replace(before, after)


def derive(raw):
    if sha256(raw).hexdigest() != ORIGINAL:
        raise ValueError("C4 source does not match the GPU1460 parent")
    source = raw.decode("utf-8")
    source = replace(source, "#include <algorithm>\n", "#include <algorithm>\n#include <atomic>\n")
    source = replace(source, "namespace rt=infinilm::runtime;namespace ex=infinilm::execution;\n",
                     """namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace c4_compact_bank {
std::atomic<bool> enabled{false};
std::array<std::atomic<uint64_t>, 8> successful_calls{};
std::array<std::atomic<uint64_t>, 8> historical_rows_avoided{};
}
""")
    source = replace(source,
                     """ auto compressed_bank=work.alloc({std::max(size_t(1),compressed_total),512},rt::DataType::kBFloat16);
 r.setDeviceMemory(compressed_bank->data(),0xff,compressed_bank->nbytes());""",
                     """ const auto device_index = r.device().index();
 const bool compact_bank = c4_compact_bank::enabled.load(std::memory_order_acquire)
     && device_index < c4_compact_bank::successful_calls.size()
     && requests.size() == 1 && tokens == 1 && pending.boundaries.empty()
     && compressed_total == requests.front().past / 4 && compressed_total > topk_
     && topk_ > 0 && topk_ <= 65536;
 rt::Tensor compressed_bank;
 if (!compact_bank) {
     compressed_bank=work.alloc({std::max(size_t(1),compressed_total),512},rt::DataType::kBFloat16);
     r.setDeviceMemory(compressed_bank->data(),0xff,compressed_bank->nbytes());
 }""")
    source = replace(source,
                     " for(auto request:requests){size_t before=request.past/4,after=(request.past+request.tokens)/4;",
                     """ if (!compact_bank) {
 for(auto request:requests){size_t before=request.past/4,after=(request.past+request.tokens)/4;""")
    source = replace(source, "  compressed_begin+=after;new_compressed_begin+=after-before;historical_begin+=before;\n }\n",
                     "  compressed_begin+=after;new_compressed_begin+=after-before;historical_begin+=before;\n }\n }\n")
    source = replace(source, " auto si=device_ids(selected)",
                     """ if (compact_bank) {
     // The unchanged selection loop validated the count, distinctness and causal prefix.
     // There is one request, so ci contains row-local history IDs in Top-K order.
     const size_t kept = size_t(lengths[1]);
     if (kept != topk_) throw std::logic_error("DeepSeek compact C4 selection count differs");
     std::vector<std::pair<size_t, size_t>> entries;
     entries.reserve(kept);
     for (size_t j = 0; j < kept; ++j) {
         entries.emplace_back(requests.front().row, size_t(ci[j]));
         ci[j] = int64_t(j);
     }
     compressed_bank = compressor_->gather(r, entries);
     if (!compressed_bank || compressed_bank->device() != r.device()
         || compressed_bank->dtype() != rt::DataType::kBFloat16
         || compressed_bank->shape() != rt::Shape{kept, 512}
         || !compressed_bank->is_contiguous()) {
         throw std::logic_error("DeepSeek compact C4 bank layout differs");
     }
     work.held.push_back(compressed_bank);
 }
 auto si=device_ids(selected)""")
    source = replace(source,
                     "r.syncStream();inner_profile.finish_after_original_drain();return projected.hidden;",
                     """r.syncStream();inner_profile.finish_after_original_drain();
 if (compact_bank) {
     c4_compact_bank::successful_calls[device_index].fetch_add(1, std::memory_order_relaxed);
     c4_compact_bank::historical_rows_avoided[device_index].fetch_add(
         compressed_total - size_t(lengths[1]), std::memory_order_relaxed);
 }
 return projected.hidden;""")
    return (source + EXPORTS).encode("utf-8")


EXPORTS = r'''
extern "C" int deep_c4_compact_bank_set_v1(int mode) noexcept {
    namespace model = infinilm::models::deepseek_v4;
    if (mode != 0 && mode != 1) return 1;
    try {
        std::lock_guard<std::mutex> lock(model::wide_r3443::policy_mutex);
        if (model::wide_r3443::in_flight || deep_owner_r8900::window.load()
            || deep_attention_inner_r9193::running.load()) return 2;
        model::c4_compact_bank::enabled.store(mode == 1, std::memory_order_release);
        return 0;
    } catch (...) { return 3; }
}

extern "C" const char *deep_c4_compact_bank_stats_v1() noexcept {
    namespace model = infinilm::models::deepseek_v4;
    static thread_local std::string out;
    try {
        std::lock_guard<std::mutex> lock(model::wide_r3443::policy_mutex);
        if (model::wide_r3443::in_flight || deep_owner_r8900::window.load()
            || deep_attention_inner_r9193::running.load()) return nullptr;
        auto rows = nlohmann::json::array();
        for (size_t rank = 0; rank < model::c4_compact_bank::successful_calls.size(); ++rank) {
            rows.push_back({{"device", rank},
                            {"calls", model::c4_compact_bank::successful_calls[rank].load()},
                            {"historical_rows_avoided", model::c4_compact_bank::historical_rows_avoided[rank].load()}});
        }
        out = nlohmann::json{{"enabled", model::c4_compact_bank::enabled.load()}, {"owners", rows}}.dump();
        return out.c_str();
    } catch (...) { return nullptr; }
}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = derive(args.original.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(result)
