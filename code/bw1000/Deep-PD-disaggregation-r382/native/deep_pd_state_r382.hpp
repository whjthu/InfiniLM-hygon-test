#pragma once
#include "cache/base_cache.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 { class TextModelR20; }
namespace infinilm::runtime { class Runtime; }
namespace infinilm::pd_r382 {
using Json = nlohmann::json;
inline constexpr const char *ABI = "deep-pd-r382-c1-tp8-raw-pages-v1";
// All host buffers are private to a session; each owner touches only its shard.
// No GPU addresses, owner pointers, descriptors or weights cross the wire.
struct Block { Json layout; std::vector<uint8_t> bytes; uint64_t received = 0; };
struct Shard { std::vector<Block> blocks; };
enum class Action { Export, Describe, Import };
struct Operation {
    Action action; uint64_t past; std::array<Shard, 8> shards;
    explicit Operation(Action a, uint64_t p) : action(a), past(p) {}
};
std::string sha256(const void *, size_t);
Json layout_only(const Json &);
Json describe(const Operation &, uint64_t epoch, uint64_t generation);
void check_manifest(const Json &, const Operation &);
void validate_payload(const Operation &);
}
namespace infinilm::cache {
class DeepPDControlR382 final : public CacheConfig {
public:
    std::shared_ptr<pd_r382::Operation> operation;
    explicit DeepPDControlR382(std::shared_ptr<pd_r382::Operation> op) : operation(std::move(op)) {}
    std::unique_ptr<CacheConfig> unique_copy() const override {
        return std::make_unique<DeepPDControlR382>(operation);
    }
};
}
namespace infinilm::models::deepseek_v4 {
class PdStateCodec {
public:
    static void run(runtime::Runtime &, TextModelR20 &, std::vector<size_t> &outer_lengths,
                    pd_r382::Operation &, unsigned rank);
};
}
