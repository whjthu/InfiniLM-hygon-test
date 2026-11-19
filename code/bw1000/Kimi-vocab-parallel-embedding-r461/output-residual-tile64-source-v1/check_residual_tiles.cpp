// CPU wiring fixture for the exact production method body.
#include "output_residual_tiles.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

using Shape = std::vector<std::size_t>;
struct Device {
    enum class Type { kHygon, kOther };
    Type kind = Type::kHygon;
    Type type() const { return kind; }
};
enum class DType { kBFloat16 };
struct GraphManager { enum class CaptureState { kInactive, kActive }; };
struct Runtime {
    Device dev;
    bool capture = false;
    std::size_t fail_at_copy = 0;
    std::vector<std::size_t> copies, norm_rows, norm_fences;
    std::size_t norm_live = 0, softmax_calls = 0, gemm_calls = 0;
    Device device() const { return dev; }
    bool isGraphRecording() const { return capture; }
    GraphManager::CaptureState graphCaptureState() const {
        return capture ? GraphManager::CaptureState::kActive : GraphManager::CaptureState::kInactive;
    }
    void syncStream() { norm_fences.push_back(norm_live); }
};

struct Storage;
struct Tensor {
    std::shared_ptr<Storage> node;
    explicit operator bool() const { return bool(node); }
    Storage *operator->() const { return node.get(); }
    static Tensor empty(Runtime &, const Shape &, DType);
};

struct Storage {
    Shape dimensions;
    std::size_t start = 0;
    int tag = 0;
    bool packed = true;
    Runtime *normalized_owner = nullptr;
    std::shared_ptr<std::vector<int>> values;
    ~Storage() {
        if (normalized_owner) --normalized_owner->norm_live;
    }
    DType dtype() const { return DType::kBFloat16; }
    const Shape &shape() const { return dimensions; }
    bool is_contiguous() const { return packed; }
    std::size_t ndim() const { return dimensions.size(); }
    std::size_t size(std::size_t i) const { return dimensions.at(i); }
    Tensor metadata(const Shape &shape, std::size_t offset) const {
        auto view = std::make_shared<Storage>();
        view->dimensions = shape;
        view->start = offset;
        view->tag = tag;
        view->packed = packed;
        view->values = values;
        return {view};
    }
    Tensor narrow(std::initializer_list<std::array<std::size_t, 3>> spans) const {
        assert(spans.size() == 1);
        const auto &range = *spans.begin();
        auto dims = dimensions;
        assert(range[1] + range[2] <= dims.at(range[0]));
        dims[range[0]] = range[2];
        return metadata(dims, start + (range[0] == 0 ? range[1] : 0));
    }
    Tensor view(const Shape &shape) const { return metadata(shape, start); }
    Tensor squeeze(std::size_t axis) const {
        auto next = dimensions;
        assert(next[axis] == 1);
        next.erase(next.begin() + axis);
        return metadata(next, start);
    }
    void copy_from(Runtime &runtime, const Tensor &source) {
        assert(tag == 0 && source->dimensions == dimensions && source->start == 0);
        runtime.copies.push_back(start);
        if (runtime.copies.size() == runtime.fail_at_copy) throw std::runtime_error("test copy failure");
        std::copy(source->values->begin(), source->values->end(), values->begin() + start * 9);
    }
};

Tensor Tensor::empty(Runtime &, const Shape &shape, DType) {
    auto s = std::make_shared<Storage>();
    s->dimensions = shape;
    s->values = std::make_shared<std::vector<int>>(shape[0] * 9);
    return {s};
}

struct Norm {
    Tensor forward(Runtime &r, const Tensor &t) {
        assert(t->tag == 1 && r.norm_live == 0);
        r.norm_rows.push_back(t->size(0));
        auto out = Tensor::empty(r, t->shape(), t->dtype());
        out->start = t->start;
        out->tag = 2;
        ++r.norm_live;
        out->normalized_owner = &r;
        return out;
    }
};
struct Linear {
    Tensor forward(Runtime &r, const Tensor &t) {
        assert(t->tag == 2 && r.norm_live == 1 && t->size(1) == 9);
        auto scores = Tensor::empty(r, {t->size(0), 9, 1}, t->dtype());
        for (std::size_t i = 0; i < scores->values->size(); ++i)
            (*scores->values)[i] = static_cast<int>((t->start * 9 + i) % 137);
        return scores;
    }
};

namespace infini::ops {
using DataType = ::DType;
struct Softmax { static int DefaultConfig(Device::Type) { return 0; } };
struct Gemm { static int DefaultConfig(Device::Type) { return 0; } };
}
namespace infinilm::runtime {
using Runtime = ::Runtime;
using Tensor = ::Tensor;
using Shape = ::Shape;
using DataType = ::DType;
using Device = ::Device;
}
namespace infinilm::layers {
struct Embedding { virtual ~Embedding() = default; };
struct VocabParallelEmbeddingR461 : Embedding {};
}
namespace infinilm::execution {
namespace graph { using GraphManager = ::GraphManager; }
enum class RecordingPolicy { DeviceOnly };
template<class T> void submit(Runtime &r, int, const Tensor &scores, std::int64_t dim,
                              std::optional<DType>, const Tensor &dest) {
    assert(dim == 1 && scores.node == dest.node && scores->size(1) == 9);
    assert(r.norm_live == (r.copies.empty() ? 1U : 0U));
    ++r.softmax_calls;
}
template<class T> void submit(Runtime &r, RecordingPolicy, int, const Tensor &scores,
                              const Tensor &values, std::optional<Tensor>, std::optional<float>,
                              std::optional<float>, std::optional<int>, std::optional<int>, const Tensor &out) {
    assert(values->tag == 1 && values->size(0) == scores->size(0));
    ++r.gemm_calls;
    *out->values = *scores->values;
}
}
#define INFINILM_TRACE_OP(operator_name, tensor_name) do { } while (false)

struct KimiK3TextModel {
    std::size_t tp_size_ = 32, pp_size_ = 1, pp_stage_ = 0;
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_ =
        std::make_shared<infinilm::layers::VocabParallelEmbeddingR461>();
    std::shared_ptr<Norm> output_attn_res_norm_ = std::make_shared<Norm>();
    std::shared_ptr<Linear> output_attn_res_proj_ = std::make_shared<Linear>();
    Tensor apply_output_attn_res(Runtime &, const Tensor &, const Tensor &, std::size_t) const;
};

#include "actual_method.inc"

void setting(const char *value) {
#ifdef _WIN32
    assert(_putenv_s("INFINILM_KIMI_OUTPUT_RESIDUAL_TILE64", value) == 0);
#else
    assert(setenv("INFINILM_KIMI_OUTPUT_RESIDUAL_TILE64", value, 1) == 0);
#endif
}

std::vector<int> model(KimiK3TextModel &m, Runtime &r, std::size_t tokens) {
    const auto hidden = Tensor::empty(r, {1, tokens, 7168}, DType::kBFloat16);
    auto residual = Tensor::empty(r, {tokens, 9, 7168}, DType::kBFloat16);
    residual->tag = 1;
    const auto out = m.apply_output_attn_res(r, hidden, residual, 8);
    assert(out->shape() == hidden->shape() && r.norm_live == 0);
    assert(r.softmax_calls == 1 && r.gemm_calls == 1);
    return *out->values;
}

int main() {
    KimiK3TextModel m;
    Runtime baseline;
    setting("0");
    const auto original = model(m, baseline, 512);
    assert(baseline.norm_rows == std::vector<std::size_t>({512}) && baseline.copies.empty() && baseline.norm_fences.empty());
    setting("1");
    Runtime candidate;
    assert(model(m, candidate, 512) == original);
    assert(candidate.copies == std::vector<std::size_t>({0, 64, 128, 192, 256, 320, 384, 448}));
    assert(candidate.norm_rows == std::vector<std::size_t>(8, 64));
    assert(candidate.norm_fences == std::vector<std::size_t>(8, 1));
    Runtime failed;
    failed.fail_at_copy = 4;
    bool rejected = false;
    try { model(m, failed, 512); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected && failed.copies.size() == 4 && failed.norm_live == 0);
    assert(failed.norm_fences == std::vector<std::size_t>(4, 1) && failed.softmax_calls == 0);
    for (std::size_t tokens : {1, 511}) {
        Runtime fallback;
        model(m, fallback, tokens);
        assert(fallback.norm_rows == std::vector<std::size_t>({tokens}) && fallback.norm_fences.empty());
    }
    Runtime graph;
    graph.capture = true;
    model(m, graph, 512);
    assert(graph.norm_rows == std::vector<std::size_t>({512}) && graph.norm_fences.empty());
    Runtime wrong_tp;
    m.tp_size_ = 16;
    model(m, wrong_tp, 512);
    assert(wrong_tp.norm_rows == std::vector<std::size_t>({512}) && wrong_tp.norm_fences.empty());
    setting("10");
    m.tp_size_ = 32;
    Runtime invalid_setting;
    model(m, invalid_setting, 512);
    assert(invalid_setting.norm_rows == std::vector<std::size_t>({512}) && invalid_setting.norm_fences.empty());
    std::cout << "8 ordered slices, exact fixture reconstruction, live-owner fences, copy failure recovery, 5 fallback cases\n";
}
