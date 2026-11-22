#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "inverse_output_width_policy.hpp"

using std::size_t;
namespace inverse_output = infinilm::models::deepseek_v4::inverse_output;

namespace rt {
enum class DataType { kBFloat16, kFloat32, kInt64 };
using Shape = std::vector<size_t>;
struct Runtime {
    std::string trace;
    bool recording = false;
    bool stream_available = true;
    int device() const { return 0; }
    bool isGraphRecording() const { return recording; }
    void assertOwnerThread() { trace += 'A'; }
    void* stream() {
        trace += 'S';
        return stream_available ? this : nullptr;
    }
    void syncStream() { trace += 'Y'; }
    void memcpyD2H(void* destination, const void* source, size_t bytes) {
        if (bytes != sizeof(int64_t)) throw std::logic_error("position transfer differs");
        std::memcpy(destination, source, bytes);
        trace += 'R';
    }
    void memcpyH2D(void* destination, const void* source, size_t bytes, bool sync) {
        if (sync || bytes != 64 * sizeof(float)) throw std::logic_error("table transfer differs");
        std::memcpy(destination, source, bytes);
        trace += 'U';
    }
};
struct Value;
using Tensor = std::shared_ptr<Value>;
struct Value {
    Shape dims;
    DataType type;
    bool contiguous = true;
    std::vector<float> payload = std::vector<float>(64);
    int device() const { return 0; }
    DataType dtype() const { return type; }
    size_t ndim() const { return dims.size(); }
    size_t size(size_t i) const { return dims.at(i); }
    Shape shape() const { return dims; }
    bool is_contiguous() const { return contiguous; }
    int64_t position = 0;
    void* data() { return type == DataType::kInt64 ? static_cast<void*>(&position) : payload.data(); }
    void copy_from(Runtime& runtime, const Tensor&) { runtime.trace += 'C'; }
};
}

struct Work {
    rt::Runtime& runtime;
    explicit Work(rt::Runtime& runtime) : runtime(runtime) {}
    ~Work() { runtime.syncStream(); }
    rt::Tensor alloc(rt::Shape shape, rt::DataType type) {
        return std::make_shared<rt::Value>(rt::Value{std::move(shape), type});
    }
};

struct AttentionPrepare {
    bool ready_ = true;
    int device_ = 0;
    size_t local_heads_ = 8, max_position_ = 32768, original_ = 0;
    double theta_ = 160000., factor_ = 16., beta_fast_ = 32., beta_slow_ = 1.;
    rt::Tensor inverse_output_rope(rt::Runtime&, const rt::Tensor&, const rt::Tensor&) const;
    void rotate(rt::Runtime& r, const rt::Tensor&, std::nullopt_t, const rt::Tensor&, bool inverse) const {
        if (!inverse) throw std::logic_error("unexpected forward RoPE");
        r.trace += 'L';
    }
};

int launch_status = 0;
int record_launch(void* out, const void* in, const void* table, size_t tokens, size_t heads,
                  void* stream, char provider) {
    if (out == in || !stream || tokens != 1 || heads != 8 || (static_cast<const float*>(table)[31] != 1.f || static_cast<const float*>(table)[63] != 0.f)) {
        throw std::logic_error("candidate call arguments differ");
    }
    auto* runtime = static_cast<rt::Runtime*>(stream);
    runtime->trace += provider;
    return launch_status;
}

extern "C" int deep_inverse_rope_h1_pack32_v1(void* out, const void* in, const void* table, size_t tokens,
                                              size_t heads, void* stream) {
    return record_launch(out, in, table, tokens, heads, stream, 'P');
}

extern "C" int deep_inverse_rope_h1_pack64_v1(void* out, const void* in, const void* table, size_t tokens,
                                              size_t heads, void* stream) {
    return record_launch(out, in, table, tokens, heads, stream, 'Q');
}

#include "rendered_method.inc"

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string scenario = argv[1];
    const std::string expected_trace = argv[2];
    const std::string expected_error = argv[3];
    AttentionPrepare caller;
    rt::Runtime runtime;
    auto x = std::make_shared<rt::Value>(rt::Value{{1, 8, 512}, rt::DataType::kBFloat16});
    auto pos = std::make_shared<rt::Value>(rt::Value{{1}, rt::DataType::kInt64});
    if (scenario == "bad-position") pos->position = 32768;
    if (scenario == "bad-position-shape") pos->dims[0] = 2;
    if (scenario == "no-position") pos.reset();
    if (scenario == "multi") x->dims[0] = 2;
    if (scenario == "other-heads") x->dims[1] = caller.local_heads_ = 4;
    if (scenario == "noncontiguous") x->contiguous = false;
    if (scenario == "no-stream") runtime.stream_available = false;
    if (scenario == "bad-dtype") x->type = rt::DataType::kFloat32;
    if (scenario == "graph") runtime.recording = true;
    if (scenario == "provider-failure") launch_status = 31;
    std::string error;
    try {
        auto output = caller.inverse_output_rope(runtime, x, pos);
        if (!output || output == x || output->shape() != x->shape()) return 2;
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    if (runtime.trace != expected_trace || error != expected_error) {
        std::cerr << "trace=" << runtime.trace << " error=" << error << '\n';
        return 3;
    }
    return 0;
}
