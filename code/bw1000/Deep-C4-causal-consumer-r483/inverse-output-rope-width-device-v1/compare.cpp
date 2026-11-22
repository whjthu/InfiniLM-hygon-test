#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int deep_inverse_rope_h1_pack32_v1(void*, const void*, const void*, size_t, size_t, void*);
extern "C" int deep_inverse_rope_h1_pack64_v1(void*, const void*, const void*, size_t, size_t, void*);

namespace {
constexpr size_t width = 512;
constexpr size_t prefix_width = 448;
constexpr size_t tail_width = width - prefix_width;
constexpr size_t guard = 128;
constexpr size_t memory_floor = size_t{16} << 30;
constexpr unsigned launches = 512;
using Kernel = int (*)(void*, const void*, const void*, size_t, size_t, void*);
const std::array<Kernel, 2> kernels{deep_inverse_rope_h1_pack32_v1, deep_inverse_rope_h1_pack64_v1};
using Clock = std::chrono::steady_clock;

void need(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value), "create event"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <class T>
class Buffer {
public:
    explicit Buffer(size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&start_), count * sizeof(T)), "allocate buffer");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (start_) cudaFree(start_); }
    T* at(size_t index) { return start_ + index; }
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        need(host.size() == count_, "upload extent differs");
        check(cudaMemcpyAsync(start_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    }
    std::vector<T> read(cudaStream_t stream) {
        std::vector<T> host(count_);
        check(cudaMemcpyAsync(host.data(), start_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
        check(cudaStreamSynchronize(stream), "download completion");
        return host;
    }
private:
    T* start_{};
    size_t count_;
};

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint16_t encode(float value) {
    const uint32_t raw = bits(value);
    const uint32_t low = raw & 0xffffu;
    const uint32_t high = raw >> 16;
    return static_cast<uint16_t>(high + (low > 0x8000u || (low == 0x8000u && (high & 1u))));
}

float decode(uint16_t value) {
    const uint32_t raw = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

struct Case {
    size_t tokens;
    size_t heads;
};
constexpr std::array<Case, 6> cases{{{1, 8}, {17, 8}, {128, 8}, {512, 8}, {1, 64}, {17, 64}}};
constexpr std::array<size_t, 3> offset_words{0, 2, 1};

class Inputs {
public:
    Inputs(Case shape, size_t offset)
        : shape(shape), start(guard + offset), elements(shape.tokens * shape.heads * width),
          x(start + elements + guard, 0xa57c), expected(x.size(), 0x5ab3), canary(expected),
          table(guard + shape.tokens * tail_width + guard, -2.0f),
          input(x.size()), output(x.size()), coefficients(table.size()) {
        constexpr std::array<uint16_t, 8> prefix_values{0x0000, 0x8000, 0x7fc1, 0x0001, 0x3f80, 0xbf80, 0xffa5, 0x7f7f};
        constexpr std::array<float, 8> values{0.25f, -0.75f, 0.5f, 1.0f, -0.25f, 0.875f, 0.125f, -0.5f};
        for (size_t token = 0; token < shape.tokens; ++token) {
            for (size_t lane = 0; lane < tail_width; ++lane) {
                const size_t phase = lane >= tail_width / 2 ? 3 : 0;
                table[guard + token * tail_width + lane] = values[(token * 3 + lane * 5 + phase) % values.size()];
            }
        }
        for (size_t head = 0; head < shape.tokens * shape.heads; ++head) {
            const size_t row = start + head * width;
            for (size_t column = 0; column < prefix_width; ++column) {
                const size_t i = head * width + column;
                uint16_t value = static_cast<uint16_t>((i * 0x7153u + i / 19u) & 0xffffu);
                if (column < prefix_values.size()) value = prefix_values[column];
                expected[row + column] = x[row + column] = value;
            }
            for (size_t lane = 0; lane < tail_width / 2; ++lane) {
                const size_t base = guard + head / shape.heads * tail_width;
                const size_t column = row + prefix_width + 2 * lane;
                const int xv = static_cast<int>((head * 13 + lane * 17) % 701) - 350;
                const int yv = static_cast<int>((head * 19 + lane * 23) % 613) - 306;
                x[column] = encode(static_cast<float>(xv) / 32.0f);
                x[column + 1] = encode(static_cast<float>(yv) / 32.0f);
                const double a = decode(x[column]);
                const double b = decode(x[column + 1]);
                const double c = table[base + lane];
                const double s = -table[base + tail_width / 2 + lane];
                // Dyadic inputs have exact F32 products here. Normalize signed zero
                // only for the rotated tail, where cancellation can produce it.
                expected[column] = encode(static_cast<float>(a * c - b * s));
                expected[column + 1] = encode(static_cast<float>(b * c + a * s));
            }
        }
    }

    void launch(Kernel kernel, cudaStream_t stream) {
        need(kernel(output.at(start), input.at(start), coefficients.at(guard),
                    shape.tokens, shape.heads, stream) == 0, "original RoPE entry returned an error");
    }

    void evaluate(Kernel kernel, cudaStream_t stream) {
        input.upload(x, stream);
        coefficients.upload(table, stream);
        output.upload(canary, stream);
        launch(kernel, stream);
        const auto actual = output.read(stream);
        for (size_t index = 0; index < actual.size(); ++index) {
            const bool tail = index >= start && index < start + elements &&
                              (index - start) % width >= prefix_width;
            const bool signed_zero = tail && ((actual[index] | expected[index]) & 0x7fffu) == 0;
            need(actual[index] == expected[index] || signed_zero, "output value or redzone differs");
        }
        need(input.read(stream) == x && coefficients.read(stream) == table, "input or table mutated");
        auto* dst = output.at(start);
        auto* src = input.at(start);
        auto* phase = coefficients.at(guard);
        need(kernel(src, src, phase, shape.tokens, shape.heads, stream) == 1 &&
             kernel(dst, src, phase, 0, shape.heads, stream) == 1 &&
             kernel(dst, src, phase, shape.tokens, 7, stream) == 1,
             "required rejected inputs were accepted");
    }

    Case shape;
    size_t start;
    size_t elements;
private:
    std::vector<uint16_t> x;
    std::vector<uint16_t> expected;
    std::vector<uint16_t> canary;
    std::vector<float> table;
    Buffer<uint16_t> input;
    Buffer<uint16_t> output;
    Buffer<float> coefficients;
};

size_t available_memory() {
    size_t free{};
    size_t total{};
    check(cudaMemGetInfo(&free, &total), "query free memory");
    need(free >= memory_floor, "strict 16 GiB free memory floor");
    return free;
}

void measure(Inputs& input, Kernel kernel, int width_bits, int pair, cudaStream_t stream, std::ostream& out) {
    Event begin, end;
    for (unsigned repeat = 0; repeat < 64; ++repeat) input.launch(kernel, stream);
    check(cudaStreamSynchronize(stream), "warmup completion");
    const auto t0 = Clock::now();
    check(cudaEventRecord(begin.value, stream), "start event");
    for (unsigned repeat = 0; repeat < launches; ++repeat) input.launch(kernel, stream);
    check(cudaEventRecord(end.value, stream), "end event");
    check(cudaEventSynchronize(end.value), "measurement completion");
    const double host_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    float event_ms{};
    check(cudaEventElapsedTime(&event_ms, begin.value, end.value), "elapsed events");
    need(event_ms > 0 && host_us > 0, "nonpositive timing sample");
    out << "{\"pair\":" << pair << ",\"width_bits\":" << width_bits << ",\"launches\":" << launches
        << ",\"event_us_per_launch\":" << (1000.0 * event_ms / launches)
        << ",\"host_us_per_launch\":" << host_us / launches << "}\n" << std::flush;
}
}

int main(int argc, char** argv) {
    try {
        need(argc == 3, "expected original results pathname and exactly one admitted PCI address");
        std::ofstream out(argv[1], std::ios::trunc);
        need(out.good(), "open private result");
        out << std::setprecision(12);
        out << "{\"planned_numerical_cases\":36,\"planned_pairs\":3,\"planned_timing_arms\":6}\n" << std::flush;
        int count{};
        check(cudaGetDeviceCount(&count), "visible devices");
        need(count == 1, "expected exactly one admitted device");
        check(cudaSetDevice(0), "select device");
        std::array<char, 64> pci{};
        check(cudaDeviceGetPCIBusId(pci.data(), pci.size(), 0), "device PCI");
        std::string pci_address(pci.data());
        for (auto& digit : pci_address) if (digit >= 'A' && digit <= 'F') digit += 'a' - 'A';
        if (pci_address.size() == 7) pci_address = "0000:" + pci_address;
        need(pci_address == argv[2], "device PCI differs from admitted card");
        const size_t before = available_memory();
        size_t minimum_sampled_free = before;
        {
            Stream stream;
            for (const auto& shape : cases) {
                for (const auto offset : offset_words) {
                    Inputs inputs(shape, offset);
                    const size_t free = available_memory();
                    if (free < minimum_sampled_free) minimum_sampled_free = free;
                    for (size_t arm = 0; arm < kernels.size(); ++arm) {
                        inputs.evaluate(kernels[arm], stream.value);
                        out << "{\"tokens\":" << shape.tokens << ",\"heads\":" << shape.heads
                            << ",\"offset_bytes\":" << offset * sizeof(uint16_t)
                            << ",\"width_bits\":" << (arm == 0 ? 32 : 64)
                            << ",\"numeric_values\":" << inputs.elements << ",\"guards_ok\":true}\n" << std::flush;
                    }
                }
            }
            Inputs primary(cases.front(), 0);
            primary.evaluate(kernels[0], stream.value);
            for (int pair = 0; pair < 3; ++pair) {
                const int first = pair % 2;
                measure(primary, kernels[first], first == 0 ? 32 : 64, pair, stream.value, out);
                const int second = 1 - first;
                measure(primary, kernels[second], second == 0 ? 32 : 64, pair, stream.value, out);
            }
            const size_t free = available_memory();
            if (free < minimum_sampled_free) minimum_sampled_free = free;
        }
        out << "{\"completed\":true,\"minimum_sampled_free_bytes\":" << minimum_sampled_free
            << ",\"before_free_bytes\":" << before << ",\"after_free_bytes\":" << available_memory() << "}\n";
        need(out.good(), "write terminal result");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
