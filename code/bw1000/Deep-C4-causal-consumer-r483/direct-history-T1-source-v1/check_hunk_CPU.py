"""Compile the exact T1 history patch hunk against a CPU recording tensor."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

patch = Path(__file__).with_name("direct_history.patch").read_bytes()
assert sha256(patch).hexdigest() == "0ffa4c3d00bbc7393776adf60586c7e354320fa657c5e8749ecc1e352f286c39"
lines = patch.decode().splitlines()
old = "\n".join(line[1:] for line in lines if line.startswith("-") and not line.startswith("---"))
new = "\n".join(line[1:] for line in lines if line.startswith("+") and not line.startswith("+++"))

COMMON = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace rt {
enum class DataType { kUInt8, kFloat32 };
struct Runtime {
    size_t allocs = 0, fills = 0, copies = 0;
    void setDeviceMemory(void* dst, int value, size_t size) {
        ++fills;
        std::memset(dst, value, size);
    }
};
struct View;
using Tensor = std::shared_ptr<View>;
struct View {
    std::shared_ptr<std::vector<uint8_t>> storage;
    size_t offset, rows, row_bytes;
    size_t nbytes() const { return rows * row_bytes; }
    uint8_t* data() const { return storage->data() + offset; }
    Tensor narrow(std::array<std::array<size_t, 3>, 1> spec) const {
        auto at = spec[0];
        assert(at[0] == 0 && at[1] + at[2] <= rows);
        return std::make_shared<View>(View{storage, offset + at[1] * row_bytes, at[2], row_bytes});
    }
    void copy_from(Runtime& runtime, const Tensor& other) {
        assert(nbytes() == other->nbytes());
        ++runtime.copies;
        std::memcpy(data(), other->data(), nbytes());
    }
};
Tensor make(size_t rows, size_t row_bytes) {
    return std::make_shared<View>(View{std::make_shared<std::vector<uint8_t>>(rows * row_bytes),
                                      0, rows, row_bytes});
}
}

struct Work {
    rt::Runtime& runtime;
    std::vector<rt::Tensor> held;
    rt::Tensor alloc(std::array<size_t, 2> shape, rt::DataType type) {
        ++runtime.allocs;
        auto t = rt::make(shape[0], shape[1] * (type == rt::DataType::kUInt8 ? 1 : 4));
        held.push_back(t);
        return t;
    }
};
struct PagedIndexerCache {
    struct Packed { rt::Tensor values, scales; };
    Packed last{};
    std::vector<int64_t> slots;
    Packed gather(rt::Runtime&, const std::vector<int64_t>& ids) {
        slots = ids;
        last = {rt::make(ids.size(), 128), rt::make(ids.size(), 4)};
        for (size_t i = 0; i < ids.size(); ++i) {
            std::memset(last.values->data() + i * 128, int(ids[i] % 251), 128);
            std::memset(last.scales->data() + i * 4, int(ids[i] % 251), 4);
        }
        return last;
    }
};
struct Request { size_t row, past, tokens; };
struct Pending { PagedIndexerCache::Packed new_keys; std::vector<int64_t> slots; };
struct Result {
    std::vector<uint8_t> key, scale;
    std::vector<int64_t> gathered_slots, pending_slots;
    size_t allocs, fills, copies;
    bool aliases, held;
};
"""
ENTRY = r"""
Result FUNCTION(Request request, size_t request_count, bool batched) {
    const size_t capacity_ = 8192;
    const size_t before = request.past / 4, after = (request.past + request.tokens) / 4;
    const size_t width = std::max(size_t(1), after);
    rt::Runtime r;
    Work work{r, {}};
    PagedIndexerCache cache;
    auto* cache_ = &cache;
    std::vector<Request> requests(request_count, request);
    // The baseline hunk has no branch on request count.
    (void)requests;
    PagedIndexerCache::Packed historical_batch;
    size_t historical_begin = 0, new_begin = 0;
    Pending pending{{rt::make(after - before, 128), rt::make(after - before, 4)}, {}};
    if (after > before) {
        std::memset(pending.new_keys.values->data(), 113, pending.new_keys.values->nbytes());
        std::memset(pending.new_keys.scales->data(), 59, pending.new_keys.scales->nbytes());
    }
    if (batched) {
        std::vector<int64_t> batch_slots(before);
        std::iota(batch_slots.begin(), batch_slots.end(), int64_t(request.row * capacity_));
        historical_batch = cache.gather(r, batch_slots);
        cache.last = {};
        cache.slots.clear();
    }
BODY
    assert(new_begin == after - before);
    const bool aliases = cache.last.values && key == cache.last.values && scale == cache.last.scales;
    const bool held = std::count(work.held.begin(), work.held.end(), key) == 1 &&
                      std::count(work.held.begin(), work.held.end(), scale) == 1;
    return {{key->data(), key->data() + key->nbytes()},
            {scale->data(), scale->data() + scale->nbytes()},
            cache.slots, pending.slots, r.allocs, r.fills, r.copies, aliases, held};
}
"""
MAIN = r"""
int main() {
    size_t direct = 0, fallback = 0;
    auto test = [&](Request r, size_t requests, bool batched, bool expected_direct) {
        auto a = baseline(r, requests, batched);
        auto b = candidate(r, requests, batched);
        assert(a.key == b.key && a.scale == b.scale);
        assert(a.gathered_slots == b.gathered_slots && a.pending_slots == b.pending_slots);
        assert(!a.aliases && b.aliases == expected_direct && b.held);
        if (expected_direct) {
            assert(a.allocs == 2 && a.fills == 2 && a.copies == 2);
            assert(b.allocs == 0 && b.fills == 0 && b.copies == 0);
            ++direct;
        } else {
            assert(a.allocs == b.allocs && a.fills == b.fills && a.copies == b.copies);
            ++fallback;
        }
    };
    test({0, 4, 1}, 1, false, true);
    test({0, 8496, 1}, 1, false, true);
    test({0, 32766, 1}, 1, false, true);
    test({1, 8496, 1}, 1, false, true);
    test({0, 0, 1}, 1, false, false);
    test({0, 3, 1}, 1, false, false);
    test({0, 32767, 1}, 1, false, false);
    test({0, 4, 2}, 1, false, false);
    test({0, 8192, 512}, 1, false, false);
    test({1, 8496, 1}, 2, true, false);
    test({1, 8496, 1}, 2, false, false);
    std::cout << direct << " " << fallback << "\n";
}
"""
cpp = COMMON + ENTRY.replace("FUNCTION", "baseline").replace("BODY", old)
cpp += ENTRY.replace("FUNCTION", "candidate").replace("BODY", new) + MAIN
with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / "direct_history.cpp"
    executable = Path(directory) / "direct_history.exe"
    source.write_text(cpp, encoding="utf8", newline="\n")
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    str(source), "-o", str(executable)], check=True, timeout=30)
    result = subprocess.run([str(executable)], capture_output=True, check=True, text=True, timeout=5)
    assert result.stdout.split() == ["4", "7"]
print(json.dumps(dict(compiled_hunk=True, direct_cases=4, fallback_cases=7,
                      payload_and_slot_bytes_match=True, actual_target_compile=False)))
