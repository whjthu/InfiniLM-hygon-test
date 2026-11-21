"""Bind the fused inverse RoPE only at the archived DeepSeek output caller."""

import argparse
from hashlib import sha256
from pathlib import Path


PARENT_SHA = "3f388df12b9f8884d72cbe6d4a29f2b273934a3dd0cc49c81c93b66fe67ba56b"
HEADER_SHA = "dbd063233069b5f36953c02b1cb8170255af0ffc7eceb1df9658d3e0482461a1"
PARENT_COPY = "result->copy_from(r,x);rotate(r,result,std::nullopt,pos,true);r.syncStream();return result;"
ANCHOR = "void AttentionPrepare::rotate("
HELPER = """namespace {
std::vector<float> output_rope_table(rt::Runtime& r, const rt::Tensor& pos,
                                     rt::Device device, size_t tokens, size_t max_position,
                                     double theta, double factor, double beta_fast, double beta_slow,
                                     size_t original) {
    r.assertOwnerThread();
    if (r.device() != device || r.isGraphRecording() || !pos || pos->device() != device ||
        pos->dtype() != rt::DataType::kInt64 || !pos->is_contiguous() ||
        pos->shape() != rt::Shape{tokens}) {
        throw std::invalid_argument("DeepSeek rotary position owner/shape differs");
    }
    std::vector<int64_t> positions(tokens);
    r.memcpyD2H(positions.data(), pos->data(), tokens * sizeof(int64_t));
    return inverse_output::make_table(positions, max_position, theta, factor, beta_fast,
                                      beta_slow, original);
}
}
extern "C" int deep_inverse_rope_h1_v1(void*, const void*, const void*, size_t, size_t, void*);
extern "C" int deep_inverse_rope_h2_v1(void*, const void*, const void*, size_t, size_t, void*);
"""
LEGACY_TABLE = """    const size_t tokens = q->size(0);
    auto cache = output_rope_table(r, pos, device_, tokens, max_position_, theta_, factor_,
                                   beta_fast_, beta_slow_, original_);
    std::vector<int64_t> rows(tokens);
    std::iota(rows.begin(), rows.end(), 0);
"""
FUSED_PATH = """
    static const auto requested = inverse_output::requested_arm(std::getenv("INFINILM_DEEP_INVERSE_OUTPUT_ROPE"));
    const bool contiguous = requested != inverse_output::Arm::legacy && tokens == 1 &&
                            local_heads_ == 8 && x->is_contiguous();
    void* stream = contiguous ? r.stream() : nullptr;
    const auto arm = inverse_output::admitted_arm(requested, tokens, local_heads_, contiguous, stream);
    if (arm == inverse_output::Arm::legacy) {
        result->copy_from(r, x);
        rotate(r, result, std::nullopt, pos, true);
        r.syncStream();
        return result;
    }
    auto cache = output_rope_table(r, pos, device_, tokens, max_position_, theta_, factor_,
                                   beta_fast_, beta_slow_, original_);
    // Enqueue owns the table through the final sync, including exception unwinding.
    Work enqueue{r};
    auto table = enqueue.alloc({tokens, 64}, rt::DataType::kFloat32);
    r.memcpyH2D(table->data(), cache.data(), cache.size() * sizeof(float), false);
    const auto launch = arm == inverse_output::Arm::h1 ? deep_inverse_rope_h1_v1 : deep_inverse_rope_h2_v1;
    const int status = launch(result->data(), x->data(), table->data(), tokens, local_heads_, stream);
    if (status != 0) {
        throw std::runtime_error("DeepSeek fused inverse RoPE launch failed: " + std::to_string(status));
    }
    r.syncStream();
    return result;
"""


def render(parent, parent_header):
    if sha256(parent).hexdigest() != PARENT_SHA or sha256(parent_header).hexdigest() != HEADER_SHA:
        raise ValueError("Archived DeepSeek AttentionPrepare source or header differs")
    source = parent.decode("utf-8")
    eol = "\r\n" if "\r\n" in source else "\n"
    source = source.replace("\r\n", "\n")
    source = source.replace('#include "execution/submit.hpp"\n',
                            '#include "execution/submit.hpp"\n#include "inverse_output_policy.hpp"\n#include <cstdlib>\n', 1)
    rotate = source.index(ANCHOR)
    a = source.index("{", rotate) + 1
    b = source.index(" Work work{r};auto table=", a)
    source = source[:rotate] + HELPER + source[rotate:a] + "\n" + LEGACY_TABLE + source[b:]
    if source.count(PARENT_COPY) != 1:
        raise ValueError("Archived inverse output copy boundary differs")
    old_prefix = "Work work{r};auto result=work.alloc(x->shape(),x->dtype());"
    if source.count(old_prefix + PARENT_COPY) != 1:
        raise ValueError("Archived inverse output owner boundary differs")
    source = source.replace(
        old_prefix + PARENT_COPY, "Work work{r};auto result=work.alloc(x->shape(),x->dtype());\n"
        "    const size_t tokens = x->size(0);" + FUSED_PATH.rstrip("\n"), 1,
    )
    return source.replace("\n", eol).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("original_header", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    modified = render(args.original.read_bytes(), args.original_header.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(modified)
    print(sha256(modified).hexdigest())


if __name__ == "__main__":
    main()
