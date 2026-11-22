"""Patch only the original DeepSeek AttentionPrepare inverse-output call."""

import argparse
from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[3] / "code/bw1000/Deep-PD-disaggregation-r382/native/original"
TABLE = HERE.parent / "inverse-output-rope-caller-v1/inverse_output_policy.hpp"
PARENT_SHA = "3f388df12b9f8884d72cbe6d4a29f2b273934a3dd0cc49c81c93b66fe67ba56b"
HEADER_SHA = "dbd063233069b5f36953c02b1cb8170255af0ffc7eceb1df9658d3e0482461a1"
TABLE_SHA = "88d232b5c3e2f42c2924373751317e82488f397625384809197ab444458cfc51"
ENTRY = "rt::Tensor AttentionPrepare::inverse_output_rope("
LEGACY = ("Work work{r};auto result=work.alloc(x->shape(),x->dtype());"
          "result->copy_from(r,x);rotate(r,result,std::nullopt,pos,true);r.syncStream();return result;")
DECLARATIONS = """extern "C" int deep_inverse_rope_h1_pack32_v1(void*, const void*, const void*, size_t, size_t, void*);
extern "C" int deep_inverse_rope_h1_pack64_v1(void*, const void*, const void*, size_t, size_t, void*);
"""
FUSED = """
    static const auto requested = inverse_output::requested_arm(std::getenv("INFINILM_DEEP_INVERSE_OUTPUT_ROPE"));
    const size_t tokens = x->size(0);
    if (requested != inverse_output::Arm::legacy && tokens == 1 && local_heads_ == 8 && x->is_contiguous()) {
        if (void* stream = r.stream()) {
            if (!pos || pos->device() != device_ || pos->dtype() != rt::DataType::kInt64 ||
                !pos->is_contiguous() || pos->shape() != rt::Shape{tokens}) {
                throw std::invalid_argument("DeepSeek rotary position owner/shape differs");
            }
            std::vector<int64_t> positions(tokens);
            r.memcpyD2H(positions.data(), pos->data(), tokens * sizeof(int64_t));
            // Keep the host coefficients alive until Work drains on success or failure.
            const auto cache = inverse_output::make_table(positions, max_position_, theta_, factor_,
                                                          beta_fast_, beta_slow_, original_);
            Work work{r};
            auto result = work.alloc(x->shape(), x->dtype());
            auto table = work.alloc({tokens, 64}, rt::DataType::kFloat32);
            r.memcpyH2D(table->data(), cache.data(), cache.size() * sizeof(float), false);
            const auto launch = requested == inverse_output::Arm::pack32 ?
                                deep_inverse_rope_h1_pack32_v1 : deep_inverse_rope_h1_pack64_v1;
            const int status = launch(result->data(), x->data(), table->data(), tokens, local_heads_, stream);
            if (status != 0) {
                throw std::runtime_error("DeepSeek fused inverse RoPE launch failed: " + std::to_string(status));
            }
            r.syncStream();
            return result;
        }
    }
    """


def check(raw, expected):
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen DeepSeek caller input differs")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Frozen DeepSeek caller anchor differs")
    return text.replace(old, new, 1)


def render(original, header, table):
    for raw, digest in ((original, PARENT_SHA), (header, HEADER_SHA), (table, TABLE_SHA)):
        check(raw, digest)
    eol = "\r\n" if b"\r\n" in original else "\n"
    source = original.decode().replace("\r\n", "\n")
    include = '#include "execution/submit.hpp"\n'
    source = replace_once(source, include, include + '#include "inverse_output_width_policy.hpp"\n#include <cstdlib>\n')
    source = replace_once(source, ENTRY, DECLARATIONS + ENTRY)
    source = replace_once(source, LEGACY, FUSED + LEGACY)
    text = table.decode()
    begin = text.index("enum class Arm")
    end = text.index("inline std::vector<float> make_table")
    policy = text[:begin] + """enum class Arm { legacy, pack32, pack64 };

inline Arm requested_arm(const char* value) {
    const std::string_view name = value ? value : "";
    if (name.empty() || name == "legacy") {
        return Arm::legacy;
    }
    if (name == "h1-pack32") {
        return Arm::pack32;
    }
    if (name == "h1-pack64") {
        return Arm::pack64;
    }
    throw std::invalid_argument("Invalid DeepSeek inverse RoPE width for this candidate");
}

""" + text[end:]
    return source.replace("\n", eol).encode(), policy.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stem = ARCHIVE / "deepseek_v4_attention_prepare.cpp"
    rendered = render(stem.read_bytes(), stem.with_suffix(".hpp").read_bytes(), TABLE.read_bytes())
    args.output.mkdir()
    for name, raw in zip(("deepseek_v4_attention_prepare.cpp", "inverse_output_width_policy.hpp"), rendered):
        with (args.output / name).open("xb") as stream:
            stream.write(raw)


if __name__ == "__main__":
    main()
