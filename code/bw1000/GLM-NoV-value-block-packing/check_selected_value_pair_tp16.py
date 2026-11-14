"""Run the paired source with the original selected-value producer and recording leaves."""
import os
from pathlib import Path
import subprocess
import tempfile

from check_device_recording_tp16 import fixture


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'GLM-V-on-demand-Model-integration-r403'

ATTENTION_STUB = r"""
#include <map>
namespace infinilm::models::glm_moe_dsa {
struct AttentionPrepare {
    struct Weight {
        runtime::Tensor value;
        runtime::Tensor tensor() const {return value;}
    };
    std::size_t local_heads() const {return 4;}
    std::map<std::string, Weight> state_dict() const {
        using namespace runtime;
        return {{"kv_b_proj.weight",{std::make_shared<Node>(Shape{1792,512},DataType::kUInt8,"W")}},
                {"kv_b_proj.weight_scale_inv",{std::make_shared<Node>(Shape{224,4},DataType::kFloat32,"S")}}};
    }
    std::pair<runtime::Tensor,runtime::Tensor> expand_cached(
        runtime::Runtime& r,const runtime::Tensor& latent,const runtime::Tensor& rotary) {
        using namespace runtime;
        const auto m=latent->size(0);
        assert((latent->shape()==Shape{m,512} && rotary->shape()==Shape{m,1,64}));
        r.calls.push_back("expand_cached");
        r.syncStream();
        return {Node::empty(r,{m,4,256},DataType::kBFloat16),
                Node::empty(r,{m,4,256},DataType::kBFloat16)};
    }
};
}
"""

TEST = r"""
#include "selected_value_pair_tp16.hpp"
int main() {
    using namespace infinilm::runtime;
    using namespace glm_nov_value::tp16;
    Runtime r;
    glm_nov_value::model::AttentionPrepare prepare, other;
    Work setup(r);
    auto binding=bind_selected_pair(prepare,r,15,setup);
    auto latent=Node::empty(r,{32768,512},DataType::kBFloat16);
    auto rotary=Node::empty(r,{32768,64},DataType::kBFloat16);
    auto ids=Node::empty(r,{1313},DataType::kInt64);
    SelectedPhase original, packed;
    for(auto arm:{SelectedArm::original,SelectedArm::packed}) {
        r.calls.clear();
        Work work(r);
        auto phase=run_selected_phase(arm,prepare,r,binding,latent,rotary,ids,work);
        assert((phase.value->shape()==Shape{1313,4,256} && phase.elapsed.count()>=0));
        if (arm==SelectedArm::original) {
            original=phase;
            assert(!phase.packed_cache_dispatch.has_value());
            assert((r.calls==std::vector<std::string>{"drain","gather input","gather input","expand_cached","drain"}));
        } else {
            packed=phase;
            assert(phase.packed_cache_dispatch==true);
            assert(r.calls.size()==10 && r.calls[4]=="mode1" && r.calls.back()=="drain");
        }
    }
    original.value->word=0x3f80;
    packed.value->word=0x3f80;
    const auto equal=compare_selected_values(r,original,packed);
    assert(equal.elements==1313*4*256 && equal.exact_finite());
    packed.value->word=0x7f80;
    const auto mismatch=compare_selected_values(r,original,packed);
    assert(mismatch.different_bits==mismatch.elements && mismatch.nonfinite_packed==mismatch.elements);
    assert(mismatch.nonfinite_original==0 && !mismatch.exact_finite());
    original.value->word=0x7f80;
    const auto nonfinite=compare_selected_values(r,original,packed);
    assert(nonfinite.different_bits==0 && !nonfinite.exact_finite());
    for(auto arm:{SelectedArm::original,SelectedArm::packed}) {
        r.calls.clear();
        Work work(r);
        bool rejected=false;
        try { (void)run_selected_phase(arm,other,r,binding,latent,rotary,ids,work); }
        catch(const std::invalid_argument&) { rejected=true; }
        assert(rejected && r.calls.empty());
    }
    std::cout<<"two producer routes; three BF16 comparisons; two cross-layer rejections";
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="glm-tp16-pair-") as tmp:
        work = Path(tmp)
        (work / "glm_fp8_runtime_r809.hpp").touch()
        (work / "execution").mkdir()
        (work / "execution/submit.hpp").touch()
        (work / "glm_moe_dsa_attention_prepare.hpp").write_text(ATTENTION_STUB, encoding="utf-8")
        source = fixture().split('\nint main() {', 1)[0]
        source = source.replace(
            '    Device device() const { return {}; }\n    const Shape&',
            '    static Tensor empty(Runtime&,Shape dims,DataType dtype) {\n'
            '        return std::make_shared<Node>(std::move(dims),dtype,"input");\n'
            '    }\n    Device device() const { return {}; }\n    const Shape&',
        )
        source = source.replace(
            'using Tensor=std::shared_ptr<Node>;',
            'struct Tensor : std::shared_ptr<Node> {\n'
            '    Tensor() = default;\n'
            '    Tensor(std::shared_ptr<Node> value) : std::shared_ptr<Node>(std::move(value)) {}\n'
            '    static Tensor empty(Runtime&,Shape,DataType);\n'
            '};',
        ).replace('struct Node;\n', 'struct Node;\nstruct Runtime;\n')
        source = source.replace(
            '}\nnamespace infini::ops {',
            'Tensor Tensor::empty(Runtime& runtime,Shape dims,DataType dtype) {\n'
            '    return Node::empty(runtime,std::move(dims),dtype);\n'
            '}\n}\nnamespace infini::ops {',
        )
        source = source.replace(
            '    void syncStream() {',
            '    void memcpyD2H(void* dst,const void* src,std::size_t bytes) {\n'
            '        std::fill_n(static_cast<std::uint16_t*>(dst),bytes/2,'
            '*static_cast<const std::uint16_t*>(src));\n'
            '    }\n    void syncStream() {',
        ).replace(
            '    bool contiguous=true;',
            '    bool contiguous=true;\n'
            '    std::uint16_t word=0;\n'
            '    const void* data() const {return &word;}',
        )
        cpp = work / "pair.cpp"
        cpp.write_text(source + TEST, encoding="utf-8")
        binary = work / "pair.exe"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-I", str(HERE), "-I", str(ORIGINAL), "-I", str(work), str(cpp), "-o", str(binary)],
                       check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    main()
