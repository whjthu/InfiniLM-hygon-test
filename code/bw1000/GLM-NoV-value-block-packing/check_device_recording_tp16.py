"""Record the TP16 device helper calls without a driver or an SDK."""
import os
from pathlib import Path
import subprocess
import tempfile

from check_device_recording import CPP

HERE = Path(__file__).resolve().parent

TEST = r"""
int main() {
    using namespace infinilm::runtime;
    namespace pack = glm_nov_value::tp16;
    Runtime r, other;
    const auto weights = std::make_shared<Node>(Shape{1792,512},DataType::kUInt8,"W");
    const auto scales = std::make_shared<Node>(Shape{224,4},DataType::kFloat32,"S");
    const auto latent = std::make_shared<Node>(Shape{32768,512},DataType::kBFloat16,"latent");
    const auto rotary = std::make_shared<Node>(Shape{32768,64},DataType::kBFloat16,"rotary");
    for (const auto rank : {7u, 15u}) {
        auto packed = [&] {
            r.calls.clear();
            Work work(r);
            auto result = pack::pack_device(r, weights, scales, rank, work);
            assert(r.calls.size()==21 && r.calls.back()=="drain");
            for (std::size_t i=0; i<pack::tiles.size(); ++i) {
                const auto tile=pack::tiles[i];
                assert(r.calls[2*i]=="copy W:"+std::to_string(tile*128)+" -> t0:"+std::to_string(i*128));
                assert(r.calls[2*i+1]=="copy S:"+std::to_string(rank*14+tile)+" -> t1:"+std::to_string(i));
            }
            return result;
        }();
        for (const auto m : {65u, 1313u, 1314u, 1315u, 2048u}) {
            const auto ids=std::make_shared<Node>(Shape{m},DataType::kInt64,"ids");
            for (const bool fallback : {false,true}) {
                r.calls.clear();
                r.deny=fallback;
                Work work(r);
                auto result=pack::selected(r,packed,latent,rotary,ids,work);
                assert((result.value->dims==Shape{m,4,256}) && result.cached_fp8_dispatch!=fallback);
                std::vector<std::string> wanted{"gather latent","gather rotary","quant","mode1"};
                if(fallback) wanted.emplace_back("sdk");
                for (std::size_t head=0; head<pack::heads; ++head)
                    wanted.push_back("copy t4:"+std::to_string(pack::starts[head])+" -> t5:"+std::to_string(head*256));
                wanted.emplace_back("drain");
                assert(r.calls==wanted);
            }
        }
        const auto ids=std::make_shared<Node>(Shape{1313},DataType::kInt64,"ids");
        const auto expect_rejection=[](auto call) {
            try { call(); } catch(const std::invalid_argument&) { return; }
            assert(false);
        };
        const auto expect_no_ops=[&](Runtime& target, const Tensor& input) {
            target.calls.clear();
            {
                Work work(target);
                expect_rejection([&] { pack::selected(target,packed,latent,rotary,input,work); });
                assert(target.calls.empty());
            }
        };
        expect_no_ops(other,ids);
        expect_no_ops(r,std::make_shared<Node>(Shape{2049},DataType::kInt64,"bad"));
        expect_no_ops(r,std::make_shared<Node>(Shape{64},DataType::kInt64,"bad"));
        expect_no_ops(r,std::make_shared<Node>(Shape{1313},DataType::kBFloat16,"bad"));
        r.graph=true;
        expect_no_ops(r,ids);
        r.graph=false;
        {
            Work work(r);
            expect_rejection([&] { pack::pack_device(r,weights,scales,8,work); });
        }
    }
    std::cout<<"2 global ranks, 20 dispatch paths and 12 guarded failures";
}
"""


def fixture():
    source = CPP.split('\nint main(){', 1)[0]
    source = source.replace(
        'struct BlockQuantF8E4m3',
        'struct Fp8BlockLinear { static int DefaultConfig(infinilm::runtime::Device::Type){return 0;} };\n'
        'struct BlockQuantF8E4m3',
    ).replace('{2560,512}', '{1280,512}').replace('{20,4}', '{10,4}').replace('==2560', '==1280')
    sdk = r"""
namespace infinilm::execution {
template<class Op>
void submit(runtime::Runtime& r, RecordingPolicy, int, const runtime::Tensor& y,
            const runtime::Tensor& x, const runtime::Tensor& xs, const runtime::Tensor& w,
            const runtime::Tensor& ws, std::size_t ro, std::size_t co) {
    static_assert(std::is_same_v<Op,infini::ops::Fp8BlockLinear>);
    assert((w->dims==runtime::Shape{1280,512})&&(ws->dims==runtime::Shape{10,4}));
    assert(y->size(0)==x->size(0)&&y->size(1)==1280&&xs->size(1)==4&&ro==0&&co==0);
    r.calls.push_back("sdk");
}
}
#include "packed_value_device_tp16.hpp"
"""
    return '#include "packed_kv_b.hpp"\n' + source + sdk + TEST


def main():
    with tempfile.TemporaryDirectory(prefix="glm-tp16-value-") as tmp:
        work = Path(tmp)
        (work / "glm_fp8_runtime_r809.hpp").touch()
        (work / "execution").mkdir()
        (work / "execution/submit.hpp").touch()
        source = work / "check.cpp"
        source.write_text(fixture(), encoding="utf-8")
        binary = work / "check.exe"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-I", str(HERE), "-I", str(work), str(source), "-o", str(binary)],
                       check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    main()
