// Private default-OFF r405. No public class/SDK/Runtime ABI change.
namespace projection_r405 {
namespace rt = infinilm::runtime;
struct Control { bool enabled=false; uint64_t chosen=0,fallback=0,packs=0,packed_bytes=0; std::array<uint64_t,4> consumed{}; uint64_t epochs=0; double setup_ms=0; };
inline thread_local Control control;
class PackedQkv final : public infinilm::layers::linear::QKVParallelLinear {
    rt::Tensor packed_;
    infinilm::quantization::ParamsMap params_;
public:
    using infinilm::layers::linear::QKVParallelLinear::QKVParallelLinear;
    void prepare(rt::Runtime&r,const std::array<rt::Tensor,4>&w) {
        r.assertOwnerThread();
        if(packed_ || r.isGraphRecording()) throw std::logic_error("r405 pack once/eager only");
        const std::array<size_t,4> widths{2304,128,6,768};
        for(size_t j=0;j<w.size();++j)
            if(!w[j] || w[j]->shape()!=rt::Shape{widths[j],7168} || w[j]->dtype()!=rt::DataType::kBFloat16 || w[j]->device()!=r.device() || !w[j]->is_contiguous())
                throw std::invalid_argument("r405 exact TP16 BF16 row-major input projection weights");
        // Retained member and slices survive all asynchronous copies and unwind.
        // On failure the component owner is Failed and is destroyed, never reused.
        auto begin=std::chrono::steady_clock::now();
        packed_=rt::Tensor::empty(r,{3206,7168},rt::DataType::kBFloat16);
        std::array<rt::Tensor,4> slices;size_t offset=0;
        try {
            for(size_t j=0;j<w.size();++j) { slices[j]=packed_->narrow({{0,offset,widths[j]}});slices[j]->copy_from(r,w[j]);offset+=widths[j]; }
            r.syncStream();params_.emplace("weight",packed_);
        } catch(...) { auto e=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(e); }
        ++control.packs;control.packed_bytes=packed_->nbytes();
        control.setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    std::array<rt::Tensor,4> combined(rt::Runtime&r,rt::Tensor&input) const {
        r.assertOwnerThread();
        if(!packed_ || r.isGraphRecording() || input->shape()!=rt::Shape{1,1,7168} || input->dtype()!=rt::DataType::kBFloat16 || !input->is_contiguous() || has_bias() || alpha()!=1.f || output_dtype()!=rt::DataType::kBFloat16) {
            ++control.fallback;throw std::invalid_argument("r405 ON requires prepared C1T1 BF16/current geometry; no fallback measurement");
        }
        rt::Tensor y;std::array<rt::Tensor,4> parts;
        try {
            // Exact same current NoneQuantization/SDK dense path, one GEMM.
            y=get_quantization()->forward(r,params_,input,false,1.f);
            if(y->shape()!=rt::Shape{1,1,3206} || y->dtype()!=rt::DataType::kBFloat16 || !y->is_contiguous())throw std::logic_error("r405 packed output layout");
            auto flat=y->view({3206});size_t offset=0;
            const std::array<size_t,4> widths{2304,128,6,768};
            for(size_t j=0;j<4;++j){auto p=flat->narrow({{0,offset,widths[j]}});parts[j]=p->view({1,1,widths[j]});offset+=widths[j];
                if(!parts[j]->is_contiguous() || parts[j]->strides()!=rt::Strides{rt::Stride(widths[j]),rt::Stride(widths[j]),1})throw std::logic_error("r405 canonical singleton strides");}
            ++control.chosen;return parts;
        } catch(...) {auto e=std::current_exception();try{r.syncStream();}catch(...){std::terminate();}std::rethrow_exception(e);}
    }
};
} // namespace projection_r405
