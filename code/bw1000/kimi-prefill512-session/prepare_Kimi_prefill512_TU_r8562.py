from pathlib import Path
import hashlib
source=Path('Kimi-native-tracer-source-r8504/kimi_k3_delta_attention.cpp');s=source.read_text();assert hashlib.sha256(source.read_bytes()).hexdigest()=='a808191a2edc73d37ab3c8bb6531e5b057362b59fd5456f262bc2e8c231488da'
needle='inline std::array<std::array<std::atomic<uint64_t>,9>,8> gate_projection_counts_r4576{};'
s=s.replace(needle,'''// Default-OFF scoped uniform512 prefill candidate; no weight/layout changes.
inline std::array<std::array<std::atomic<uint64_t>,3>,8> prefill512_counts_r8562{};
extern "C" int modern_kimi_prefill512_counts_r8562(int device,uint64_t*out,size_t n){
 if(device<0||device>=8||!out||n!=3)return 1;
 for(size_t i=0;i<3;++i)out[i]=prefill512_counts_r8562[device][i].load();return 0;
}
'''+needle)
needle='    auto selected_input_projection_r4305=[&](const auto& projection,size_t output_width){'
insert='''    auto flag512_r8562=[](const char*name){const char*v=std::getenv(name);if(!v||std::string(v)=="0")return false;if(std::string(v)=="1")return true;throw std::invalid_argument("uniform512 flag must be0or1");};
    const bool mode512_r8562=flag512_r8562("INFINILM_KIMI_PREFILL512_R8562");
    // The serial owner driver derives ELIGIBLE from the immutable actual job:
    // kind=prefill, all actual pasts0, every request input length exactly512.
    // M divisibility alone is not eligibility (single1024/mixed1033 fallback).
    const bool eligible512_r8562=mode512_r8562&&flag512_r8562("INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562");
    const bool split512_r8562=eligible512_r8562&&batch_size==1&&seq_len>512&&seq_len<=8192&&seq_len%512==0;
    // Keep original projection outputs/views through this forward. Normal
    // provided-Runtime leases retain queued device uses; no new success fence.
    std::vector<infinilm::runtime::Tensor>held512_r8562;
    auto project512_r8562=[&](const auto&projection,infinilm::runtime::Tensor source,size_t width){
      runtime.assertOwnerThread();
      if(runtime.isGraphRecording()||!source||source->device()!=runtime.device()||source->dtype()!=infinilm::runtime::DataType::kBFloat16||(source->ndim()!=3||source->shape()[0]!=1||source->shape()[1]!=seq_len))
        throw std::invalid_argument("uniform512 requires eager sameowner BF16 packedinput");
      infinilm::runtime::Tensor output;
      try{
        output=infinilm::runtime::Tensor::empty(runtime,{1,seq_len,width},source->dtype());
        auto&counts=prefill512_counts_r8562.at(runtime.device().index());counts[0].fetch_add(1,std::memory_order_relaxed);counts[1].fetch_add(seq_len,std::memory_order_relaxed);
        for(size_t row=0;row<seq_len;row+=512){
          auto input_row=source->narrow({{1,row,512}});held512_r8562.push_back(input_row);
          auto value=projection->forward(runtime,input_row);held512_r8562.push_back(value);
          if(!value||value->shape()!=infinilm::runtime::Shape{1,512,width}||value->dtype()!=source->dtype()||value->device()!=runtime.device())throw std::logic_error("uniform512 childprojection geometry differs");
          output->narrow({{1,row,512}})->copy_from(runtime,value);counts[2].fetch_add(1,std::memory_order_relaxed);
        }
        return output;
      }catch(...){try{runtime.syncStream();}catch(...){std::terminate();}throw;}
    };
'''
s=s.replace(needle,insert+needle+'\n        if(split512_r8562)return project512_r8562(projection,input,output_width);')
needle='    auto selected_gate_projection_r4576=[&](const auto&projection,infinilm::runtime::Tensor source,size_t output_width,int branch){'
s=s.replace(needle,needle+'\n      if(split512_r8562)return project512_r8562(projection,source,output_width);')
p=Path('kimi_k3_delta_attention_prefill512_r8562.cpp');p.write_text(s,encoding='utf-8',newline='\n');print(hashlib.sha256(p.read_bytes()).hexdigest())
