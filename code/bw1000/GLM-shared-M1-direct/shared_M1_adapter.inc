// Included inside infinilm::models::glm_moe_dsa, after the original Work type.
// Uses only public ModelWeights::state_dict/Weight::tensor. No header/layout
// change, checkpoint conversion, new GPU code or routed-expert selector change.
extern "C" int glm_routed_gate_r19(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_down_r19(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
namespace shared_m1_direct_v1 {
struct State { std::atomic<int> enabled{0}; std::array<std::atomic<uint64_t>,5> count{}; };
std::array<State,8> states;
// counters: eligible, selected, paired_gateup_complete, down_complete, fallback.
bool select(rt::Runtime&r,size_t tokens,size_t local,size_t global){
 int d=r.device().index();if(d<0||d>=8)throw std::invalid_argument("shared M1 device");
 bool eligible=tokens==1&&local==256&&global==2048;
 if(eligible)++states[d].count[0];
 bool selected=eligible&&states[d].enabled.load(std::memory_order_relaxed);
 if(selected)++states[d].count[1];else if(states[d].enabled.load(std::memory_order_relaxed))++states[d].count[4];
 return selected;
}
void tensor(const rt::Tensor&t,const rt::Shape&shape,rt::DataType dtype,rt::Runtime&r){
 if(!t||t->device()!=r.device()||!t->is_contiguous()||t->shape()!=shape||t->dtype()!=dtype)
  throw std::invalid_argument("shared M1 actual tensor/dtype/stride contract");
}
std::pair<rt::Tensor,rt::Tensor> gateup(rt::Runtime&r,Work&work,const Fp8Projection&g,const Fp8Projection&u,const rt::Tensor&x,size_t offset){
 if(offset>1792||offset%256)throw std::invalid_argument("shared M1 TP offset");
 auto gw=g.state_dict(),uw=u.state_dict();
 auto wg=gw.at("weight").tensor(),sg=gw.at("weight_scale_inv").tensor();
 auto wu=uw.at("weight").tensor(),su=uw.at("weight_scale_inv").tensor();
 tensor(x,{1,6144},rt::DataType::kBFloat16,r);
 for(auto t:{wg,wu})tensor(t,{256,6144},rt::DataType::kUInt8,r);
 for(auto t:{sg,su})tensor(t,{16,48},rt::DataType::kFloat32,r);
 auto packed=work.alloc({1,6144},rt::DataType::kUInt8),xs=work.alloc({1,48},rt::DataType::kFloat32);
 auto gate=work.alloc({1,256},rt::DataType::kBFloat16),up=work.alloc({1,256},rt::DataType::kBFloat16);
 auto id=work.alloc({1},rt::DataType::kInt64);r.setDeviceMemory(id->data(),0,8);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),x,packed,xs);
 // The 5123 public wrapper refuses routes%8. Its already-linked raw r19
 // provider has no such guard and indexes token=route/8; route0 is token0.
 // Fixed expert0 here names the shared weight view, never a routed MoE choice.
 if(glm_routed_gate_r19(gate->data(),up->data(),packed->data(),xs->data(),id->data(),wg->data(),sg->data(),wu->data(),su->data(),1,offset,r.stream()))
  throw std::runtime_error("shared M1 raw gate/up launch");
 ++states[r.device().index()].count[2];return {gate,up};
}
void down(rt::Runtime&r,Work&work,rt::Tensor out,const rt::Tensor&x,const rt::Tensor&xs,const rt::Tensor&w,const rt::Tensor&ws,size_t offset){
 if(offset>1792||offset%256)throw std::invalid_argument("shared M1 down TP offset");
 tensor(out,{1,6144},rt::DataType::kBFloat16,r);tensor(x,{1,256},rt::DataType::kUInt8,r);tensor(xs,{1,2},rt::DataType::kFloat32,r);
 tensor(w,{6144,256},rt::DataType::kUInt8,r);tensor(ws,{48,16},rt::DataType::kFloat32,r);
 auto id=work.alloc({1},rt::DataType::kInt64);r.setDeviceMemory(id->data(),0,8);
 if(glm_routed_down_r19(out->data(),x->data(),xs->data(),id->data(),w->data(),ws->data(),1,offset,r.stream()))
  throw std::runtime_error("shared M1 raw down launch");
 ++states[r.device().index()].count[3];
}
}
