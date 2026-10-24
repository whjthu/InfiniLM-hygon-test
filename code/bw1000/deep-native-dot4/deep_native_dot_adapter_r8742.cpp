// Finite component only. Mode0 default baseline; mode1 native signed dot4.
#include "deepseek_routed_mlp_batch_r20.hpp"
#include "deep_routed_sdot4_forward_r8742.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <nlohmann/json.hpp>
namespace rt=infinilm::runtime;
namespace model=infinilm::models::deepseek_v4;
namespace {
using Clock=std::chrono::steady_clock;
thread_local std::string error,report;
struct Context{std::shared_ptr<rt::Runtime>r;int device;};
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void owner(Context*c){if(!c||!c->r)throw std::invalid_argument("context");c->r->assertOwnerThread();int d=-1;ck(cudaGetDevice(&d));if(d!=c->device||c->r->isGraphRecording())throw std::invalid_argument("owner/device/eager");}
}
extern "C" const char*deep_native_dot_error_r8742(){return error.c_str();}
extern "C" const char*deep_native_dot_report_r8742(){return report.c_str();}
extern "C" void*deep_native_dot_create_r8742(int device){try{infini::rt::DeviceGuard g{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<Context>();c->device=device;c->r=rt::Runtime::create({rt::Device::Type::kHygon,device});c->r->enableOwnerDrainFailStop();return c.release();}catch(const std::exception&e){error=e.what();return nullptr;}}
// Pointers: BF16x[32,4096], I64ids[32,6], F32weights[32,6],
// I8w13[256,512,4096], F32s13[256,512,1], I8w2[256,4096,256],
// F32s2[256,4096,1], F32out[32,4096]. Borrowed producer complete before entry.
extern "C" int deep_native_dot_forward_r8742(void*raw,int mode,void*const*p){auto*c=static_cast<Context*>(raw);try{
 owner(c);if(!p||mode<0||mode>1)throw std::invalid_argument("mode/pointers");auto&r=*c->r;std::vector<rt::Tensor>held;
 auto wrap=[&](size_t j,rt::Shape shape,rt::DataType dtype){if(!p[j])throw std::invalid_argument("null");cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p[j]));if(a.type!=cudaMemoryTypeDevice||a.device!=c->device)throw std::invalid_argument("pointer owner");auto t=rt::Tensor::from_blob(p[j],shape,dtype,r.device());held.push_back(t);return t;};
 auto x=wrap(0,{32,4096},rt::DataType::kBFloat16),ids=wrap(1,{32,6},rt::DataType::kInt64),weights=wrap(2,{32,6},rt::DataType::kFloat32),w13=wrap(3,{256,512,4096},rt::DataType::kInt8),s13=wrap(4,{256,512,1},rt::DataType::kFloat32),w2=wrap(5,{256,4096,256},rt::DataType::kInt8),s2=wrap(6,{256,4096,1},rt::DataType::kFloat32),out=wrap(7,{32,4096},rt::DataType::kFloat32);
 r.syncStream();uint64_t g=model::routed_sdot4_r8742::native_calls[0],d=model::routed_sdot4_r8742::native_calls[1];auto begin=Clock::now();rt::Tensor value;
 if(mode==0)value=model::routed_r20::forward(r,x,ids,weights,w13,s13,w2,s2);
 else value=model::routed_sdot4_r8742::forward(r,x,ids,weights,w13,s13,w2,s2);
 auto returned=Clock::now();out->copy_from(r,value);r.syncStream();auto end=Clock::now();
 report=nlohmann::json({{"mode",mode},{"whole_routed_return_seconds",std::chrono::duration<double>(returned-begin).count()},{"routed_plus_outputcopy_drain_seconds",std::chrono::duration<double>(end-begin).count()},{"native_gate_calls",model::routed_sdot4_r8742::native_calls[0]-g},{"native_down_calls",model::routed_sdot4_r8742::native_calls[1]-d}}).dump();return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->r)try{c->r->syncStream();}catch(...){std::terminate();}return 1;}}
extern "C" int deep_native_dot_destroy_r8742(void*raw,int*expired){auto*c=static_cast<Context*>(raw);try{owner(c);if(!expired)throw std::invalid_argument("expired");std::weak_ptr<rt::Runtime>w=c->r;c->r->syncStream();c->r->trimMemory();c->r.reset();*expired=w.expired();delete c;return *expired?0:1;}catch(const std::exception&e){error=e.what();return 1;}}
