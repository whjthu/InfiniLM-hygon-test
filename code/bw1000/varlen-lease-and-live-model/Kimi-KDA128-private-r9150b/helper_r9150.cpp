#include "helper_r9150.hpp"
#include "raw_r9150.hpp"
#include <infini/ops.h>
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
namespace kda_threads_r9150 {
struct Saved{int threads=0;std::shared_ptr<rt::Memory>workspace;cudaEvent_t a=nullptr,b=nullptr;bool pending=false;uint64_t serial=0;std::string json="{}";};static thread_local std::unordered_map<rt::Runtime*,Saved>saved;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void mode(rt::Runtime&r,int threads,size_t bytes){r.assertOwnerThread();if(threads!=0&&threads!=128&&threads!=256)throw std::invalid_argument("9150 mode");auto&s=saved[&r];if(s.pending)throw std::invalid_argument("9150 pending event");s.threads=threads;if(threads){if(!s.workspace||s.workspace->size()<bytes)s.workspace=r.allocateMemory(bytes);if(!s.a){ck(cudaEventCreate(&s.a));ck(cudaEventCreate(&s.b));}}}
int selected(rt::Runtime&r){auto it=saved.find(&r);return it==saved.end()?0:it->second.threads;}
void run(rt::Runtime&r,const std::array<rt::Tensor,12>&x,float scale,float lower,bool norm){r.assertOwnerThread();auto&s=saved.at(&r);if(!s.threads||r.isGraphRecording()||!s.workspace)throw std::invalid_argument("9150 eager mode/workspace/pending");std::vector<infini::ops::Tensor>v;for(auto&t:x){if(!t||t->device()!=r.device())throw std::invalid_argument("9150 wrong tensor device");v.emplace_back(const_cast<std::byte*>(t->data()),t->shape(),t->dtype(),t->device(),t->strides());}if(modern_kimi_kda_threads_raw_r9150(r.device().index(),r.stream(),s.workspace->data(),s.workspace->size(),v.data(),scale,lower,norm,s.threads,s.a,s.b))throw std::runtime_error(modern_kimi_kda_threads_error_r9150());s.pending=true;++s.serial;}
void collect(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);if(it==saved.end()||!it->second.pending)return;auto&s=it->second;float ms=0;ck(cudaEventElapsedTime(&ms,s.a,s.b));s.json=nlohmann::json{{"threads",s.threads},{"serial",s.serial},{"KDA_snapshot_and_kernel_stream_ms",ms},{"original_metadata_DtoH_syncs",3},{"full_pool_snapshot",true},{"lease",false},{"event_extra_syncs",0}}.dump();s.pending=false;}
const char*last(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);return it==saved.end()?"{}":it->second.json.c_str();}
void clear(rt::Runtime&r){r.assertOwnerThread();auto it=saved.find(&r);if(it==saved.end())return;r.syncStream();if(it->second.a)ck(cudaEventDestroy(it->second.a));if(it->second.b)ck(cudaEventDestroy(it->second.b));saved.erase(it);}
}
