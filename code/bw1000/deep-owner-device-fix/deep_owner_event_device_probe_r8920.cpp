#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <array>
#include <mutex>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <unistd.h>
static std::mutex output;static std::atomic<int>errors{0};static std::atomic<int>guarded_ready{0};
static void emit(nlohmann::json j){std::lock_guard<std::mutex>lock(output);std::puts(j.dump().c_str());std::fflush(stdout);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){alarm(85);std::array<std::thread,8>threads;for(int d=0;d<8;++d)threads[d]=std::thread([d]{nlohmann::json j={{"device",d},{"phase","createRuntime_enter"}};emit(j);try{ck(cudaSetDevice(0));auto rt=infinilm::runtime::Runtime::create({infinilm::runtime::Device::Type::kHygon,d});rt->assertOwnerThread();ck(cudaSetDevice(0));int before=-1,during=-1,after=-1;ck(cudaGetDevice(&before));cudaEvent_t a=nullptr,b=nullptr;float ms=-1;
 {infini::rt::DeviceGuard guard{rt->device().type(),rt->device().index(),true};ck(cudaGetDevice(&during));if(during!=d)throw std::runtime_error("guard device mismatch");ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,static_cast<cudaStream_t>(rt->stream())));ck(cudaEventRecord(b,static_cast<cudaStream_t>(rt->stream())));rt->syncStream();ck(cudaEventQuery(a));ck(cudaEventQuery(b));ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));}ck(cudaGetDevice(&after));if(before!=0||after!=0||!std::isfinite(ms)||ms<0)throw std::runtime_error("guard restore/finite mismatch");emit({{"phase","guarded_complete"},{"device",d},{"before",before},{"during",during},{"after",after},{"elapsed_ms",ms},{"Runtime_stream",reinterpret_cast<uintptr_t>(rt->stream())}});
 ++guarded_ready;while(guarded_ready.load()<8){if(errors.load())throw std::runtime_error("peer guarded phase failed");std::this_thread::yield();}
 // Diagnostic hazardous old path last. No old event query, retry, or drain of an unqualified record.
 ck(cudaSetDevice(0));cudaEvent_t old=nullptr;int create=static_cast<int>(cudaEventCreate(&old));emit({{"phase","unguarded_record_enter"},{"device",d},{"current",0},{"create_code",create}});int record=create? -1:static_cast<int>(cudaEventRecord(old,static_cast<cudaStream_t>(rt->stream())));emit({{"phase","unguarded_record_return"},{"device",d},{"create_code",create},{"record_code",record}});
 // Lifetime cleanup uses the resource device; event was created on0. No CCL or math work exists.
 if(old){infini::rt::DeviceGuard guard{rt->device().type(),0,true};int destroy=static_cast<int>(cudaEventDestroy(old));emit({{"phase","unguarded_destroy_return"},{"device",d},{"destroy_code",destroy}});}rt.reset();emit({{"phase","owner_complete"},{"device",d}});
 }catch(const std::exception&e){++errors;emit({{"phase","error"},{"device",d},{"message",e.what()}});}});for(auto&t:threads)t.join();emit({{"phase","all8_complete"},{"errors",errors.load()},{"no_Model_weights_CCL",true}});return errors.load()?1:0;}