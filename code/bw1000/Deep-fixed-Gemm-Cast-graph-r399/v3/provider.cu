#include "provider.hpp"
#include "native/cuda/hygon/blas.h"
#include "native/cuda/hygon/blas_utils.h"
#include "native/cuda/hygon/ops/cast/kernel.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace deep399 {namespace {
using namespace infini::ops;
void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void blas_check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("r399 BLAS status "+std::to_string(int(e)));}
class Fixed final:public PreparedOps {
 Tensor a_,b_,y_,z_;
 int device_;CUcontext context_=nullptr;
 std::thread::id owner_=std::this_thread::get_id();
 cublasHandle_t handle_=nullptr;
 std::unique_ptr<Operator<Cast,Device::Type::kHygon,0>> cast_;
 void*stream_=nullptr;void*workspace_=nullptr;bool ready_=false;
 const float alpha_=1.f,beta_=0.f; // Captured HOST scalar pointers share the prepared operator lease.
 static constexpr size_t workspace_bound_=8*1024*1024;
 void owner()const{
  int d=-1;CUcontext c=nullptr;cuda_check(cudaGetDevice(&d));
  if(cuCtxGetCurrent(&c)!=CUDA_SUCCESS||d!=device_||c!=context_||owner_!=std::this_thread::get_id())throw std::invalid_argument("r399 provider owner/device/context");
 }
 void stream(void*s)const{owner();if(!ready_||s!=stream_)throw std::logic_error("r399 fixed prepared stream");}
 void launch_gemm()const{
  // Exact current generic row-major F32 geometry/default-algo path. No provider switch,
  // TF32 policy change, mixed-precision replacement, or cublas/rocblas handle cast.
  // SetStream can reset a vendor workspace: always bind explicit workspace AFTER it.
  blas_check(cublasSetStream(handle_,static_cast<cudaStream_t>(stream_)));
  blas_check(cublasSetWorkspace(handle_,workspace_,workspace_bound_));
  using B=Blas<Device::Type::kHygon>;
  const auto type=BlasUtils<Device::Type::kHygon>::GetDataType(DataType::kFloat32);
  const auto compute=BlasUtils<Device::Type::kHygon>::GetComputeType(DataType::kFloat32);
  blas_check(B::BlasGemmStridedBatchedEx(handle_,B::BLAS_OP_N,B::BLAS_OP_N,
      512,16,2048,&alpha_,b_.data(),type,512,0,a_.data(),type,2048,0,
      &beta_,const_cast<void*>(y_.data()),type,512,0,1,compute,B::BLAS_GEMM_DEFAULT));
 }
 void launch_cast()const{
  // No operator Handle::Clone/allocation inside capture; the prepared Cast has
  // immutable input/output metadata and only its original stream field changes.
  cast_->set_stream(stream_);(*cast_)(y_,z_);
 }
 public:
 Fixed(bool opt,const Tensor&a,const Tensor&b,const Tensor&y,const Tensor&z):a_(a),b_(b),y_(y),z_(z),device_(y.device().index()){
  if(!opt)throw std::invalid_argument("r399 default OFF: explicit opt-in required");
  for(const auto*t:{&a_,&b_,&y_,&z_})if(t->device()!=y_.device()||t->device().type()!=Device::Type::kHygon||!t->IsContiguous())throw std::invalid_argument("r399 device/contiguous");
  if(a_.shape()!=Tensor::Shape{16,2048}||b_.shape()!=Tensor::Shape{2048,512}||y_.shape()!=Tensor::Shape{16,512}||z_.shape()!=y_.shape()||a_.dtype()!=DataType::kFloat32||b_.dtype()!=DataType::kFloat32||y_.dtype()!=DataType::kFloat32||z_.dtype()!=DataType::kBFloat16)throw std::invalid_argument("r399 fixed F32 PV geometry + BF16 Cast");
  if(cuCtxGetCurrent(&context_)!=CUDA_SUCCESS)throw std::runtime_error("r399 get context");owner();
  // All resources are created eagerly, not in capture_safe() or a captured call.
  blas_check(cublasCreate(&handle_));
  try{blas_check(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST));cast_=std::make_unique<Operator<Cast,Device::Type::kHygon,0>>(y_,z_);}catch(...){cublasDestroy(handle_);handle_=nullptr;throw;}
 }
 ~Fixed()noexcept override{
  // Bridge destroys this only after its original Runtime has drained and graph died.
  if(handle_){try{owner();if(stream_)cuda_check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));cast_.reset();blas_check(cublasDestroy(handle_));}catch(...){std::terminate();}}
 }
 size_t workspace_bytes()const override{return workspace_bound_;}
 void prepare(void*s,void*w,size_t n)override{
  owner();if(ready_||!s||!w||reinterpret_cast<uintptr_t>(w)%256||n<workspace_bound_)throw std::invalid_argument("r399 eager prepare workspace/once");
  cudaStreamCaptureStatus status;cuda_check(cudaStreamIsCapturing(static_cast<cudaStream_t>(s),&status));if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("r399 preparation inside capture");
  stream_=s;workspace_=w;
  // The caller explicitly acknowledges one eager mathematical warm execution.
  // A bound workspace is necessary, not proof of absent hidden vendor allocations.
  // The first actual-device qualification must check graph creation/replay/canaries.
  launch_gemm();launch_cast();cuda_check(cudaStreamSynchronize(static_cast<cudaStream_t>(s)));ready_=true;
 }
 bool capture_ready()const override{owner();return ready_;}
 void gemm(void*s)override{stream(s);launch_gemm();}
 void cast(void*s)override{stream(s);launch_cast();}
 std::string metadata()const override{return "fixed16x2048x512 F32 current default GEMM + original BF16 Cast; dedicated cublas lease; HOST scalars; explicit Runtime workspace8MiB; hidden vendor allocations UNKNOWN until device qualification";}
};
}
std::unique_ptr<PreparedOps> make_prepared(bool opt,const infini::ops::Tensor&a,const infini::ops::Tensor&b,const infini::ops::Tensor&y,const infini::ops::Tensor&z){return std::make_unique<Fixed>(opt,a,b,y,z);}
}
