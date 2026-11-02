#pragma once
#include <infini/ops.h>
#include <cstddef>
#include <memory>
#include <string>
namespace deep399 {
// Explicit private factory; never registered in or interposed over the shared Ops factory.
class PreparedOps {
 public:
 virtual ~PreparedOps()=default;
 virtual size_t workspace_bytes()const=0;
 virtual void prepare(void*stream,void*workspace,size_t bytes)=0;
 virtual bool capture_ready()const=0;
 virtual void gemm(void*stream)=0;
 virtual void cast(void*stream)=0;
 virtual std::string metadata()const=0;
};
std::unique_ptr<PreparedOps> make_prepared(bool explicit_optin,const infini::ops::Tensor&a,const infini::ops::Tensor&b,const infini::ops::Tensor&y,const infini::ops::Tensor&z);
}
