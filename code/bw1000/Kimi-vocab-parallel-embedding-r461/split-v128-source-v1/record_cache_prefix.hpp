#pragma once
#include "contract.hpp"
#include <exception>
#include <memory>
#include <vector>
namespace infinilm::runtime {
using Shape=std::vector<size_t>;enum class DataType{kBFloat16};
struct Device {enum class Type{kHygon,kOther};Type type()const{return Type::kHygon;}};
inline int alive=0;struct Impl {Shape shape;Impl(Shape s):shape(s){++alive;}~Impl(){--alive;}size_t nbytes()const{size_t n=2;for(auto x:shape)n*=x;return n;}void*data(){return this;}};
struct Runtime {int zero_calls=0,sync_calls=0;bool fail=false;void assertOwnerThread(){}Device device(){return{};}bool isGraphRecording(){return false;}void*stream(){return nullptr;}void setDeviceMemoryAsync(void*,int value,size_t count,void*){if(value||count!=64880640){throw std::logic_error("record cache bytes");}++zero_calls;if(fail){throw std::runtime_error("record injected failure");}}void syncStream(){if(alive!=1){throw std::logic_error("cache released before drain");}++sync_calls;}};
struct Tensor {std::shared_ptr<Impl> p;Impl*operator->()const{return p.get();}static Tensor empty_parameter(Runtime&,Shape s,DataType){return {std::make_shared<Impl>(s)};}};
}
