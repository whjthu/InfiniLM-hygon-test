#pragma once
#include <memory>
#include <cstdint>
#include <stdexcept>
namespace infinilm::runtime {
enum class DataType{kInt64,kFloat32};
struct Device{int n=0;int index()const{return n;}bool operator!=(const Device&o)const{return n!=o.n;}};
struct Value{Device dev;DataType type=DataType::kInt64;size_t count=3;bool contiguous=true;int rank=1;int64_t storage[8]{};Device device()const{return dev;}DataType dtype()const{return type;}size_t ndim()const{return rank;}size_t size(int)const{return count;}bool is_contiguous()const{return contiguous;}void*data(){return storage;}};
using Tensor=std::shared_ptr<Value>;
struct Runtime{Device dev;void*s=(void*)0x10;bool graph=false,owner=true;void assertOwnerThread(){if(!owner)throw std::runtime_error("owner");}bool isGraphRecording()const{return graph;}void*stream(){return s;}Device device()const{return dev;}};
}
