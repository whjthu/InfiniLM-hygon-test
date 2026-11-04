#pragma once
// Recording metadata/address/ownership leaf ONLY. No SDK, device or weight data.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
namespace infinilm::execution::graph { struct GraphManager {enum class CaptureState{kInactive};}; }
namespace infinilm::runtime {
using Size=std::size_t;using Shape=std::vector<Size>;
enum class DataType{kFloat32,kBFloat16,kUInt8,kInt64};
struct Device {
    enum class Type{kCpu,kHygon};Type kind;int ordinal;
    Type type()const{return kind;}int index()const{return ordinal;}
    bool operator==(const Device&o)const{return kind==o.kind&&ordinal==o.ordinal;}
    bool operator!=(const Device&o)const{return !(*this==o);}
};
inline std::size_t width(DataType d){return d==DataType::kBFloat16?2:d==DataType::kUInt8?1:d==DataType::kInt64?8:4;}
struct Runtime {
    Device dv{Device::Type::kHygon,0};std::thread::id owner=std::this_thread::get_id();
    struct Copy {std::uintptr_t dst,src;std::size_t bytes;bool async;};
    std::vector<Copy> copies;std::size_t syncs=0;bool throw_copy=false;
    void assertOwnerThread()const{if(owner!=std::this_thread::get_id())throw std::logic_error("recording owner");}
    bool isGraphRecording()const{return false;}
    execution::graph::GraphManager::CaptureState graphCaptureState()const{return execution::graph::GraphManager::CaptureState::kInactive;}
    Device device()const{return dv;}
    void syncStream(){assertOwnerThread();++syncs;}
    void memcpyH2D(void*d,const void*s,std::size_t n,bool async){
        assertOwnerThread();copies.push_back({reinterpret_cast<std::uintptr_t>(d),reinterpret_cast<std::uintptr_t>(s),n,async});
        if(throw_copy)throw std::runtime_error("recorded copy failure after enqueue");
    }
};
struct Storage {std::uintptr_t base;std::size_t bytes;};
inline std::uintptr_t next_base=0x100000000ULL;
inline std::vector<Shape> allocations;
class TensorImpl;class Tensor {
    std::shared_ptr<TensorImpl> p_;
public:
    Tensor()=default;explicit Tensor(std::shared_ptr<TensorImpl>p):p_(std::move(p)){}
    TensorImpl*operator->()const{return p_.get();}bool empty()const{return !p_;}
    explicit operator bool()const{return bool(p_);}void reset(){p_.reset();}
    void reset(std::shared_ptr<TensorImpl>p){p_=std::move(p);}std::size_t use_count()const{return p_.use_count();}
    static Tensor empty_parameter(Runtime&,const Shape&,DataType);
    static Tensor source(const Shape&,DataType,std::size_t=0);
};
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    std::shared_ptr<Storage> backing;Shape dimensions;DataType dt;Device dv;
    std::size_t byte_offset=0;bool contiguous_flag=true;
    Shape shape()const{return dimensions;}std::size_t size(std::size_t d)const{return dimensions.at(d);}
    std::size_t ndim()const{return dimensions.size();}
    DataType dtype()const{return dt;}Device device()const{return dv;}
    bool is_contiguous()const{return contiguous_flag;}
    std::size_t numel()const{std::size_t n=1;for(auto d:dimensions)n*=d;return n;}
    std::size_t nbytes()const{return numel()*width(dt);}
    void*data()const{return reinterpret_cast<void*>(backing->base+byte_offset);}
    std::string info()const{return "recording Tensor metadata, not SDK";}
    Tensor view(const Shape&s)const {auto t=std::make_shared<TensorImpl>(*this);t->dimensions=s;return Tensor(t);}
    Tensor contiguous(Runtime&)const{return Tensor(std::make_shared<TensorImpl>(*this));}
    Tensor permute(std::initializer_list<int>order)const{
        auto t=std::make_shared<TensorImpl>(*this);Shape s;for(int d:order)s.push_back(dimensions.at(std::size_t(d)));
        t->dimensions=s;t->contiguous_flag=false;return Tensor(t);
    }
    Tensor narrow(const std::vector<std::array<std::size_t,3>>&ranges)const{
        auto t=std::make_shared<TensorImpl>(*this);
        for(const auto&a:ranges){
            const auto dim=a[0],offset=a[1],length=a[2];
            if(dim>=ndim()||offset>size(dim)||length>size(dim)-offset)throw std::out_of_range("recording narrow");
            std::size_t stride=width(dt);for(std::size_t d=dim+1;d<ndim();++d)stride*=size(d);
            t->byte_offset+=offset*stride;t->dimensions[dim]=length;
            if(dim!=0)t->contiguous_flag=false;
        }
        return Tensor(t);
    }
    void copy_from(Runtime&r,const Tensor&s){r.memcpyH2D(data(),s->data(),nbytes(),false);}
};
inline Tensor Tensor::source(const Shape&s,DataType dtype,std::size_t offset){
    auto t=std::make_shared<TensorImpl>();t->dimensions=s;t->dt=dtype;t->dv={Device::Type::kCpu,0};
    t->byte_offset=offset;t->backing=std::make_shared<Storage>(Storage{next_base,t->nbytes()+offset});
    next_base+=t->backing->bytes+4096;return Tensor(t);
}
inline Tensor Tensor::empty_parameter(Runtime&r,const Shape&s,DataType dtype){
    allocations.push_back(s);auto t=Tensor::source(s,dtype);t->dv=r.device();return t;
}
}
