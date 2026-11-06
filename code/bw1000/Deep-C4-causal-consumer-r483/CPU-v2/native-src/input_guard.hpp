#pragma once
// Pure host byte contract shared by the actual device input owner and local C++ test.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace c4_input_guard_r483 {
constexpr size_t guard_bytes=256;
struct Layout {size_t payload_bytes,element_bytes,total_bytes,prefix_elements,payload_elements,total_elements;};
template<class Shape> Layout layout(size_t bytes,const Shape&shape,size_t element_bytes){
 if(!bytes||!(element_bytes==1||element_bytes==2||element_bytes==4||element_bytes==8))
  throw std::invalid_argument("input guard byte width");
 size_t elements=1;
 if(shape.empty())throw std::invalid_argument("input guard shape");
 for(auto raw:shape){
  if(raw<=0)throw std::invalid_argument("input guard dimension");
  size_t d=size_t(raw);
  if(elements>std::numeric_limits<size_t>::max()/d)throw std::overflow_error("input guard shape overflow");
  elements*=d;
 }
 if(elements>std::numeric_limits<size_t>::max()/element_bytes||elements*element_bytes!=bytes)
  throw std::invalid_argument("input guard shape byte mismatch");
 if(bytes>std::numeric_limits<size_t>::max()-2*guard_bytes)
  throw std::overflow_error("input guard extent overflow");
 return {bytes,element_bytes,bytes+2*guard_bytes,guard_bytes/element_bytes,elements,(bytes+2*guard_bytes)/element_bytes};
}
struct Buffer {
 Layout geometry;
 std::vector<uint8_t> expected,actual;
 Buffer(const void*p,Layout g):geometry(g){
  if(!p)throw std::invalid_argument("missing actual input bytes");
  expected.assign(g.total_bytes,uint8_t(0xa5));
  std::memcpy(expected.data()+guard_bytes,p,g.payload_bytes);
  std::fill(expected.begin()+guard_bytes+g.payload_bytes,expected.end(),uint8_t(0x5a));
  actual.resize(g.total_bytes);
 }
 void verify()const{
  if(actual.size()!=expected.size())throw std::runtime_error("actual GPU input extent differs");
  auto prefix_end=expected.begin()+guard_bytes;
  auto payload_end=prefix_end+geometry.payload_bytes;
  if(!std::equal(expected.begin(),prefix_end,actual.begin()))throw std::runtime_error("actual GPU input prefix changed");
  if(!std::equal(prefix_end,payload_end,actual.begin()+guard_bytes))throw std::runtime_error("actual GPU input payload changed");
  if(!std::equal(payload_end,expected.end(),actual.begin()+guard_bytes+geometry.payload_bytes))throw std::runtime_error("actual GPU input suffix changed");
 }
};
}
