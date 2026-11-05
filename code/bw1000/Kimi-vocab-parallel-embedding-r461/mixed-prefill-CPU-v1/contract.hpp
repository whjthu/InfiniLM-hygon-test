#pragma once
#include <cstddef>
#include <stdexcept>
namespace kimi_mixed_r461 {
struct Geometry { std::size_t m,k,n; };
inline Geometry geometry(int id) {
 if(id==0)return {4608,7168,1};
 if(id==1)return {512,7168,3584};
 throw std::invalid_argument("mixed fixed actual projection geometry");
}
inline void validate(int id,int mode,std::size_t nx,std::size_t nw,std::size_t ny) {
 const auto g=geometry(id);
 if((mode!=0&&mode!=1)||nx!=g.m*g.k||nw!=g.n*g.k||ny!=g.m*g.n)
  throw std::invalid_argument("mixed exact mode and complete BF16 buffers");
}
}
