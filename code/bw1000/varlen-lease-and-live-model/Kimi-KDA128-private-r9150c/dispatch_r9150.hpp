#pragma once
#include <vector>
#include <set>
#include <cstddef>
#include <cstdint>
namespace kda_dispatch_r9150 {
struct Input{size_t T=0,C=0,H=0,D=0,pool=0;bool eager=false,bf16=false,same_device=false,metadata_typed=false,metadata_contiguous=false;std::vector<int64_t>offsets,reads,writes;};
inline bool eligible(const Input&g){if(!g.eager||!g.bf16||!g.same_device||!g.metadata_typed||!g.metadata_contiguous||g.H!=6||g.D!=128||!g.C||g.C>32||g.T<=g.C||g.T>64||(g.pool!=17&&g.pool!=33&&g.pool!=65)||g.offsets.size()!=g.C+1||g.reads.size()!=g.C||g.writes.size()!=g.C||g.offsets.front()!=0||g.offsets.back()!=int64_t(g.T))return false;std::set<int64_t>written;for(size_t j=0;j<g.C;++j){auto x=g.offsets[j],y=g.offsets[j+1];if(x<0||y<=x||y-x>8||y>int64_t(g.T)||g.reads[j]<0||g.writes[j]<0||size_t(g.reads[j])>=g.pool||size_t(g.writes[j])>=g.pool||!written.insert(g.writes[j]).second)return false;}return true;}
}
