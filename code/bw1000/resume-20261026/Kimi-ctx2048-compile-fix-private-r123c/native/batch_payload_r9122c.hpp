#pragma once
#include <cstdint>
#include <cstddef>
#include <set>
#include <stdexcept>
namespace kimi_batch_payload_r9122c {
inline void validate(const int64_t*v,size_t n,size_t expected_pool,int64_t op,size_t max_context=1024){
 if(max_context!=1024&&max_context!=2048)throw std::invalid_argument("r123 invalid owner context");
 if(!v||n<8||n>232||(expected_pool!=17&&expected_pool!=33&&expected_pool!=65)||v[0]!=91220001||v[1]!=1||v[2]!=op||op<1||op>4||v[4]<0||v[4]>32||v[6]!=int64_t(expected_pool)||v[7]!=0||n!=size_t(8+7*v[4]))throw std::invalid_argument("9122c typed payload header");
 if(op==3||op==4){if(v[3]||v[4]||v[5])throw std::invalid_argument("9122c clear/stats nonempty");return;}
 const auto half=int64_t((expected_pool-1)/2);if(v[3]<1||v[4]<1||v[4]>half||v[5]<1||v[5]>64)throw std::invalid_argument("9122c typed payload capacity");size_t total=0;std::set<int64_t>logical,rows;
 for(int64_t i=0;i<v[4];++i){const auto*x=v+8+7*i;if(x[0]<1||x[0]>half||!logical.insert(x[0]).second||!((x[2]==x[0]&&x[3]==x[0]+half)||(x[3]==x[0]&&x[2]==x[0]+half))||!rows.insert(x[2]).second||!rows.insert(x[3]).second||x[1]<1||x[1]>int64_t(max_context)||x[4]!=int64_t(total)||x[5]<1||x[5]>8||x[1]+x[5]>int64_t(max_context)||(op==1?x[6]!=0:x[6]<1||x[6]>x[5]))throw std::invalid_argument("9122c typed payload independent owner/past/prefix");total+=size_t(x[5]);}
 if(total!=size_t(v[5])||total>64)throw std::invalid_argument("9122c typed payload flatten total");
}
}
