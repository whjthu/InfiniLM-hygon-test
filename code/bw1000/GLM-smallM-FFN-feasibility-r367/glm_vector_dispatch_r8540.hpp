#pragma once
#include <cstddef>
#include <cstdint>
extern "C" int modern_glm_vector_policy_r8540(int device,int mode);
extern "C" int modern_glm_vector_get_r8540(int device);
extern "C" int modern_glm_vector_counts_r8540(int device,std::uint64_t*out,std::size_t n);
namespace glm_vector_r8540 {
// Process-local per-device policy, held constant across both projections.
struct Frame {
 int device; bool enabled;
 explicit Frame(int d);
 ~Frame();
 Frame(const Frame&)=delete;Frame&operator=(const Frame&)=delete;
};
int gate(const Frame&,void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
int down(const Frame&,void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*);
}
