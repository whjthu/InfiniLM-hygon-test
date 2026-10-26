#include <cstdint>
#include <cstddef>
#include <cassert>
#include <iostream>
extern "C" int modern_glm_down_row2_policy_r9281(int,int);extern "C" int modern_glm_down_row2_get_r9281(int);extern "C" int modern_glm_down_row2_counts_r9281(int,uint64_t*,size_t);
extern "C" int glm_vec_down_r8528(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
static int ref=0,pairc=0;
extern "C" int glm_vec_down_reference_r9281(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*){++ref;assert(modern_glm_down_row2_policy_r9281(0,1)==-2);return 0;}
extern "C" int glm_vec_down_row2_r9281(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*){++pairc;assert(modern_glm_down_row2_policy_r9281(0,0)==-2);return 0;}
int main(){void*p=(void*)16;assert(modern_glm_down_row2_get_r9281(0)==0);assert(modern_glm_down_row2_policy_r9281(1,1)==-1);assert(modern_glm_down_row2_policy_r9281(0,2)==-1);assert(glm_vec_down_r8528(p,p,p,p,p,p,128,0,p)==0);assert(modern_glm_down_row2_policy_r9281(0,1)==0);assert(glm_vec_down_r8528(p,p,p,p,p,p,64,1792,p)==0);assert(modern_glm_down_row2_policy_r9281(0,0)==0);assert(glm_vec_down_r8528(p,p,p,p,p,p,128,0,p)==0);assert(glm_vec_down_r8528(p,p,p,p,p,p,128,SIZE_MAX,p)==-10);assert(glm_vec_down_r8528(nullptr,p,p,p,p,p,128,0,p)==-10);assert(glm_vec_down_r8528(p,p,p,p,p,p,129,0,p)==-10);uint64_t c[4];assert(modern_glm_down_row2_counts_r9281(0,c,4)==0);assert(c[0]==2&&c[1]==1&&c[2]==3&&c[3]==2&&ref==2&&pairc==1);assert(modern_glm_down_row2_counts_r9281(0,c,3)==-1);std::cout<<"{\"passed\":true,\"busy_refused\":true,\"default_OFF\":true,\"stubbed_raw_kernels\":true}";}
