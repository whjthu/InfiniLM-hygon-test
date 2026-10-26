#include "glm_fp8_cache_m15_r119.hpp"
#include <cassert>
#include <thread>
#include <cstdint>
#include <cstdio>
struct Plan{size_t m,n;};
extern "C" const char*glm_fp8_error_r776(){return "mock776";}
extern "C" void*glm_fp8_create_r776(int,int,size_t m,size_t n,size_t,size_t,size_t,size_t,size_t,void*){return new Plan{m,n};}
extern "C" int glm_fp8_run_r776(void*,int,void*out,const void*x,const void*,const void*w,const void*){*static_cast<int*>(out)=*static_cast<const int*>(x)+*static_cast<const int*>(w);return 0;}
extern "C" int glm_fp8_destroy_r776(void*p){delete static_cast<Plan*>(p);return 0;}
int main(){int identity=1,x=3,w=7,sc=1,y=0;void*stream=reinterpret_cast<void*>(1234);void*c=glm_fp8_cache_create_r119(0,stream,&identity);assert(c);
 auto run=[&](size_t m,size_t n,int mode=1,const void*id=nullptr,void*st=nullptr){return glm_fp8_cache_run_r119(c,id?id:&identity,st?st:stream,mode,1,m,n,128,1,1,0,0,&y,&x,&sc,&w,&sc);};
 assert(run(15,64)==0&&y==10);w=8;assert(run(15,64)==0&&y==11);int before=y;assert(run(16,64)!=0&&y==before);assert(run(15,64,99)!=0&&y==before);assert(run(15,64,1,&x)!=0&&y==before);assert(run(15,64,1,nullptr,reinterpret_cast<void*>(9))!=0&&y==before);
 uint64_t stats[10]{};int wrong=0;std::thread t([&]{wrong=glm_fp8_cache_stats_r119(c,stats,10);});t.join();assert(wrong==1);
 for(size_t n=1;n<=11;++n)assert(run(15,n)==0);assert(glm_fp8_cache_stats_r119(c,stats,10)==0&&stats[4]>0&&stats[5]<=10&&stats[6]<=192ull*1024*1024);
 assert(glm_fp8_cache_clear_r119(c,&identity,stream)==0);assert(glm_fp8_cache_stats_r119(c,stats,10)==0&&stats[0]==stats[1]&&stats[5]==0&&stats[6]==0);assert(glm_fp8_cache_destroy_r119(c,&identity,stream)==0);
 std::puts("{\"passed\":true,\"CPU_mock_only\":true,\"GPU_math_proof\":false,\"tests\":\"M15/otherM/owner/stream/thread/invalid/outputpreserve/freshweight/LRU/normalclear\"}");}
