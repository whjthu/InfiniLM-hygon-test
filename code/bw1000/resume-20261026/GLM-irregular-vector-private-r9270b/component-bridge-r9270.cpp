#include "glm_vector_dispatch_r8540.hpp"
#include <cstddef>
#include <type_traits>
static_assert(sizeof(glm_vector_r8540::Frame)==8 && alignof(glm_vector_r8540::Frame)==4,"original Frame ABI");
static_assert(offsetof(glm_vector_r8540::Frame,enabled)==4,"original Frame field offset");
extern "C" int glm_component_gate_r9270(int d,void*g,void*u,const void*x,const void*xs,const void*ids,const void*w,const void*ws,const void*up,const void*us,std::size_t routes,std::size_t offset,void*stream){
 if(d<0||d>=8||!g||!u||!x||!xs||!ids||!w||!ws||!up||!us||!routes||routes>128||routes%8||offset>1792||offset%256)return -10;
 try{glm_vector_r8540::Frame f(d);return glm_vector_r8540::gate(f,g,u,x,xs,ids,w,ws,up,us,routes,offset,stream);}catch(...){return -11;}
}
extern "C" int glm_component_down_r9270(int d,void*out,const void*x,const void*xs,const void*ids,const void*w,const void*ws,std::size_t routes,std::size_t offset,void*stream){
 if(d<0||d>=8||!out||!x||!xs||!ids||!w||!ws||!routes||routes>128||routes%8||offset>1792||offset%256)return -10;
 try{glm_vector_r8540::Frame f(d);return glm_vector_r8540::down(f,out,x,xs,ids,w,ws,routes,offset,stream);}catch(...){return -11;}
}
