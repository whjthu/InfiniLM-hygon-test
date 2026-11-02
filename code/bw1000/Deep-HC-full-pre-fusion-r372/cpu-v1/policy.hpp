#pragma once
#include <stdexcept>
#include <cstring>
namespace hc372 {
inline bool selected(int mode,const char*flag){
 if(mode<0||mode>1)throw std::invalid_argument("HC372 mode");
 if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::invalid_argument("HC372 flag");
 return mode==1&&flag&&std::strcmp(flag,"1")==0;
}
}
