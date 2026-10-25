#pragma once
#include "runtime/runtime.hpp"
#include <array>
#include <string>
#include "dispatch_r9150.hpp"
namespace kda_threads_r9150 {namespace rt=infinilm::runtime;
void record_input(rt::Runtime&,size_t,size_t,size_t,const int32_t*,const int32_t*,const int32_t*,const std::array<rt::Tensor,3>&);bool eligible(rt::Runtime&,size_t,size_t,size_t,size_t,bool,const std::array<rt::Tensor,3>&);void dispatch_counts(rt::Runtime&,uint64_t*);
void mode(rt::Runtime&,int,size_t);int selected(rt::Runtime&);void clear(rt::Runtime&);void collect(rt::Runtime&);const char*last(rt::Runtime&);void run(rt::Runtime&,const std::array<rt::Tensor,12>&,float,float,bool);
}
