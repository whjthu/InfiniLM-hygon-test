#pragma once
#include "runtime/runtime.hpp"
#include <array>
#include <string>
namespace kda_threads_r9150 {namespace rt=infinilm::runtime;
void mode(rt::Runtime&,int,size_t);int selected(rt::Runtime&);void clear(rt::Runtime&);void collect(rt::Runtime&);const char*last(rt::Runtime&);void run(rt::Runtime&,const std::array<rt::Tensor,12>&,float,float,bool);
}
