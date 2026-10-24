#pragma once
#include "runtime/runtime.hpp"
namespace kimi_model_events_r8669zq {
using R=infinilm::runtime::Runtime;
void prepare(R&,int tp_rank,int tp_size,int pp_stage,int pp_size);
void begin_task(R&);
void mark(R&,const char*,int layer=-1,bool kda=false,bool moe=false);
void end_task(R&);
void original_drain_begin(R&);
void original_drain_end(R&);
void flush_after_original_drain(R&);
void close_after_original_drain(R&);
void abandon_noexcept(R&) noexcept;
}
