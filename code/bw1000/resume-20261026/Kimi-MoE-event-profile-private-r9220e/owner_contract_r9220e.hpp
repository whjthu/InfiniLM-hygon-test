#pragma once
#include <cstddef>
#include <stdexcept>
namespace owner_contract_r9220e {inline void before_runtime_create(size_t inherited,size_t owners=8,size_t width=4){if(inherited<owners*width)throw std::invalid_argument("Runtime2392 requires full inherited owner pool BEFORE create");}inline void after_runtime_create(size_t actual,size_t width=4){if(actual!=width)throw std::invalid_argument("Runtime2392 actual owner subset width");}}
