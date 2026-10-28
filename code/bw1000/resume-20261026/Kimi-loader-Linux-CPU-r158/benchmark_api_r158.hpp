#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
struct CountersR158{std::uint64_t blocks,bytes,leaves,scalar;};
using ShapeR158=std::vector<std::size_t>;
using StridesR158=std::vector<std::ptrdiff_t>;
extern "C" void baseline_r158(std::byte*,const std::byte*,const ShapeR158&,const StridesR158&,const StridesR158&,CountersR158*);
extern "C" void candidate_r158(std::byte*,const std::byte*,const ShapeR158&,const StridesR158&,const StridesR158&,CountersR158*);
