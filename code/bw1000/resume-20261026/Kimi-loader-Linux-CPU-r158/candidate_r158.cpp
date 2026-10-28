#include "benchmark_api_r158.hpp"
#include "cpu_strided_copy_r2515.hpp"
#include "candidate_rows_r157.hpp"
extern "C" void candidate_r158(std::byte*d,const std::byte*s,const ShapeR158&shape,const StridesR158&ds,const StridesR158&ss,CountersR158*out){
 infinilm::runtime::detail::CpuCopyStatsR2515 st;
 infinilm::runtime::detail::copy_cpu_strided_r157_candidate(d,s,shape,ds,ss,1,0,st);
 *out={st.blocks,st.block_bytes,st.scalar_leaves,st.scalar_bytes};
}
