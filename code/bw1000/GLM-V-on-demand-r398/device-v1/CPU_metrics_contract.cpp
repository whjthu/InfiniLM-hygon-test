#include "bf16_metrics_r398.hpp"
#include <cassert>
#include <iostream>
int main(){
 uint16_t a[]={0x3f80,0x4000,0x0000,0xc000},b[]={0x3f80,0x4040,0x0000,0xc000};
 r398_metrics::Accumulator m;m.add(a,b,2);m.add(a+2,b+2,2);
 assert(m.finite&&m.elements==4&&m.different==1&&std::abs(m.NRMSE()-1./3.)<1e-14&&m.maxabs==1.);
 uint16_t bad[]={0x7f80};m.add(a,bad,1);assert(!m.finite);
 std::cout<<"{\"passed\":true,\"actual_streamed_BF16_metric\":true,\"finite_and_nonfinite\":true,\"CPU_F64_metrics_only\":true,\"GPU\":0}\n";
}
