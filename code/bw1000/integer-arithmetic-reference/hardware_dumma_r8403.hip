#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <stdint.h>
#include <type_traits>
namespace dm=du::dumma;

// One complete wave per block. Operands differ across chains/blocks; retaining
// all outputs prevents duplicate-chain elimination. Register-resident arithmetic
// reference only: no unpack/quantization/routing, GEMM tiling or HBM-rate claim.
template<int K,int Chains>
__global__ void integer_mma(const uint8_t* ap,const uint8_t* bp,int32_t* out,int loops){
    using In=typename std::conditional<K==32,signed char,dm::experimental::precision::s4>::type;
    dm::DUFragment<dm::matrix_a,16,16,K,In,dm::row_major> a[Chains];
    dm::DUFragment<dm::matrix_b,16,16,K,In,dm::col_major> b[Chains];
    dm::DUFragment<dm::accumulator,16,16,K,int> c[Chains];
    #pragma unroll
    for(int j=0;j<Chains;++j){
        const uint8_t* aa=ap+(size_t(blockIdx.x)*Chains+j)*512;
        const uint8_t* bb=bp+(size_t(blockIdx.x)*Chains+j)*512;
        if constexpr(K==32){
            dm::du_load_matrix_sync(a[j],reinterpret_cast<const signed char*>(aa),K);
            dm::du_load_matrix_sync(b[j],reinterpret_cast<const signed char*>(bb),K);
        }else{
            dm::du_load_matrix_sync(a[j],static_cast<const void*>(aa),K);
            dm::du_load_matrix_sync(b[j],static_cast<const void*>(bb),K);
        }
        dm::du_fill_fragment(c[j],j);
    }
    #pragma unroll 1
    for(int l=0;l<loops;++l){
        #pragma unroll
        for(int j=0;j<Chains;++j)dm::du_mma_sync(c[j],a[j],b[j],c[j]);
    }
    #pragma unroll
    for(int j=0;j<Chains;++j)
        dm::du_store_matrix_sync(out+(size_t(blockIdx.x)*Chains+j)*256,c[j],16,dm::mem_row_major);
}

extern "C" int hardware_dumma_r8403(int op,int chains,const void* a,const void* b,void* out,int blocks,int loops,void* stream){
    if((op!=0&&op!=1)||(chains!=1&&chains!=4)||!a||!b||!out||blocks<1||blocks>2560||loops<1||loops>32768)return 999;
    const auto aa=static_cast<const uint8_t*>(a);const auto bb=static_cast<const uint8_t*>(b);auto cc=static_cast<int32_t*>(out);auto st=static_cast<hipStream_t>(stream);
    if(op==0&&chains==1)hipLaunchKernelGGL((integer_mma<32,1>),dim3(blocks),dim3(64),0,st,aa,bb,cc,loops);
    if(op==0&&chains==4)hipLaunchKernelGGL((integer_mma<32,4>),dim3(blocks),dim3(64),0,st,aa,bb,cc,loops);
    if(op==1&&chains==1)hipLaunchKernelGGL((integer_mma<64,1>),dim3(blocks),dim3(64),0,st,aa,bb,cc,loops);
    if(op==1&&chains==4)hipLaunchKernelGGL((integer_mma<64,4>),dim3(blocks),dim3(64),0,st,aa,bb,cc,loops);
    return int(hipGetLastError());
}
