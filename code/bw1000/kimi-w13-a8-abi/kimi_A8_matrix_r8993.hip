#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <cstdint>
namespace {
namespace dm=du::dumma;
using B=uint16_t;
constexpr unsigned E=896,T=16,H=3584,I=192,N=384,R=256,KGroups=112;
// Integer mantissa is exactly twice E2M1; no lossy weight requantization.
__device__ int8_t code_i8(unsigned c) {
    const int values[8]={0,1,2,3,4,6,8,12};
    const int v=values[c&7];return int8_t(c&8?-v:v);
}
__global__ void quant32(const B*x,int8_t*q,float*scale,int*error) {
    const unsigned row=blockIdx.x/KGroups,kg=blockIdx.x%KGroups,lane=threadIdx.x;
    __shared__ float amax[32],s;
    const unsigned index=row*H+kg*32+lane;const float xval=__uint_as_float(uint32_t(x[index])<<16);
    if(!isfinite(xval))atomicOr(error,1);
    amax[lane]=isfinite(xval)?fabsf(xval):0.f;__syncthreads();
    for(unsigned d=16;d;d>>=1){if(lane<d)amax[lane]=fmaxf(amax[lane],amax[lane+d]);__syncthreads();}
    if(!lane){s=amax[0]>0.f?amax[0]/127.f:1.f;scale[row*KGroups+kg]=s;}__syncthreads();
    const float v=isfinite(xval)?xval/s:0.f;
    q[index]=int8_t(max(-127,min(127,__float2int_rn(v))));
}
__global__ void sorted16(const int*counts,const int*routes,int*sorted,int*expert,int*groups){
    if(threadIdx.x||blockIdx.x)return;int g=0;
    for(unsigned e=0;e<E;++e)for(int p=0;p<counts[e];p+=16){
        expert[g]=int(e);for(int j=0;j<16;++j)sorted[g*16+j]=(p+j<counts[e])?routes[e*R+p+j]:int(R);++g;
    }*groups=g;
}
// One wave, one M16/N16 tile. No assumed fragment register-coordinate mapping:
// use the same DU store-row-major interface numerically qualified in702.
__global__ void w13_a8(float*raw,const int8_t*q,const float*sx,const uint8_t*w,
    const uint8_t*sw,const int*sorted,const int*expert,const int*groups,int*error){
    __shared__ int8_t A[16*32],W[16*32];
    __shared__ int C[16*16];
    __shared__ float xs[16],ws[16];
    const unsigned tid=threadIdx.x;
    for(unsigned work=blockIdx.x;work<unsigned(*groups)*(N/16);work+=gridDim.x){
        const unsigned group=work/(N/16),nb=(work%(N/16))*16,e=unsigned(expert[group]);
        float sum[4]={0.f,0.f,0.f,0.f};
        for(unsigned kg=0;kg<KGroups;++kg){
            for(unsigned j=tid;j<16*32;j+=64){
                const unsigned row=j/32,k=j%32,route=unsigned(sorted[group*16+row]);
                A[j]=route<R?q[(route/16)*H+kg*32+k]:int8_t(0);
                // W is column-major [K32,N16], leading dimension32.
                const unsigned col=j/32;
                const uint8_t packed=w[(size_t(e)*N+nb+col)*(H/2)+kg*16+k/2];
                W[j]=code_i8((packed>>(4*(k%2)))&15);
            }
            if(tid<16){
                const unsigned route=unsigned(sorted[group*16+tid]);
                xs[tid]=route<R?sx[(route/16)*KGroups+kg]:0.f;
                const unsigned scale=sw[(size_t(e)*N+nb+tid)*KGroups+kg];
                if(scale==255){atomicOr(error,2);ws[tid]=nanf("");}
                else ws[tid]=ldexpf(1.f,int(scale)-128); // integer code =2*E2M1
            }
            __syncthreads();
            dm::DUFragment<dm::matrix_a,16,16,32,signed char,dm::row_major> af;
            dm::DUFragment<dm::matrix_b,16,16,32,signed char,dm::col_major> bf;
            dm::DUFragment<dm::accumulator,16,16,32,int> cf;
            dm::du_load_matrix_sync(af,A,32);dm::du_load_matrix_sync(bf,W,32);
            dm::du_fill_fragment(cf,0);dm::du_mma_sync(cf,af,bf,cf);
            dm::du_store_matrix_sync(C,cf,16,dm::mem_row_major);__syncthreads();
            #pragma unroll
            for(unsigned j=0;j<4;++j){const unsigned index=tid+j*64;
                // Each K32 partial is independently rescaled BEFORE F32 addition.
                sum[j]+=(float(C[index])*xs[index/16])*ws[index%16];
            }__syncthreads();
        }
        #pragma unroll
        for(unsigned j=0;j<4;++j){const unsigned index=tid+j*64,route=unsigned(sorted[group*16+index/16]);
            if(route<R)raw[size_t(route)*N+nb+index%16]=sum[j];
        }__syncthreads();
    }
}
}
extern "C" int kimi_A8_HIP_matrix_r8993(int device,const void*input,const void*w13,const void*s13,const void*counts,const void*routes,void*extra,void*stream){
 int current=-1;if(hipGetDevice(&current)!=hipSuccess||current!=device||stream)return int(hipErrorInvalidValue);
 auto st=static_cast<hipStream_t>(stream);auto*p=static_cast<uint8_t*>(extra);
 auto*raw=reinterpret_cast<float*>(p);auto*sorted=reinterpret_cast<int*>(p+393216);
 auto*expert=reinterpret_cast<int*>(p+409600);auto*groups=reinterpret_cast<int*>(p+410624);
 auto*q=reinterpret_cast<int8_t*>(p+410880);auto*sx=reinterpret_cast<float*>(p+468224);auto*error=reinterpret_cast<int*>(p+475392);
 auto code=hipMemsetAsync(error,0,sizeof(int),st);if(code!=hipSuccess)return int(code);
 quant32<<<T*KGroups,32,0,st>>>(static_cast<const B*>(input),q,sx,error);
 sorted16<<<1,1,0,st>>>(static_cast<const int*>(counts),static_cast<const int*>(routes),sorted,expert,groups);
 w13_a8<<<640,64,0,st>>>(raw,q,sx,static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),sorted,expert,groups,error);
 return int(hipGetLastError());
}
