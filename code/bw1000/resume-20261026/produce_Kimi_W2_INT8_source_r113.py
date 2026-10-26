import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent; d=r/'Kimi-W2-INT8-private-r113'; d.mkdir(exist_ok=False); sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
base=r/'Kimi-W2-pair-private-r9280/kimi_W2_pair_provider_r9280.cu'; raw=base.read_text(encoding='utf8'); assert sha(base)=='4c67808661077dfa8f6c4d1712b31c01e33c36e2bb8a5f556b4c6cd95b43a63a'
insert=r'''
namespace i8r113 {
namespace dm=du::dumma;
struct Binding{int device=-1;size_t T=0;void*stream=nullptr;unsigned char*base=nullptr;size_t bytes=0;};
thread_local Binding binding;
std::atomic<uint64_t> counts[8][3]{};
size_t bytes(size_t T){return T*16*(192+24+3584*4+4)+4;}
__device__ __forceinline__ signed char decode(uint8_t c){constexpr int q[8]={0,1,2,3,4,6,8,12};int x=q[c&7];return static_cast<signed char>((c&8)?-x:x);}
__global__ void constructed(int32_t*ids,size_t R){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<R)ids[i]=int32_t(i%896);}
__global__ void quantize(const __nv_bfloat16*a,signed char*q,float*scale,int*error,size_t R){
 size_t rg=blockIdx.x;if(rg>=R*6)return;unsigned k=threadIdx.x;__shared__ float vals[32];float x=mxfp4Load(a,rg*32+k);if(!isfinite(x)){atomicExch(error,1);x=0.f;}vals[k]=fabsf(x);__syncthreads();
 for(unsigned s=16;s;s/=2){if(k<s)vals[k]=fmaxf(vals[k],vals[k+s]);__syncthreads();}float sa=vals[0]>0?vals[0]/127.f:1.f;
 if(!isfinite(sa)||!(sa>0)){atomicExch(error,1);sa=1.f;}if(k==0)scale[rg]=sa;int v=int(nearbyintf(x/sa));q[rg*32+k]=static_cast<signed char>(max(-127,min(127,v)));
}
__global__ void matrix(float*partial,const signed char*q,const float*as,const int32_t*cnt,const int32_t*routes,const int32_t*expert,const int32_t*start,const int32_t*jobs,const uint8_t*w,const uint8_t*sc,size_t T){
 __shared__ __align__(16) signed char aa[512];__shared__ __align__(16) signed char bb[512];__shared__ __align__(16) int tmp[256];__shared__ int rr[16];
 unsigned lane=threadIdx.x;size_t R=T*16;
 for(size_t work=blockIdx.x;work<size_t(jobs[0])*224;work+=gridDim.x){size_t job=work/224;unsigned n0=(work%224)*16;int e=expert[job],s=start[job];
  if(lane<16)rr[lane]=(s+int(lane)<cnt[e])?routes[size_t(e)*R+s+lane]:-1;__syncthreads();float sum[4]={0.f,0.f,0.f,0.f};
  #pragma unroll
  for(unsigned g=0;g<6;++g){
   for(unsigned p=lane;p<512;p+=64){unsigned m=p/32,k=p%32;int rid=rr[m];aa[p]=rid>=0?q[size_t(rid)*192+g*32+k]:0;unsigned n=p/32;uint8_t byte=w[(size_t(e)*3584+n0+n)*96+g*16+k/2];bb[p]=decode(uint8_t((byte>>(4*(k&1)))&15));}__syncthreads();
   dm::DUFragment<dm::matrix_a,16,16,32,signed char,dm::row_major>A;
   dm::DUFragment<dm::matrix_b,16,16,32,signed char,dm::col_major>B;
   dm::DUFragment<dm::accumulator,16,16,32,int>C;
   dm::du_load_matrix_sync(A,aa,32);dm::du_load_matrix_sync(B,bb,32);dm::du_fill_fragment(C,0);dm::du_mma_sync(C,A,B,C);dm::du_store_matrix_sync(tmp,C,16,dm::mem_row_major);__syncthreads();
   #pragma unroll
   for(unsigned i=0;i<4;++i){unsigned p=lane+64*i,m=p/16,n=p%16;int rid=rr[m];float av=rid>=0?as[size_t(rid)*6+g]:0.f;float sw=ldexpf(.5f,int(sc[(size_t(e)*3584+n0+n)*6+g])-127);sum[i]+=float(tmp[p])*(av*sw);}__syncthreads();
  }
  #pragma unroll
  for(unsigned i=0;i<4;++i){unsigned p=lane+64*i,m=p/16,n=p%16;int rid=rr[m];if(rid>=0)partial[size_t(rid)*3584+n0+n]=sum[i];}__syncthreads();
 }
}
__global__ void reduce(__nv_bfloat16*out,const float*partial,const int32_t*ids,const float*weight,size_t T){size_t p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=T*3584)return;size_t t=p/3584,n=p%3584;float v=0.f;for(unsigned k=0;k<16;++k){int e=ids[t*16+k];if(e>=0&&e<896)v+=weight[t*16+k]*partial[(t*16+k)*3584+n];}out[p]=mxfp4Store<__nv_bfloat16>(v);}
__global__ void layout(const signed char*a,const signed char*b,int*out){dm::DUFragment<dm::matrix_a,16,16,32,signed char,dm::row_major>A;dm::DUFragment<dm::matrix_b,16,16,32,signed char,dm::col_major>B;dm::DUFragment<dm::accumulator,16,16,32,int>C;dm::du_load_matrix_sync(A,a,32);dm::du_load_matrix_sync(B,b,32);dm::du_fill_fragment(C,0);dm::du_mma_sync(C,A,B,C);dm::du_store_matrix_sync(out,C,16,dm::mem_row_major);}
}
extern "C" size_t modern_kimi_i8_bytes_r113(size_t T){return T>=8&&T<=64?i8r113::bytes(T):0;}
extern "C" int modern_kimi_i8_bind_r113(int device,size_t T,void*base,size_t bytes,void*stream){if(!base){i8r113::binding={};return 0;}if(device<0||device>=8||T<8||T>64||bytes!=i8r113::bytes(T)||reinterpret_cast<uintptr_t>(base)%16)return 1;i8r113::binding={device,T,stream,static_cast<unsigned char*>(base),bytes};return 0;}
extern "C" int modern_kimi_i8_counts_r113(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=3)return 1;for(size_t i=0;i<3;++i)out[i]=i8r113::counts[d][i].load();return 0;}
extern "C" int modern_kimi_i8_layout_r113(const void*a,const void*b,void*out,void*stream){i8r113::layout<<<1,64,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const signed char*>(a),static_cast<const signed char*>(b),static_cast<int*>(out));return int(cudaGetLastError());}
'''
raw=raw.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <du_mma.h>')
# Insert after original namespaces close, before counters. All original functions remain unchanged.
anchor='namespace{std::atomic<uint64_t>pair_counts_r9280[8][2]{};}'
raw=raw.replace(anchor,'using namespace infini::ops::mxfp4_detail;\n'+insert+'\n'+anchor)
needle='auto st=static_cast<cudaStream_t>(stream);auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);'
replacement='''const char*flag=std::getenv("INFINILM_KIMI_W2_INT8_R113");const bool i8=flag&&flag[0]=='1'&&flag[1]=='\\0';
 const char*ctl=std::getenv("INFINILM_KIMI_W2_CONSTRUCTED_R113");const bool low=ctl&&ctl[0]=='1'&&ctl[1]=='\\0';
 auto bind=i8r113::binding;if((i8||low)&&(bind.device!=device||bind.T!=T||bind.stream!=stream||!bind.base||bind.bytes!=i8r113::bytes(T)))return int(cudaErrorInvalidValue);
 const size_t R=T*16;auto*qA=reinterpret_cast<signed char*>(bind.base);auto*sA=reinterpret_cast<float*>(bind.base?bind.base+R*192:nullptr);auto*partial=reinterpret_cast<float*>(bind.base?bind.base+R*216:nullptr);auto*newids=reinterpret_cast<int32_t*>(bind.base?bind.base+R*(216+3584*4):nullptr);auto*error=reinterpret_cast<int*>(bind.base?bind.base+R*(216+3584*4+4):nullptr);
 auto st=static_cast<cudaStream_t>(stream);if(low){i8r113::constructed<<<(R+255)/256,256,0,st>>>(newids,R);ids=newids;}
 i8r113::counts[device][i8?1:0].fetch_add(1);if(low)i8r113::counts[device][2].fetch_add(1);
 auto code=cudaMemsetAsync(counts,0,E*sizeof(int32_t),st);'''
assert needle in raw;raw=raw.replace(needle,replacement)
needle='const char*toggle=std::getenv("INFINILM_KIMI_W2_PAIR_R9280");'
replacement='''if(i8){
  code=cudaMemsetAsync(error,0,sizeof(int),st);if(code!=cudaSuccess)return int(code);
  i8r113::quantize<<<R*6,32,0,st>>>(static_cast<const B*>(activation),qA,sA,error,R);
  code=cudaMemsetAsync(jobs,0,sizeof(int32_t),st);if(code!=cudaSuccess)return int(code);
  buildJobsI192R8669y<16><<<(E+255)/256,256,0,st>>>(static_cast<const int32_t*>(counts),static_cast<int32_t*>(jobExpert),static_cast<int32_t*>(jobStart),static_cast<int32_t*>(jobs),E);
  i8r113::matrix<<<640,64,0,st>>>(partial,qA,sA,static_cast<const int32_t*>(counts),static_cast<const int32_t*>(routes),static_cast<const int32_t*>(jobExpert),static_cast<const int32_t*>(jobStart),static_cast<const int32_t*>(jobs),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),T);
  i8r113::reduce<<<(T*3584+255)/256,256,0,st>>>(static_cast<B*>(out),partial,static_cast<const int32_t*>(ids),static_cast<const float*>(weights),T);return int(cudaGetLastError());
 }
 const char*toggle=std::getenv("INFINILM_KIMI_W2_PAIR_R9280");'''
assert needle in raw;raw=raw.replace(needle,replacement)
(d/'kimi_W2_INT8_provider_r113.cu').write_text(raw,encoding='utf8')
runner=(r/'Kimi-W2-pair-private-r9280/W2_pair_runner_r9280.cpp').read_text()
runner=runner.replace('modern_kimi_W2_pair_result_r9280','modern_kimi_W2_INT8_result_r113').replace('modern_kimi_W2_pair_run_r9280','modern_kimi_W2_INT8_run_r113')
runner=runner.replace('namespace {using J=', 'extern "C" size_t modern_kimi_i8_bytes_r113(size_t);\nextern "C" int modern_kimi_i8_bind_r113(int,size_t,void*,size_t,void*);\nextern "C" int modern_kimi_i8_counts_r113(int,uint64_t*,size_t);\nnamespace {using J=')
runner=runner.replace('for(const size_t T:std::array<size_t,2>{8,64}){','for(int scenario=0;scenario<3;++scenario){const size_t T=scenario==0?8:64;const bool constructed=scenario==2;\n const size_t scratchbytes=modern_kimi_i8_bytes_r113(T);auto i8storage=rt::Tensor::empty(r,{scratchbytes+512},rt::DataType::kUInt8);std::vector<uint8_t> init(scratchbytes+512,0xa5);r.memcpyH2D(i8storage->data(),init.data(),init.size(),false);r.syncStream();auto*scratch=static_cast<unsigned char*>(i8storage->data())+256;if(modern_kimi_i8_bind_r113(rank,T,scratch,scratchbytes,r.stream()))throw std::runtime_error("i8scratch bind");\n struct Unbind{rt::Runtime&r;int rank;~Unbind(){try{r.syncStream();}catch(...){std::terminate();}modern_kimi_i8_bind_r113(rank,0,nullptr,0,nullptr);}}unbind{r,rank};')
runner=runner.replace('unsetenv("INFINILM_KIMI_PROFILE_R9125");','unsetenv("INFINILM_KIMI_PROFILE_R9125");setenv("INFINILM_KIMI_W2_PAIR_R9280","1",1);setenv("INFINILM_KIMI_W2_CONSTRUCTED_R113",constructed?"1":"0",1);')
runner=runner.replace('uint64_t count_before[2]','uint64_t i8before[3],i8after[3];if(modern_kimi_i8_counts_r113(rank,i8before,3))throw std::runtime_error("i8 counts before");uint64_t count_before[2]')
runner=runner.replace('setenv("INFINILM_KIMI_W2_PAIR_R9280",enabled?"1":"0",1)','setenv("INFINILM_KIMI_W2_INT8_R113",enabled?"1":"0",1)')
runner=runner.replace('enabled?"W2_pair_ON":"W2_original_OFF"','enabled?"W2_INT8_ON":"W2_pair_baseline_OFF"')
runner=runner.replace('{{"T",T},{"arm",arm}', '{{"T",T},{"constructed_low_reuse",constructed},{"extra_scratch_bytes",scratchbytes},{"arm",arm}')
# Numeric error and canaries after eacharm, before recording result (outside measured wall).
needle='row["arms"].push_back('
check='''std::array<uint8_t,256>pre{},post{};r.memcpyD2H(pre.data(),i8storage->data(),256);r.memcpyD2H(post.data(),scratch+scratchbytes,256);for(auto v:pre)if(v!=0xa5)throw std::runtime_error("i8prefix canary");for(auto v:post)if(v!=0xa5)throw std::runtime_error("i8tail canary");if(enabled){int qerror=-1;r.memcpyD2H(&qerror,scratch+scratchbytes-4,4);if(qerror)throw std::runtime_error("i8quant nonfinite/range");}
  '''
runner=runner.replace(needle,check+needle)
runner=runner.replace('if(pair_after[0]-pair_before[0]!=34||pair_after[1]-pair_before[1]!=34)', 'if(pair_after[0]-pair_before[0]!=0||pair_after[1]-pair_before[1]!=34)')
needle='row["current_branch"].push_back('
runner=runner.replace(needle,'if(modern_kimi_i8_counts_r113(rank,i8after,3)||i8after[0]-i8before[0]!=34||i8after[1]-i8before[1]!=34||i8after[2]-i8before[2]!=(constructed?68:0))throw std::runtime_error("i8 actual arm coverage");\n '+needle)
runner=runner.replace('{{"T",T},{"I192_calls"','{{"T",T},{"constructed_low_reuse",constructed},{"I8_OFF_calls",i8after[0]-i8before[0]},{"I8_ON_calls",i8after[1]-i8before[1]},{"I192_calls"')
runner=runner.replace('setenv("INFINILM_KIMI_W2_PAIR_R9280","0",1);','setenv("INFINILM_KIMI_W2_PAIR_R9280","0",1);setenv("INFINILM_KIMI_W2_INT8_R113","0",1);setenv("INFINILM_KIMI_W2_CONSTRUCTED_R113","0",1);')
(d/'W2_INT8_runner_r113.cpp').write_text(runner,encoding='utf8')
# Verify unchanged W13 and pair kernels are included byte-for-byte as text.
for a,z in [('template<int TT,int Tile>__global__ void groupedW13I192R8669y','template <typename T>'),('template <typename T>\n__global__ void pairW2I192R9280','\n}\nnamespace')]:
 s=base.read_text();part=s[s.index(a):s.index(z,s.index(a))];assert part in raw,a
q=dict(CPU_only=True,GPU=False,defaultOFF=True,baseline_native='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd',baseline_provider_source_SHA256=sha(base),new_provider_source_SHA256=sha(d/'kimi_W2_INT8_provider_r113.cu'),wrapper_source_SHA256=sha(d/'W2_INT8_runner_r113.cpp'),producer_SHA256=sha(__file__),new_scratch_formula='R*(192+24+3584*4+4)+4 +512canary',scratch_lifetime='Runtime UInt8 Tensor beforewarm; sameownerstream binding;drain+unbind before Tensor destruction; no providerallocation',constructed='scenario2 T64 ids[r]=r%896 => pertoken top16unique,896experts M1/2; privatebuffer, originalselectedidsreadonly',scope='onefixedHIPdataflow +wrapper; originalW13/pairW2kernel unchanged; rootpublishedbeforeGPU')
(d/'CPU-source-contract-r113.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
