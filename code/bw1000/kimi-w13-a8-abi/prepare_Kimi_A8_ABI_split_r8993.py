import pathlib,hashlib,ast,json
r=pathlib.Path.cwd();b=r/'Kimi-W13-A8-ABI-private-r8993';b.mkdir(exist_ok=True)
old=(r/'Kimi-W13-A8-private-r8991/kimi_W13_A8_r8991.cu').read_text()
# Keep algorithm bodies byte-identical apart from BF16 bit-load and API isolation.
hip=old[old.index('namespace {'):old.index('__global__ void situ')]
hip=hip.replace('using B=__nv_bfloat16;','using B=uint16_t;').replace('__bfloat162float(x[index])','__uint_as_float(uint32_t(x[index])<<16)')
hip='#include <hip/hip_runtime.h>\n#include <du_mma.h>\n#include <cstdint>\n'+hip+'}\n'
hip+='''extern "C" int kimi_A8_HIP_matrix_r8993(int device,const void*input,const void*w13,const void*s13,const void*counts,const void*routes,void*extra,void*stream){
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
'''
(b/'kimi_A8_matrix_r8993.hip').write_bytes(hip.encode())
start=old.index('namespace infini::ops::kimi_i192_r8669y');end=old.index('namespace {',start)
cuda='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cstdint>\n#include "base/fused_moe_mxfp4.h"\n#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"\n'+old[start:end]
cuda+='extern "C" int kimi_A8_HIP_matrix_r8993(int,const void*,const void*,const void*,const void*,const void*,void*,void*);\n'
cuda+='namespace {\nusing B=__nv_bfloat16;\nconstexpr unsigned E=896,T=16,H=3584,I=192,N=384,R=256;\n'
cuda+=old[old.index('__global__ void situ'):old.index('// Same explicit pointer ABI')]
adapter=old[old.index('// Same explicit pointer ABI'):]
adapter=adapter.replace('modern_kimi_A8_W13_r8991','modern_kimi_A8_W13_r8993')
adapter=adapter.replace('!extra)return','!extra||stream)return')
a=adapter.index(' auto st=');z=adapter.index(' situ<<<')
adapter=adapter[:a]+''' auto st=static_cast<cudaStream_t>(stream);auto*raw=static_cast<float*>(extra);
 auto code=cudaMemsetAsync(counts,0,E*sizeof(int),st);if(code!=cudaSuccess)return int(code);
 infini::ops::kimi_i192_r8669y::packRoutesI192R8669y<<<1,256,0,st>>>(static_cast<const int*>(ids),static_cast<int*>(counts),static_cast<int*>(routes),E,T);
 if(cudaGetLastError()!=cudaSuccess)return int(cudaErrorLaunchFailure);
 const int hip_status=kimi_A8_HIP_matrix_r8993(device,input,w13,s13,counts,routes,extra,stream);
 if(hip_status)return 10000+hip_status;
'''+adapter[z:]
cuda+=adapter
(b/'kimi_A8_adapter_r8993.cu').write_bytes(cuda.encode())
assert 'cuda'not in hip.lower()and 'infini'not in hip.lower()and '__nv_bfloat16'not in hip
assert 'hip/hip_'not in cuda and 'du_mma.h'not in cuda
runner=(r/'Kimi-W13-A8-private-r8991/Kimi_W13_A8_real_r8991.py').read_text().replace('modern_kimi_A8_W13_r8991','modern_kimi_A8_W13_r8993')
runner=runner.replace("stream=torch.cuda.current_stream().cuda_stream","stream=torch.cuda.current_stream().cuda_stream;assert stream==0,'default_stream_only_ABI_scope'")
runner=runner.replace("'A8_workspace':","'stream_ABI_scope':'default stream0 only; cross-runtime same-default-stream interoperability not yet GPU verified','A8_workspace':")
ast.parse(runner);(b/'Kimi_A8_ABI_real_r8993.py').write_bytes(runner.encode())
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in b.iterdir()if p.is_file()}))
