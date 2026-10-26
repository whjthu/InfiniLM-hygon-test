import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent;d=r/'Kimi-W2-INT8-private-r113';p=d/'kimi_W2_INT8_provider_r113.cu';s=p.read_text();s=s.replace('#if defined(__HIP_DEVICE_COMPILE__)\n#include <du_mma.h>\n#endif','')
s=s.replace('#if defined(__HIP_DEVICE_COMPILE__)\nnamespace dm=du::dumma;\n#endif',r'''
#if defined(__CUDA_ARCH__)
// Exact gfx936 DU mapping, bound to DTK du_mma.hpp SHA19126534...:
// A rowmajor/B colmajor: lane&15 outputrow/column, (lane>>4)*8 K-byte block.
// Accumulator rowmajor: row=lane&15, col=(lane>>4)+4*i.
using I4r113=int __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void atom(const signed char*a,const signed char*b,int*out){
 unsigned l=threadIdx.x&63,row=l&15,k0=(l>>4)*8;union Bits{uint64_t u;long s;};Bits ax{0},bx{0};
 #pragma unroll
 for(unsigned i=0;i<8;++i){ax.u|=uint64_t(uint8_t(a[row*32+k0+i]))<<(8*i);bx.u|=uint64_t(uint8_t(b[row*32+k0+i]))<<(8*i);}
 I4r113 z={0,0,0,0};I4r113 c=__builtin_amdgcn_mmac_i32_16x16x32i8(ax.s,bx.s,z);
 #pragma unroll
 for(unsigned i=0;i<4;++i)out[row*16+(l>>4)+4*i]=c[i];
}
#endif
''')
s=s.replace('#if defined(__HIP_DEVICE_COMPILE__)','#if defined(__CUDA_ARCH__)')
a=s.index('   dm::DUFragment<dm::matrix_a,16,16,32');z=s.index('__syncthreads();',a)+len('__syncthreads();');s=s[:a]+'   atom(aa,bb,tmp);__syncthreads();'+s[z:]
a=s.index('__global__ void layout(');body=s.index('#if defined(__CUDA_ARCH__)',a);z=s.index('#endif',body);s=s[:body]+'#if defined(__CUDA_ARCH__)\natom(a,b,out);\n'+s[z:];assert 'dm::'not in s;p.write_text(s)
f=d/'CPU_build_r113.py';s=f.read_text();s=s.replace("candidates/Kimi-W2-INT8-r113c","candidates/Kimi-W2-INT8-r113final")
a=s.index("includes=[x for x in cmd if x.startswith('-I')]");z=s.index("lines=asm.read_text()",a)
s=s[:a]+"asm=b/'provider.s';acmd=cmd[:];acmd.remove('-c');acmd.remove('-MMD');i=acmd.index('-MF');del acmd[i:i+2];acmd[acmd.index('-o')+1]=str(asm);acmd+=['-S'];run(acmd,'assembly_nvcc',180)\n "+s[z:]
a=s.index(" for name in ['du_mma.h','du_mma.hpp']:");z=s.index(' q.update(completed=True',a)
s=s[:a]+" q['integer_mapping_reference_SHA256']={str(p):sha(p)for p in [pathlib.Path('/opt/dtk/include/du_mma.hpp'),pathlib.Path('/opt/dtk/cuda/cuda/include/crt/mma.hpp')]}\n assert q['integer_mapping_reference_SHA256']['/opt/dtk/include/du_mma.hpp']=='191265343da30927d9a02d7ad3d64ca5f52b9f41f80008f979c9ae05a03d0d47'\n assert q['integer_mapping_reference_SHA256']['/opt/dtk/cuda/cuda/include/crt/mma.hpp']=='bada0b8317ef5426cc362abbe0a9dacdb9c1b23f8e8b5d95ae41f4ece53c86b3'\n"+s[z:];f.write_text(s)
q=json.loads((d/'CPU-source-contract-r113.json').read_text());q['new_provider_source_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();q['intrinsic_binding']='ActualCUDAdevice macro __CUDA_ARCH__=750/gfx936; directbuiltin copiedDUregistermapping; noHIPheader or fakehosttype';q['final_binding_producer_SHA256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();(d/'CPU-source-contract-r113.json').write_text(json.dumps(q,indent=2)+'\n');compile(s,'CPU_build_r113.py','exec')
for f in d.glob('*.sh'):f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'));assert b'\r'not in f.read_bytes()
print(q['new_provider_source_SHA256'])
