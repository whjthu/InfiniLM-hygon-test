"""v2 source preparation only. No compiler, SSH, GPU, Slurm, weights or Git."""
import pathlib,json,hashlib,difflib
r=pathlib.Path(__file__).resolve().parent;v1=r/'Kimi-W2-INT8-private-r113';d=r/'Kimi-W2-INT8-register-v2-private-r113c';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=v1/'kimi_W2_INT8_provider_r113.cu';raw=p.read_bytes();assert sha(p)=='1db5b9c937f448cdeb41bd7007f4a8405313de459f425d7e576b106c77a9c76e';s=raw.decode();lines=s.splitlines(keepends=True);first=next(i for i,x in enumerate(lines)if x.startswith('__global__ void matrix('));last=next(i for i,x in enumerate(lines[first+1:],first+1)if x.startswith('__global__ void reduce('));prefix=''.join(lines[:first]);oldmatrix=''.join(lines[first:last]);suffix=''.join(lines[last:]);eol='\r\n'if b'\r\n'in raw else'\n'
new=r'''__global__ void matrix(float*partial,const signed char*q,const float*as,const int32_t*cnt,const int32_t*routes,const int32_t*expert,const int32_t*start,const int32_t*jobs,const uint8_t*w,const uint8_t*sc,size_t T){
#if defined(__CUDA_ARCH__)
 // v2 changes only ownership of C/scale accumulation and the shared transpose.
 // All64 lanes must execute MMAC, including paddedA rows: their B fragments
 // still contribute to other valid rows. The validrow predicate starts AFTER it.
 __shared__ __align__(16) signed char aa[512];__shared__ __align__(16) signed char bb[512];__shared__ __align__(16) float tmp[256];__shared__ int rr[16];
 const unsigned lane=threadIdx.x,mnative=lane&15,colbase=lane>>4,k0=colbase*8;size_t R=T*16;
 for(size_t work=blockIdx.x;work<size_t(jobs[0])*224;work+=gridDim.x){size_t job=work/224;unsigned n0=(work%224)*16;int e=expert[job],s=start[job];
  if(lane<16)rr[lane]=(s+int(lane)<cnt[e])?routes[size_t(e)*R+s+lane]:-1;__syncthreads();const int rid=rr[mnative];float sum[4]={0.f,0.f,0.f,0.f};
  #pragma unroll
  for(unsigned g=0;g<6;++g){
   for(unsigned p=lane;p<512;p+=64){unsigned m=p/32,k=p%32;int r=rr[m];aa[p]=r>=0?q[size_t(r)*192+g*32+k]:0;unsigned n=p/32;uint8_t byte=w[(size_t(e)*3584+n0+n)*96+g*16+k/2];bb[p]=decode(uint8_t((byte>>(4*(k&1)))&15));}__syncthreads();
   // Same bytepacking/builtin as qualified750/751 atom, now retain C in registers.
   union Bits{uint64_t u;long s;};Bits ax{0},bx{0};
   #pragma unroll
   for(unsigned i=0;i<8;++i){ax.u|=uint64_t(uint8_t(aa[mnative*32+k0+i]))<<(8*i);bx.u|=uint64_t(uint8_t(bb[mnative*32+k0+i]))<<(8*i);}
   I4r113 z={0,0,0,0};I4r113 c=__builtin_amdgcn_mmac_i32_16x16x32i8(ax.s,bx.s,z);
   if(rid>=0){const float av=as[size_t(rid)*6+g];
    #pragma unroll
    for(unsigned i=0;i<4;++i){unsigned n=colbase+4*i;float sw=ldexpf(.5f,int(sc[(size_t(e)*3584+n0+n)*6+g])-127);sum[i]+=float(c[i])*(av*sw);}
   }
   // Keep conservative A/B reuse barrier; no pergroup I32 shared roundtrip.
   __syncthreads();
  }
  if(rid>=0){
   #pragma unroll
   for(unsigned i=0;i<4;++i)tmp[mnative*16+colbase+4*i]=sum[i];
  }
  __syncthreads();
  // Exactly one final F32 rowmajor transpose, then the original coalesced egress.
  #pragma unroll
  for(unsigned i=0;i<4;++i){unsigned p=lane+64*i,m=p/16,n=p%16;int r=rr[m];if(r>=0)partial[size_t(r)*3584+n0+n]=tmp[p];}__syncthreads();
 }
#endif
}

'''.replace('\n',eol)
newraw=(prefix+new+suffix).encode();out=d/'kimi_W2_INT8_register_provider_r113v2.cu';out.write_bytes(newraw);assert newraw.startswith(prefix.encode())and newraw.endswith(suffix.encode());assert prefix.encode()+oldmatrix.encode()+suffix.encode()==raw
# Preserve v1 runner ABI and every consumer byte; nohost/source/control/math change.
for name in ['W2_INT8_runner_r113.cpp','run_GPU_r113.py','expected-headers-r9125.json']:(d/name).write_bytes((v1/name).read_bytes());assert sha(d/name)==sha(v1/name)
(d/'v1-v2-matrix-only.diff').write_bytes(''.join(difflib.unified_diff(s.splitlines(True),(prefix+new+suffix).splitlines(True),fromfile='v1-qualified750-provider',tofile='v2-register-scale-provider')).encode())
# Address/ownership proof, no performance sampling or device execution.
checks=0;oldmap={};newmap={}
for lane in range(64):
 for i in range(4):
  p0=lane+64*i;oldmap[(p0//16,p0%16)]=(lane,i)
  newmap[(lane&15,(lane>>4)+4*i)]=(lane,i)
assert len(oldmap)==len(newmap)==256 and oldmap.keys()==newmap.keys()
for validM in range(17):
 writes={};reads={}
 for m in range(16):
  for n in range(16):
   ln,ii=newmap[m,n];assert ln&15==m and(ln>>4)+4*ii==n
   if m<validM:
    writes[m*16+n]=(ln,ii);ol,oi=oldmap[m,n];assert ol+64*oi==m*16+n
    for g in range(6):
     assert (m,n,g,'float(C)*(A_scale*W_scale)')==(ln&15,(ln>>4)+4*ii,g,'float(C)*(A_scale*W_scale)');checks+=1
 for lane in range(64):
  for i in range(4):
   pp=lane+64*i;m,n=divmod(pp,16)
   if m<validM:reads[pp]=(lane,i)
 assert writes.keys()==reads.keys() and len(writes)==validM*16
assert new.count('__builtin_amdgcn_mmac_i32_16x16x32i8')==1;assert new.index('I4r113 c=')<new.index('if(rid>=0)');assert 'sum[i]+=float(c[i])*(av*sw)'in new;assert 'tmp[p]=c'not in new
proof={'schema':1,'status':'SOURCE_ONLY_REGISTER_SCALE_V2_WAIT_ROOT_CLAIM_PUBLICATION_BEFORE_COMPILE','GPU':False,'compile':False,'new_Slurm':False,'weights_payload':False,'Git_mutation':False,'v1_source_SHA256':sha(p),'v2_source_SHA256':sha(out),'outside_matrix_raw_bytes_identical':True,'unchanged_runner_SHA256':sha(d/'W2_INT8_runner_r113.cpp'),'unchanged_consumer_SHA256':sha(d/'run_GPU_r113.py'),'mapping_checks':checks,'ownership':'old coalesced owner lane=(16m+n)%64,i=(16m+n)//64; new native owner lane=m+16*(n%4),i=n//4; both bijections256','native_C_mapping':'C[i] -> row=lane&15,col=(lane>>4)+4*i','scale_chain':'Every valid(m,n) has six independentzeroI32group32 dots; float(Cg)*(A_scale[rid,g]*ldexp(.5,W_scale[e,n,g]-127)), F32 sum g0..5 ascending, same expression asv1; invalidrows skipscale only','MMAC_fullwave':'Builtin sourcecall before rid predicate; all64 lane B fragments required eveninvalidA rows. Inspect emittedISA toreject compiler sinking MMAC under validrow execmask. Source placementalone not ISA proof.','shared_transpose':'Only validrows store F32 tmp[m*16+n]; barrier; existingp=lane+64*i egressreads onlysamevalidrow slots; no uninitializedinvalidslotread','static_per_CTA_jobNtile_only':{'v1_tmp_I32_LDS_store_read_bytes':12288,'v2_tmp_F32_LDS_store_read_bytes_formula':'128*validM, max2048','v1_scalar_scaled_elements':1536,'v2_scalar_scaled_elements_formula':'96*validM','v1_source_barriers_per_job':20,'v2_source_barriers_per_job':15,'static_LDS_bytes_same':2112},'dynamic_bottleneck_or_speedup_proven':False,'VGPR_spills_and_actual_instructioncount':'UNKNOWN untilboundedcompileISA; no promise lowerregisters','parameters_unchanged':'T8/T64historical +T64constructed; M16,N16,K32atom,Ntiles224,Kgroups6,launch640CTA/64threads; noT/tile/reusegrid','baseline':'W2pair fixedON andINT8 separateflag; v1result preservedPARK; newv2 changes dataflow notprecision','actual_qualified_references':{'GPU750_provider_object_SHA256':'bbd747043eeded2b50ffda2fa236c929d90c18f9889ba4dd08c332c0d8bd3ebf','native750_SHA256':'0231e49914507a431627f77c53747466ff12bbbfbca7f5a44141ce3550de1566','runner751_SHA256':'341a35663e6433cbb24ebcdf18f9d33288464af434e38a16bd56c64598fd4e94','DU_mapping_header_SHA256':'191265343da30927d9a02d7ad3d64ca5f52b9f41f80008f979c9ae05a03d0d47','CUDA_mma_header_SHA256':'bada0b8317ef5426cc362abbe0a9dacdb9c1b23f8e8b5d95ae41f4ece53c86b3','macro_receipt_SHA256':sha(r/'Kimi-INT8-device-macros-r113.json'),'v1_terminal_SHA256':sha(v1/'terminal-canonical-r113b.json')},'compile_plan':'Afterrootclaimordinarypush only: same750nvcc/arch/flags,1providerTU+sameTU-S; replaceonlyproviderobject in750native link; unchanged751runner source/ABI rebuiltagainstnewprivateabsoluteDT_NEEDED (oldrunnerDSO hardcodesv1native); exactoldMD/allotherobjectsSHA; ld-r/symbol/criticalheaderclosure; noGPU/Model/profile/newCPUgate','compile_STOP':'15minbound; ISA mustshow real6W2I8+1unchangedwarmI8,noF64; inspect no validrowmask aroundMMAC and no pergroupI32 tmp LDSroundtrip; spill/resourceissue or uncertainmapping =>STOP, noframeworkchange/grid','producer_SHA256':sha(__file__)}
(d/'sourceproof-r113v2.json').write_text(json.dumps(proof,indent=2)+'\n')
plan='''# W2 INT8 v2：寄存器scale/累加（仅CPU源码）

v1已PARK：T8四对全负、合成低复用T64四对全负；高复用历史T64正例不推Model。v1源码和结果均未覆盖。

v2只替换matrix函数。使用750/751/753已验证的DU accumulator位置：每lane固定row=lane&15，四个col=(lane>>4)+4*i。在同一寄存器持有对应四个FP32 sum；六个独立I32 group32各自scale后按g升序累加。rid<0只跳过scalar scale，MMAC必须仍由全64lane执行，因为这些lane的B碎片也用于有效A行。六组结束后一次FP32 rowmajor shared transpose，再用原coalesced写partial。quant/jobs/W13/reduce、activation量化、全局scratch及三个scenario不变。

sourceproof枚举每输出的旧/新owner和g链，证明256输出双射、有效行写读集合一致、无无效tmp读取；矩阵函数外原始字节及runner/consumer/header原字节一致。静态tmp流量从每job×Ntile 12288B变为128×validM B（最多2048B），scalar scale从1536个元素变为96×validM，source barrier20→15；这些是代码数据流计数，不能据此声称动态瓶颈或速度收益。LDS仍2112B，VGPR/spill变化未知。

等待root将新机制认领普通push后才编译。最小编译：使用实际750 nvcc/flags及同TU -S，只替换provider.o；其余旧对象和MD逐项SHA核验。旧751 runner DSO绝对DT_NEEDED绑定v1 native，因此需要从原字节不变的wrapper源码按原命令链接新private native，ABI不变，不创造新HOST流程。检查真实6条W2 I8及原warm原语、无F64、关键header/ldd-r闭包，另必须核实编译器没有把MMAC下沉到validrow exec mask中、没有每组I32 tmp往返。若无法确认或出现spill/闭包扩展，STOP。

当前未SSH、未编译、未Slurm、未GPU、未读权重，也没有新CPU性能/独立门。后续GPU仍需源码发布及root单独GO。
''';(d/'PLAN.zh-CN.md').write_text(plan)
index={'private':str(d),'status':proof['status'],'sourcefiles_SHA256':{x.name:sha(x)for x in d.iterdir()if x.is_file()},'producer_SHA256':sha(__file__)};(d/'source-producer-index-r113v2.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps({'private':str(d),'provider_SHA256':sha(out),'sourceproof_SHA256':sha(d/'sourceproof-r113v2.json'),'index_SHA256':sha(d/'source-producer-index-r113v2.json'),'producer_SHA256':sha(__file__),'mapping_checks':checks,'compile':False,'GPU':False}))
