import pathlib,json,hashlib,subprocess,time,sys,struct,collections,os
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();b=r/'candidates/deep-sdot4-r8740';assert not b.exists();b.mkdir();src=r/'incoming/deep_routed_sdot4_r8740.cu';spec=json.loads((r/'incoming/Deep-sdot4-source-r8740.json').read_text());assert sha(src)==spec['candidate_source_SHA256'];base=r/'candidates/models-5f91ad7-r3675/Deep/helpers';assert sha(base/'src/deepseek_routed_i8_kernels_r20.cu')==spec['baseline_source_SHA256'];assert sha(base/'routed.o')=='08ed7c8bfb7fe56cee66a6d6bdc5a5d569b05574ae68b2e7741d755a563ea756';obj=b/'routed-sdot4.o';flags=['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1'];start=time.monotonic();args=flags+['-c',str(src),'-o',str(obj)];p=subprocess.run(args,capture_output=True,text=True,timeout=120);(b/'compile.log').write_text(p.stdout+p.stderr);header=pathlib.Path('/opt/dtk/cuda/cuda/include/sm_61_intrinsics.hpp');report={'completed':p.returncode==0,'CPU_only':True,'GPU_initialized':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_SHA256':sha(src),'parent_actual_object_SHA256':sha(base/'routed.o'),'compiler_returncode':p.returncode,'compile_argv':args,'compiler_diagnostics':p.stderr[-5000:],'CPU_seconds':time.monotonic()-start,'DTK_dp4a_header_SHA256':sha(header),'DTK_dp4a_definition':header.read_text().splitlines()[18:44],'math':'Same signed integer dot, W/launch/tree/scales/BF16 unchanged. K4096 maxabsintegerproductsum67108864<int32max; K2564194304. No saturation needed.','GPU_NOT_RUN':True,'ISA':[]}
if p.returncode==0:
 raw=obj.read_bytes();report['object_SHA256']=sha(obj)
 for i in range(len(raw)-64):
  if raw[i:i+4]!=b'\x7fELF'or raw[i+4]!=2 or struct.unpack_from('<H',raw,i+18)[0]!=224:continue
  off=struct.unpack_from('<Q',raw,i+40)[0];ents,num=struct.unpack_from('<HH',raw,i+58)
  if not off or ents!=64 or off+num*64>len(raw)-i:continue
  end=off+num*64
  for k in range(num):
   so=i+off+k*64;typ=struct.unpack_from('<I',raw,so+4)[0];a,z=struct.unpack_from('<QQ',raw,so+24)
   if typ!=8:end=max(end,a+z)
  if end>len(raw)-i:continue
  elf=b/f'gpu-{i}.elf';elf.write_bytes(raw[i:i+end]);z=subprocess.run(['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936',str(elf)],capture_output=True,text=True,timeout=30);isa=elf.with_suffix('.isa.txt');isa.write_text(z.stdout);report['ISA'].append({'path':str(isa),'SHA256':sha(isa),'exit':z.returncode,'native_dot_lines':[s.strip()for s in z.stdout.splitlines()if 'v_dot'in s],'MMAC_lines':[s.strip()for s in z.stdout.splitlines()if 'mmac'in s.lower()],'software_byte_multiply_lines':sum('v_mul_i32_i24_sdwa'in s for s in z.stdout.splitlines())})
report['native_scalar_dot4_confirmed']=any(v['native_dot_lines']for v in report['ISA']);report['eligible_for_GPU']=report['completed']and report['native_scalar_dot4_confirmed'];out=r/'results/deep-sdot4-CPU-r8740.json'
with out.open('x')as f:json.dump(report,f,indent=2)
print(json.dumps({'completed':report['completed'],'compiler_returncode':p.returncode,'CPU_seconds':report['CPU_seconds'],'native_scalar_dot4_confirmed':report['native_scalar_dot4_confirmed'],'eligible_for_GPU':report['eligible_for_GPU'],'receipt_SHA256':sha(out),'diagnostics':p.stderr[-3000:]}));sys.exit(0 if p.returncode==0 else 1)
