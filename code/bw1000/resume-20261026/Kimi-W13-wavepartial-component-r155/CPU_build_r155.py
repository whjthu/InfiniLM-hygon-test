"""Bounded CPU compilation only: 2 HOST TUs, 1 unchanged-math provider TU."""
from pathlib import Path
import ast,hashlib,json,os,shlex,subprocess,time,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-MoE-phase-component-r155';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(cmd,label,seconds):
 q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-6000:])
try:
 source=json.loads((D/'source-manifest-r155.json').read_bytes())
 for rel,h in source['files_SHA256'].items():assert sha(D/rel)==h
 run(['c++','-std=c++17','-O0','-Wall','-Wextra','-pedantic',str(D/'CPU_contract_r155.cpp'),'-o',str(B/'CPU_contract')],'CPU_contract_compile',60)
 run([str(B/'CPU_contract')],'CPU_contract_run',20)
 run(['c++','-std=c++17','-O2','-fno-fast-math',str(D/'CPU_reduction_contract_r155.cpp'),'-o',str(B/'CPU_reduction_contract')],'CPU_reduction_compile',60)
 run([str(B/'CPU_reduction_contract')],'CPU_reduction_run',20)
 p=json.loads((D/'parent-build-pins-r155.json').read_bytes());ctx=json.loads((D/'parent-GPU-context-r155.json').read_bytes())
 base=json.loads((R/'candidates/Kimi-MoE-batch-profile-r9125/CPU-build.json').read_bytes())
 for dependencies in [p['provider_MD'],p['adapter_MD'],base['dependencies']['moe.d']]:
  for path,h in dependencies.items():
   if Path(path).suffix in ['.h','.hpp','.cuh']:assert sha(path)==h,('original MD header changed',path)
 native=R/'candidates/Kimi-W2-pair-r9280/libKimi_W2_pair_r9280.so';assert sha(native)=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
 for path,h in p['all_linked_objects'].items():assert sha(path)==h
 oldmoe=R/'candidates/Kimi-I192-batch-Model-r9129/CPU-build.json';m=json.loads(oldmoe.read_bytes());assert m['moe_object_SHA256']=='55fb31b5f4fa1510f06e8b62894d64d52926862371f43e1e2d8c5aaeb2624800'
 cc=m['commands'][0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_moe_phase_r155.cpp');cc[cc.index('-o')+1]=str(B/'moe.o');cc[cc.index('-MF')+1]=str(B/'moe.d');run(cc,'moe',180)
 cc=p['provider'][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_provider_phase_r155.cu');cc[cc.index('-o')+1]=str(B/'provider.o');cc[cc.index('-MF')+1]=str(B/'provider.d');run(cc,'provider',240)
 newnative=B/'libKimi_MoE_phase_r155.so';cc=p['link'][:]
 for old,new in [(str(R/'candidates/Kimi-I192-batch-Model-r9129/moe.o'),str(B/'moe.o')),(str(R/'candidates/Kimi-W2-pair-r9280/provider.o'),str(B/'provider.o'))]:assert cc.count(old)==1;cc[cc.index(old)]=new
 cc[cc.index('-o')+1]=str(newnative);cc=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in cc];run(cc,'link',120)
 wrapper=B/'libKimi_MoE_component_r155.so';cc=p['adapter'][:];cc.insert(1,'-I'+str(D));cc=[str(D/'component_runner_r155.cpp')if x.endswith('/W2_pair_runner_r9280.cpp')else str(newnative)if x==str(native)else x for x in cc];cc[cc.index('-o')+1]=str(wrapper);cc[cc.index('-MF')+1]=str(B/'adapter.d');run(cc,'adapter',180)
 deps={}
 for path in B.glob('*.d'):
  deps[path.name]={s:sha(s)for s in shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1])if Path(s).is_file()}
 critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert deps['adapter.d'][critical]==p['critical_MoE_header_SHA256']==deps['moe.d'][critical]
 ld=subprocess.run(['ldd','-r',str(wrapper)],env=dict(os.environ,LD_LIBRARY_PATH=p['LD_LIBRARY_PATH']),capture_output=True,text=True,timeout=45);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr and 'undefined symbol:'not in ld.stdout+ld.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(newnative)],text=True);assert all(x in nm for x in ['modern_kimi_moe_phase_mark_r155','modern_kimi_W13_wavepartial_counts_r155','modern_kimi_moe_event_arm_r9220','modern_kimi_grouped_I192_r8669y'])
 index=Path('/shared/Kimi-K3/model.safetensors.index.json');assert sha(index)=='a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd'
 weightmap=json.loads(index.read_bytes())['weight_map'];headers={}
 for layer in [1,47]:
  prefix='language_model.model.layers.%d.block_sparse_moe.'%layer;files=sorted({v for k,v in weightmap.items()if k.startswith(prefix)});assert files;values={}
  for name in files:
   with (index.parent/name).open('rb')as stream:size=int.from_bytes(stream.read(8),'little');assert 0<size<8*1024*1024;raw=stream.read(size);assert len(raw)==size
   values[name]=hashlib.sha256(raw).hexdigest()
  headers[str(layer)]=values
 assert headers['1']==json.loads((D/'original_L1_header.json').read_bytes())
 q.update(completed=True,native=str(newnative),native_SHA256=sha(newnative),library=str(wrapper),library_SHA256=sha(wrapper),parent_native_SHA256=sha(native),unchanged_object_SHA256={x:sha(x)for x in p['link']if x.endswith('.o') and not x.endswith('/Kimi-I192-batch-Model-r9129/moe.o') and not x.endswith('/Kimi-W2-pair-r9280/provider.o')},new_object_SHA256={str(x):sha(x)for x in [B/'moe.o',B/'provider.o']},new_MD=deps,layer_headers=headers,LD_LIBRARY_PATH=p['LD_LIBRARY_PATH'],source_manifest_SHA256=sha(D/'source-manifest-r155.json'),elapsed_s=time.monotonic()-start,math_kernel_source_unchanged=False,new_GPU_TU_scope='one HIP provider TU includes baseline and new wave-first reduction kernel; TT2/Tile4 FP4/Situ and W2 bodies unchanged; summation order differs',actual_GPU_qualified=False)
except BaseException as error:q.update(error_class=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','elapsed_s','native_SHA256','library_SHA256','error_class']}));assert q['completed']
