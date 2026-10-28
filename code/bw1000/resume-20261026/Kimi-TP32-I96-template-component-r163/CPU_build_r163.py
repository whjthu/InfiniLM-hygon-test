"""Private compile/link only: preserve original provider and all other objects."""
from pathlib import Path
import ast,json,hashlib,os,subprocess,time,shlex,traceback,sys
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-TP32-I96-template-r163'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();B.mkdir(exist_ok=False)
q={'completed':False,'CPU_only':True,'GPU':False,'Native_loaded':False,'actual_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),'commands':[]}
def run(cmd,label,seconds):
 q['commands'].append(cmd);z=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-6000:]);return z.stdout
try:
 m=json.loads((D/'source-manifest-r163.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 for p in D.glob('*.py'):ast.parse(p.read_bytes())
 geometry=json.loads(run([sys.executable,'-B',str(D/'CPU_geometry_r163.py')],'geometry',20));assert geometry['passed'];q['geometry']=geometry
 bp=R/'candidates/Kimi-MoE-phase-component-r155/CPU-build.json';assert sha(bp)=='c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd'
 p=json.loads(bp.read_bytes());assert p['completed']and p['CPU_only']and not p['GPU']and p['native_SHA256']=='d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b';assert sha(p['native'])==p['native_SHA256']
 for name,h in {**p['unchanged_object_SHA256'],**p['new_object_SHA256']}.items():assert sha(name)==h,name
 for deps in p['new_MD'].values():
  for name,h in deps.items():
   if Path(name).suffix in ['.h','.hpp','.cuh']:assert sha(name)==h,name
 cc=[x for x in p['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_moe_phase_r155.cpp')];assert len(cc)==1;cc=cc[0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_moe_I96_r163.cpp');cc[cc.index('-o')+1]=str(B/'moe.o');cc[cc.index('-MF')+1]=str(B/'moe.d');run(cc,'moe',180)
 cc=[x for x in p['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_provider_phase_r155.cu')];assert len(cc)==1;cc=cc[0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'provider_I96_r163.cu');cc[cc.index('-o')+1]=str(B/'I96.o');cc[cc.index('-MF')+1]=str(B/'I96.d');run(cc,'I96',240)
 native=B/'libKimi_I96_r163.so';cc=[x for x in p['commands']if '-shared'in x and '-o'in x and x[x.index('-o')+1]==p['native']];assert len(cc)==1;cc=cc[0][:];old=str(R/'candidates/Kimi-MoE-phase-component-r155/moe.o');assert cc.count(old)==1;cc[cc.index(old)]=str(B/'moe.o');cc.insert(cc.index('-o'),str(B/'I96.o'));cc[cc.index('-o')+1]=str(native);cc=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in cc];assert str(R/'candidates/Kimi-MoE-phase-component-r155/provider.o')in cc
 if '-Wl,--no-undefined'not in cc:cc.insert(cc.index('-o'),'-Wl,--no-undefined')
 run(cc,'Native_link',120)
 wrapper=B/'libKimi_I96_component_r163.so';cc=[x for x in p['commands']if '-shared'in x and '-o'in x and x[x.index('-o')+1]==p['library']];assert len(cc)==1;cc=cc[0][:];cc.insert(1,'-I'+str(D));hits=[i for i,x in enumerate(cc)if x.endswith('/component_runner_r155.cpp')];assert len(hits)==1;cc[hits[0]]=str(D/'TP32_I96_MoE_r163.cpp');assert cc.count(p['native'])==1;cc[cc.index(p['native'])]=str(native);cc[cc.index('-o')+1]=str(wrapper);cc[cc.index('-MF')+1]=str(B/'adapter.d');cc=[('-Wl,-Map,'+str(B/'adapter.map'))if x.startswith('-Wl,-Map')else x for x in cc]
 if '-Wl,--no-undefined'not in cc:cc.insert(cc.index('-o'),'-Wl,--no-undefined')
 run(cc,'adapter',180)
 md={}
 for f in B.glob('*.d'):
  names=shlex.split(f.read_text().replace('\\\n',' ').split(':',1)[1]);md[f.name]={n:sha(n)for n in names if Path(n).is_file()}
 header='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert md['adapter.d'][header]==md['moe.d'][header]=='3f415dcb1585d2236397bde988bdbbec4a0d73dc48fe2e2b371310b706ac67fd'
 nm=run(['nm','-D','--defined-only',str(native)],'Native_nm',30);assert all(s in nm for s in ['modern_kimi_grouped_I96_r163','modern_kimi_I96_mode_r163','modern_kimi_I96_counts_r163'])
 exported=run(['nm','-D','--defined-only',str(wrapper)],'adapter_nm',30);assert all(s in exported for s in ['modern_kimi_TP32_moe_r163','modern_kimi_TP32_moe_result_r163'])
 ld=subprocess.run(['ldd','-r',str(wrapper)],env=dict(os.environ,LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH']),capture_output=True,text=True,timeout=45);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr and 'undefined symbol:'not in ld.stdout+ld.stderr
 q.update(completed=True,parent_receipt_SHA256=sha(bp),parent_native_SHA256=p['native_SHA256'],native=str(native),native_SHA256=sha(native),library=str(wrapper),library_SHA256=sha(wrapper),LD_LIBRARY_PATH=str(B)+':'+p['LD_LIBRARY_PATH'],new_MD=md,layer_headers=p['layer_headers'],source_manifest_SHA256=sha(D/'source-manifest-r163.json'),new_objects={str(B/n):sha(B/n)for n in ['moe.o','I96.o']},original_provider_preserved_SHA256=sha(R/'candidates/Kimi-MoE-phase-component-r155/provider.o'),math_change='I192 TT2Tile4/W2pair parameterizedI96 only; originaldecode/reduction/Situ, FP32/BF16, noF64',actual_GPU_qualified=False)
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','elapsed_s','native_SHA256','library_SHA256','error_class']}));assert q['completed']
