"""Bounded CPU build: reuse503MoE/device kernels; only quadHIP+componentHOST new."""
from pathlib import Path
import os,json,hashlib,subprocess,shlex,struct,time,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-W2-quad4-component-r166';B.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'new_GPU_TUs':1,'new_HOST_adapter_TUs':1,'HOST_MoE_recompiled':False,'device_math_baseline':'CPU503 d559 originalW13/reducer/pairW2; no161directdecode'}
def run(cmd,name,seconds):
 q['commands'].append(cmd);z=subprocess.run(cmd,cwd=B,text=True,capture_output=True,timeout=seconds);(B/(name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(name,z.stderr[-4000:])
def dep(p):return {f:sha(f)for f in shlex.split(Path(p).read_text().replace(chr(92)+chr(10),' ').split(':',1)[1])if Path(f).is_file()}
def device_sections(path):
 raw=Path(path).read_bytes();assert raw[:6]==b'\x7fELF\x02\x01';off=struct.unpack_from('<Q',raw,40)[0];size,count,names=struct.unpack_from('<HHH',raw,58);assert size==64;hs=[struct.unpack_from('<IIQQQQIIQQ',raw,off+i*64)for i in range(count)];h=hs[names];strings=raw[h[4]:h[4]+h[5]];out={}
 for h in hs:
  name=strings[h[0]:].split(b'\0',1)[0].decode()
  if h[2]&4 or 'fatbin'in name.lower():out[name]=hashlib.sha256(raw[h[4]:h[4]+h[5]]).hexdigest()
 assert out;return out
try:
 source=json.loads((D/'source-manifest-r166.json').read_bytes())
 for n,h in source['files_SHA256'].items():assert sha(D/n)==h
 original=R/'candidates/Kimi-MoE-phase-component-r155/CPU-build.json';assert sha(original)=='c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd';b=json.loads(original.read_bytes());assert b['completed']and b['native_SHA256']=='d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b'
 oldmoe=R/'candidates/Kimi-MoE-phase-component-r155/moe.o';oldprovider=R/'candidates/Kimi-MoE-phase-component-r155/provider.o';assert sha(oldmoe)==b['new_object_SHA256'][str(oldmoe)]=='79185c1d02b48ed64bd946e9d87be2ca8b925b77565a3f9e20be4f33832b1864';assert sha(oldprovider)==b['new_object_SHA256'][str(oldprovider)]=='7f6ccd4b97011c07b3fedd818ddded1d8be66888866ef74f2be41a612be7fbaa'
 p=json.loads((D/'parent-build-pins-r166.json').read_bytes());native=R/'candidates/Kimi-W2-pair-r9280/libKimi_W2_pair_r9280.so'
 for f,h in p['all_linked_objects'].items():assert sha(f)==h
 for deps in b['new_MD'].values():
  for f,h in deps.items():assert sha(f)==h
 # A single host CAPI rename permits the new selector to coexist with originaldevice stubs.
 renamed=B/'parent503_provider_renamed.o';run(['objcopy','--redefine-sym','modern_kimi_grouped_I192_r8669y=modern_kimi_grouped_I192_parent503_r166',str(oldprovider),str(renamed)],'host_symbol_rename',30);sections=device_sections(oldprovider);assert sections==device_sections(renamed)
 nm=subprocess.check_output(['nm','--defined-only',str(renamed)],text=True);assert ' modern_kimi_grouped_I192_parent503_r166'in nm and ' modern_kimi_grouped_I192_r8669y\n'not in nm
 run(['c++','-std=c++17','-O2','-fno-fast-math',str(D/'CPU_quad_contract_r166.cpp'),'-o',str(B/'CPU_quad_contract')],'CPU_contract_compile',60);run([str(B/'CPU_quad_contract')],'CPU_contract',20)
 cc=p['provider'][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'quad_provider_r166.cu');cc[cc.index('-o')+1]=str(B/'quad.o');cc[cc.index('-MF')+1]=str(B/'quad.d');run(cc,'quad_compile',240)
 undefined=subprocess.check_output(['nm','-C','--undefined-only',str(B/'quad.o')],text=True);assert all(n in undefined for n in ['__device_stub__packRoutesI192R8669y','__device_stub__buildJobs4531<2>','__device_stub__groupedW13I192R8669y<2, 4, false>','__device_stub__pairW2I192R9280'])
 newnative=B/'libKimi_W2_quad4_r166.so';cc=p['link'][:]
 for old,new in [(str(R/'candidates/Kimi-I192-batch-Model-r9129/moe.o'),str(oldmoe)),(str(R/'candidates/Kimi-W2-pair-r9280/provider.o'),str(renamed))]:assert cc.count(old)==1;cc[cc.index(old)]=new
 cc.insert(cc.index('-o'),str(B/'quad.o'));cc[cc.index('-o')+1]=str(newnative);cc=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in cc];run(cc,'link',120)
 wrapper=B/'libKimi_W2_quad4_component_r166.so';cc=p['adapter'][:];cc=[str(D/'component_runner_r166.cpp')if x.endswith('/W2_pair_runner_r9280.cpp')else str(newnative)if x==str(native)else x for x in cc];cc[cc.index('-o')+1]=str(wrapper);cc[cc.index('-MF')+1]=str(B/'adapter.d');run(cc,'adapter',180)
 ldpath=str(B)+':'+p['LD_LIBRARY_PATH'];ld=subprocess.run(['ldd','-r',str(wrapper)],env=dict(os.environ,LD_LIBRARY_PATH=ldpath),capture_output=True,text=True,timeout=45);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr and 'undefined symbol:'not in ld.stdout+ld.stderr
 nm=subprocess.check_output(['nm','-D','--defined-only',str(newnative)],text=True);assert all(n in nm for n in ['modern_kimi_grouped_I192_r8669y','modern_kimi_W2_quad_counts_r166','modern_kimi_W2_event_arm_r166','modern_kimi_W2_event_read_r166','modern_kimi_W2_event_close_r166','modern_kimi_moe_phase_mark_r155','modern_kimi_moe_event_arm_r9220'])
 q.update(completed=True,native=str(newnative),native_SHA256=sha(newnative),library=str(wrapper),library_SHA256=sha(wrapper),source_manifest_SHA256=sha(D/'source-manifest-r166.json'),parent_CPU503_SHA256=sha(original),parent_native_SHA256=b['native_SHA256'],reused_moe_object_SHA256=sha(oldmoe),reused_provider_object_SHA256=sha(oldprovider),renamed_provider_SHA256=sha(renamed),device_sections_unchanged=sections,quad_object_SHA256=sha(B/'quad.o'),new_MD={f.name:dep(f)for f in B.glob('*.d')},layer_headers={'1':b['layer_headers']['1']},LD_LIBRARY_PATH=ldpath,elapsed_s=time.monotonic()-start,actual_GPU_qualified=False)
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','native_SHA256','library_SHA256','error_class']}));assert q['completed']
