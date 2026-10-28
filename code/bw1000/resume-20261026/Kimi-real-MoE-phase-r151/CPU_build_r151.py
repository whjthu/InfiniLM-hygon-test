"""No GPU import/dlopen: replace exactly current MoE/provider objects."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,time,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;B=R/'candidates/Kimi-real-MoE-phase-r151';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]}
def run(args,label,seconds):
 q['commands'].append(args);z=subprocess.run(args,capture_output=True,text=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-6000:])
try:
 source=json.loads((D/'source-manifest-r151.json').read_bytes())
 for rel,h in source['files_SHA256'].items():assert sha(D/rel)==h
 p=json.loads((D/'parent-build-pins-r151.json').read_bytes());native=R/'candidates/Kimi-W2-pair-r9280/libKimi_W2_pair_r9280.so'
 assert sha(native)=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
 for path,h in p['all_linked_objects'].items():assert sha(path)==h
 assert p['all_linked_objects']['/refactor/candidates/Kimi-Model-events-r8669zq1/kimi_decoder_model_events_r8669zq.o']=='ae34f72d2c5751a048417c9ef9320efdb4f53bae9c6ea918c06ae1c943ee6600'
 original_events=R/'candidates/Kimi-Model-events-r8669zq1/src';event_pins={'kimi_model_events_r8669zq.hpp':'2a36c755719cb723530795d57c14b2b5311a54cd0e6f737de532bee483cfffc2','kimi_model_events_impl_r8669zq.hpp':'da687e2628bbc2463c19a9976e83fa79235359475c7d809848ec90135a9faf3a','kimi_decoder_model_events_r8669zq.cpp':'26119225b7efb6128eaa31b17afe96cc4081f8aa86bf50465341b6fecfabd640','kimi_rank_worker_model_events_r8669zq.cpp':'2332ee0b217f8ba4d7ccec901c0c138f2ada384ff62156972846ee9913b8cd43'}
 for name,h in event_pins.items():assert sha(original_events/name)==h
 decoded=(original_events/'kimi_decoder_model_events_r8669zq.cpp').read_text();assert decoded.count('kimi_model_events_r8669zq::mark(')==4
 assert sha(D/'kimi_model_events_r8669zq.hpp')==event_pins['kimi_model_events_r8669zq.hpp']
 nm=subprocess.check_output(['nm','-D','-C','--defined-only',str(native)],text=True);assert 'kimi_model_events_r8669zq::mark('in nm
 base=json.loads((R/'candidates/Kimi-MoE-batch-profile-r9125/CPU-build.json').read_bytes())
 for deps in [base['dependencies']['moe.d'],p['all_new_MD']['provider.d'],p['all_new_MD']['adapter.d']]:
  for path,h in deps.items():
   if Path(path).suffix in ['.h','.hpp','.cuh']:assert sha(path)==h,('original header MD changed',path)
 run(['c++','-std=c++17','-O0','-Wall','-Wextra','-pedantic',str(D/'CPU_context_contract_r151.cpp'),'-o',str(B/'CPU_contract')],'CPU_contract_compile',60);run([str(B/'CPU_contract')],'CPU_contract_run',20)
 m=json.loads((R/'candidates/Kimi-I192-batch-Model-r9129/CPU-build.json').read_bytes());assert m['moe_object_SHA256']=='55fb31b5f4fa1510f06e8b62894d64d52926862371f43e1e2d8c5aaeb2624800'
 cc=m['commands'][0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_moe_real_phase_r151.cpp');cc[cc.index('-o')+1]=str(B/'moe.o');cc[cc.index('-MF')+1]=str(B/'moe.d');run(cc,'moe',180)
 cc=p['commands'][0][:];cc.insert(1,'-I'+str(D));cc[cc.index('-c')+1]=str(D/'kimi_provider_real_phase_r151.cu');cc[cc.index('-o')+1]=str(B/'provider.o');cc[cc.index('-MF')+1]=str(B/'provider.d');run(cc,'provider',240)
 newnative=B/'libKimi_real_MoE_phase_r151.so';link=p['commands'][1][:];replaced={str(R/'candidates/Kimi-I192-batch-Model-r9129/moe.o'):str(B/'moe.o'),str(R/'candidates/Kimi-W2-pair-r9280/provider.o'):str(B/'provider.o')}
 for old,new in replaced.items():assert link.count(old)==1;link[link.index(old)]=new
 link[link.index('-o')+1]=str(newnative);link=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in link];run(link,'link',120)
 md={}
 for f in B.glob('*.d'):md[f.name]={s:sha(s)for s in shlex.split(f.read_text().replace('\\\n',' ').split(':',1)[1])if Path(s).is_file()}
 critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert md['moe.d'][critical]==p['critical_MoE_header_SHA256']
 olddefs={x.split()[-1]for x in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()if len(x.split())>=3};newdefs={x.split()[-1]for x in subprocess.check_output(['nm','-D','--defined-only',str(newnative)],text=True).splitlines()if len(x.split())>=3};assert olddefs<=newdefs and 'modern_kimi_moe_phase_mark_r151'in newdefs
 z=subprocess.run(['ldd','-r',str(newnative)],env=dict(os.environ,LD_LIBRARY_PATH=p['LD_LIBRARY_PATH']),text=True,capture_output=True,timeout=45);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr
 q.update(HOST_TUs_compiled=1,HIP_TUs_compiled=1,CPU_contract_TUs_compiled=1,device_function_source_unchanged=True,kernel_ISA_identity_claim=False,completed=True,native=str(newnative),native_SHA256=sha(newnative),parent_native_SHA256=sha(native),all_linked_object_SHA256={x:sha(x)for x in link if x.endswith('.o')},unchanged_objects={x:h for x,h in p['all_linked_objects'].items()if x not in replaced},replaced_objects=replaced,new_MD=md,event_source_SHA256=event_pins,existing_model_event_object_reused=True,source_manifest_SHA256=sha(D/'source-manifest-r151.json'),elapsed_s=time.monotonic()-start,LD_LIBRARY_PATH=p['LD_LIBRARY_PATH'],GPU_qualified=False,histogram='DEFERRED',new_sync=False,new_D2H=False)
except BaseException as error:q.update(error_class=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
(B/'CPU-build.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','actual_step','CPU_only','GPU','elapsed_s','native_SHA256','error_class']}));assert q['completed']
