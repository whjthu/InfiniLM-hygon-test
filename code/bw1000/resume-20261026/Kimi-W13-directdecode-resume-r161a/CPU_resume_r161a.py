"""CPU510 resume: reuse passed MoE/contracts, replace one provider jobstub only."""
from pathlib import Path
import json,hashlib,os,subprocess,shlex,time,argparse,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;OLD=R/'candidates/Kimi-MoE-phase-component-r161';B=R/'candidates/Kimi-MoE-phase-component-r161a';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--moe-sha',required=True);p.add_argument('--moe-md-sha',required=True);a=p.parse_args();assert len(a.moe_sha)==64 and len(a.moe_md_sha)==64
assert sha(OLD/'CPU-build.json')=='09263fb2d5c5f6930f184e423dc0631bfc98ea82496dc8dba65af4916fc4b2e0';failed=json.loads((OLD/'CPU-build.json').read_bytes());assert not failed['completed']and failed['actual_step']=='910407.510'and len(failed['commands'])==6
assert sha(OLD/'moe.o')==a.moe_sha and sha(OLD/'moe.d')==a.moe_md_sha
B.mkdir(exist_ok=False);start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'resumed_CPU510_failed_receipt_SHA256':sha(OLD/'CPU-build.json'),'reused_passed_moe_object_SHA256':a.moe_sha,'reused_passed_moe_MD_SHA256':a.moe_md_sha,'reused_CPU_contracts_in_first4commands':True,'recompiled_HOST_MoE_TUs':0,'new_HIP_TUs':1,'pending_HOST_adapter_TUs':1}
def run(cmd,label,seconds):
 q['commands'].append(cmd);z=subprocess.run(cmd,cwd=B,text=True,capture_output=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-4000:])
try:
 manifest=json.loads((D/'source-manifest-r161.json').read_bytes())
 for n,h in manifest['files_SHA256'].items():assert sha(D/n)==h
 p=json.loads((D/'parent-build-pins-r161.json').read_bytes());native=R/'candidates/Kimi-W2-pair-r9280/libKimi_W2_pair_r9280.so';assert sha(native)=='3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'
 packed=R/'candidates/kimi-packed-reuse-r6136/packed.o';assert sha(packed)=='355607f22e0232b6422cef6f81224af24f62b585e5e251fbb656e7eb0da4700c';assert str(packed)in p['link']
 nm=subprocess.check_output(['nm','-C','--defined-only',str(packed)],text=True);assert '__device_stub__buildJobs4531<2>'in nm
 for path,h in p['all_linked_objects'].items():assert sha(path)==h
 # Verify the reused HOST's complete generated include/source closure before borrowing it.
 def dependencies(path):
  return {f:sha(f)for f in shlex.split(Path(path).read_text().replace(chr(92)+chr(10),' ').split(':',1)[1])if Path(f).is_file()}
 olddeps=dependencies(OLD/'moe.d');critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert olddeps[critical]==p['critical_MoE_header_SHA256'];q['reused_moe_current_MD']=olddeps
 for path,h in olddeps.items():
  if path.startswith('/refactor/incoming/Kimi-W13-directdecode-component-r161/'):
   same=D/Path(path).name;assert same.is_file()and sha(same)==h,('reused_HOST_source_changed',Path(path).name)
 cc=failed['commands'][-1][:];assert '-save-temps'in cc or '--keep'in cc
 cc=[('-I'+str(D))if x=='-I/refactor/incoming/Kimi-W13-directdecode-component-r161'else x for x in cc];cc[cc.index('-c')+1]=str(D/'kimi_provider_phase_r161.cu');cc[cc.index('-o')+1]=str(B/'provider.o');cc[cc.index('-MF')+1]=str(B/'provider.d');run(cc,'provider_resume',240)
 undefined=subprocess.check_output(['nm','-C','--undefined-only',str(B/'provider.o')],text=True);assert '__device_stub__buildJobs4531<2>'in undefined and 'buildJobsI192R8669y'not in undefined
 newnative=B/'libKimi_MoE_phase_r161a.so';cc=p['link'][:]
 for old,new in [(str(R/'candidates/Kimi-I192-batch-Model-r9129/moe.o'),str(OLD/'moe.o')),(str(R/'candidates/Kimi-W2-pair-r9280/provider.o'),str(B/'provider.o'))]:assert cc.count(old)==1;cc[cc.index(old)]=new
 cc[cc.index('-o')+1]=str(newnative);cc=[('-Wl,-Map,'+str(B/'link.map'))if x.startswith('-Wl,-Map')else x for x in cc];run(cc,'link',120)
 defined=subprocess.check_output(['nm','-D','-C','--defined-only',str(newnative)],text=True);assert '__device_stub__buildJobs4531<2>'in defined
 wrapper=B/'libKimi_MoE_component_r161a.so';cc=p['adapter'][:];cc.insert(1,'-I'+str(D));cc=[str(D/'component_runner_r161.cpp')if x.endswith('/W2_pair_runner_r9280.cpp')else str(newnative)if x==str(native)else x for x in cc];cc[cc.index('-o')+1]=str(wrapper);cc[cc.index('-MF')+1]=str(B/'adapter.d');run(cc,'pending_adapter',180)
 deps={path.name:dependencies(path)for path in B.glob('*.d')};deps['reused_moe.d']=olddeps;assert deps['adapter.d'][critical]==p['critical_MoE_header_SHA256']
 ldpath=str(B)+':'+p['LD_LIBRARY_PATH'];ld=subprocess.run(['ldd','-r',str(wrapper)],env=dict(os.environ,LD_LIBRARY_PATH=ldpath),text=True,capture_output=True,timeout=45);assert ld.returncode==0 and 'not found'not in ld.stdout+ld.stderr and 'undefined symbol:'not in ld.stdout+ld.stderr
 # Reuse already-qualified metadata headers rather than reread payload or model.
 original=R/'candidates/Kimi-MoE-phase-component-r155/CPU-build.json';assert sha(original)=='c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd';headers=json.loads(original.read_bytes())['layer_headers']
 retained=[p for p in B.rglob('*')if p.is_file()and p.suffix in ['.s','.ll']]
 q.update(completed=True,native=str(newnative),native_SHA256=sha(newnative),library=str(wrapper),library_SHA256=sha(wrapper),parent_native_SHA256=sha(native),unchanged_object_SHA256={x:sha(x)for x in p['link']if x.endswith('.o')and not x.endswith('/Kimi-I192-batch-Model-r9129/moe.o')and not x.endswith('/Kimi-W2-pair-r9280/provider.o')},new_object_SHA256={str(OLD/'moe.o'):sha(OLD/'moe.o'),str(B/'provider.o'):sha(B/'provider.o')},new_MD=deps,layer_headers=headers,LD_LIBRARY_PATH=ldpath,source_manifest_SHA256=sha(D/'source-manifest-r161.json'),elapsed_s=time.monotonic()-start,math_kernel_source_unchanged=False,new_GPU_TU_scope='r161 directFP4 math unchanged; only reference existingpacked buildJobs4531 instead of recompile current equivalent jobbuilder',actual_GPU_qualified=False,retained_ISA=[{'path':str(p),'bytes':p.stat().st_size,'SHA256':sha(p)}for p in retained],reused_job_stub_object_SHA256=sha(packed),original_failed_receipt_preserved=sha(OLD/'CPU-build.json')=='09263fb2d5c5f6930f184e423dc0631bfc98ea82496dc8dba65af4916fc4b2e0')
except BaseException as e:q.update(error_class=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','native_SHA256','library_SHA256','error_class']}));assert q['completed']
