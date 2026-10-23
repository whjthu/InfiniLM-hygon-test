import ctypes,pathlib,json,hashlib,os,torch
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,n,False)
assert not os.getenv('INFINI_OPS_TRACE_CALLS');m=json.loads((r/'incoming/native-runtime-GPU-review-r8411.json').read_text());b=json.loads((r/'results/native-runtime-CPU-r8407.json').read_text());assert b['completed']and not b['dlopen']and sha(r/'results/native-runtime-CPU-r8407.json')==m['CPU_receipt_SHA256'];libpath=pathlib.Path(b['library']);assert sha(libpath)==m['library_SHA256'];
for n,h in b['actual_deps'].items():assert sha(pathlib.Path(n))==h,n
for n,h in b['actual_dynamic_dependency_SHA256'].items():assert sha(pathlib.Path(n))==h,n
lib=ctypes.CDLL(str(libpath));lib.native_runtime_probe_r8407.argtypes=[ctypes.c_char_p];lib.native_runtime_probe_result_r8407.restype=ctypes.c_char_p
class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int;bindings={}
for n in ['native_runtime_probe_r8407','native_runtime_probe_result_r8407']:
 info=Dl();assert dl(ctypes.cast(getattr(lib,n),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.fname.decode()).resolve();assert p==libpath;bindings[n]={'path':str(p),'SHA256':sha(p)}
maps=pathlib.Path('/proc/self/maps').read_text();actual={}
for n,h in b['actual_dynamic_dependency_SHA256'].items():assert n in maps,n;actual[n]=h
out=r/'results'/('native-runtime-GPU-r8411-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not out.exists();report={'completed':False,'actual_deps':actual,'bindings':bindings,'manifest_SHA256':sha(r/'incoming/native-runtime-GPU-review-r8411.json'),'GPUF64':'blas-policy-r271 preloaded','scope':m['scope']}
def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(out)
save();rc=lib.native_runtime_probe_r8407(json.dumps({'cases':m['cases']}).encode());report['native_returncode']=rc;report['native']=json.loads(lib.native_runtime_probe_result_r8407());report['completed']=rc==0 and report['native']['completed'];save();assert report['completed'];seen=set();affinities={}
for owner in report['native']['reports']:
 assert owner['Runtime_closed'] and owner['cases'];sets={tuple(c['actual_owner_CPU_affinity'])for c in owner['cases']};assert len(sets)==1;cpus=set(next(iter(sets)));assert len(cpus)==4 and not(seen&cpus);seen|=cpus;affinities[owner['rank']]=sorted(cpus)
 for case in owner['cases']:
  assert len(case['phases'])==8 and all(p['untimed_own_arm_oracle_max_abs']==0 for p in case['phases'])
assert len(seen)==32;report['actual_disjoint_owner_CPU_affinity']=affinities;report['all_arms_analytic_BF16_oracle_passed']=True;save();print(json.dumps({'completed':True,'path':str(out),'SHA256':sha(out)}))
