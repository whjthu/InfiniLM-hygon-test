"""Actual CDLL/dladdr/maps qualification, GPU hidden and no factory/forward."""
from pathlib import Path
import ctypes,hashlib,json,os,sys,time,types,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-TP8PP4-ordinary-r410-provider-CPU-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def kfd():
 p=Path('/sys/class/kfd/kfd/proc')
 try:return dict(available=p.is_dir(),PIDs=sorted(int(x.name)for x in p.iterdir()if x.name.isdigit())if p.is_dir()else None)
 except OSError as e:return dict(available=False,error=repr(e),PIDs=None)
def main():
 assert all(os.environ.get(k)==''for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mf=D/'SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes())
 for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
 assert not B.exists();B.mkdir();started=time.monotonic();q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,Native_dlopen_attempted=False,Native_dlopen=False,Native_factory_calls=0,Native_forward_calls=0,Runtime_created=False,weights_loaded=False,compile=0,link=0,source_manifest_SHA256=sha(mf),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],KFD_before=kfd())
 try:
  # Same process dependency ordering as the real prepare; no CUDA/device query.
  import torch
  assert not torch.cuda.is_initialized();q['Torch_imported']=True;q['Torch_CUDA_initialized_before']=torch.cuda.is_initialized()
  from bootstrap import RealDependencies
  from native_binding import bind,SIGNATURES
  from lifecycle import Gate
  pins=json.loads((D/'bindings.json').read_bytes());specs=json.loads((D/'CAPI_PROVIDERS.json').read_bytes())['symbols'];assert set(specs)==set(SIGNATURES)
  for key,value in pins['required_environment'].items():assert os.environ.get(key)==value,key
  native=Path(pins['candidate_receipt']);assert sha(native)==pins['candidate_receipt_SHA256'];r=json.loads(native.read_bytes());assert r['completed']and r['passed']and r['native_SHA256']==pins['candidate_Native_SHA256']and sha(r['native'])==r['native_SHA256']
  gate=Gate(types.SimpleNamespace(deadline=time.time()+230,cancel=B/'not-created.cancel'))
  q['Native_dlopen_attempted']=True;lib,actual=bind(r['native'],gate);q['Native_dlopen']=True
  q['all20_actual_dladdr']={name:dict(path=actual[name],SHA256=sha(actual[name]),ELF_definition=specs[name]['ELF_definition'],argtypes=[str(t)for t in getattr(lib,name).argtypes],restype=str(getattr(lib,name).restype))for name in SIGNATURES}
  assert all(Path(actual[n]).resolve()==Path(specs[n]['path']).resolve()and q['all20_actual_dladdr'][n]['SHA256']==specs[n]['SHA256']for n in SIGNATURES)
  q['actual_maps']=RealDependencies.verify_maps(pins['actual_dependency_library_SHA256']);q['deadline_checks']=gate.counts
  assert set(gate.counts)=={'Native:modern_kimi_vocab_head_get_r8341','Native:modern_kimi_vocab_head_set_r8341'}
  q['Torch_CUDA_initialized_after']=torch.cuda.is_initialized();assert not q['Torch_CUDA_initialized_after']
  q['KFD_after']=kfd();assert not q['KFD_after'].get('PIDs')or os.getpid()not in q['KFD_after']['PIDs']
  q.update(completed=True,passed=True,Native_SHA256=r['native_SHA256'],CAPI_provider_count=20,Native_owned_symbols=19,Runtime_owned_symbols=1,qualification='Actual candidateCDLL+20exactdladdr+maps+SHA under same final LD/preload policy; GPUhidden/0factory/0forward/0Runtime/0weight. No Model/physical constructor or performance proof.')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  q.setdefault('KFD_after',kfd());q['elapsed_s']=time.monotonic()-started;(B/'CPU-result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in['completed','passed','actual_CPU','elapsed_s','error','Native_SHA256','CAPI_provider_count']}))
if __name__=='__main__':main()
