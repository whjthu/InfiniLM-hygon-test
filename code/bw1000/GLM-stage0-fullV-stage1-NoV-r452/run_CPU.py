from pathlib import Path
import hashlib,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/GLM-stage0-fullV-stage1-NoV-r452-CPU-v1')
def main():
 B.mkdir(exist_ok=False);q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Native_dlopen=False,GPU=0,weights=False,compile=False)
 try:
  assert all(os.environ.get(k)=='-1'for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  raw=(D/'SOURCE_MANIFEST.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==os.environ['GLM_R452_SOURCE_SHA256'];m=json.loads(raw);assert not m['GPU_GO']and not m['Model_GO']
  for n,h in m['files_SHA256'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,n
  from check_local_recording import run
  q['new_HOST_consumer']=run();q['source_manifest_SHA256']=hashlib.sha256(raw).hexdigest();q['provider_manifest_SHA256']=m['provider_manifest_SHA256'];q['hidden']={k:os.environ[k]for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']};q['passed']=True
  q['project_maps']=[s for s in Path('/proc/self/maps').read_text().splitlines()if any(n in s for n in['libGLM_','libinfinilm_runtime','_infinilm.cpython','libinfiniops'])];assert not q['project_maps']
 except BaseException as e:q.update(passed=False,error=repr(e),traceback=traceback.format_exc())
 finally:q['completed']=True;(B/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
