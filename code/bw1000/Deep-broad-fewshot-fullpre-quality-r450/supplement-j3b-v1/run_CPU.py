from pathlib import Path
import hashlib,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Deep450-deployment-CPU-v2');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 B.mkdir(exist_ok=False);start=time.monotonic();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,weights=False,Model=False,compile=False,encoding_repeated=False)
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  mf=H(D/'SOURCE_MANIFEST.json');assert mf==os.environ['DEEP450_DEPLOY_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert not m['GPU_GO']and not m['Model_GO']
  for name,h in m['files_SHA256'].items():assert H(D/name)==h,name
  q['source_manifest_SHA256']=mf
  from check_metadata_CPU import run
  q['actual_changed_metadata_consumer']=run()
  maps=Path('/proc/self/maps').read_text();matched=[s for s in maps.splitlines()if any(n in s for n in ['libDeep_','libinfinilm_runtime','_infinilm.cpython','libinfiniops','libinfiniccl','libinfinirt'])];assert not matched
  assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
  q.update(passed=True,project_maps=matched,hidden={k:os.environ[k]for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']})
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start);(B/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','elapsed_s','error']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
