"""Exact GPU-library environment resolution only; no Native calls/compile/weights."""
from pathlib import Path
import os,json,hashlib,subprocess,time,re,sys,traceback
D=Path(__file__).resolve().parent;O=Path('/refactor/candidates/Kimi-splitV128-loader-closure-CPU-v2');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists(),'Original candidate exists, reconcile only';O.mkdir();start=time.monotonic();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Native_calls=0,explicit_SDK_calls=0,production_compile=0,probe_compile=0,weights=False,GPU_job=False,preload_child_initialization='UNKNOWN; ldd child receives exact GPU preload, no independent constructor trace',commands=[])
 def save():(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(argv,name,env):
  left=140-(time.monotonic()-start);assert left>2;q['commands'].append(argv);save();z=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=left-1);(O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr);q[name]=dict(rc=z.returncode,stdout_SHA256=H(O/(name+'.stdout')),stderr_SHA256=H(O/(name+'.stderr')),stdout_bytes=len(z.stdout.encode()),stderr_bytes=len(z.stderr.encode()));save();assert z.returncode==0,(name,z.stderr[-1500:]);return z.stdout+z.stderr
 try:
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['KIMI_LOADER_CLAIM_SHA'];assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items());assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert not os.environ.get('LD_PRELOAD')
  parent=json.loads((D/'PARENT.json').read_bytes());receipt=Path(parent['CPU1080_receipt']);assert H(receipt)==parent['CPU1080_receipt_SHA256'];c=json.loads(receipt.read_bytes());assert c['passed']and c['actual_CPU']=='910407.1080';assert H(c['probe'])==c['probe_SHA256']==parent['probe_SHA256'];assert H(c['parent_Native']['path'])==c['parent_Native']['SHA256']==parent['Native_SHA256']
  for p in c['actual_dependency_library_SHA256'].values():assert H(p['path'])==p['SHA256']
  e=os.environ.copy();e['LD_PRELOAD']=os.environ['KIMI_LOADER_EXPECTED_PRELOAD'];assert e['LD_PRELOAD']=='/refactor/candidates/blas-policy-r271/libblas_policy_r271.so';q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'),ENV_SHA256=H(D/'ENV.sh'),LD_LIBRARY_PATH=e['LD_LIBRARY_PATH'],LD_PRELOAD=e['LD_PRELOAD'],preload_SHA256=H(e['LD_PRELOAD']),parent_receipt_SHA256=H(receipt),probe=c['probe'],probe_SHA256=c['probe_SHA256'],Native=c['parent_Native']);save()
  text=run(['/usr/bin/ldd','-r',c['probe']],'ldd_full_streams',e);assert 'not found'not in text and 'undefined symbol:'not in text,text[-2500:];paths=sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M)));assert paths
  libs={p:H(p)for p in paths};q['all_transitive_resolved_library_SHA256']=libs
  required=[c['parent_Native'],*c['actual_dependency_library_SHA256'].values()]
  for a in required:assert any(Path(p).resolve()==Path(a['path']).resolve()and h==a['SHA256']for p,h in libs.items()),('Resolved identity differs',a['path'])
  helper='/refactor/candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so';assert helper in libs;q['grouped_prefill']={'path':helper,'SHA256':libs[helper]};q['parent_process_Runtime_maps']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(x in line for x in ['libKimi','libinfinilm_runtime','libinfiniops','libcudart'])];assert not q['parent_process_Runtime_maps']and 'torch'not in sys.modules;q['passed']=True
 except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
