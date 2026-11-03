"""Recompute the original CPU-only compiler/layout receipt; no device imports."""
import argparse,base64,hashlib,json
from pathlib import Path

SOURCE='6b9b53a4dc2c42842fb6e370f85d21fd85b2b6678d22fe5fa4441e12de36ab5a'
def digest(b):return hashlib.sha256(b).hexdigest()
def audit(path):
 packed=Path(path).read_bytes();q=json.loads(packed);files={}
 assert q['actual_CPU']=='910407.900'and q['new_jobs']==0
 for p,v in q['files'].items():
  b=base64.b64decode(v['bytes'],validate=True)
  assert len(b)==v['nbytes']and digest(b)==v['SHA256'];files[p]=b
 def one(suffix):
  found=[b for p,b in files.items()if p.endswith(suffix)];assert len(found)==1;return found[0]
 mf=one('/SOURCE_MANIFEST.json');assert digest(mf)==SOURCE
 local=Path(__file__).resolve().parent
 assert (local/'SOURCE_MANIFEST.json').read_bytes()==mf
 for n,h in json.loads(mf)['files_SHA256'].items():assert digest((local/n).read_bytes())==h,n
 rb=one('/result.json');r=json.loads(rb)
 assert r['passed']and r['completed']and r['actual_CPU']=='910407.900'
 assert r['source_manifest_SHA256']==SOURCE and r['project_Native_maps']==[]
 assert r['GPU']==0 and not r['Native_dlopen']and not r['Runtime_execution']and not r['Model']
 assert len(r['actual_CPU_affinity'])==4 and set(r['visible_env'].values())=={'-1'}
 names=['production-pre-MD','production-actual-Dense-and-AllGather-compile-only','pure-geometry-compile','pure-geometry-run','CPU-Torch-layout-and-reduction-reference']
 assert [c['name']for c in r['commands']]==names
 for c in r['commands']:
  assert c['rc']==0
  for ext in ['stdout','stderr']:assert digest(one('/'+c['name']+'.'+ext))==c[ext+'_SHA256']
 def deps(b):return set(b.decode().replace('\\\n',' ').split(':',1)[1].split())
 assert deps(one('/compile_contract.d'))==deps(one('/compile_contract.pre.d'))
 assert digest(one('/compile_contract.d'))==r['actual_production_compile']['MD_SHA256']
 # CPU's pre/post byte snapshots use Linux resolve() for symlinked SDK headers.
 # This offline Windows reader does not pretend raw MD spellings are realpaths.
 assert r['actual_production_compile']['dependency_SHA256']
 obj=next(v for p,v in q['artifact_metadata'].items()if p.endswith('/compile_contract.o'))
 assert obj['SHA256']==r['actual_production_compile']['object_SHA256']and not obj['downloaded']
 geometry=json.loads(one('/pure-geometry-run.stdout'));assert geometry==r['geometry']and geometry['passed']and geometry['cases']==35
 math=json.loads(one('/CPU-Torch-layout-and-reduction-reference.stdout'));assert math==r['CPU_math_reference']and math['passed']
 assert not math['CUDA_initialized']and not math['GPU_execution']and not math['performance_gain_qualified']
 assert [(c['N'],c['K'],c['local_N'])for c in math['cases']]==[(3584,7168,224),(7168,3584,448)]
 assert all(c['wrong_order_rejected']and c['F32_reconstruction_NRMSE']<2e-6 for c in math['cases'])
 assert any(s.startswith('910407.900|')and '|COMPLETED|0:0|'in s for s in q['account'])
 return {'offline_audit_passed':True,'actual_CPU':'910407.900','account':q['account'],'source_manifest_SHA256':SOURCE,'archive_SHA256':digest(packed),'archive_bytes':len(packed),'archive_files':len(files),'receipt_SHA256':digest(rb),'HOST_TUs':2,'production_Dense_AllGather_compile_only':True,'production_dependency_files':len(r['actual_production_compile']['dependency_SHA256']),'geometry':geometry,'CPU_math_reference':math,'CPU_worker_wall_s':r['wall_s'],'GPU':0,'SDK_Runtime_executions':0,'Native_dlopen':False,'actual_collective':False,'Model':False,'performance_qualified':False,'limits':'Actual production inline overload compile and CPU synthetic BF16/F32 row reconstruction only; no device GEMM rounding/AllGather/TP identity/cost or fullMoE claim.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');a=p.parse_args();v=audit(a.archive);s=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(s,encoding='utf8')
 print(s)
