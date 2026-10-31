"""Actual CPU-container path checks; stop original Deep prefix at first Torch import."""
import pathlib,json,hashlib,sys,os,time,datetime,argparse
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent
HOST='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('phase',choices=['control','candidate']);a=p.parse_args()
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['PATH295_MANIFEST_SHA']
assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
assert not os.environ.get('LD_PRELOAD') and not os.environ.get('APPTAINERENV_LD_PRELOAD')
assert all(not os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'])
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'phase':a.phase,'container_cwd':str(pathlib.Path.cwd()),'CPU_only':True,'source_manifest_SHA256':os.environ['PATH295_MANIFEST_SHA'],'GPU':0,'Model':0,'HTTP':0,'Native_dlopen':0,'weights_opened':0,'LD_PRELOAD_cleared':True,'full_GPU_entry_executed':False,'path_cases':[]}
# Initialize only stdlib ctypes' Python self handle before blocking all later dlopen.
import ctypes
q['stdlib_ctypes_self_handle_before_audit']=True
q['container_HOME']=os.environ.get('HOME');q['related_mounts']=[]
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
 parts=line.split();point=parts[4];root=parts[3]
 if point in ['/old','/refactor','/root'] or HOST in point or HOST in root:q['related_mounts'].append({'root':root,'mountpoint':point,'filesystem':line.split(' - ',1)[1].split()[0]})
class StopBeforeTorch(ImportError):pass
blocked=[]
def audit(event,args):
 if event=='ctypes.dlopen':raise RuntimeError('CPU audit forbids ctypes.dlopen')
 if event=='import':
  root=args[0].split('.')[0]
  if root in {'torch','infinilm','_infinilm','transformers','tokenizers'}:
   blocked.append(args[0])
   if root=='torch':raise StopBeforeTorch('Expected original Deep first Torch import boundary')
   raise RuntimeError('Forbidden CPU import: '+args[0])
 if event=='open' and isinstance(args[0],str) and (args[0].startswith('/shared/') or args[0].startswith('/old/models/')):raise RuntimeError('CPU audit forbids weight/model payload open')
sys.addaudithook(audit)
cases=[(HOST+'/candidates/upstream-b7b370-r2284/nlohmann-json-5ed07097faa6/include/nlohmann/detail/conversions/from_json.hpp','7bb417c7b3b7fe1731e999c63fcd8ae4ab7bed5047ffeafb7462d3875d7d7a6e'),('/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/third_party/json/include/nlohmann/detail/meta/identity_tag.hpp','b43dc21a675c4cdf7cec46c3925a9deceb3d56070bb9c72b0c2a1ba2e53a24ea')]
for name,pin in cases:
 try:actual=sha(name)
 except FileNotFoundError:
  assert a.phase=='control';q['path_cases'].append({'path':name,'expected_namespace_failure':True});continue
 if a.phase!='candidate' or actual!=pin:
  q['path_cases'].append({'path':name,'actual_SHA256':actual,'unexpected_access_or_hash':True});q.update(completed=False,passed=False)
  with(R/'candidates/container-host-alias-r295'/('CPU-'+a.phase+'.json')).open('x')as f:json.dump(q,f,indent=2)
  raise AssertionError('Control path unexpectedly accessible or candidate SHA mismatch')
 q['path_cases'].append({'path':name,'SHA256':actual})
if a.phase=='candidate':
 glm=R/'incoming/GLM-latest-full-owner-r290';deep=R/'incoming/Deep-latest-full-owner-r287'
 for d,pin in [(glm,'903053b1f6d8031bc03c698c7bd9498a8cab8b0802ff19f4c88a7b4d698e7674'),(deep,'12a4160d5703a9364929c41d5b9b69772a600254f2084ec2774d585023b17df1')]:
  assert sha(d/'SOURCE_MANIFEST.json')==pin;source=json.loads((d/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(d/n)==h for n,h in source['files_SHA256'].items())
 sys.path.insert(0,str(glm))
 from native_chain_r186 import verify
 q['GLM_actual_static_chain']=verify()
 sys.path[0]=str(deep)
 import native_binding_r206b
 original=native_binding_r206b.binding
 def observed_binding():
  b,c,n=original();q['Deep_actual_binding']={'CPU_receipt_SHA256':b['CPU_receipt_SHA256'],'Native_SHA256':c['library_SHA256'],'objects':len(n['link']['full']['objects']),'bindings':len(c['actual_DTK_bindings'])};return b,c,n
 native_binding_r206b.binding=observed_binding
 from preweight_r206b import run
 cpu_path=R/'candidates/Deep-agent-regular32K-r206b/CPU-preflight.json';cpu=json.loads(cpu_path.read_bytes());counts={k:len(cpu[k])for k in ['baseline_files_SHA256','overlay_files_SHA256']};assert list(counts.values())==[68,9];q['Deep_CPU_preflight_counts_actual']=counts;q['Deep_CPU_preflight_SHA256']=sha(cpu_path)
 os.environ['DEEP_SOURCE_MANIFEST_SHA']='12a4160d5703a9364929c41d5b9b69772a600254f2084ec2774d585023b17df1';os.environ['DEEP_R277_GPUSTART_UNIX']=str(time.time())
 try:run()
 except StopBeforeTorch:q['Deep_original_preweight_reached_first_Torch_import']=True
 else:raise AssertionError('Original Deep preweight did not stop at forbidden GPU boundary')
 assert blocked==['torch'] and 'Deep_actual_binding'in q
 q['Deep_CPU_baseline68_overlay9_prefix_passed']=True
assert not any(v.split('.')[0]in {'torch','infinilm','_infinilm'}for v in sys.modules)
q.update(completed=True,passed=True,expected_blocked_imports=blocked,qualification_scope='Two actual static dependency chains plus original Deep pre-first-Torch checks only; no GLM full prefix/preengine/GPU ABI/Model qualification')
out=R/'candidates/container-host-alias-r295'/('CPU-'+a.phase+'.json')
with out.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
