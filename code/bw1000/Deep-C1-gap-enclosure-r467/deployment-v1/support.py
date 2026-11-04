"""Pure source/receipt gates; pending CPU never authorizes Slurm/Native."""
from pathlib import Path
import hashlib,json,os,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=Path(__file__).resolve().parent;PACKAGE=D.parent/'profile-source-v1'
RUN='current-a314-profile-a';LABEL='Deep-r467-current-a314-profile';NODE='a02r4n04'
ENTRY_MF='27892f2a9c62bada471942fcdcd53b0be7e3843f53effaa434be165add1855e8'
OUTPUT=R/'results/Deep-r467-current-profile'/RUN
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(path):
 p=Path(path);return R/p.relative_to('/refactor')if p==Path('/refactor')or p.is_relative_to('/refactor')else p
def publish(path,q):
 with Path(path).open('x')as f:json.dump(q,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and(not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','DEEP_','INFINILM_DEEP_'))or k in['SLURM_CONF','SLURM_CONF_SERVER'])}
def tree(root,digest):
 root=Path(root);assert sha(root/'SOURCE_MANIFEST.json')==digest;m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():p=root/n;assert p.resolve().is_relative_to(root.resolve())and sha(p)==h,n
 return m
def validate_config(c):
 assert c['run']==RUN and c['node']==NODE and c['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 assert(c['TP'],c['PP'],c['rows'],c['context'],c['HC_mode'],c['fullpre'],c['r366'],c['precision_policy'])==(8,1,1,32768,0,0,0,0)
 assert(c['GPU_seconds'],c['Native_seconds'],c['service_seconds'],c['keeper_seconds'])==(2700,2580,2400,3000)
 assert c['HTTP']==0 and c['requests']==5 and c['observer_schedule']==[False,False,False,True,False]and c['samples']==[8,87]
 assert c['gpu_start_epoch']is None and c['actual_step']is None and not c['execute_authorized']
 assert c['output_root']=='/refactor/results/Deep-r467-current-profile/'+RUN
 return c

def verify(digest,*,require_ready=False,directory=None):
 d=D if directory is None else Path(directory);binding=None
 if sha(d/'SOURCE_MANIFEST.json')==digest:m=tree(d,digest)
 else:
  assert sha(d/'GPU_MANIFEST.json')==digest;o=json.loads((d/'GPU_MANIFEST.json').read_bytes());m=tree(d,o['deployment_source_manifest_SHA256']);assert o['entry_source_manifest_SHA256']==ENTRY_MF
  assert o['execution_READY']and o['GPU_GO'];binding=o['entry_CPU'];assert re.fullmatch('[0-9a-f]{64}',binding['SHA256'])and re.fullmatch(r'910407[.]\d+',binding['actual_CPU'])
  p=actual(binding['path']);assert sha(p)==binding['SHA256'];q=json.loads(p.read_bytes());assert q['completed']and q['passed']and q['source_manifest_SHA256']==ENTRY_MF and q['actual_CPU']==binding['actual_CPU']
  assert q['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
  assert q['actual_ELF_CPU750_50objects_SDK_profile_sources_verified']and q['actual_changed_entry_consumer']['passed']and q['actual_changed_entry_consumer']['actual_entry_main_execute']
  assert not any(q[k]for k in['Native_dlopen','SDK_Runtime','weights_opened','GPU','Model_READY'])
 assert m['entry_source_manifest_SHA256']==ENTRY_MF;tree(d.parent/'profile-source-v1',ENTRY_MF);validate_config(json.loads((d/'config.json').read_bytes()))
 pending=json.loads((d/'ENTRY_CPU_BINDING.json').read_bytes());assert pending['SHA256']is None and pending['actual_CPU']is None and not pending['passed']
 if require_ready:assert binding is not None,'Real published CPU receipt and new GPUmanifest required; source-only never READY'
 return dict(m,execution_READY=binding is not None,GPU_GO=binding is not None,entry_CPU=binding)
