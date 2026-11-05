"""Pure source/receipt gates; pending CPU never authorizes Slurm/Native."""
from pathlib import Path
import hashlib,json,os,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=Path(__file__).resolve().parent;PACKAGE=D.parent/'hc-pre-entry-v1'
RUN='hc-pre-fivephase-20261105-a';LABEL='Deep-r467-current7f-HCpre';NODE='a02r4n04'
ENTRY_MF='413dff01e160851b494c4f1ce2b4314e982b1909a00bf0d11faf5aecc0d8b5e4'
OUTPUT=R/'results/Deep-r467-HCpre-phase'/RUN
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
 assert c['run']==RUN and c['node']==NODE and c['Native_SHA256']=='5fd76082aa15f039515728c16bcb320b3b1c245952843e8f93f2038d17c5e0a6'
 assert(c['TP'],c['PP'],c['rows'],c['context'],c['HC_mode'],c['fullpre'],c['r366'],c['precision_policy'])==(8,1,1,32768,0,0,0,0)
 assert(c['GPU_seconds'],c['Native_seconds'],c['service_seconds'],c['keeper_seconds'])==(2700,2580,2400,3000)
 assert c['HTTP']==0 and c['requests']==5 and c['observer_schedule']==[False,False,False,True,False]and c['samples']==[8,87]
 assert c['gpu_start_epoch']is None and c['actual_step']is None and not c['execute_authorized']
 assert c['output_root']=='/refactor/results/Deep-r467-HCpre-phase/'+RUN
 return c

def verify(digest,*,require_ready=False,directory=None):
 d=D if directory is None else Path(directory);binding=None
 if sha(d/'SOURCE_MANIFEST.json')==digest:m=tree(d,digest)
 else:
  assert sha(d/'GPU_MANIFEST.json')==digest;o=json.loads((d/'GPU_MANIFEST.json').read_bytes());m=tree(d,o['deployment_source_manifest_SHA256']);assert o['entry_source_manifest_SHA256']==ENTRY_MF
  assert o['execution_READY']and o['GPU_GO'];binding=o['entry_CPU'];assert re.fullmatch('[0-9a-f]{64}',binding['SHA256'])and re.fullmatch(r'910407[.]\d+',binding['actual_CPU'])
  p=actual(binding['path']);assert sha(p)==binding['SHA256'];q=json.loads(p.read_bytes());assert q['completed']and q['passed']and q['source_manifest_SHA256']=='b368d0c33db11b65a7bc72d60c0ed38cf397d748dcd602758d34e5b62f7efff9' and q['actual_CPU']==binding['actual_CPU']
  assert q['library_SHA256']=='5fd76082aa15f039515728c16bcb320b3b1c245952843e8f93f2038d17c5e0a6'
  assert q['new_HOST']==1 and q['retained_count']==49 and q['new_GPU']==0 and q['fresh_M_equals_MD']and q['old_strong_exports_retained']and q['SDK_unchanged']and q['no_new_sync']and q['strip_equals_original']
  assert not any(q[k]for k in['Native_dlopen','Runtime','weights','GPU','Model'])
 assert m['entry_source_manifest_SHA256']==ENTRY_MF;tree(d.parent/'hc-pre-entry-v1',ENTRY_MF);validate_config(json.loads((d/'config.json').read_bytes()))
 pending=json.loads((d/'ENTRY_CPU_BINDING.json').read_bytes());assert pending['SHA256']=='97702f1120ebf6fdd7d56c01d58dfc1be4ab03b7b3177ea3c496b33fff3b5a00'and pending['actual_CPU']=='910407.1033'and pending['passed']
 if require_ready:assert binding is not None,'Real published CPU receipt and new GPUmanifest required; source-only never READY'
 return dict(m,execution_READY=binding is not None,GPU_GO=binding is not None,entry_CPU=binding)
