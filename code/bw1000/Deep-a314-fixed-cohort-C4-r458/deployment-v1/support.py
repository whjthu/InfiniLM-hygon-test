"""One immutable C4 residency; CPU-only source and evidence verification."""
from pathlib import Path
import hashlib,json,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-a314-fixed-cohort-C4-r458/deployment-v1'
PACKAGE=R/'incoming/Deep-a314-fixed-cohort-C4-r458'
RUN='fixed-c4-20261104-a';LABEL='Deep-r458-fixed-C4';NODE='a02r4n05'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(path):
 p=Path(path)
 return R/p.relative_to('/refactor')if p==Path('/refactor')or p.is_relative_to('/refactor')else p
def publish(path,value):
 with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in ['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and(not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','DEEP_','INFINILM_DEEP_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER'])}
def verify_tree(root,digest):
 root=actual(root);assert sha(root/'SOURCE_MANIFEST.json')==digest
 m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():
  p=root/name;assert p.resolve().is_relative_to(root.resolve())and sha(p)==h,name
 return m
def verify(digest,*,require_ready=False):
 if sha(D/'SOURCE_MANIFEST.json')==digest:
  m=verify_tree(D,digest)
 else:
  assert sha(D/'GPU_MANIFEST.json')==digest
  overlay=json.loads((D/'GPU_MANIFEST.json').read_bytes());m=verify_tree(D,overlay['source_manifest_SHA256'])
  assert overlay['entry_manifest_SHA256']==m['entry_manifest_SHA256']
  cpu=actual(overlay['glue_CPU_receipt']);assert sha(cpu)==overlay['glue_CPU_receipt_SHA256']
  proof=json.loads(cpu.read_bytes());assert proof['passed']and proof['completed']and proof['source_manifest_SHA256']==overlay['source_manifest_SHA256']
  assert proof['actual_changed_session']and not proof['Native_dlopen']and not proof['GPU']
  m=dict(m,**{k:overlay[k]for k in ['execution_READY','GPU_GO','performance_qualification']})
 verify_tree(PACKAGE,m['entry_manifest_SHA256'])
 assert m['entry_manifest_SHA256']=='4c782ade4e6f28c71e210de3344bf1630533dfdb39481b12a144464dd9c4d474'
 receipt=actual('/refactor/candidates/Deep-a314-fixed-cohort-C4-r458-CPU-v1/CPU-result.json')
 assert sha(receipt)=='937199ee08d4a94f13286b3adf851d4e70bc1ff5dbd5775706c9dcbd41f3d047'
 q=json.loads(receipt.read_bytes());assert q['passed']and q['completed']and q['actual_CPU']=='910407.967'
 assert q['actual_four_CPU945_inputs']and q['source_manifest_SHA256']==m['entry_manifest_SHA256']
 assert not any(q[k]for k in ['Native_dlopen','Runtime_execution','SDK_execution','GPU','Model_READY'])
 c=json.loads((D/'config.json').read_bytes())
 assert c['run']==RUN and c['node']==NODE and c['Native_seconds']==5280 and c['CPUkeeper_seconds']==5700 and c['GPU_seconds']==5400
 assert c['rows']==4 and c['capacity_per_row']==32768 and c['cohort_modes']==['serial','cohort','serial','cohort','cohort','serial']
 assert c['HC_mode']==c['r366']==c['fullpre']==0 and c['HTTP']==0 and c['logical_requests']==24
 assert c['gpu_start_epoch']is None and c['actual_step']is None
 if require_ready:assert m['GPU_GO']and m['execution_READY']and not m['performance_qualification']
 return m
