"""Inherited1378 keeper with exact new entry and completed CPU handoff."""
from pathlib import Path
import hashlib,json,os,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-layer2-3-boundary-diagnostic-r447/controller-v1'
RUN='boundary-20261104-a'
LABEL='Deep-r447-layer2-3-boundary'
NODE='a02r4n01'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def publish(path,value):
 with Path(path).open('x')as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
def clean_environment(base):
 return {k:v for k,v in base.items()if k not in ['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and(not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','DEEP_','INFINILM_DEEP_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER'])}
def host(path):
 p=Path(path)
 return R/p.relative_to('/refactor')if p==Path('/refactor')or p.is_relative_to('/refactor')else p
def verify_tree(root,expected):
 root=Path(root);root=root.parent if root.name=='SOURCE_MANIFEST.json'else root;root=host(root).resolve()
 assert sha(root/'SOURCE_MANIFEST.json')==expected
 m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():
  p=root/name;assert p.resolve().is_relative_to(root)and sha(p)==h,name
 return m
def verify(expected,*,require_ready=False):
 m=verify_tree(D,expected)
 if not require_ready:return m
 assert m['execution_READY']and m['GPU_GO']and not m['Model_GO']and not m['performance_qualification']and not m['quality_qualification']
 cfg=json.loads((D/'config.json').read_bytes());pin=json.loads((D/'binding-pin.json').read_bytes());cpu=json.loads((D/'ENTRY_CPU.json').read_bytes())
 assert cpu['passed']and cpu['completed']and cpu['actual_changed_entry_CLI']and cpu['new_CABI']==7
 assert not cpu['Native_dlopen']and not cpu['GPU']and not cpu['actual_input_encode_replayed']
 assert sha(host(cpu['original_receipt']))==cpu['original_receipt_SHA256']
 actual=json.loads(host(cpu['original_receipt']).read_bytes())
 assert actual['completed']and actual['passed']and actual['actual_CPU']==cpu['actual_CPU']
 assert actual['source_manifest_SHA256']==m['entry_manifest_SHA256']and actual['binding_pin_SHA256']==sha(D/'binding-pin.json')
 assert actual['actual_CLI']['returncode']==0 and actual['actual_typed_entry']['new_CABI']==7 and actual['actual_typed_entry']['high64_CPUbyte_pointers']
 assert actual['reused_Native_derivation_CPU924']['original_overall_passed']is False
 assert cpu['CPU913_actual_input_SHA256']=='ed9728d88c06f1ef4ce7fac887c706c9caf4e4265ee3e1b713b4ef87c86dbd1c'
 previous=host('/refactor/candidates/Deep438-diagnostic-entry-CPU-v3/result.json');assert sha(previous)==cpu['CPU913_actual_input_SHA256']
 original=json.loads(previous.read_bytes());assert original['passed']and original['actual_input_processor']and original['actual_input']['prompt_count']==8489
 assert original['actual_input']['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'
 assert cfg['run']==RUN and cfg['node']==NODE and cfg['gpu_start_epoch']is None and cfg['actual_step']is None
 assert cfg['execute_authorized']and cfg['Native_seconds']==2700 and cfg['CPUkeeper_seconds']==3000
 assert cfg['modes']==[0,0]and cfg['max_forwards']==184 and cfg['raw_limit']==33554432 and cfg['capture_outputs']==[82,83]
 assert cfg['HTTP']==cfg['free_generation']==0 and cfg['HC_mode']==0
 assert cfg['Native_SHA256']==pin['library_SHA256']=='8b1d1dacee6ba4b4f4d08dd38aa46c7bc529e5d3d89dc1ab753c204fe1b8084b'
 assert cfg['output_root']=='/refactor/results/Deep447-boundary-private/'+RUN
 assert pin['actual_CPU']=='910407.921'and pin['reused_parent_objects']==48 and pin['new_HOST_TUs']==1
 assert sha(host(pin['CPU_receipt']))==pin['CPU_receipt_SHA256']and sha(host(pin['library']))==pin['library_SHA256']
 verify_tree('/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/entry-v2',m['entry_manifest_SHA256'])
 verify_tree(cfg['boundary_root'],pin['source_manifest_SHA256'])
 verify_tree(cfg['diagnostic_root'],'cf570e363016fac276037184ffdc3bdcd03f3bc3b8896ddae1c1031e0d3b552d')
 verify_tree(pin['parent_binding_root'],pin['parent_binding_manifest_SHA256'])
 assert sha(host(cfg['context_package_root'])/'CONTEXT_DERIVATION.json')==actual['context_proof']['SHA256']
 return m
