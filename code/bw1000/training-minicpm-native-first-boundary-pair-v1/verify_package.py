"""Source/asset/CPU-prerequisite verification before any new model load."""
import hashlib,json,shutil,stat,time
from pathlib import Path
from scope import ROOT

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for raw in iter(lambda:stream.read(1024**2),b''):h.update(raw)
 return h.hexdigest()

def verify():
 start=time.monotonic();src=ROOT/'src';intent=json.loads((ROOT/'intent.json').read_bytes());assert sha(src/'SOURCE_FILES.json')==intent['source_manifest_SHA256'];mf=json.loads((src/'SOURCE_FILES.json').read_bytes())
 for name,pin in mf['files'].items():
  p=src/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],name
 deps=json.loads((src/'DEPENDENCIES.json').read_bytes());binding=deps['framework'];fmp=Path(deps['framework_files_manifest']);assert sha(fmp)==binding['source_manifest_SHA256'];fm=json.loads(fmp.read_bytes());assert fm['framework_commit']==binding['framework_commit']=='f9aaaa8f89624fae3c0ed960780b62e1fee375da' and len(fm['files'])==353
 source=Path(binding['source_directory'])
 for name,pin in fm['files'].items():
  p=source/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'],name
 cp=Path(binding['CPU_result_path']);assert sha(cp)==binding['CPU_result_SHA256']=='461cbbfce8462e243efeb95c31fe666649ac274af8253f0def51e92b0f5b2b3e';cpu=json.loads(cp.read_bytes());assert cpu['passed'] and cpu['tests_run']==26 and cpu['source_files_verified']==353 and cpu['plugin_registered_and_base_constructor_verified'] and cpu['CUDA_initialized'] is False and cpu['owner']['step']=='1087656.182'
 assert deps['swift']['swift_script']==str(source/'vlm/runtime/swift_entry.py')
 swift=Path(deps['swift']['host_runtime_root'])/'site/swift/cli';assert sha(swift/'main.py')==deps['swift']['console_source_SHA256'] and sha(swift/'sft.py')==deps['swift']['sft_source_SHA256']
 tf=Path(deps['processor']['host_transformers_site']);assert sha(tf/'transformers/models/qwen3_5/modeling_qwen3_5.py')==deps['qwen_provider_source_SHA256']
 assert sha(tf/'transformers/models/minicpmv4_6/modeling_minicpmv4_6.py')==deps['minicpm_provider_source_SHA256']
 for pin in deps['boundary_CPU_providers'].values():
  name=pin['path'];path=(tf/name[len('/minicpm-transformers/'):]) if name.startswith('/minicpm-transformers/') else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1')/name.lstrip('/')
  assert path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256']
 model=deps['model'];mr=Path(model['model_directory'])
 for name,pin in model['files'].items():
  p=mr/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],name
 assets=Path(deps['inputs']['host_root'])
 for pin in deps['original_asset_plan']['files']:
  p=assets/pin['name'];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],pin['name']
 data=deps['inputs']['host_dataset'];assert Path(data['path']).stat().st_size==data['bytes'] and sha(data['path'])==data['SHA256']
 checkpoint=json.loads((src/'INITIAL_CHECKPOINT.json').read_bytes());assert checkpoint==deps['initial_checkpoint'] and checkpoint['checkpoint_step']==10
 for name,pin in checkpoint['checkpoint_files'].items():
  p=Path(checkpoint['checkpoint'])/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'],name
 assert sha(src/'original-launch.json')=='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789' and shutil.disk_usage(ROOT).free>=80*1024**3
 return dict(passed=True,framework_files=353,CPU_prerequisite='1087656.182',original_checkpoint_files=14,model_files=10,original_input_files=67,seconds=time.monotonic()-start,GPU_calls=0,qualification='Source and actual input bytes only; normal native recovery still unexecuted at preflight.')
if __name__=='__main__':print(json.dumps(verify()))