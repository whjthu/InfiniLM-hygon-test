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
 deps=json.loads((src/'DEPENDENCIES.json').read_bytes());binding=deps['framework'];fmp=Path(deps['framework_files_manifest']);assert sha(fmp)==binding['source_manifest_SHA256'];fm=json.loads(fmp.read_bytes());assert fm['framework_commit']==binding['framework_commit']=='480cee6ed85a8499a57b0890c5cbbb56f77da9ce' and len(fm['files'])==304
 source=Path(binding['source_directory'])
 for name,pin in fm['files'].items():
  p=source/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'],name
 cp=Path(binding['CPU_result_path']);assert sha(cp)==binding['CPU_result_SHA256']=='04ce22901a3dbf6b6ed52b265bed07453ed95c4ac601cf2a90bd61a074893c34';cpu=json.loads(cp.read_bytes());assert cpu['passed'] and cpu['tests_run']==14 and cpu['source_files_verified']==304 and len(cpu['registered_case_ids'])==17 and cpu['CUDA_initialized'] is False and cpu['owner']['step']=='1087656.149'
 assert deps['swift']['swift_script']==str(source/'vlm/runtime/swift_entry.py')
 swift=Path(deps['swift']['host_runtime_root'])/'site/swift/cli';assert sha(swift/'main.py')==deps['swift']['console_source_SHA256'] and sha(swift/'sft.py')==deps['swift']['sft_source_SHA256']
 tf=Path(deps['processor']['host_transformers_site']);assert sha(tf/'transformers/models/qwen3_5/modeling_qwen3_5.py')==deps['qwen_provider_source_SHA256']
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
 return dict(passed=True,framework_files=304,CPU_prerequisite='1087656.149',original_checkpoint_files=14,model_files=10,original_input_files=67,seconds=time.monotonic()-start,GPU_calls=0,qualification='Source and actual input bytes only; normal native recovery still unexecuted at preflight.')
if __name__=='__main__':print(json.dumps(verify()))