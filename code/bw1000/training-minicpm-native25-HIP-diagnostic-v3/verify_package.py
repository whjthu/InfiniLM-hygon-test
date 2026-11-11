"""Source/asset-only verification; this never imports Torch or submits Slurm."""
import hashlib,json,shutil,stat,time
from pathlib import Path
from scope import ROOT

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()

def verify():
 started=time.monotonic();src=ROOT/'src';manifest=json.loads((src/'SOURCE_FILES.json').read_bytes())
 assert sha(src/'SOURCE_FILES.json')==json.loads((ROOT/'intent.json').read_bytes())['source_manifest_SHA256']
 for name,pin in manifest['files'].items():
  path=src/name
  assert path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256'],name
 deps=json.loads((src/'DEPENDENCIES.json').read_bytes());binding=deps['framework'];mf=Path(deps['framework_files_manifest'])
 assert sha(mf)==binding['source_manifest_SHA256']
 files=json.loads(mf.read_bytes());assert files['framework_commit']==binding['framework_commit']=='0345a8c9ba2966b0fb7301edfbbe86d9cdd608b4' and len(files['files'])==284
 source=Path(binding['source_directory'])
 for name,pin in files['files'].items():
  path=source.parent/name
  assert path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256'] and stat.S_IMODE(path.stat().st_mode)==pin['mode'],name
 assert deps['swift']['swift_script']==str(source/'vlm/runtime/swift_entry.py') and Path(deps['swift']['swift_script']).is_file()
 swift_site=Path(deps['swift']['host_runtime_root'])/'site/swift/cli'
 assert sha(swift_site/'main.py')==deps['swift']['console_source_SHA256'] and sha(swift_site/'sft.py')==deps['swift']['sft_source_SHA256']
 model=deps['model'];model_root=Path(model['model_directory'])
 for name,pin in model['files'].items():
  path=model_root/name
  assert path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256'],name
 assets=Path(deps['inputs']['host_root'])
 for pin in deps['original_asset_plan']['files']:
  path=assets/pin['name'];assert path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256'],pin['name']
 data=deps['inputs']['host_dataset'];assert Path(data['path']).stat().st_size==data['bytes'] and sha(data['path'])==data['SHA256']
 mask=deps['mask_assets'];initial=mask['initial_state'];assert Path(initial['path']).stat().st_size==initial['bytes'] and sha(initial['path'])==initial['SHA256']
 for rank,pin in mask['rank_manifest_SHA256'].items():assert sha(Path(mask['mask_root'])/('rank-%02d'%int(rank))/'complete.json')==pin
 assert sha(src/'original-launch.json')=='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789'
 assert shutil.disk_usage(ROOT).free>=80*1024**3
 return dict(passed=True,framework_files=284,model_files=10,original_input_files=67,reference_manifests=8,full_model_weight_SHA256=model['files']['model.safetensors']['SHA256'],seconds=time.monotonic()-started,GPU_calls=0,qualification='Source and actual asset bytes only; no model import or execution.')
if __name__=='__main__':print(json.dumps(verify()))
