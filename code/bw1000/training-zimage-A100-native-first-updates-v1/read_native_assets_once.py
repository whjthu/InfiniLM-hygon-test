"""Read actual imported-source closure and the two new private Turbo renders."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
code=b'''from pathlib import Path
import hashlib,json,signal,sys
signal.alarm(60);w=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-A100-native-first-updates-v1')
binding=json.loads((w/'src/BINDING.json').read_bytes());source=Path(binding['source_root'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
providers={}
for mode in ('continuous','resume20'):
 p=w/mode/'providers.json';assert p.is_file() and p.stat().st_size<2*1024**2
 values=json.loads(p.read_bytes());assert 1<=len(values)<=512
 total=0
 for name,pin in values.items():
  part=Path(pin['relative_path']);assert not part.is_absolute() and '..' not in part.parts
  f=source/part;assert f.is_file() and not f.is_symlink() and f.stat().st_size==pin['bytes'] and pin['bytes']<=2*1024**2
  total+=pin['bytes'];assert total<=40*1024**2 and sha(f)==pin['SHA256']
 providers[mode]=dict(provider_count=len(values),manifest_SHA256=sha(p),checked_source_bytes=total,source_files=values)
from PIL import Image
renders={}
for name in ('turbo-before.private.png','turbo-after.private.png'):
 p=w/'continuous'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2*1024**2
 with Image.open(p) as im:im.load();assert im.size==(256,256) and im.mode=='RGB'
 renders[name]=dict(bytes=p.stat().st_size,SHA256=sha(p),shape=[256,256,3],pixel_values_exported=False)
import ast,math
path=w/'framework/generation/src/gencuda/backends/zimage_training.py'
fn=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='_image_regression')
tree=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(tree);scope=dict(Path=Path,math=math)
exec(compile(tree,str(path),'exec'),scope)
metrics=scope['_image_regression'](w/'continuous/turbo-before.private.png',w/'continuous/turbo-after.private.png')
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,providers=providers,renders=renders,original_single_example_pixel_regression=metrics,
 raw_images_exported=False,CLIP_Score=None,FID=None,heldout_effect_qualified=False,training_improvement_qualified=False,new_CPU_or_GPU_runs=0,GPU_initialized=False)))
'''
start=time.monotonic();r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-zimage/bin/python -B -u -'],input=code,capture_output=True,timeout=80)
(D/(tag+'-assets.stdout.private.json')).write_bytes(r.stdout);(D/(tag+'-assets.stderr.private.txt')).write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
with (D/(tag+'-assets-result.json')).open('x') as f:json.dump(dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,
 source_SHA256=hashlib.sha256(code).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),value=v),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed') if v else None,
 providers={n:p['provider_count'] for n,p in v['providers'].items()} if v else None,
 legacy_pixel_regression=v.get('original_single_example_pixel_regression') if v else None,CLIP_Score=None,FID=None,heldout_effect_qualified=False,new_GPU_runs=0)))
