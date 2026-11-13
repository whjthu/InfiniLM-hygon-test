from pathlib import Path
import hashlib,json,os,sys,time
W=Path(__file__).resolve().parent.parent
root=W/'src'
launch=root/'ORIGINAL_LAUNCH.json'
binding=json.loads((root/'BINDING.json').read_bytes())
assert hashlib.sha256(launch.read_bytes()).hexdigest()==binding['launch_SHA256']
original=json.loads(launch.read_bytes())
assets={p:p for p in (original['weights'],original['source'],original['config']['training_cache'])}
mapping=W/'assets.json'
with mapping.open('x') as f:json.dump(assets,f)
import torch,importlib.metadata as metadata
assert os.environ['CUDA_VISIBLE_DEVICES']==binding['gpu_index']
assert torch.__version__==original['torch'] and torch.version.cuda==original['cuda']
assert all(metadata.version(k)==v for k,v in original['packages'].items())
torch.set_num_threads(original['torch_threads'])
torch.use_deterministic_algorithms(original['deterministic_algorithms'])
torch.backends.cuda.matmul.allow_tf32=original['tf32']
torch.backends.cudnn.allow_tf32=original['tf32']
assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==original['gpu']
assert torch.cuda.get_device_properties(0).total_memory==original['gpu_capacity_bytes']
sys.path.insert(0,str(W/'framework/tools'))
from hygon_hunyuanvideo_first_update import main
rc=main(['--capture-root',str(W/'capture'),'--capture-timeout',str(float(os.environ['CONTRACT_GPU_DEADLINE'])-time.time()),
    '--a100-launch',str(launch),'--capture-accelerator','cuda','--run','--asset-map',str(mapping),
    '--cards',binding['gpu_index'],'--source-root',str(Path(original['source']).parent),'--output-root',str(W/'outputs'),
    '--run-name','native25','--lock-root','/home/whj/nvidia-baseline/alignment-assets/locks','--lock-host',os.uname().nodename])
assert rc==0
capture=json.loads((W/'capture/capture-result.json').read_bytes())
assert capture['actual_global_step']==1 and len(capture['artifacts'])==6
for rec in capture['artifacts']:
    p=W/'capture'/rec['file']
    assert p.stat().st_size==rec['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==rec['SHA256']
out=W/'outputs/hunyuanvideo-1.5/train/native25'
steps=[json.loads(line) for line in (out/'steps.ndjson').read_text().splitlines()]
assert [s['global_step'] for s in steps]==list(range(1,26))
assert all((out/f'checkpoint-{n}/state.pt').is_file() for n in (10,20,25))
with (W/'GPU-result.json').open('x') as f:json.dump(dict(passed=True,capture=capture,
    completed_global_steps=25,first_loss=steps[0]['loss'],first_grad_norm=steps[0]['grad_norm'],
    checkpoints=[10,20,25],performance_eligible=False),f,indent=2)
