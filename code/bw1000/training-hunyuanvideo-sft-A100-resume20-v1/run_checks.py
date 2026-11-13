from pathlib import Path
import hashlib,json,os,sys

W=Path(__file__).resolve().parent.parent
ORIGINAL=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1')
BINDING=json.loads((W/'src/BINDING.json').read_bytes())
STEP=20
def verify(path, pin):
    assert path.stat().st_size==pin['bytes']
    with path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==pin['SHA256']

prior=ORIGINAL/'outputs/hunyuanvideo-1.5/train/native25'
ck=prior/'checkpoint-20/state.pt'
for path,key in [(ck,'resume'),(prior/'steps.ndjson','steps'),(prior/'checkpoint-25/state.pt','final')]:
    verify(path,BINDING[key])
assert hashlib.sha256((ORIGINAL/'outcome.json').read_bytes()).hexdigest()==BINDING['original_outcome_SHA256']
launch=ORIGINAL/'src/ORIGINAL_LAUNCH.json'
assert hashlib.sha256(launch.read_bytes()).hexdigest()=='3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f'
original=json.loads(launch.read_bytes())
assets={p:p for p in (original['weights'],original['source'],original['config']['training_cache'])}
mapping=W/'assets.json'
with mapping.open('x') as f:json.dump(assets,f)
import importlib.metadata as metadata
import torch
assert torch.__version__==original['torch'] and torch.version.cuda==original['cuda']
assert all(metadata.version(k)==v for k,v in original['packages'].items())
torch.set_num_threads(original['torch_threads'])
torch.use_deterministic_algorithms(original['deterministic_algorithms'])
torch.backends.cuda.matmul.allow_tf32=original['tf32']
torch.backends.cudnn.allow_tf32=original['tf32']
assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)==original['gpu']
assert torch.cuda.get_device_properties(0).total_memory==original['gpu_capacity_bytes']
sys.path.insert(0,str(W/'framework/tools'))
import hygon_training_recipe as recipe
run=recipe.run_generation

def resume(plan,options):
    assert plan['case']=='T09.sft' and not options.dry_run and 'resume_from_checkpoint' not in plan['config']
    config=dict(plan['config'],resume_from_checkpoint=str(ck))
    return run(dict(plan,config=config),options)

recipe.run_generation=resume
try:
    rc=recipe.main(['T09.sft','--a100-launch',str(launch),'--run','--asset-map',str(mapping),
        '--cards',os.environ['CUDA_VISIBLE_DEVICES'],'--source-root',str(Path(original['source']).parent),
        '--output-root',str(W/'outputs'),'--run-name','resume20',
        '--lock-root','/home/whj/nvidia-baseline/alignment-assets/locks','--lock-host',os.uname().nodename],
        generation_accelerator='cuda')
finally:
    recipe.run_generation=run
assert rc==0
sys.path.insert(0,str(W/'framework/generation/src'))
from gencuda import hunyuanvideo_resume_checkpoint as checkpoint
out=W/'outputs/hunyuanvideo-1.5/train/resume20'
summary=json.loads((out/'summary.json').read_bytes())
assert summary['external_resume'] is True and summary['completed_global_step_at_start']==STEP
reference=[json.loads(line) for line in (prior/'steps.ndjson').read_bytes().splitlines()]
observed=[json.loads(line) for line in (out/'steps.ndjson').read_bytes().splitlines()]
assert [s['global_step'] for s in reference]==list(range(1,26))
assert [s['global_step'] for s in observed]==list(range(21,26))
errors={key:max(abs(a[key]-b[key]) for a,b in zip(observed,reference[20:])) for key in ('loss','grad_norm')}
lhs=checkpoint.load_checkpoint(out/'checkpoint-25/state.pt')
rhs=checkpoint.load_checkpoint(prior/'checkpoint-25/state.pt')
same={key:checkpoint.tree_sha256(lhs[key])==checkpoint.tree_sha256(rhs[key]) for key in ('metadata','state','rng','adapter_sha256','integrity_sha256')}
result=dict(passed=all(same.values()) and max(errors.values())==0,completed_global_step_at_start=STEP,
    completed_steps=[s['global_step'] for s in observed],max_absolute_error=errors,
    final_state_exact=same,performance_eligible=False,raw_checkpoint_bytes_exported=0)
with (W/'GPU-result.json').open('x') as f:json.dump(result,f)
assert result['passed']
