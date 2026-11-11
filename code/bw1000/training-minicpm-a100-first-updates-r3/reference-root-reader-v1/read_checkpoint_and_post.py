"""Audit the three original checkpoints and independent owned-PID/GPU cleanup."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
code=r'''
import datetime,hashlib,json,os,signal,socket,subprocess
from pathlib import Path
import torch
signal.alarm(180);torch.set_num_threads(2)
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1' and not torch.cuda.is_initialized()
root=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-a100-first-updates-r3')
raw=(root/'outcome.json').read_bytes();outcome=json.loads(raw)
assert outcome['claim_commit']=='e8898fdd6d006a60e0960dd52eba17da6a28d287' and outcome['native_returncode']==0 and outcome['keeper_passed']
assert outcome['native25_rank_records']==outcome['capture_rank_records']==8
remaining=[]
for pid,v in outcome['owners'].items():
    try:
        stat=(Path('/proc')/pid/'stat').read_text().split(') ',1)[1].split()
        if int(stat[19])==v['startticks'] and stat[0]!='Z':remaining.append({'PID':int(pid),'startticks':v['startticks']})
    except (FileNotFoundError,ProcessLookupError):pass
assert not remaining
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
assert not {int(x.split(',')[1]) for x in apps}.intersection(int(x) for x in outcome['owners'])
try:
    with socket.socket() as p:p.bind(('127.0.0.1',29873))
    port_free=True
except OSError:port_free=False
checkpoints=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for step in (10,20,25):
    folder=root/'train'/('checkpoint-%d'%step)
    assert folder.is_dir()
    files={p.name:{'bytes':p.stat().st_size,'SHA256':sha(p)} for p in sorted(folder.iterdir()) if p.is_file()}
    required={'adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json','training_args.bin',*['rng_state_%d.pth'%i for i in range(8)]}
    assert required.issubset(files),required-set(files)
    trainer=json.loads((folder/'trainer_state.json').read_bytes());assert trainer['global_step']==step
    optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
    steps=[float(v['step']) for v in optimizer['state'].values()];assert steps and min(steps)==max(steps)==step
    for state in optimizer['state'].values():
        assert set(state)=={'step','exp_avg','exp_avg_sq'}
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    scheduler=torch.load(folder/'scheduler.pt',map_location='cpu',weights_only=True);assert scheduler['last_epoch']==step
    rng=[]
    for rank in range(8):
        # These are this just-completed, SHA-bound native Trainer's own RNG states.
        state=torch.load(folder/('rng_state_%d.pth'%rank),map_location='cpu',weights_only=False)
        assert set(state)=={'python','numpy','cpu','cuda'}
        assert isinstance(state['cpu'],torch.Tensor) and state['cpu'].dtype==torch.uint8
        cuda=state['cuda'];cuda=[cuda] if isinstance(cuda,torch.Tensor) else cuda
        assert cuda and all(isinstance(x,torch.Tensor) and x.dtype==torch.uint8 for x in cuda)
        rng.append({'rank':rank,'cpu_rng_bytes':state['cpu'].numel(),'device_rng_states':len(cuda),'device_rng_bytes':[x.numel() for x in cuda],'python_state_type':type(state['python']).__name__,'numpy_state_type':type(state['numpy']).__name__})
    checkpoints.append({'global_step':step,'files':files,'optimizer_state_count':len(steps),'optimizer_step_min':min(steps),'optimizer_step_max':max(steps),'scheduler_last_epoch':scheduler['last_epoch'],'rank_RNG':rng})
assert not torch.cuda.is_initialized()
out={'observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'ORIGINAL_THREE_CHECKPOINTS_AND_OWNED_POST_VERIFIED','outcome_SHA256':hashlib.sha256(raw).hexdigest(),'owned_identities_checked':len(outcome['owners']),'owned_processes_remaining':remaining,'independent_post_compute_apps':apps,'original_master_port_free':port_free,'checkpoints':checkpoints,'fresh_process_restore_qualified':False,'GPU_execution':False}
print(json.dumps(out))
'''
cmd='CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /home/whj/nvidia-baseline/env-vlm/bin/python -B -'
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26',cmd],input=code.encode(),capture_output=True,timeout=210)
with (BASE/'checkpoint-post-reader.stderr').open('xb') as f:f.write(r.stderr)
assert r.returncode==0,r.stderr.decode(errors='replace')[-900:]
v=json.loads(r.stdout)
with (BASE/'independent-original-checkpoints-and-post.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k!='checkpoints'}))
print(json.dumps({'checkpoint_steps':[x['global_step'] for x in v['checkpoints']],'rank_RNG_per_checkpoint':[len(x['rank_RNG']) for x in v['checkpoints']]}))
