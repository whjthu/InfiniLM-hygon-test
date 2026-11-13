"""Audit the exact original HunyuanVideo checkpoint state on its A100 host."""
import hashlib
import json
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parent
D = B/'hunyuan-sft-first-update-A100-v1'
terminal = json.loads((D/'independent-terminal.json').read_bytes())
assert terminal['passed'] and terminal['completed_steps'] == 25

REMOTE = r'''
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch

root = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1')
assert hashlib.sha256((root/'outcome.json').read_bytes()).hexdigest() == OUTCOME
sys.path.insert(0,str(root/'framework/generation/src'))
from gencuda import hunyuanvideo_resume_checkpoint as ck
torch.set_num_threads(1)
run = root/'outputs/hunyuanvideo-1.5/train/native25'
rng_globals = [np.ndarray,np.dtype,np.core.multiarray._reconstruct,type(np.dtype(np.uint32))]
contract = None
evidence = []
for step in (10,20,25):
    pin = PINS[str(step)]
    with (run/f'checkpoint-{step}'/'state.pt').open('rb') as f:
        raw = f.read(pin['bytes']+1)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    with torch.serialization.safe_globals(rng_globals):
        payload = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    assert set(payload)=={'metadata','state','rng','adapter_sha256','integrity_sha256'}
    assert ck.tree_sha256({k:v for k,v in payload.items() if k!='integrity_sha256'})==payload['integrity_sha256']
    metadata=payload['metadata']
    current=metadata['contract']
    assert current['device_ids']==[0] and current['total_global_steps']==25 and current['warmup_steps']==5
    assert current['data_order']==[CACHE]
    assert contract is None or current==contract
    contract=current
    assert metadata==ck.boundary_metadata(contract,step)
    state=payload['state']
    assert set(state)=={'trainable','buffers','modes','optimizer','optimizer_parameter_names',
                        'optimizer_placements','optimizer_implementation'}
    params=state['trainable']
    assert len(params)==216 and sum(t.numel() for t in params.values())==1769472
    assert ck.tree_sha256(params)==payload['adapter_sha256']
    optimizer=state['optimizer']
    group,=optimizer['param_groups']
    names,=state['optimizer_parameter_names']
    assert names==list(params) and group['params']==list(range(len(names)))
    assert set(optimizer['state'])==set(state['optimizer_placements'])==set(group['params'])
    assert state['optimizer_implementation']['qualname']=='Muon'
    for index,name in enumerate(names):
        entry=optimizer['state'][index]
        param=params[name]
        mom=entry['momentum_buffer']
        assert set(entry)=={'use_muon','momentum_buffer'} and entry['use_muon'] is True
        assert param.dtype==mom.dtype and param.shape==mom.shape and param.ndim==2
        assert param.device.type==mom.device.type=='cpu'
        assert bool(torch.isfinite(param).all() and torch.isfinite(mom).all())
        assert state['optimizer_placements'][index]=={'momentum_buffer':'cuda:0'}
    rng = payload['rng']
    rng_validation=ck.common.validate_rng(rng,[0],native=False)
    evidence.append(dict(global_step=step,completed_measured_steps=metadata['completed_measured_steps'],
        next_coordinate=ck.next_coordinate(metadata), checkpoint=pin, trainable_tensors=len(params),
        muon_momentum_tensors=len(optimizer['state']),module_modes=len(state['modes']),
        buffers=len(state['buffers']), adapter_SHA256=payload['adapter_sha256'],
        rng_validation=rng_validation,rng_streams=sorted(rng),rng_SHA256=ck.tree_sha256(rng)))
assert len({r['adapter_SHA256'] for r in evidence})==3
assert len({r['rng_SHA256'] for r in evidence})==3
print(json.dumps(dict(verified=True,checkpoints=evidence,source_outcome_SHA256=OUTCOME,
    contract_SHA256=ck.json_sha256(contract),scheduler_state_present=False,
    gradient_accumulation_steps=1,rank_count=1,device_calls=0,raw_checkpoint_bytes_exported=0,
    restored_in_new_process=False)))
'''
args = dict(OUTCOME=terminal['outcome_SHA256'],PINS=terminal['checkpoints'],
            CACHE='9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b')
config='C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config'
command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',config,'a100-27',
         'timeout 90 /home/whj/nvidia-baseline/env-hunyuan/bin/python -B -u -']
r = subprocess.run(command,input=(''.join(f'{k}={v!r}\n' for k,v in args.items())+REMOTE).encode(),
                   capture_output=True,timeout=105)
if r.returncode:
    import re
    print(json.dumps(dict(rc=r.returncode, lines=re.findall(rb'File "<stdin>", line (\d+)',r.stderr)[-1:][0].decode() if b'<stdin>' in r.stderr else None,
                          errors=[n for n in ('UnpicklingError','AssertionError','KeyError','ValueError','FileNotFoundError') if n.encode() in r.stderr],
                          stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())))
    raise SystemExit(1)
v=json.loads(r.stdout)
assert v['verified'] and [row['global_step'] for row in v['checkpoints']]==[10,20,25]
with (B/'hunyuan-A100-native25-checkpoint-state-audit.json').open('x',encoding='utf-8') as f:
    json.dump(v,f,indent=2)
print(json.dumps(v))
