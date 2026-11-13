"""Audit the captured first update in place. Export only aggregate evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

B = Path(__file__).resolve().parent
D = B/'hunyuan-sft-first-update-A100-v1'
intent = json.loads((D/'launch-intent.json').read_bytes())
verified = json.loads((D/'independent-terminal.json').read_bytes())
assert verified['passed'] and verified['completed_steps']==25
remote = r'''
import hashlib
import json
from pathlib import Path
import torch

W = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1')
torch.set_num_threads(1)
intent = json.loads((W/'intent.json').read_bytes())
assert (intent['claim'], intent['request_id']) == (CLAIM, REQUEST)
out = W/'outcome.json'
assert hashlib.sha256(out.read_bytes()).hexdigest()==OUTCOME_SHA and json.loads(out.read_bytes())['passed']
expected_keys = {
    'before': {'params','optimizer','latents','condition','text','text_mask','byt5','byt5_mask'},
    'inputs': {'noise','timestep','hidden_states','target'},
    'forward': {'prediction','loss'},
    'preclip': {'gradients'},
    'postclip': {'gradients','norm'},
    'updated': {'params','optimizer'},
}
def load(phase):
    path = W/'capture'/(phase+'.pt')
    with path.open('rb') as f:
        raw_sha = hashlib.file_digest(f,'sha256').hexdigest()
        assert path.stat().st_size==PHASES[phase]['bytes'] and raw_sha==PHASES[phase]['SHA256']
        f.seek(0)
        value = torch.load(f,map_location='cpu',weights_only=True)
    assert set(value)==expected_keys[phase]
    return value

def check_tensor(t):
    assert isinstance(t, torch.Tensor) and t.device.type=='cpu'
    assert not (t.is_floating_point() or t.is_complex()) or bool(torch.isfinite(t).all())
    return dict(shape=list(t.shape),dtype=str(t.dtype),elements=t.numel())

before, inputs, forward, pre, post, updated = (load(k) for k in expected_keys)
fixed = {k: check_tensor(v) for k,v in before.items() if k not in ('params','optimizer')}
fixed.update({k:check_tensor(v) for k,v in inputs.items()})
a, b = before['params'], updated['params']
grad_a, grad_b = pre['gradients'], post['gradients']
assert a.keys() == b.keys() == grad_a.keys() == grad_b.keys()
assert len(a)==216 and sum(t.numel() for t in a.values())==1769472
changed_params = 0
changed_elements = 0
grad_changed_tensors = 0
pre_l2_sq = 0.0
post_l2_sq = 0.0
for name, old in a.items():
    new = b[name]
    g1, g2 = grad_a[name], grad_b[name]
    assert check_tensor(old)==check_tensor(new)==check_tensor(g1)==check_tensor(g2)
    elements = int(torch.count_nonzero(old!=new))
    changed_params += elements>0
    changed_elements += elements
    grad_changed_tensors += not torch.equal(g1,g2)
    pre_l2_sq += float(g1.double().square().sum())
    post_l2_sq += float(g2.double().square().sum())
assert changed_params>0 and changed_elements>0 and pre_l2_sq>0 and post_l2_sq>0
loss = forward['loss']
pred = forward['prediction']
target = inputs['target']
assert check_tensor(pred)['shape']==check_tensor(target)['shape']
assert check_tensor(loss)['elements']==check_tensor(post['norm'])['elements']==1
before_opt, updated_opt = before['optimizer'],updated['optimizer']
assert set(before_opt)==set(updated_opt)=={'state','param_groups'}
assert before_opt['param_groups']==updated_opt['param_groups']
assert len(before_opt['state'])==len(updated_opt['state'])==216
assert set(before_opt['state'])==set(updated_opt['state'])
state_keys = set()
state_tensors = 0
for pid,state in updated_opt['state'].items():
    assert before_opt['state'][pid]=={'use_muon':True} and state['use_muon'] is True
    state_keys.update(state)
    for value in state.values():
        if isinstance(value,torch.Tensor):
            check_tensor(value)
            state_tensors+=1
assert state_tensors>0
print(json.dumps(dict(verified=True,first_loss=float(loss),recorded_grad_norm=float(post['norm']),
    phases=6, trainable_tensors=len(a),changed_parameter_tensors=changed_params,
    changed_parameter_elements=changed_elements,changed_gradient_tensors=grad_changed_tensors,
    preclip_L2=pre_l2_sq**.5, postclip_L2=post_l2_sq**.5,
    muon_state_keys=sorted(state_keys),muon_state_tensor_count=state_tensors,
    input_shapes=fixed,prediction=check_tensor(pred),raw_tensor_bytes_exported=0,device_calls=0)))
'''
values = dict(CLAIM=intent['claim'],REQUEST=intent['request_id'],PHASES=verified['phases'],
              OUTCOME_SHA=verified['outcome_SHA256'])
config='C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config'
command = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',config,
           'a100-27','timeout 55 /home/whj/nvidia-baseline/env-hunyuan/bin/python -B -u -']
r = subprocess.run(command,input=(''.join(f'{k}={v!r}\n' for k,v in values.items())+remote).encode(),
                   capture_output=True,timeout=70)
if r.returncode:
    error = next((name for name in ('AssertionError','TypeError','KeyError','ModuleNotFoundError','FileNotFoundError','RuntimeError','Timeout') if name.encode() in r.stderr),None)
    print(json.dumps(dict(rc=r.returncode,error=error,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())))
    raise SystemExit(1)
summary = json.loads(r.stdout)
assert summary['first_loss']==verified['first_loss']
assert summary['recorded_grad_norm']==verified['first_grad_norm']
with (B/'hunyuan-A100-first-update-values-independent.json').open('x',encoding='utf8') as f:
    json.dump(dict(request_id=intent['request_id'],outcome_SHA256=values['OUTCOME_SHA'],**summary),f,indent=2)
print(json.dumps(summary))
