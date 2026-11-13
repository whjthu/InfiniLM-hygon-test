from pathlib import Path
import hashlib
import json
import subprocess
import sys

D = Path(__file__).resolve().parent
intent = json.loads((D / 'launch-intent.json').read_bytes())
config = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
remote = 'CLAIM = ' + repr(intent['claim']) + '\nREQUEST = ' + repr(intent['request_id']) + '\n'
remote += r'''
from pathlib import Path
import hashlib
import json
import math
import subprocess

W = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-first-update-v1')

def read(name, limit=2*1024**2):
    with (W / name).open('rb') as f:
        raw = f.read(limit+1)
    assert len(raw) <= limit
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def pin(path):
    with path.open('rb') as f:
        return {'bytes':path.stat().st_size,'SHA256':hashlib.file_digest(f, 'sha256').hexdigest()}

def alive(owner):
    try:
        parts = Path('/proc/%d/stat' % owner['PID']).read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return False
    return int(parts[19]) == owner['startticks'] and parts[0] != 'Z'

intent, intent_hash = read('intent.json')
assert (intent['claim'], intent['request_id']) == (CLAIM, REQUEST)
path = W / 'outcome.json'
if not path.is_file():
    child, _ = read('GPU-child-owner.json') if (W / 'GPU-child-owner.json').is_file() else (None, None)
    step_log = W / 'outputs/hunyuanvideo-1.5/train/native25/steps.ndjson'
    complete_lines = sum(line.endswith(b'\n') for line in step_log.open('rb')) if step_log.is_file() else 0
    print(json.dumps(dict(state='OUTCOME_PENDING',owner_live=alive(intent['owner']),child_live=bool(child and alive(child)),saved_step_rows=complete_lines)))
else:
    try:
        out, outcome_hash = read('outcome.json')
    except json.JSONDecodeError:
        print(json.dumps(dict(state='OUTCOME_WRITE_IN_PROGRESS')))
        raise SystemExit(0)
    for k in ('claim','request_id','epoch','deadline','owner'):
        assert out[k] == intent[k]
    child, child_hash = read('GPU-child-owner.json') if (W / 'GPU-child-owner.json').is_file() else (None,None)
    assert out['child'] == child
    v = {k:out[k] for k in ('passed','returncode','stop_reason','owned_group_reaped','GPU_post_no_compute_process','peak_group_RSS_bytes','wall_seconds','GPU_cards')}
    v.update(state='TERMINAL',intent_SHA256=intent_hash,outcome_SHA256=outcome_hash,child_SHA256=child_hash,child_live=bool(child and alive(child)))
    log = W / 'GPU.private.log'
    if log.is_file():
        v['private_log'] = pin(log)
        if not out['passed']:
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16384))
                tail = f.read().decode(errors='replace')
            for token in ('OutOfMemoryError','RuntimeError','ModuleNotFoundError','AssertionError','PermissionError','FileNotFoundError','ValueError','ImportError','TypeError','AttributeError'):
                if token in tail:
                    v.setdefault('failure_indicators',[]).append(token)
    if out['passed']:
        assert not v['child_live'] and v['owned_group_reaped'] and v['GPU_post_no_compute_process'] and v['returncode']==0
        gpu, gpu_hash = read('GPU-result.json')
        assert gpu == out['GPU_result'] and gpu_hash == out['GPU_result_SHA256']
        capture, capture_hash = read('capture/capture-result.json')
        assert capture == gpu['capture'] and capture['actual_global_step'] == 1
        expected_phases = {'before','inputs','forward','preclip','postclip','updated'}
        phases = {}
        for rec in capture['artifacts']:
            part = Path(rec['file'])
            assert len(part.parts) == 1 and part.suffix == '.pt'
            phase = next(name for name in expected_phases if part.stem.endswith(name))
            assert phase not in phases
            actual = pin(W/'capture'/part)
            assert actual == {k:rec[k] for k in actual}
            phases[phase] = actual
        assert phases.keys() == expected_phases
        root = W/'outputs/hunyuanvideo-1.5/train/native25'
        steps = root/'steps.ndjson'
        rows = [json.loads(line) for line in steps.read_bytes().splitlines()]
        assert [r['global_step'] for r in rows] == list(range(1,26))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows)
        assert rows[0]['loss'] == gpu['first_loss'] and rows[0]['grad_norm'] == gpu['first_grad_norm']
        ckpts = {step:pin(root/f'checkpoint-{step}'/'state.pt') for step in (10,20,25)}
        v.update(phases=phases,capture_SHA256=capture_hash,completed_steps=len(rows),steps=pin(steps),checkpoints=ckpts,first_loss=gpu['first_loss'],first_grad_norm=gpu['first_grad_norm'])
    print(json.dumps(v))
'''
completed = subprocess.run(
    ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-27','timeout 120 python3 -B -u -'],
    input=remote.encode(), capture_output=True, timeout=135,
)
if completed.returncode:
    print(json.dumps(dict(state='READ_UNAVAILABLE',rc=completed.returncode,stderr_SHA256=hashlib.sha256(completed.stderr).hexdigest())))
    sys.exit(1)
value = json.loads(completed.stdout)
if value['state']=='TERMINAL':
    with (D / 'independent-terminal.json').open('x') as f:
        json.dump(value, f, indent=2)
print(json.dumps(value))
