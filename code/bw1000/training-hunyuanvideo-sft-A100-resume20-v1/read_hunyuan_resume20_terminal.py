from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parent
D=B/'hunyuan-sft-A100-resume20-v1'
I=json.loads((D/'launch-intent.json').read_bytes())
cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',
    'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config',
    'a100-27','timeout 30 python3 -B -u -']
code=r'''
from pathlib import Path
import hashlib,json
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sft-resume20-v1')
def read(path):
    raw=(W/path).read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()
def live(owner):
    if not owner:return False
    try:
        fields=Path('/proc/%d/stat'%owner['PID']).read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0]!='Z' and int(fields[19])==owner['startticks']
i,ih=read('intent.json')
assert (i['claim'],i['request_id'])==(CLAIM,REQUEST)
if not (W/'outcome.json').is_file():
    child,_=read('GPU-child-owner.json') if (W/'GPU-child-owner.json').is_file() else (None,None)
    print(json.dumps(dict(terminal=False,owner_live=live(i['owner']),child_live=live(child))))
else:
    out,oh=read('outcome.json')
    assert all(i[k]==out[k] for k in ('claim','request_id','epoch','deadline','owner'))
    v={k:out[k] for k in ('passed','returncode','stop_reason','owned_group_reaped','GPU_post_no_compute_process','wall_seconds')}
    v.update(terminal=True,request_id=REQUEST,claim=CLAIM,intent_SHA256=ih,outcome_SHA256=oh,child_live=live(out['child']))
    if out['GPU_result'] is not None:
        result,rh=read('GPU-result.json')
        assert rh==out['GPU_result_SHA256'] and result==out['GPU_result']
        v['result']=result
    if out['passed']:
        assert not v['child_live'] and v['owned_group_reaped'] and v['GPU_post_no_compute_process'] and v['returncode']==0
        assert result['completed_steps']==[21,22,23,24,25] and all(result['final_state_exact'].values())
    log=W/'GPU.private.log'
    if log.is_file():
        with log.open('rb') as f:
            v['private_log_SHA256']=hashlib.file_digest(f,'sha256').hexdigest()
            if not v['passed']:
                f.seek(max(0,log.stat().st_size-12000))
                tail=f.read()
                v['failure_types']=[x for x in ('CheckpointValidationError','RuntimeError','AssertionError','ValueError','OutOfMemoryError') if x.encode() in tail]
    print(json.dumps(v))
'''
r=subprocess.run(cmd,input=(f"CLAIM={I['claim']!r}\nREQUEST={I['request_id']!r}\n"+code).encode(),capture_output=True,timeout=45)
if r.returncode:
    print(json.dumps(dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())))
    raise SystemExit(1)
v=json.loads(r.stdout)
if v['terminal']:
    with (D/'independent-terminal.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
