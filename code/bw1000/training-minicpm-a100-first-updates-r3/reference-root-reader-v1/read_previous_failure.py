"""Independently consume the original failed run and verify owned process cleanup."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
config=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
code=r'''
import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-a100-first-updates-r2')
raw=(root/'outcome.json').read_bytes();outcome=json.loads(raw)
assert outcome['claim_commit']=='f5f27a02d8e0b499df7fb1308e1f5cde09dd3a14'
assert outcome['GPU_launched'] and outcome['native_returncode']==1 and not outcome['keeper_passed']
assert outcome['deadline']-outcome['epoch']==900 and not outcome['timed_out'] and outcome['controller_error'] is None
assert outcome['cleanup']=={'owned_processes_remaining':0,'termination_needed':False}
assert outcome['capture_rank_records']==outcome['native25_rank_records']==0
remaining=[]
for pid,v in outcome['owners'].items():
    p=Path('/proc')/pid
    try:
        values=(p/'stat').read_text().split(') ',1)[1].split()
        if int(values[19])==v['startticks'] and values[0]!='Z':remaining.append({'PID':int(pid),'startticks':v['startticks'],'same_UID':p.stat().st_uid==os.getuid()})
    except (FileNotFoundError,ProcessLookupError):pass
assert not remaining
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
assert not {int(x.split(',')[1]) for x in apps}.intersection(int(x) for x in outcome['owners'])
log=(root/'native.log').read_bytes();assert hashlib.sha256(log).hexdigest()==outcome['native_log_SHA256']
assert b"AttributeError: 'Seq2SeqTrainingArguments' object has no attribute 'lora_rank'" in log
out={k:v for k,v in outcome.items() if k not in ('owners','post')}
out.update(state='TERMINAL_REFERENCE_CALLBACK_ARGUMENT_API_FAILURE',observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),outcome_SHA256=hashlib.sha256(raw).hexdigest(),owned_identities_checked=len(outcome['owners']),owned_remaining=remaining,independent_post_compute_apps=apps,original_post=outcome['post'],training_qualified=False,reference_updates_completed=0)
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(config),'a100-26','python3'],input=code.encode(),capture_output=True,timeout=50)
assert r.returncode==0,r.stderr.decode(errors='replace')[-600:]
v=json.loads(r.stdout)
with (BASE/'independent-original-terminal-failure.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k!='original_post'}))
