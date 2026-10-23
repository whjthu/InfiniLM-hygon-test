"""Read-only observer, reuses unique supervisor outputs, no new audit/KFD."""
import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
d=json.loads((r/'Kimi-causal-r8521-launch.json').read_text());run=d['run']
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,intent_state=d['state'])
p=r/'Kimi-causal-r8521-watch.json'
if p.exists():out['watch']=json.loads(p.read_text())
out['actual_steps']=[v for v in subprocess.check_output(['sacct','-S','2026-10-23','-j','910375','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,Elapsed'],text=True,timeout=30).splitlines()if run in v]
out['owners']=[]
for n in range(4):
 p=r/'results'/f'kimi-causal-r8521-{run}-node{n}.json'
 if p.exists():
  q=json.loads(p.read_text());c=q.get('native_causal',{})
  out['owners'].append(dict(node=n,**{k:q[k]for k in ['phase','completed','error','shutdown_error','loading_seconds','loaded_keys','historical_B16M4_anchor_passed','all8_owned_workers_shutdown_completed']if k in q},QA_count=len(q.get('QA',[])),causal={k:c[k]for k in ['status','first_differing_stage_observed','first_differing_layer_observed']if k in c},causal_phase_count=len(c.get('phases',[]))))
p=r/'results'/f'Kimi-causal-r8521-audit-{run}.json'
if p.exists():out['audit']=json.loads(p.read_text())
print(json.dumps(out))
