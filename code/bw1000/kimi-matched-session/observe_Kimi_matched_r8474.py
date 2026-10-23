"""Read-only observer, reuses unique supervisor outputs, no new audit/KFD."""
import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
d=json.loads((r/'Kimi-matched-r8474-launch.json').read_text());run=d['run']
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,intent_state=d['state'])
p=r/'Kimi-matched-r8474-watch.json'
if p.exists():out['watch']=json.loads(p.read_text())
out['actual_steps']=[v for v in subprocess.check_output(['sacct','-S','2026-10-23','-j','910375','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,Elapsed'],text=True,timeout=30).splitlines()if run in v]
out['owners']=[]
for n in range(4):
 p=r/'results'/f'kimi-matched-r8472-{run}-node{n}.json'
 if p.exists():
  q=json.loads(p.read_text());out['owners'].append(dict(node=n,**{k:q[k]for k in ['phase','completed','error','shutdown_error','loading_seconds','loaded_keys','historical_B16M4_anchor_passed','matched_trace_result','arrival_reference_qualification','live_trace_matched_CPU_oracle_qualified','all8_owned_workers_shutdown_completed']if k in q},QA_count=len(q.get('QA',[]))))
p=r/'results'/f'Kimi-matched-r8474-audit-{run}.json'
if p.exists():out['audit']=json.loads(p.read_text())
print(json.dumps(out))
