"""Read same unique r8485 intent/watch/reports only; no launch/audit/cleanup."""
import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'Kimi-bounded-r8485-launch.json'
if not p.exists():print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),intent='NOT_CREATED')));raise SystemExit
intent=json.loads(p.read_text());run=intent['run'];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,intent_state=intent['state'])
wp=r/'Kimi-bounded-r8485-watch.json'
if wp.exists():out['watch']=json.loads(wp.read_text())
text=subprocess.check_output(['sacct','-S','2026-10-23','-j','910375','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,Elapsed'],text=True,timeout=30)
out['actual_steps']=[v for v in text.splitlines()if run in v];out['owners']=[]
for node in range(4):
 p=r/'results'/f'kimi-bounded-r8485-{run}-node{node}.json'
 if p.exists():
  q=json.loads(p.read_text());out['owners'].append(dict(node=node,**{k:q[k]for k in ['phase','completed','error','shutdown_error','loading_seconds','loaded_keys','historical_B16M4_anchor_passed','bounded_wait_four_epochs_qualified','all8_owned_workers_shutdown_completed']if k in q},QA_count=len(q.get('QA',[])),epoch_qualification=q.get('bounded_wait_qualification',{}),oracle_checked_counts={k:len(v)for k,v in q.get('bounded_wait_oracle_checks',{}).items()},epoch_labels=[v['label']for v in q.get('actual_arrival_epochs',[])]))
p=r/'results'/f'Kimi-bounded-r8485-audit-{run}.json'
if p.exists():out['audit']=json.loads(p.read_text())
print(json.dumps(out))
