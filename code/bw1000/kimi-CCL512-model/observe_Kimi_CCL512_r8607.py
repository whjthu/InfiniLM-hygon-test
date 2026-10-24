"""Read-only observer, reuses unique supervisor outputs, no new audit/KFD."""
import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
d=json.loads((r/'Kimi-CCL512-r8607-launch.json').read_text());run=d['run']
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,intent_state=d['state'])
p=r/'Kimi-CCL512-r8607-watch.json'
if p.exists():out['watch']=json.loads(p.read_text())
out['actual_steps']=[v for v in subprocess.check_output(['sacct','-S','2026-10-23','-j','910375','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,Elapsed'],text=True,timeout=30).splitlines()if run in v]
out['owners']=[]
for n in range(4):
 p=r/'results'/f'kimi-CCL512-r8607-{run}-node{n}.json'
 if p.exists():
  q=json.loads(p.read_text());c=q.get('prefill512_capture_placeholder',{})
  out['owners'].append(dict(node=n,**{k:q[k]for k in ['phase','completed','error','shutdown_error','loading_seconds','loaded_keys','historical618_exact_O8','static_final_IDs_equal','generation_acceptance','all8_owned_workers_shutdown_completed']if k in q},QA_count=len(q.get('QA',[])),comparison_count=len(q.get('model_comparisons',[])),paired_count=len(q.get('paired_timing',[])),causal={k:c[k]for k in ['status']if k in c},causal_phase_count=len(c.get('cases',[]))))
p=r/'results'/f'Kimi-CCL512-r8607-audit-{run}.json'
if p.exists():out['audit']=json.loads(p.read_text())
print(json.dumps(out))
