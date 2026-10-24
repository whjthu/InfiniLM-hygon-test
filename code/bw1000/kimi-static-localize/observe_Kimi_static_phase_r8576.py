"""Only reads original621 watcher and per-owner reports. No new audit/probe."""
import pathlib,json,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
run=json.loads((r/'Kimi-static-localize-r8572-launch.json').read_text())['run']
w=json.loads((r/'Kimi-static-localize-r8572-watch.json').read_text())
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,watch=w,owners=[])
for node in range(4):
 p=r/'results'/f'kimi-static-localize-r8572-{run}-node{node}.json'
 if not p.exists():continue
 q=json.loads(p.read_text());x={k:q[k]for k in ('phase','completed','loaded_keys','loading_seconds','error','shutdown_error','all8_owned_workers_shutdown_completed','selected_detail_complete','detail_budget_exceeded','selected_detail_capture_wall_s','actual_snapshot_bytes')if k in q}
 x.update(node=node,report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=[dict(label=v['label'],wall_s=v['wall_s'],all_jobs_completed=v['all_jobs_completed'],record_count=len(v['records']))for v in q.get('diagnostic_cases',[])],decisions=[{k:v[k]for k in ('status','detail_layer','first','error')if k in v}for v in q.get('localization_decisions',[])])
 out['owners'].append(x)
print(json.dumps(out))
