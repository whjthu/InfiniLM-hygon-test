import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';run='kimi-W2pair-A-r9282-4108c1be3d';hp=r/'results'/('Kimi-W2pair-A-model-r9282-'+run+'.json');out={'run':run,'health_exists':hp.exists(),'sacct':[]}
for parent in ['910406','910379','910371']:
 rows=subprocess.check_output(['sacct','-S','2026-10-25','-j',parent,'-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True,timeout=25).splitlines();out['sacct'].extend(row.split('|')for row in rows if 'Kimi-W2pair-Model-keeper-r9282'in row or run in row)
if hp.exists():
 w=json.loads(hp.read_bytes());out['health']={k:w.get(k)for k in ['CPU_actual_step','hostname','PID','ready','completed','GPU_submitted','manifest_SHA256','heartbeat_UTC','preflight_pins_passed','capacity_passed_before_draft','passed','audit','post_KFD_passed','observation_errors']};out['groups']=[{k:v.get(k)for k in ['parent','name','actual_handle','actual_row','terminal','client_returncode','signal']}for v in w.get('groups',[])];out['draft']={k:w.get('draft',{}).get(k)for k in ['parent','name','actual_handle','actual_row','terminal','status','ready_seen']};out['capacity']=[{k:x.get(k)for k in ['node','phase','all_target_min_free_bytes','actual_engine_key_stage_range','actual_all93_coverage_validated']}for x in w.get('all4_capacity_projections',[])];out['fresh_completed']=len(w.get('fresh_before',[]))
if hp.exists():out['health']['error_present']=bool(w.get('error'))

out['node_progress']=[]
for node in range(4):
 p=r/'results'/('Kimi-W2pair-A-r9282-'+run+'-node%d.json'%node);v={'node':node,'exists':p.exists()}
 if p.exists():
  v['bytes']=p.stat().st_size
  if v['bytes']<=8*1024*1024:
   q=json.loads(p.read_bytes());v.update({k:q.get(k)for k in ['phase','loaded_keys','loading_elapsed_seconds','loading_seconds','loader_flushes','completed']});v['error_present']=bool(q.get('error')or q.get('shutdown_error'));v['case_count']=len(q.get('batch_cases',[]))
 out['node_progress'].append(v)
print(json.dumps(out))

