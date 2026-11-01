"""Follow only the single349 label/handle; never submit or cancel."""
import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='GLM-W4-cross-workload-r349';q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Slurm':{},'logs':{},'tail_owner':None}
for parent in ['910376','910407']:
 s=subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
 q['Slurm'][parent]='\n'.join(x for x in s.splitlines()if label in x)
w=r/'results'/(label+'-watch.json')
if w.exists():
 try:q['watch']=json.loads(w.read_bytes())
 except json.JSONDecodeError:q['watch_partial_JSON']=True
for p in(r/'logs').glob(label+'*.log'):
 q['logs'][str(p)]=p.read_text(errors='replace')[-2500:]
p=r/'results'/('GLM-latest-full-owner-r290-'+label+'-stage3-rank0.json')
if p.exists():
 try:
  a=json.loads(p.read_bytes());rows=a.get('service_requests',[]);keep=['completed','actual_step','attempted_Native','completed_Native','normal_shutdown','normal_HTTP_shutdown','service_error','worker_shutdown_completed','service_ready','source_manifest_SHA256']
  tail={k:a[k]for k in keep if k in a};tail['rows']=[{k:x[k]for k in ['fixture_id','global_index','arm_index','prefill_control_window','paired_warmup','prompt_tokens','generated_tokens','wall_s','reason','generated_natural_EOS','actual_prompt_token_ids_SHA256','generated_token_ids_SHA256','Native_forward_host_summary']if k in x}for x in rows];q['tail_owner']=tail
 except json.JSONDecodeError:q['tail_owner_partial_JSON']=True
print(json.dumps(q))
