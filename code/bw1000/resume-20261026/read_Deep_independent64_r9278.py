import pathlib,json,hashlib,subprocess,os,re,datetime
R=pathlib.Path.home()/'hygon-inference-refactor-20261014';env={k:v for k,v in os.environ.items()if not(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])};raw=subprocess.check_output(['sacct','-S','2026-10-25','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%100,State%35,ExitCode,NodeList%100,Elapsed,ElapsedRaw'],env=env,text=True);actual={v.split('|')[0]:v.split('|')for v in raw.splitlines()if len(v.split('|'))>=7};out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'GPU_submitted':False,'replicas':{}}
for n in ['n01','n04','n05','n07']:
 p=R/('results/Deep-independent64-control-r9278-'+n+'.json');z={'control_exists':p.exists()};out['replicas'][n]=z
 if not p.exists():continue
 c=json.loads(p.read_bytes());z['control']={k:c[k]for k in ['CPU_step','GPU_submitted','completed','phase','heartbeat_UTC','manifest_SHA256','watch_ready_before_GPU','GPU_handle','GPU_terminal','GPU_srun_returncode','audit_returncode','audit_stdout','qualified_terminal','error','traceback']if k in c};z['CPU_actual']=actual.get(c.get('CPU_step'));h=c.get('GPU_handle')
 if 'post_KFD'in c:z['post']={k:c['post_KFD'][k]for k in ['returncode','rows','actual_terminal']if k in c['post_KFD']}
 if not h:continue
 z['GPU_actual']=actual.get(h);p=R/('results/deep-c4-independent64-r9278-'+n+'-'+h+'.json')
 if p.exists():
  v=json.loads(p.read_bytes());z['report_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();z['report']={k:v[k]for k in ['completed','phase','keys','loading_seconds','planned_count','admitted_count','bilateral_capacity_rejections','live_generation','grid_complete','performance_qualified','outcome','engine_worker_shutdown_completed','error','shutdown_error','second8_not_executed']if k in v};z['report']['formal_trials']=len(v.get('quality_trials',[]));z['report']['paired_cohorts']=len(v.get('paired_cohorts',[]))
  for k in ['arms','first8_stop']:
   if k in v:z[k]=v[k]
  def strip(obj):
   if isinstance(obj,dict):return {k:strip(v)for k,v in obj.items()if k not in ['user_metrics','ITL_seconds']}
   if isinstance(obj,list):return [strip(x)for x in obj]
   return obj
  if 'arms'in z:z['arms']=strip(z['arms'])
  if 'first8_stop'in z:z['first8_stop']=strip(z['first8_stop'])
 p=R/('logs/deep-c4-independent64-r9278-'+n+'-'+h+'.log')
 if p.exists():
  tail=p.read_text(errors='replace')[-131072:];x=re.findall(r'NATIVE_ENGINE_LOAD (\d+) (\d+) ([0-9.]+)',tail);z['last_weight_progress']={'loaded':int(x[-1][0]),'total':int(x[-1][1]),'loading_elapsed':float(x[-1][2])}if x else None
print(json.dumps(out))
