import pathlib,json,hashlib,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='kimi-cap8-micro1-r9293-5efe17cbd6';p=r/'results'/('Kimi-cap8-micro1-sequential-audit-r9293-'+run+'.json');raw=p.read_bytes();a=json.loads(raw);assert a['passed']and a['initial50_complete']and a['cases']==50 and a['normal_target_owners']==32 and not a['unexecuted_planned_case_ids'];out={'schema':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_original_SHA256':hashlib.sha256(raw).hexdigest(),'audit_original_path':str(p),'audit':{k:v for k,v in a.items()if k not in ['case_summaries']}}
for c in a['case_summaries']:
 for u in c['users']:
  u['quality']={k:u['quality'].get(k)for k in ['id','sample_index','task','domain','correct','extracted','expected']}
out['cases']=a['case_summaries']
h=json.loads((r/'results'/('Kimi-cap8micro1-A-model-r9293-'+run+'.json')).read_bytes());out['coordinator']={k:h.get(k)for k in ['completed','passed','post_KFD_passed','manifest_SHA256','CPU_actual_step','heartbeat_UTC']};out['coordinator']['error_present']=bool(h.get('error'));out['coordinator']['draft_normal_terminal']=h['draft'].get('terminal');out['coordinator']['targets_terminal']=[g.get('terminal')for g in h['groups']]
np=r/'results'/('Kimi-cap8-micro1-r9293-'+run+'-node2.json');d=json.loads(np.read_bytes());out['runtime_scalar']={k:d.get(k)for k in ['loading_elapsed_seconds','create_seconds','completed','serving_protocol_complete','all8_owned_workers_shutdown_completed']};out['case_minimum_free_bytes']=[{'label':c['label'],'minimum_free_bytes':c.get('memory_minimum_case',{}).get('minimum_free_bytes')}for c in d['batch_cases']]
out['Slurm']=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-26','-j','910379,910371,910406','-n','-P','-o','JobID,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines()if x.split('|')[0]in['910379.474','910371.586','910406.277']]
print(json.dumps(out))
