from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-minicpm-original143-values-CPU-v2';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
task=json.loads((G/'coordination/tasks/training-minicpm-original143-values-CPU-v2.json').read_bytes());original=json.loads((D/'attempts/0002/transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_exec_acknowledged']
expected=dict(request=original['request_id'],claim=original['claim'],source={n:task['source_SHA256'][n] for n in task['remote_files']},result=original['value']['CPU-result.json']['SHA256'],progress=original['value']['progress.json']['SHA256'])
ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
remote=r'''from pathlib import Path
import datetime,hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original143-values-CPU-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,pin in E['source'].items():assert sha(W/'source'/name)==pin
v=json.loads((W/'outcome.json').read_bytes());intent=json.loads((W/'intent.json').read_bytes());result=json.loads((W/'evidence/CPU-result.json').read_bytes());progress=json.loads((W/'evidence/progress.json').read_bytes());plan=json.loads((W/'source/PLAN.json').read_bytes())
assert v['passed'] and v['returncode']==0 and not v['forced_cleanup'] and result['passed']
assert intent['claim']==v['claim']==E['claim'] and intent['request_id']==E['request']
assert intent['epoch']==v['epoch']==result['epoch'] and intent['deadline']==v['deadline']==v['epoch']+720 and result['deadline']==v['epoch']+600
assert sha(W/'CPU.log')==v['log_SHA256']
for name,key in [('CPU-result.json','result'),('progress.json','progress')]:
 p=W/'evidence'/name;assert sha(p)==v[name]['SHA256']==E[key] and p.stat().st_size==v[name]['bytes']
assert result['completed_value_files']==progress['completed_value_files']==136 and progress['completed_ranks']==8 and not progress['failed'] and progress['stage']=='complete'
assert result['source_file_bytes_verified']==18892807208 and result['plan_SHA256']==E['source']['PLAN.json'] and result['module_SHA256']==E['source']['vlm/runtime/first_update_audit.py']
assert result['CPU_tests']==4 and not result['GPU_initialized'] and result['GPU_cards']==0 and result['raw_tensor_transfer_bytes']==0
ranks=result['rank_summaries'];assert len(ranks)==8 and {x['rank'] for x in ranks}==set(range(8))
assert all(len(x['forwards'])==8 and len(x['updates'])==2 and x['at_least_one_nonzero_saved_parameter_update'] for x in ranks)
assert len({x['initial_parameters']['SHA256'] for x in ranks})==1
for i in range(2):
 for key in ('parameters_after','preclip_gradients','postclip_gradients','optimizer_state_after'):
  assert len({r['updates'][i][key]['SHA256'] for r in ranks})==1
 assert all(r['updates'][i]['optimizer_after_steps']['minimum']==r['updates'][i]['optimizer_after_steps']['maximum']==i+1 for r in ranks)
assert all(r['updates'][0]['before_group_learning_rates']==[0.0] and r['updates'][0]['parameter_update']['changed_elements']==0 for r in ranks)
assert all(r['updates'][1]['before_group_learning_rates']==[2e-5] and r['updates'][1]['parameter_update']['changed_tensors']==372 and r['updates'][1]['parameter_update']['changed_elements']==10822432 for r in ranks)
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.147','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=10).splitlines() if x.startswith('1087656.147|')]
assert rows==v['accounting'] and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]=='f11r4n10' and result['owner']['step']==rows[0][0]
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%L'],text=True,timeout=10).strip()
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_original147_verified=True,source_pins=E['source'],outcome_SHA256=sha(W/'outcome.json'),intent_SHA256=sha(W/'intent.json'),result_SHA256=E['result'],progress_SHA256=E['progress'],log_SHA256=v['log_SHA256'],accounting_columns=v['accounting_columns'],accounting=rows,parent_snapshot=parent,completed_value_files=136,source_file_bytes_verified=result['source_file_bytes_verified'],rank_summaries=ranks,peak_RSS_KiB=result['peak_RSS_KiB'],audit_wall_seconds=result['wall_seconds'],torch=result['torch'],GPU_cards=0,raw_tensor_transfer_bytes=0,original143_failure_preserved=True,cross_platform_numerical_accepted=False,fresh_recovery_qualified=False,formal_performance=False,Model_GO=False)))
'''
with (D/'independent147-read-intent.json').open('x') as f:json.dump(dict(scope='Exact147 source/receipt/log/accounting independent closure; no repeated tensor reads',expected=expected),f)
r=subprocess.run(ns['ssh'],input=('E='+repr(expected)+'\n'+remote).encode(),capture_output=True,timeout=45)
if r.returncode:
 print(json.dumps(dict(rc=r.returncode,error_tail=r.stderr.decode(errors='replace')[-800:])));raise SystemExit(1)
v=json.loads(r.stdout)
with (D/'independent-original147.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:v[k] for k in ('independent_original147_verified','accounting','parent_snapshot','completed_value_files','source_file_bytes_verified','peak_RSS_KiB','audit_wall_seconds')}))
