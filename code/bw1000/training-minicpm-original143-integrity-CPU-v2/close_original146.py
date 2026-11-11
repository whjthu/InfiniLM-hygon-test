from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).parent;B=D.parent;ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
pins={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('audit.py','ARTIFACT_PLAN.json')}
remote=r'''from pathlib import Path
import datetime,hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original143-integrity-CPU-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,pin in PINS.items():assert sha(W/'source'/name)==pin
v=json.loads((W/'outcome.json').read_bytes());assert v['passed'] and v['returncode']==0 and not v['forced_cleanup']
assert sha(W/'CPU.log')==v['log_SHA256']
for name in ('CPU-result.json','progress.json'):
 p=W/'evidence'/name;assert p.stat().st_size==v[name]['bytes'] and sha(p)==v[name]['SHA256']
result=v['CPU-result.json']['value'];plan=json.loads((W/'source/ARTIFACT_PLAN.json').read_bytes());progress=v['progress.json']['value'];intent=json.loads((W/'intent.json').read_bytes())
assert intent['claim']=='7fe6e6f09b4b6191b0ceaf390fc2c2a4a5607776' and intent['epoch']==v['epoch']==result['epoch'] and intent['deadline']==v['deadline']==v['epoch']+300 and result['deadline']==v['epoch']+240
assert result['passed'] and result['verified_capture_files']==progress['completed_files']==200 and not progress['failed'] and result['verified_capture_bytes']==plan['expected_bytes']==19318791784
assert result['capture_plan_SHA256']==PINS['ARTIFACT_PLAN.json'] and not result['torch_imported'] and result['GPU_cards']==0 and result['checkpoint_expected_files_complete']
assert len(result['baselines'])==8 and {x['rank'] for x in result['baselines']}==set(range(8)) and {x['global_step'] for x in result['baselines']}=={25}
assert [x['step'] for x in result['checkpoint_inventory']]==[10,20,25] and all(x['actual_global_step']==x['step'] and not x['missing'] and len(x['files'])==13 for x in result['checkpoint_inventory'])
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.146','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=10).splitlines() if x.startswith('1087656.146|')];assert rows==v['accounting'] and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'] and result['owner']['step']==rows[0][0]
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_original146_verified=True,source_pins=PINS,outcome_SHA256=sha(W/'outcome.json'),accounting_columns=v['accounting_columns'],accounting=rows,original=v,verified_capture_files=200,verified_capture_bytes=result['verified_capture_bytes'],checkpoint_files_verified=39,rank_RNG_files_verified=24,final_rank_parameter_SHA_unique=len({x['final_trainable_sha256'] for x in result['baselines']}),recovery_qualified=False,numerical_qualified=False,formal_performance=False,original143_failure_preserved=True,Model_GO=False)))
'''
with (D/'independent146-read-intent.json').open('x') as f:json.dump({'scope':'Exact146 source/receipt/log/accounting independent closure; rawfiles not rehashed'},f)
r=subprocess.run(ns['ssh'],input=('PINS='+repr(pins)+'\n'+remote).encode(),capture_output=True,timeout=35);assert r.returncode==0
v=json.loads(r.stdout)
with (D/'independent-original146.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:v[k] for k in ('independent_original146_verified','accounting','verified_capture_files','verified_capture_bytes','checkpoint_files_verified','rank_RNG_files_verified','final_rank_parameter_SHA_unique')}))
