from pathlib import Path
import hashlib,json,stat,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-minicpm-first-update-recorder-CPU-v2';started=time.monotonic()
def load(p,cap=200000):
 assert p.stat().st_size<cap
 raw=p.read_bytes();return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
out,outpin=load(W/'outcome.json');assert out==EXPECTED and out['passed'] and out['returncode']==0 and not out['forced_cleanup']
assert out['deadline']==out['epoch']+240 and out['wall_seconds']<240
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.130','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if x.strip()]
assert rows==out['accounting']==[['1087656.130','training-minicpm-first-update-recorder-CPU-v2','COMPLETED','0:0','13','f11r4n09']]
intent,intpin=load(W/'intent.json');assert intent['claim']=='a885a7e612a481836188e9b34bffa3a9e4584b22' and intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']
assert len(intent['files'])==4
for name,h in intent['files'].items():
 p=W/'source'/name;assert p.stat().st_size<1000000 and hashlib.sha256(p.read_bytes()).hexdigest()==h
mf,mfpin=load(W/'source/FRAMEWORK_FILES.json');assert mf['framework_commit']=='668bfe950a74c6c4a61e97a08630da5edc4049be' and len(mf['files'])==282
for name,pin in mf['files'].items():
 p=W/'source'/name;assert p.stat().st_size==pin['bytes']<2000000 and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
for name in ('CPU-result.json','owned-node.json','entry-verification.json'):
 value,pin=load(W/'evidence'/name);assert value==out[name]['value'] and pin['SHA256']==out[name]['SHA256'] and pin['bytes']==out[name]['bytes']
c=out['CPU-result.json']['value'];e=out['entry-verification.json']['value'];owner=out['owned-node.json']['value']
assert c['owner']==owner and owner['step']=='1087656.130'
assert [x['tests'] for x in c['tests']]==[4,7,3,4] and all(x['errors']==x['failures']==0 for x in c['tests'])
assert c['fixture_only'] and c['synthetic_CPU_updates']==25 and c['production_plugin_on_train_begin_invoked'] is False and c['CUDA_initialized'] is False and c['original_model_loaded'] is False and c['Model_GO'] is False
assert e['framework_commit']==mf['framework_commit'] and len(e['case_ids'])==len(set(e['case_ids']))==17 and e['entry_returncode']==0 and e['python'].startswith('3.10.12')
log=W/'CPU.log';assert log.stat().st_size<200000 and hashlib.sha256(log.read_bytes()).hexdigest()==out['log_SHA256']
for count in (4,7,3):assert ('Ran '+str(count)+' tests').encode() in log.read_bytes()
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%l|%L|%N'],text=True,timeout=15).strip().split('|');assert parent[:2]==['RUNNING','40'];nodes=subprocess.check_output(['scontrol','show','hostnames',parent[4]],text=True,timeout=15).splitlines();assert len(nodes)==len(set(nodes))==40
print(json.dumps(dict(passed=True,accounting_columns=out['accounting_columns'],accounting=rows,original_outcome=outpin,intent=intpin,source_files_verified=282,source_manifest=mfpin,framework_commit=mf['framework_commit'],target_CPU=c,target_entry=e,current_parent=dict(parent='1087656',state=parent[0],nodes=40,time_limit=parent[2],time_left=parent[3],node_names=nodes),CPU_log_SHA256=out['log_SHA256'],reader_seconds=time.monotonic()-started,old_CPU129_failure_retained=True,Model_GO=False)))
