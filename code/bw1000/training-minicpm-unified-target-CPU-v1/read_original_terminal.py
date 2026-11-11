import hashlib,json,os,stat,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-minicpm-unified-target-CPU-v1';F=R/'results/training-framework-bd24ec3-v1'
start=time.monotonic()
def read(p,cap):
 assert p.stat().st_size<=cap
 raw=p.read_bytes();return raw,dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def load(p,cap):
 raw,pin=read(p,cap);return json.loads(raw),pin
out,outpin=load(W/'outcome.json',20000);assert out==EXPECTED
assert out['passed'] and out['returncode']==0 and not out['forced_cleanup'] and out['GPU_cards']==0
assert out['deadline']==out['epoch']+240 and out['wall_seconds']<240
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.128','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if x.strip()]
assert rows==out['accounting'] and rows==[['1087656.128','training-minicpm-unified-target-CPU-v1','COMPLETED','0:0','27','f11r4n09']]
intent,intpin=load(W/'intent.json',12000);owned,ownedpin=load(W/'owned-srun.json',2000)
assert intent['claim']=='dfa28d10f6f3652bd591322f257f170b9fc52513' and intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']
for n,h in HELPERS.items():
 raw,pin=read(W/'helpers'/n,30000);assert pin['SHA256']==h==intent['files'][n]
for n in ('target-CPU-result.json','entry-verification.json','owned-node.json'):
 v,pin=load(W/'evidence'/n,10000);assert pin['SHA256']==out[n]['SHA256'] and pin['bytes']==out[n]['bytes'] and v==out[n]['value']
c=out['target-CPU-result.json']['value'];e=out['entry-verification.json']['value'];o=out['owned-node.json']['value']
assert c['owner']==o and o['step']=='1087656.128' and o['node']=='f11r4n09'
assert c['tests']==[dict(name='test_initial_trainables.py',tests=4,failures=0,errors=0),dict(name='test_hygon_vlm_recipe.py',tests=6,failures=0,errors=0)]
assert c['CUDA_initialized'] is False and c['original_model_loaded'] is False and c['fixture_only'] is True and c['Model_GO'] is False
assert len(e['case_ids'])==len(set(e['case_ids']))==17 and e['entry_returncode']==0 and e['framework_commit']=='bd24ec33a2fde6eb284dc3461822217dabe4cba5' and e['python'].startswith('3.10.12')
log,logpin=read(W/'CPU.log',1000000);assert logpin['SHA256']==out['log_SHA256']
assert b'Ran 4 tests' in log and b'Ran 6 tests' in log
mf,mfpin=load(F/'FRAMEWORK_FILES.json',200000);assert mfpin['SHA256']=='cd831960f81de4cdaf41ceee390ef48bae4e17a60612238391da2c80750ebcda'
assert len(mf['files'])==277
for name,pin in mf['files'].items():
 path=F/name;raw,actual=read(path,5000000)
 assert actual['bytes']==pin['bytes'] and actual['SHA256']==pin['SHA256'] and stat.S_IMODE(path.stat().st_mode)==pin['mode']
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%l|%L|%N'],text=True,timeout=15).strip().split('|')
assert parent[:2]==['RUNNING','40']
nodes=subprocess.check_output(['scontrol','show','hostnames',parent[4]],text=True,timeout=15).splitlines();assert len(nodes)==len(set(nodes))==40
print(json.dumps(dict(passed=True,qualification='Target compute-container CPU entry and initialization only; no training/GPU qualification',original_outcome=outpin,intent=intpin,owned_srun=ownedpin,original_log=logpin,accounting_columns=out['accounting_columns'],accounting=rows,source_files_verified=277,source_manifest=mfpin,actual_parent_resources=intent['actual_parent_resources'],current_parent=dict(state=parent[0],nodes=40,time_limit=parent[2],time_left=parent[3],node_names=nodes),target=c,entry=e,reader_seconds=time.monotonic()-start,original_login_entry_timeout_retained=True,Model_GO=False)))
