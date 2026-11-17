import json,pathlib,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-native-CPU-v2')
def read(n):
 b=(r/n).read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()
o,oh=read('outcome.json');i,ih=read('intent.json');c,ch=read('source/CLAIM.json');mf,mh=read('source/SOURCE_FILES.json')
assert ch=='9a0dcf49671a4527cb0cc8200fb428c40b77478188137c9c554c829044302af6'
for n,pin in mf['files'].items():
 b=(r/'source'/n).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
raw=subprocess.check_output(['sacct','-n','-P','-j','1087656','-o',o['accounting_columns']],text=True,timeout=9)
rows=[line.split('|') for line in raw.splitlines() if line.split('|')[0]=='1087656.326']
assert rows==[o['accounting']['row']] and rows[0][2:4]==['COMPLETED','0:0']
log=(r/'step.log').read_bytes();assert hashlib.sha256(log).hexdigest()==o['bounded_log_SHA256']
metrics=[json.loads(line) for line in log.splitlines() if line.startswith(b'{') and b'NATIVE_CPU_PREPROCESSING_AND_COLLATION_PASS' in line]
assert metrics==[o['metrics']] and o['passed'] is True
assert o['metrics']['actual_GPU_cards']==0 and o['metrics']['actual_updates']==0 and o['metrics']['rows']==160
assert hashlib.sha256((r/'inputs.jsonl').read_bytes()).hexdigest()==o['metrics']['dataset_SHA256']
print(json.dumps(dict(root_independent_audit_passed=True,actual_accounting=rows[0],original_intent_SHA256=ih,original_outcome_SHA256=oh,source_manifest_SHA256=mh,claim_SHA256=ch,source_inventory_verified=True,log_SHA256=o['bounded_log_SHA256'],epoch=o['epoch'],deadline=o['deadline'],metrics=o['metrics'],formal_training_acceptance=False,actual_new_training_updates=0)))
