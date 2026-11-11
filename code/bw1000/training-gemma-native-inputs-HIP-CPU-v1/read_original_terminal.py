"""Read the one original Gemma CPU terminal and independently compare all rows."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys
B=Path(__file__).parent;D=B/'training-gemma-native-inputs-HIP-CPU-v1';P=D/'package';W='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-native-inputs-HIP-CPU-v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
assert not (D/'original-terminal-read-intent.json').exists();transport=json.loads((D/'transport-outcome.json').read_bytes());assert transport.get('value',{}).get('terminal',{}).get('terminal_visible');dump(D/'original-terminal-read-intent.json',dict(request_id=transport['request_id'],scope='Independent original CPU terminal read and full311-source/all64-field verification; no replay'))
remote='''from pathlib import Path
import base64,datetime,hashlib,json,stat,subprocess
W=Path('''+repr(W)+''');label=W.name;columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|'+label+'|' in x];assert len(rows)==1 and rows[0][2] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL')
files={}
for name in ['intent.json','owned-srun.json','outcome.json','evidence/CPU-result.json','evidence/owner.json','source/SOURCE_FILES.json','source/FRAMEWORK_FILES.json']:
 raw=(W/name).read_bytes();assert len(raw)<=2000000;files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
for name in ['evidence/comparison.json','evidence/encoded/input-identities.json','evidence/encoded/result.json','evidence/encoded/progress.json','evidence/encoded/failure.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=2000000;files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
fm=json.loads((W/'source/FRAMEWORK_FILES.json').read_bytes());framework=[]
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'];framework.append(name)
mf=json.loads((W/'source/SOURCE_FILES.json').read_bytes());sources=[]
for name,pin in mf['files'].items():
 p=W/'source'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'];sources.append(name)
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting_columns=columns,accounting=rows,files=files,source_files_verified=len(sources)+1,framework_files_verified=len(framework),parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%L'],text=True,timeout=12).strip())))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote.encode(),capture_output=True,timeout=45);(D/'original-terminal-reader-stdout.private.json').write_bytes(r.stdout);(D/'original-terminal-reader-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0
out=json.loads(r.stdout);files={}
for name,pin in out['files'].items():
 raw=base64.b64decode(pin['base64']);assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'];files[name]=json.loads(raw)
intent=files['intent.json'];original=files['outcome.json'];value=files['evidence/CPU-result.json'];owner=files['evidence/owner.json'];row=out['accounting'][0];assert original==transport['value'] and intent['request_id']==transport['request_id'] and intent['claim']==transport['claim'] and original['accounting']==out['accounting'];assert original['CPU-result.json']['SHA256']==out['files']['evidence/CPU-result.json']['SHA256'] and original['CPU-result.json']['value']==value
assert intent['epoch']==original['epoch']==value['epoch'] and intent['deadline']==original['deadline']==intent['epoch']+300 and value['deadline']==intent['epoch']+220
assert row[0]==owner['step']==value['owner']['step'] and row[1]=='training-gemma-native-inputs-HIP-CPU-v1' and row[5]==owner['node']=='a14r1n17' and out['framework_files_verified']==311
assert sha((P/'FRAMEWORK_FILES.json').read_bytes())==out['files']['source/FRAMEWORK_FILES.json']['SHA256'] and sha((P/'SOURCE_FILES.json').read_bytes())==out['files']['source/SOURCE_FILES.json']['SHA256']
result=dict(observed_UTC=out['observed_UTC'],independent_original_terminal_verified=True,actual_CPU_step=row[0],accounting_columns=out['accounting_columns'],accounting=out['accounting'],parent=out['parent'],framework_files_verified=311,source_files_verified=out['source_files_verified'],epoch=original['epoch'],deadline=original['deadline'],forced_cleanup=original['forced_cleanup'],owned_srun_reaped=original['owned_srun_reaped'],CPU_result=value,original_receipts={n:{k:v for k,v in p.items() if k!='base64'} for n,p in out['files'].items()},GPU_cards=0,Model_GO=False)
if 'evidence/encoded/input-identities.json' in files:
 sys.path.insert(0,str(B/'hygon-training-integration'));from vlm.runtime.template_inputs import compare_input_identities
 comparison=compare_input_identities(json.loads((P/'ORIGINAL_REFERENCE.json').read_bytes()),files['evidence/encoded/input-identities.json']);assert comparison==files['evidence/comparison.json']==value['comparison'];result['independent_comparison']=comparison
 assert sha(json.dumps(files['evidence/encoded/input-identities.json'],indent=2).encode())==value['target_identities_SHA256']
 result['passed']=bool(original['passed'] and comparison['all_fields_exact'] and row[2:4]==['COMPLETED','0:0'])
else:result['passed']=False
result['qualification']='Actual native CPU preprocessing comparison only,not model,first-update,25step,recovery,performance or effects acceptance.'
dump(D/'independent-original-terminal.json',result);print(json.dumps(dict(passed=result['passed'],CPU_step=row[0],accounting=out['accounting'],rows=result.get('independent_comparison',{}).get('compared_rows'),exact_rows=result.get('independent_comparison',{}).get('exact_rows'),stage=value.get('stage'),error_type=value.get('error_type'),capture_failure=value.get('capture_failure'),framework_files=311,Model_GO=False)))
