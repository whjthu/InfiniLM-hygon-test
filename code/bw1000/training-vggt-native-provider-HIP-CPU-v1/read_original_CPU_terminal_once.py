from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-vggt-native-provider-HIP-CPU-v1';o=json.loads((D/'transport-outcome.json').read_bytes());v=o['value'];task_source='code/bw1000/training-vggt-native-provider-HIP-CPU-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)];t=json.loads(subprocess.check_output(git+['show',o['claim']+':coordination/tasks/training-vggt-native-provider-HIP-CPU-v1.json']));expected={n:t['source_SHA256'][n] for n in t['remote_files']}
code='''from pathlib import Path
import hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vggt-native-provider-HIP-CPU-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
i=json.loads((W/'intent.json').read_bytes());o=json.loads((W/'outcome.json').read_bytes());assert i['claim']==o['claim']==CLAIM and i['request_id']==o['request_id']==REQUEST and i['source_files']==EXPECTED and o['owned_srun_reaped'] and not o['forced_cleanup']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert o['accounting_columns']==columns
rows=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|'+W.name+'|' in line];assert rows==o['accounting'] and len(rows)==1 and rows[0][2] not in ['RUNNING','PENDING','CONFIGURING','COMPLETING']
for n,pin in EXPECTED.items():assert (W/'source'/n).is_file() and sha(W/'source'/n)==pin
fm=json.loads((W/'source/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==321
for n,pin in fm['files'].items():
 p=W/'framework'/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
up=json.loads((W/'source/UPSTREAM_FILES.json').read_bytes());assert len(up['files'])==69
for pin in up['files']:
 p=W/'upstream'/pin['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
raw=(W/'evidence/CPU-result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==o['CPU-result.json']['SHA256'];c=json.loads(raw);assert c==o['CPU-result.json']['value'] and c['owner']['step']==rows[0][0] and c['owner']['node']=='f11r4n19' and c['GPU_cards']==c['model_weight_bytes_read']==c['raw_input_bytes_read']==0
assert sha(W/'CPU.log')==o['log_SHA256'];tests=W/'evidence/tests.private.txt';test_raw=tests.read_bytes() if tests.is_file() else b'';assert len(test_raw)<=20000
value=dict(terminal_verified=True,CPU_passed=o['passed'],original_outcome=o,original_outcome_SHA256=sha(W/'outcome.json'),original_intent_SHA256=sha(W/'intent.json'),fresh_accounting=rows,accounting_columns=columns,framework_files=321,framework_manifest_SHA256=sha(W/'source/FRAMEWORK_FILES.json'),upstream_source_files=69,upstream_source_bytes=552106,upstream_manifest_SHA256=sha(W/'source/UPSTREAM_FILES.json'),tests_log_SHA256=hashlib.sha256(test_raw).hexdigest(),tests_log=test_raw.decode(),model_weight_bytes_read=0,raw_input_bytes_read=0,GPU_cards=0,CPU_replayed=False)
provider=W/'evidence/TARGET_PROVIDER_EVIDENCE.json'
if provider.is_file():value['target_provider_evidence_SHA256']=sha(provider);value['target_provider_evidence']=json.loads(provider.read_bytes())
print(json.dumps(value))
'''
payload=('CLAIM='+repr(o['claim'])+'\nREQUEST='+repr(o['request_id'])+'\nEXPECTED='+repr(expected)+'\n'+code).encode();pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'readreuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'ordinary_cluster_ssh','exec'),ns);r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=45);(D/'independent-reader-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0,r.stderr[-250:];q=json.loads(r.stdout);(D/'independent-original-CPU-terminal.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());c=q['original_outcome']['CPU-result.json']['value'];print(json.dumps(dict(terminal_verified=True,CPU_passed=q['CPU_passed'],accounting=q['fresh_accounting'],phase=c['phase'],error_type=c.get('error_type'),framework_files=321,upstream_source_files=69,tests_log=q['tests_log'][-7000:]),ensure_ascii=False))
