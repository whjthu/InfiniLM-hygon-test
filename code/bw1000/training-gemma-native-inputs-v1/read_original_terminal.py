from pathlib import Path
import ast,base64,hashlib,json,subprocess,sys,time
B=Path(__file__).parent;D=B/'training-gemma-native-inputs-v1';P=D/'package'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
assert not (D/'independent-read-intent.json').exists();dump(D/'independent-read-intent.json',dict(scope='Read exact original source capture/archive and target staging receipts; no replay or new model operation'))
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
remote='''from pathlib import Path
import base64,hashlib,json
out={}
for label,root in [('capture','/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-inputs-v1'),('archive','/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-processor-archive-v1')]:
 out[label]={}
 for name in ['intent.json','outcome.json']:
  raw=(Path(root)/name).read_bytes();assert len(raw)<=2000000;out[label][name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(out))
'''
r=subprocess.run(ssh,input=remote.encode(),capture_output=True,timeout=25);(D/'independent-source-stdout.private.json').write_bytes(r.stdout);(D/'independent-source-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0
original=json.loads(r.stdout);decoded={}
for label,records in original.items():
 decoded[label]={}
 for name,item in records.items():
  raw=base64.b64decode(item['base64']);assert len(raw)==item['bytes'] and sha(raw)==item['SHA256'];decoded[label][name]=json.loads(raw)
localcapture=json.loads((D/'original-CPU-capture.json').read_bytes());localarchive=json.loads((D/'processor-archive-result.json').read_bytes());assert decoded['capture']['outcome.json']==localcapture and decoded['archive']['outcome.json']==localarchive
for label in ('capture','archive'):
 intent=json.loads((D/(label+'-launch-intent.json')).read_bytes());assert decoded[label]['intent.json']['request_id']==intent['request_id'] and decoded[label]['outcome.json']['request_id']==intent['request_id'] and decoded[label]['outcome.json']['passed']
assert localcapture['epoch']==decoded['capture']['intent.json']['epoch'] and localcapture['deadline']==localcapture['epoch']+300
rows=localcapture['input_identities'];assert len(rows)==64 and [x['row_ordinal'] for x in rows]==list(range(1,65))
sys.path.insert(0,str(P/'framework'));from vlm.runtime.template_inputs import compare_input_identities
comparison=compare_input_identities(rows,rows);assert comparison['all_fields_exact'] and comparison['compared_rows']==64
assert all(set(x['fields'])=={'input_ids','labels','pixel_values','token_type_ids'} and 0<x['sequence_length']<=512 for x in rows)
assert localcapture['capture']['model_loaded'] is False and localcapture['capture']['CUDA_initialized'] is False and localcapture['tests']==5 and localcapture['original_input_files_verified']==67
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
remote='''from pathlib import Path
import base64,datetime,hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-processor-assets-v1');out={}
for name in ['intent.json','outcome.json','MODEL_PROCESSOR_FILES.json']:
 raw=(W/name).read_bytes();assert len(raw)<=100000;out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
plan=json.loads((W/'MODEL_PROCESSOR_FILES.json').read_bytes());model=W/'models/gemma-3-4b-it';files={}
for name,pin in plan['files'].items():
 p=model/name;s=p.stat();assert not p.is_symlink() and s.st_size==pin['bytes'];files[name]=dict(bytes=s.st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
assert set(p.name for p in model.iterdir())==set(plan['files'])
print(json.dumps(dict(receipts=out,actual_files=files,parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%l|%L'],text=True,timeout=12).strip(),observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''
r=subprocess.run(ns['ssh'],input=remote.encode(),capture_output=True,timeout=35);(D/'independent-target-stdout.private.json').write_bytes(r.stdout);(D/'independent-target-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0
h=json.loads(r.stdout);target={}
for name,item in h['receipts'].items():
 raw=base64.b64decode(item['base64']);assert len(raw)==item['bytes'] and sha(raw)==item['SHA256'];target[name]=json.loads(raw)
assert target['outcome.json']==json.loads((D/'target-processor-stage.json').read_bytes()) and target['outcome.json']['passed'];assert target['intent.json']['request_id']==json.loads((D/'receive-launch-intent.json').read_bytes())['request_id']
pins=json.loads((P/'MODEL_PROCESSOR_FILES.json').read_bytes());assert h['actual_files']==pins['files'] and target['MODEL_PROCESSOR_FILES.json']==pins
assert target['outcome.json']['archive_SHA256']==localarchive['archive_SHA256'] and target['outcome.json']['archive_bytes']==localarchive['archive_bytes']
result=dict(passed=True,source_capture_request=localcapture['request_id'],archive_request=localarchive['request_id'],target_request=target['outcome.json']['request_id'],source_original_receipts={k:{n:{q:v for q,v in item.items() if q!='base64'} for n,item in rec.items()} for k,rec in original.items()},target_original_receipts={n:{q:v for q,v in item.items() if q!='base64'} for n,item in h['receipts'].items()},rows=64,fields=256,sequence_length_min=min(x['sequence_length'] for x in rows),sequence_length_max=max(x['sequence_length'] for x in rows),files=14,file_bytes=39352681,parent=h['parent'],observed_UTC=h['observed_UTC'],capture_wall_seconds=localcapture['wall_seconds'],target_wall_seconds=target['outcome.json']['wall_seconds'],framework_commit=localcapture['framework_commit'],source_versions=localcapture['capture']['versions'],settings=localcapture['capture']['settings'],identities_SHA256=localcapture['capture']['identities_SHA256'],all_target_files_independently_SHA_verified=True,target_CPU_inputs_executed=False,GPU_cards=0,Model_GO=False,qualification='Original CPU64 pre-model template reference and exact14 target processor assets only; no cross-platform CPU input comparison, model numerical,update,recovery,performance or effects acceptance.')
dump(D/'independent-terminal.json',result);print(json.dumps(result))
