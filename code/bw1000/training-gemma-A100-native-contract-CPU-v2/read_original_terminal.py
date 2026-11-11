from pathlib import Path
import base64,hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-gemma-A100-native-contract-CPU-v2';P=D/'package'
def sha(raw):return hashlib.sha256(raw).hexdigest()
assert not (D/'terminal-read-intent.json').exists();t=json.loads((D/'transport-outcome.json').read_bytes());assert t.get('value');(D/'terminal-read-intent.json').write_bytes((json.dumps(dict(request_id=t['request_id'],scope='Original CPU contract result and actual full314 source/binding read only; no replay'))+'\n').encode())
remote='''from pathlib import Path
import base64,datetime,hashlib,json,stat
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-gemma-native-update-contract-CPU-v2');out={}
for name in ['intent.json','outcome.json','CPU-result.json','src/FRAMEWORK_FILES.json','src/SOURCE_FILES.json','GEMMA_NATIVE_BINDING.json']:
 p=W/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<2000000;out[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());checked=[]
for name,pin in fm['files'].items():
 p=Path(fm.get('qualified_unchanged_source_reuse',str(W/'framework')))/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode'];checked.append(name)
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=out,framework_files_verified=len(checked))))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');a=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(a,input=remote.encode(),capture_output=True,timeout=35);(D/'terminal-reader-stdout.private.json').write_bytes(r.stdout);(D/'terminal-reader-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0
out=json.loads(r.stdout);files={}
for name,pin in out['files'].items():
 raw=base64.b64decode(pin['base64']);assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'];files[name]=json.loads(raw)
assert files['outcome.json']==t['value'] and files['intent.json']['request_id']==t['request_id'] and files['intent.json']['claim']==t['claim'];original=files['outcome.json'];cpu=files.get('CPU-result.json');assert cpu==original.get('CPU_result') and out['framework_files_verified']==314
assert out['files']['src/FRAMEWORK_FILES.json']['SHA256']==sha((P/'FRAMEWORK_FILES.json').read_bytes()) and out['files']['src/SOURCE_FILES.json']['SHA256']==sha((P/'SOURCE_FILES.json').read_bytes())
result=dict(passed=bool(original['passed']),independent_original_CPU_result_verified=True,observed_UTC=out['observed_UTC'],request_id=t['request_id'],claim=t['claim'],epoch=original['epoch'],deadline=original['deadline'],framework_files_verified=314,original_receipts={name:{k:v for k,v in pin.items() if k!='base64'} for name,pin in out['files'].items()},CPU_result=cpu,GPU_cards=0,Model_GO=False)
if original['passed']:
 assert original['epoch']==cpu['epoch'] and original['deadline']==cpu['epoch']+300 and cpu['deadline']==cpu['epoch']+260 and cpu['normal_contract']==1 and len(cpu['rejected_boundaries'])==15 and cpu['GPU_initialized'] is False and cpu['model_weight_values_loaded'] is False
 binding=files['GEMMA_NATIVE_BINDING.json'];assert binding==original['binding'] and out['files']['GEMMA_NATIVE_BINDING.json']['SHA256']==cpu['binding_SHA256']==original['binding_SHA256'];raw=base64.b64decode(out['files']['GEMMA_NATIVE_BINDING.json']['base64']);(D/'original-GEMMA_NATIVE_BINDING.json').write_bytes(raw);m=binding['trainable_manifest'];assert sum(x['numel'] for x in m)==cpu['native_trainable_parameters'] and len(m)==cpu['native_trainable_tensors'];result['binding_SHA256']=sha(raw)
(D/'independent-original-CPU-terminal.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(passed=result['passed'],framework_files=314,phase=cpu.get('phase') if cpu else None,error_type=cpu.get('error_type') if cpu else None,message=cpu.get('message') if cpu else None,trainable_parameters=cpu.get('native_trainable_parameters') if cpu else None,trainable_tensors=cpu.get('native_trainable_tensors') if cpu else None,normal_contract=cpu.get('normal_contract') if cpu else None,rejected=len(cpu.get('rejected_boundaries',[])) if cpu else None,wall_seconds=cpu.get('wall_seconds') if cpu else None)))
